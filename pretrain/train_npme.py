import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.voxel_dataset import NpyDataset
from losses.npme_losses import HybridLoss, to_one_hot
from models.npme import ResNetAutoEncoder
from utils.utils import load_config


CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "config.yaml")


def _strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_training_checkpoint(model, optimizer, scheduler, scaler, device, local_rank, resume_path):
    if not resume_path:
        return 0, 0, 999.0

    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"npme.resume_checkpoint does not exist: {resume_path}")

    checkpoint = torch.load(resume_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model_state = checkpoint["state_dict"]
    else:
        model_state = checkpoint

    model.module.load_state_dict(_strip_module_prefix(model_state))

    iter_idx = 0
    epoch = 0
    best_dice = 999.0
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        iter_idx = int(checkpoint.get("iter_idx", 0))
        epoch = int(checkpoint.get("epoch", 0))
        best_dice = float(checkpoint.get("best_dice", 999.0))

    if local_rank == 0:
        print(
            f"Resumed checkpoint from {resume_path}. "
            f"iter_idx={iter_idx}, epoch={epoch}, best_dice={best_dice:.4f}"
        )

    return iter_idx, epoch, best_dice


def save_training_checkpoint(model, optimizer, scheduler, scaler, iter_idx, epoch, best_dice, path):
    torch.save(
        {
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "iter_idx": iter_idx,
            "epoch": epoch,
            "best_dice": best_dice,
        },
        path,
    )


def setup():
    dist.init_process_group(backend="nccl")
    torch.backends.cudnn.benchmark = True


def cleanup():
    dist.destroy_process_group()


@torch.no_grad()
def evaluate(
    model,
    val_dataset,
    criterion,
    device,
    iter_idx,
    num_samples,
    batch_size,
    num_workers,
    pin_memory,
    num_classes_seg,
):
    model.eval()
    sample_count = min(num_samples, len(val_dataset))
    indices = np.random.choice(len(val_dataset), sample_count, replace=False)
    subset = torch.utils.data.Subset(val_dataset, indices)

    val_loader = DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    total_loss = 0.0
    total_dice = 0.0
    count = 0

    for batch_data, geometry in val_loader:
        inputs = batch_data.to(device, non_blocking=True)
        geometry = geometry.to(device, non_blocking=True).float()
        inputs_oh = to_one_hot(inputs, num_classes_seg)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            recon, _, geo_pred = model(inputs_oh, geometry)
            loss, dice, _, _ = criterion(recon, inputs, iter_idx, geo_pred, geometry)

        total_loss += loss.item() * inputs.size(0)
        total_dice += dice.item() * inputs.size(0)
        count += inputs.size(0)

    model.train()
    return total_loss / count, total_dice / count


def train():
    cfg = load_config(CONFIG_PATH)
    np = cfg["npme"]

    num_classes_seg = np["num_classes_seg"]
    input_shape = tuple(np["input_shape"])
    latent_spatial_shape = tuple(np["latent_spatial_shape"])
    geometry_conditioner_hidden_dims = tuple(np["geometry_conditioner_hidden_dims"])
    geometry_readout_output_dims = tuple(np["geometry_readout_output_dims"])

    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if local_rank == 0:
        print("Master process is scanning directories...")

    train_files = sorted(glob.glob(os.path.join(np["train_dir"], "*.npy")))
    val_files = sorted(glob.glob(os.path.join(np["val_dir"], "*.npy")))

    if local_rank == 0:
        print(f"Scan complete. Train files: {len(train_files)}, Val files: {len(val_files)}")

    model = ResNetAutoEncoder(
        num_classes=num_classes_seg,
        base_ch=np["cnn_base_channels"],
        latent_channels=np["latent_channels"],
        input_shape=input_shape,
        geometry_dim=np["geometry_dim"],
        latent_spatial_shape=latent_spatial_shape,
        geometry_conditioner_hidden_dims=geometry_conditioner_hidden_dims,
        geometry_conditioner_channels=np["geometry_conditioner_channels"],
        geometry_conditioner_dropout=np["geometry_conditioner_dropout"],
        geometry_readout_output_dims=geometry_readout_output_dims,
        geometry_readout_dropout=np["geometry_readout_dropout"],
    ).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    train_dataset = NpyDataset(
        train_files,
        target_shape=input_shape,
        voxel_size=np["voxel_size"],
        surface_level=np["surface_level"],
        boundary_step_size=np["boundary_step_size"],
    )
    val_dataset = NpyDataset(
        val_files,
        target_shape=input_shape,
        voxel_size=np["voxel_size"],
        surface_level=np["surface_level"],
        boundary_step_size=np["boundary_step_size"],
    )

    sampler = DistributedSampler(train_dataset, shuffle=True)
    dataloader = DataLoader(
        train_dataset,
        batch_size=np["batch_size_per_gpu"],
        sampler=sampler,
        num_workers=np["train_num_workers"],
        pin_memory=np["pin_memory"],
        prefetch_factor=np["prefetch_factor"],
        persistent_workers=np["persistent_workers"],
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=np["lr_max"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=np["lr_max"],
        total_steps=np["max_iters"],
        pct_start=np["onecycle_pct_start"],
    )
    scaler = GradScaler()
    criterion = HybridLoss(
        num_classes_seg,
        np["grad_loss_start_iter"],
        np["geo_loss_weight"],
    ).to(device)

    eval_batch_size = np["batch_size_per_gpu"]
    eval_num_workers = np["eval_num_workers"]

    save_dir = np["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)
    iter_idx, epoch, best_dice = load_training_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        device,
        local_rank,
        np.get("resume_checkpoint", ""),
    )

    while iter_idx < np["max_iters"]:
        sampler.set_epoch(epoch)

        for batch_data, geometry in dataloader:
            if iter_idx >= np["max_iters"]:
                break

            inputs = batch_data.to(device, non_blocking=True)
            geometry = geometry.to(device, non_blocking=True).float()
            inputs_oh = to_one_hot(inputs, num_classes_seg)

            model.train()
            optimizer.zero_grad()

            with autocast(dtype=torch.bfloat16):
                recon, z, geo_pred = model(inputs_oh, geometry)
                loss, dice, grad_l, geo_l = criterion(
                    recon, inputs, iter_idx, geo_pred, geometry
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if local_rank == 0:
                if iter_idx % np["log_interval"] == 0:
                    print(
                        f"[Iter {iter_idx}] Loss: {loss.item():.4f} | "
                        f"Dice: {dice.item():.4f} | Geo: {geo_l.item():.4f} | "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                    )

                if iter_idx > 0 and iter_idx % np["eval_interval"] == 0:
                    val_loss, val_dice = evaluate(
                        model,
                        val_dataset,
                        criterion,
                        device,
                        iter_idx,
                        np["eval_num_samples"],
                        eval_batch_size,
                        eval_num_workers,
                        np["pin_memory"],
                        num_classes_seg,
                    )
                    print(
                        f"--- [EVALUATION] Iter {iter_idx}: "
                        f"Avg Loss: {val_loss:.4f}, Avg Dice: {val_dice:.4f} ---"
                    )

                    torch.save(model.module.state_dict(), os.path.join(save_dir, "model_latest.pth"))
                    save_training_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        iter_idx + 1,
                        epoch,
                        best_dice,
                        os.path.join(save_dir, "checkpoint_latest.pth"),
                    )
                    if val_dice < best_dice:
                        best_dice = val_dice
                        torch.save(model.module.state_dict(), os.path.join(save_dir, "model_best.pth"))
                        save_training_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            iter_idx + 1,
                            epoch,
                            best_dice,
                            os.path.join(save_dir, "checkpoint_best.pth"),
                        )
                        print(f"New Best Dice achieved: {best_dice:.4f}, model saved.")

            iter_idx += 1

        epoch += 1

    if local_rank == 0:
        torch.save(model.module.state_dict(), os.path.join(save_dir, "model_final.pth"))
        save_training_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            iter_idx,
            epoch,
            best_dice,
            os.path.join(save_dir, "checkpoint_final.pth"),
        )
        print("Training Finished. Final model saved.")

    cleanup()


if __name__ == "__main__":
    train()
