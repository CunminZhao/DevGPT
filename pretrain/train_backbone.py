import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.sequence_dataset import UnlabeledCellDataset, collate_unlabeled, list_npy_files
from losses.transformer_losses import compute_ntp_loss
from models.transformer import CausalBackbone, NTPHead, PretrainModel
from utils.utils import build_scheduler, get_device, load_config, set_seed, setup_logging


CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "config.yaml")
log = logging.getLogger(__name__)


def train_epoch(model, loader, optimizer, scheduler, device, loss_type, grad_clip_norm):
    model.train()
    total, n = 0.0, 0
    for x, mask in loader:
        x, mask = x.to(device), mask.to(device)

        optimizer.zero_grad()
        next_pred = model(x, src_key_padding_mask=mask)
        loss = compute_ntp_loss(next_pred, x, mask, loss_type=loss_type)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()

        batch_size = x.size(0)
        total += loss.item() * batch_size
        n += batch_size
    return total / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, loss_type):
    model.eval()
    total, n = 0.0, 0
    for x, mask in loader:
        x, mask = x.to(device), mask.to(device)
        next_pred = model(x, src_key_padding_mask=mask)
        loss = compute_ntp_loss(next_pred, x, mask, loss_type=loss_type)
        batch_size = x.size(0)
        total += loss.item() * batch_size
        n += batch_size
    return total / max(1, n)


def build_model(cfg):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    max_len = data_cfg["max_seq_len"] + model_cfg["max_len_extra"]
    backbone = CausalBackbone(
        input_dim=data_cfg["input_dim"],
        d_model=model_cfg["d_model"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        ffn_dim=model_cfg["ffn_dim"],
        dropout=model_cfg["dropout"],
        max_len=max_len,
    )
    ntp_head = NTPHead(
        model_cfg["d_model"],
        data_cfg["input_dim"],
        dropout=model_cfg["dropout"],
    )
    return PretrainModel(backbone, ntp_head)


def main():
    cfg = load_config(CONFIG_PATH)
    setup_logging(cfg["logging"])
    set_seed(cfg["project"]["seed"], cfg["project"]["deterministic"])

    data_cfg = cfg["data"]
    pre_cfg = cfg["pretrain"]
    device = get_device(cfg["project"]["device"])
    log.info("[Pretrain] Device: %s", device)

    all_files = list_npy_files(data_cfg["train_dir"])
    if not all_files:
        raise FileNotFoundError(f"No .npy files found under {data_cfg['train_dir']}")
    log.info("[Pretrain] Total files: %d", len(all_files))

    train_files, val_files = train_test_split(
        all_files,
        test_size=pre_cfg["val_size"],
        random_state=cfg["project"]["seed"],
    )
    log.info("[Pretrain] Train: %d  Val: %d", len(train_files), len(val_files))

    train_ds = UnlabeledCellDataset(
        train_files,
        data_cfg["max_seq_len"],
        data_cfg["normalization_eps"],
    )
    val_ds = UnlabeledCellDataset(
        val_files,
        data_cfg["max_seq_len"],
        data_cfg["normalization_eps"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=pre_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_unlabeled,
        num_workers=pre_cfg["num_workers"],
        pin_memory=pre_cfg["pin_memory"],
        drop_last=pre_cfg["drop_last"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=pre_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_unlabeled,
        num_workers=pre_cfg["num_workers"],
        pin_memory=pre_cfg["pin_memory"],
        drop_last=False,
    )

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("[Pretrain] Params: %s", f"{n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=pre_cfg["lr"],
        weight_decay=pre_cfg["weight_decay"],
    )
    scheduler = build_scheduler(
        optimizer,
        pre_cfg["epochs"],
        pre_cfg["warmup_epochs"],
        len(train_loader),
    )

    save_path = pre_cfg["save_path"]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    best_val = float("inf")
    patience_count = 0
    for epoch in range(1, pre_cfg["epochs"] + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            pre_cfg["ntp_loss_type"],
            pre_cfg["grad_clip_norm"],
        )
        val_loss = evaluate(model, val_loader, device, pre_cfg["ntp_loss_type"])
        lr_now = optimizer.param_groups[0]["lr"]
        log.info(
            "[Pretrain %03d/%03d] train_ntp=%.5f val_ntp=%.5f lr=%.2e",
            epoch,
            pre_cfg["epochs"],
            train_loss,
            val_loss,
            lr_now,
        )

        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state": model.backbone.state_dict(),
                    "ntp_head_state": model.ntp_head.state_dict(),
                    "cfg": cfg,
                    "best_val_ntp": best_val,
                },
                save_path,
            )
            log.info("New best saved: val_ntp=%.5f", best_val)
        else:
            patience_count += 1
            if patience_count >= pre_cfg["patience"]:
                log.info("[Pretrain] Early stop at epoch %d", epoch)
                break

    log.info("[Pretrain] Best val NTP: %.5f", best_val)
    log.info("[Pretrain] Saved to: %s", save_path)


if __name__ == "__main__":
    main()
