import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.sequence_dataset import TimeToEndDataset, collate_regression, list_npy_files
from losses.transformer_losses import compute_ntp_loss
from models.transformer import (
    CausalBackbone,
    FinetuneRegressionModel,
    NTPHead,
    RegressionHead,
)
from utils.utils import (
    build_scheduler,
    compute_regression_metrics,
    decode_regression_prediction,
    encode_regression_target,
    format_within,
    get_device,
    load_config,
    set_seed,
    setup_logging,
)


CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "config.yaml")
log = logging.getLogger(__name__)


def set_backbone_trainable(model, trainable):
    for p in model.backbone.parameters():
        p.requires_grad = trainable
    if model.ntp_head is not None:
        for p in model.ntp_head.parameters():
            p.requires_grad = trainable


def build_regression_loss(loss_type, huber_delta):
    if loss_type == "mae":
        return nn.L1Loss()
    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type == "huber":
        return nn.SmoothL1Loss(beta=huber_delta)
    raise ValueError(f"Unknown regression_loss: {loss_type}")


def train_epoch(
    model,
    loader,
    reg_criterion,
    optimizer,
    scheduler,
    device,
    cfg,
):
    model.train()
    total_reg, total_ntp, n = 0.0, 0.0, 0
    all_pred, all_target = [], []

    for x, y, mask, lengths, _ in loader:
        x, y, mask, lengths = x.to(device), y.to(device), mask.to(device), lengths.to(device)
        y_train = encode_regression_target(y, cfg["use_log_target"])

        optimizer.zero_grad()
        pred_train, next_pred = model(x, src_key_padding_mask=mask, lengths=lengths)
        reg_loss = reg_criterion(pred_train, y_train)
        if next_pred is not None and cfg["aux_ntp_weight"] > 0:
            ntp_loss = compute_ntp_loss(next_pred, x, mask, loss_type=cfg["ntp_loss_type"])
            loss = reg_loss + cfg["aux_ntp_weight"] * ntp_loss
            total_ntp += ntp_loss.item() * y.size(0)
        else:
            loss = reg_loss

        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            cfg["grad_clip_norm"],
        )
        optimizer.step()
        scheduler.step()

        batch_size = y.size(0)
        total_reg += reg_loss.item() * batch_size
        n += batch_size
        all_pred.append(decode_regression_prediction(pred_train.detach(), cfg["use_log_target"]).cpu())
        all_target.append(y.detach().cpu())

    pred = torch.cat(all_pred)
    target = torch.cat(all_target)
    mae, rmse, within = compute_regression_metrics(pred, target, cfg["tolerances"])
    return total_reg / max(1, n), total_ntp / max(1, n), mae, rmse, within


@torch.no_grad()
def evaluate(model, loader, reg_criterion, device, cfg):
    model.eval()
    total_reg, total_ntp, n = 0.0, 0.0, 0
    all_pred, all_target = [], []

    for x, y, mask, lengths, _ in loader:
        x, y, mask, lengths = x.to(device), y.to(device), mask.to(device), lengths.to(device)
        y_eval = encode_regression_target(y, cfg["use_log_target"])

        pred_eval, next_pred = model(x, src_key_padding_mask=mask, lengths=lengths)
        reg_loss = reg_criterion(pred_eval, y_eval)
        if next_pred is not None and cfg["aux_ntp_weight"] > 0:
            ntp_loss = compute_ntp_loss(next_pred, x, mask, loss_type=cfg["ntp_loss_type"])
            total_ntp += ntp_loss.item() * y.size(0)

        batch_size = y.size(0)
        total_reg += reg_loss.item() * batch_size
        n += batch_size
        all_pred.append(decode_regression_prediction(pred_eval, cfg["use_log_target"]).cpu())
        all_target.append(y.cpu())

    pred = torch.cat(all_pred)
    target = torch.cat(all_target)
    mae, rmse, within = compute_regression_metrics(pred, target, cfg["tolerances"])
    return total_reg / max(1, n), total_ntp / max(1, n), mae, rmse, within


def build_model(cfg, pretrained_ckpt):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    cycle_cfg = cfg["finetune4cellcycle"]
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
    backbone.load_state_dict(pretrained_ckpt["backbone_state"])
    reg_head = RegressionHead(model_cfg["d_model"], dropout=model_cfg["dropout"])

    if cycle_cfg["aux_ntp_weight"] > 0:
        ntp_head = NTPHead(model_cfg["d_model"], data_cfg["input_dim"], dropout=model_cfg["dropout"])
        ntp_head.load_state_dict(pretrained_ckpt["ntp_head_state"])
    else:
        ntp_head = None

    return FinetuneRegressionModel(backbone, reg_head, ntp_head)


def make_dataset(files, data_cfg, cycle_cfg, sampling_mode):
    return TimeToEndDataset(
        files,
        max_seq_len=data_cfg["max_seq_len"],
        normalization_eps=data_cfg["normalization_eps"],
        min_obs_steps=cycle_cfg["min_obs_steps"],
        min_obs_ratio=cycle_cfg["min_obs_ratio"],
        max_obs_ratio=cycle_cfg["max_obs_ratio"],
        sampling_mode=sampling_mode,
        fixed_obs_ratio=cycle_cfg["eval_obs_ratio"],
    )


def make_loader(dataset, cycle_cfg, shuffle):
    return DataLoader(
        dataset,
        batch_size=cycle_cfg["batch_size"],
        shuffle=shuffle,
        collate_fn=collate_regression,
        num_workers=cycle_cfg["num_workers"],
        pin_memory=cycle_cfg["pin_memory"],
        drop_last=cycle_cfg["drop_last"] if shuffle else False,
    )


def main():
    cfg = load_config(CONFIG_PATH)
    setup_logging(cfg["logging"])
    set_seed(cfg["project"]["seed"], cfg["project"]["deterministic"])

    data_cfg = cfg["data"]
    cycle_cfg = cfg["finetune4cellcycle"]
    device = get_device(cfg["project"]["device"])
    log.info("[CellCycle] Device: %s", device)

    ckpt_path = cycle_cfg["pretrained_path"]
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
    pretrained_ckpt = torch.load(ckpt_path, map_location=device)
    log.info("[CellCycle] Loaded pretrained checkpoint: %s", ckpt_path)

    all_files = list_npy_files(data_cfg["cellcycle_train_dir"])
    if not all_files:
        raise FileNotFoundError(f"No .npy files found under {data_cfg['cellcycle_train_dir']}")

    if cycle_cfg["test_size"] > 0:
        train_files, test_files = train_test_split(
            all_files,
            test_size=cycle_cfg["test_size"],
            random_state=cfg["project"]["seed"],
        )
    else:
        train_files, test_files = all_files, []
        log.info("[CellCycle] test_size <= 0, skipping test split.")
    train_files, val_files = train_test_split(
        train_files,
        test_size=cycle_cfg["val_size"],
        random_state=cfg["project"]["seed"],
    )
    log.info("[CellCycle] Train: %d  Val: %d  Test: %d", len(train_files), len(val_files), len(test_files))

    train_loader = make_loader(make_dataset(train_files, data_cfg, cycle_cfg, "random"), cycle_cfg, True)
    val_loader = make_loader(make_dataset(val_files, data_cfg, cycle_cfg, "fixed"), cycle_cfg, False)
    test_loader = None
    if test_files:
        test_loader = make_loader(make_dataset(test_files, data_cfg, cycle_cfg, "fixed"), cycle_cfg, False)

    model = build_model(cfg, pretrained_ckpt).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("[CellCycle] Total params: %s", f"{n_params:,}")

    back_params = list(model.backbone.parameters())
    if model.ntp_head is not None:
        back_params += list(model.ntp_head.parameters())
        log.info("[CellCycle] Aux NTP enabled: weight=%.4f", cycle_cfg["aux_ntp_weight"])
    head_params = list(model.reg_head.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": back_params, "lr": cycle_cfg["backbone_lr"]},
            {"params": head_params, "lr": cycle_cfg["head_lr"]},
        ],
        weight_decay=cycle_cfg["weight_decay"],
    )
    scheduler = build_scheduler(
        optimizer,
        cycle_cfg["epochs"],
        cycle_cfg["warmup_epochs"],
        len(train_loader),
    )
    reg_criterion = build_regression_loss(cycle_cfg["regression_loss"], cycle_cfg["huber_delta"])

    save_path = cycle_cfg["save_path"]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    freeze_epochs = cycle_cfg["freeze_epochs"]
    if freeze_epochs > 0:
        set_backbone_trainable(model, False)
        log.info("[CellCycle] Backbone frozen for first %d epochs", freeze_epochs)

    best_val_mae = float("inf")
    patience_count = 0
    for epoch in range(1, cycle_cfg["epochs"] + 1):
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            set_backbone_trainable(model, True)
            log.info("[CellCycle] Backbone unfrozen at epoch %d", epoch)

        tr_reg, tr_ntp, tr_mae, tr_rmse, tr_within = train_epoch(
            model, train_loader, reg_criterion, optimizer, scheduler, device, cycle_cfg
        )
        va_reg, va_ntp, va_mae, va_rmse, va_within = evaluate(
            model, val_loader, reg_criterion, device, cycle_cfg
        )
        lrs = [g["lr"] for g in optimizer.param_groups]
        log.info(
            "[CC %03d/%03d] train reg=%.4f ntp=%.4f mae=%.4f rmse=%.4f %s | "
            "val reg=%.4f ntp=%.4f mae=%.4f rmse=%.4f %s | lr=%.2e/%.2e",
            epoch,
            cycle_cfg["epochs"],
            tr_reg,
            tr_ntp,
            tr_mae,
            tr_rmse,
            format_within(tr_within),
            va_reg,
            va_ntp,
            va_mae,
            va_rmse,
            format_within(va_within),
            lrs[0],
            lrs[1],
        )

        if va_mae < best_val_mae:
            best_val_mae = va_mae
            patience_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "cfg": cfg,
                    "best_val_mae": best_val_mae,
                    "task": "cellcycle_time_to_end_regression",
                },
                save_path,
            )
            log.info("New best saved: val_mae=%.4f", best_val_mae)
        else:
            patience_count += 1
            if patience_count >= cycle_cfg["patience"]:
                log.info("[CellCycle] Early stop at epoch %d", epoch)
                break

    best_ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    if test_loader is not None:
        te_reg, te_ntp, te_mae, te_rmse, te_within = evaluate(
            model, test_loader, reg_criterion, device, cycle_cfg
        )
        log.info(
            "[CellCycle] TEST reg=%.4f ntp=%.4f mae=%.4f rmse=%.4f %s",
            te_reg,
            te_ntp,
            te_mae,
            te_rmse,
            format_within(te_within),
        )
    else:
        log.info("[CellCycle] TEST skipped because test_size <= 0.")


if __name__ == "__main__":
    main()
