import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.sequence_dataset import LabeledCellDataset, collate_labeled, get_label_from_path, list_npy_files
from losses.transformer_losses import compute_ntp_loss
from models.transformer import CausalBackbone, ClsHead, FinetuneModel, NTPHead
from utils.utils import build_scheduler, get_device, load_config, set_seed, setup_logging


CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "config.yaml")
log = logging.getLogger(__name__)


def set_backbone_trainable(model, trainable):
    for p in model.backbone.parameters():
        p.requires_grad = trainable
    if model.ntp_head is not None:
        for p in model.ntp_head.parameters():
            p.requires_grad = trainable


def train_epoch(
    model,
    loader,
    ce_criterion,
    optimizer,
    scheduler,
    device,
    grad_clip_norm,
    aux_ntp_weight,
    ntp_loss_type,
):
    model.train()
    tot_cls, tot_ntp, tot_correct, n = 0.0, 0.0, 0, 0
    for x, y, mask, lengths in loader:
        x, y, mask, lengths = x.to(device), y.to(device), mask.to(device), lengths.to(device)

        optimizer.zero_grad()
        cls_logits, next_pred = model(x, src_key_padding_mask=mask, lengths=lengths)
        cls_loss = ce_criterion(cls_logits, y)
        if next_pred is not None and aux_ntp_weight > 0:
            ntp_loss = compute_ntp_loss(next_pred, x, mask, loss_type=ntp_loss_type)
            loss = cls_loss + aux_ntp_weight * ntp_loss
            tot_ntp += ntp_loss.item() * y.size(0)
        else:
            loss = cls_loss

        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            grad_clip_norm,
        )
        optimizer.step()
        scheduler.step()

        batch_size = y.size(0)
        tot_cls += cls_loss.item() * batch_size
        tot_correct += (cls_logits.argmax(1) == y).sum().item()
        n += batch_size
    return tot_cls / max(1, n), tot_ntp / max(1, n), tot_correct / max(1, n)


@torch.no_grad()
def evaluate(model, loader, ce_criterion, device, aux_ntp_weight, ntp_loss_type):
    model.eval()
    tot_cls, tot_ntp, tot_correct, n = 0.0, 0.0, 0, 0
    for x, y, mask, lengths in loader:
        x, y, mask, lengths = x.to(device), y.to(device), mask.to(device), lengths.to(device)
        cls_logits, next_pred = model(x, src_key_padding_mask=mask, lengths=lengths)
        cls_loss = ce_criterion(cls_logits, y)
        if next_pred is not None and aux_ntp_weight > 0:
            ntp_loss = compute_ntp_loss(next_pred, x, mask, loss_type=ntp_loss_type)
            tot_ntp += ntp_loss.item() * y.size(0)
        batch_size = y.size(0)
        tot_cls += cls_loss.item() * batch_size
        tot_correct += (cls_logits.argmax(1) == y).sum().item()
        n += batch_size
    return tot_cls / max(1, n), tot_ntp / max(1, n), tot_correct / max(1, n)


def build_model(cfg, pretrained_ckpt, num_classes):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    finetune_cfg = cfg["finetune4cellfate"]
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
    cls_head = ClsHead(model_cfg["d_model"], num_classes, dropout=model_cfg["dropout"])

    if finetune_cfg["aux_ntp_weight"] > 0:
        ntp_head = NTPHead(model_cfg["d_model"], data_cfg["input_dim"], dropout=model_cfg["dropout"])
        ntp_head.load_state_dict(pretrained_ckpt["ntp_head_state"])
    else:
        ntp_head = None

    return FinetuneModel(backbone, cls_head, ntp_head)


def main():
    cfg = load_config(CONFIG_PATH)
    setup_logging(cfg["logging"])
    set_seed(cfg["project"]["seed"], cfg["project"]["deterministic"])

    data_cfg = cfg["data"]
    finetune_cfg = cfg["finetune4cellfate"]
    device = get_device(cfg["project"]["device"])
    log.info("[Finetune] Device: %s", device)

    ckpt_path = finetune_cfg["pretrained_path"]
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
    pretrained_ckpt = torch.load(ckpt_path, map_location=device)
    log.info("[Finetune] Loaded pretrained checkpoint: %s", ckpt_path)
    log.info("[Finetune] Pretrain best val NTP: %s", pretrained_ckpt.get("best_val_ntp", "?"))

    all_files = list_npy_files(data_cfg["train_dir"])
    if not all_files:
        raise FileNotFoundError(f"No .npy files found under {data_cfg['train_dir']}")
    all_labels = [
        get_label_from_path(f, data_cfg["label_delimiter"], data_cfg["label_index"])
        for f in all_files
    ]

    counts = Counter(all_labels)
    valid_classes = {k for k, v in counts.items() if v >= finetune_cfg["min_samples_per_class"]}
    paired = [(f, label) for f, label in zip(all_files, all_labels) if label in valid_classes]
    files, labels = map(list, zip(*paired))
    classes = sorted(valid_classes)
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}

    log.info("[Finetune] Raw class distribution:")
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        suffix = "" if label in valid_classes else " removed"
        log.info("  %-20s %5d%s", label, count, suffix)
    log.info("[Finetune] Classes: %d  Samples: %d", len(classes), len(files))

    if finetune_cfg["test_size"] > 0:
        train_files, test_files, train_labels, test_labels = train_test_split(
            files,
            labels,
            test_size=finetune_cfg["test_size"],
            stratify=labels,
            random_state=cfg["project"]["seed"],
        )
    else:
        train_files, train_labels = files, labels
        test_files, test_labels = [], []
        log.info("[Finetune] test_size <= 0, skipping test split.")
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files,
        train_labels,
        test_size=finetune_cfg["val_size"],
        stratify=train_labels,
        random_state=cfg["project"]["seed"],
    )
    log.info(
        "[Finetune] Train: %d  Val: %d  Test: %d",
        len(train_files),
        len(val_files),
        len(test_files),
    )

    weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.array(classes),
        y=np.array(train_labels),
    )
    class_weights = torch.tensor(weights_np, dtype=torch.float).to(device)

    train_ds = LabeledCellDataset(
        train_files,
        train_labels,
        label2idx,
        data_cfg["max_seq_len"],
        data_cfg["normalization_eps"],
    )
    val_ds = LabeledCellDataset(
        val_files,
        val_labels,
        label2idx,
        data_cfg["max_seq_len"],
        data_cfg["normalization_eps"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=finetune_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_labeled,
        num_workers=finetune_cfg["num_workers"],
        pin_memory=finetune_cfg["pin_memory"],
        drop_last=finetune_cfg["drop_last"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=finetune_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_labeled,
        num_workers=finetune_cfg["num_workers"],
        pin_memory=finetune_cfg["pin_memory"],
        drop_last=False,
    )
    test_loader = None
    if test_files:
        test_ds = LabeledCellDataset(
            test_files,
            test_labels,
            label2idx,
            data_cfg["max_seq_len"],
            data_cfg["normalization_eps"],
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=finetune_cfg["batch_size"],
            shuffle=False,
            collate_fn=collate_labeled,
            num_workers=finetune_cfg["num_workers"],
            pin_memory=finetune_cfg["pin_memory"],
            drop_last=False,
        )

    model = build_model(cfg, pretrained_ckpt, len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("[Finetune] Total params: %s", f"{n_params:,}")

    back_params = list(model.backbone.parameters())
    if model.ntp_head is not None:
        back_params += list(model.ntp_head.parameters())
        log.info("[Finetune] Aux NTP enabled: weight=%.4f", finetune_cfg["aux_ntp_weight"])
    head_params = list(model.cls_head.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": back_params, "lr": finetune_cfg["backbone_lr"], "name": "backbone"},
            {"params": head_params, "lr": finetune_cfg["head_lr"], "name": "head"},
        ],
        weight_decay=finetune_cfg["weight_decay"],
    )
    scheduler = build_scheduler(
        optimizer,
        finetune_cfg["epochs"],
        finetune_cfg["warmup_epochs"],
        len(train_loader),
    )
    ce_criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=finetune_cfg["label_smoothing"],
    )

    save_path = finetune_cfg["save_path"]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    freeze_epochs = finetune_cfg["freeze_epochs"]
    if freeze_epochs > 0:
        set_backbone_trainable(model, False)
        log.info("[Finetune] Backbone frozen for first %d epochs", freeze_epochs)

    best_val_acc = 0.0
    patience_count = 0
    for epoch in range(1, finetune_cfg["epochs"] + 1):
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            set_backbone_trainable(model, True)
            log.info("[Finetune] Backbone unfrozen at epoch %d", epoch)

        train_cls, train_ntp, train_acc = train_epoch(
            model,
            train_loader,
            ce_criterion,
            optimizer,
            scheduler,
            device,
            finetune_cfg["grad_clip_norm"],
            finetune_cfg["aux_ntp_weight"],
            finetune_cfg["ntp_loss_type"],
        )
        val_cls, val_ntp, val_acc = evaluate(
            model,
            val_loader,
            ce_criterion,
            device,
            finetune_cfg["aux_ntp_weight"],
            finetune_cfg["ntp_loss_type"],
        )
        lrs = [g["lr"] for g in optimizer.param_groups]
        log.info(
            "[FT %03d/%03d] train cls=%.4f ntp=%.4f acc=%.4f | val cls=%.4f ntp=%.4f acc=%.4f | lr=%.2e/%.2e",
            epoch,
            finetune_cfg["epochs"],
            train_cls,
            train_ntp,
            train_acc,
            val_cls,
            val_ntp,
            val_acc,
            lrs[0],
            lrs[1],
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "label2idx": label2idx,
                    "idx2label": idx2label,
                    "cfg": cfg,
                    "best_val_acc": best_val_acc,
                },
                save_path,
            )
            log.info("New best saved: val_acc=%.4f", best_val_acc)
        else:
            patience_count += 1
            if patience_count >= finetune_cfg["patience"]:
                log.info("[Finetune] Early stop at epoch %d", epoch)
                break

    best_ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    if test_loader is not None:
        test_cls, test_ntp, test_acc = evaluate(
            model,
            test_loader,
            ce_criterion,
            device,
            finetune_cfg["aux_ntp_weight"],
            finetune_cfg["ntp_loss_type"],
        )
        log.info("[Finetune] TEST cls=%.4f ntp=%.4f acc=%.4f", test_cls, test_ntp, test_acc)
    else:
        log.info("[Finetune] TEST skipped because test_size <= 0.")


if __name__ == "__main__":
    main()
