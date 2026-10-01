import logging
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.sequence_dataset import (
    InferenceDataset,
    collate_inference,
    get_label_from_path,
    list_npy_files,
    normalize_array,
)
from models.transformer import CausalBackbone, ClsHead, FinetuneModel, NTPHead
from utils.utils import get_device, load_config, save_predictions_csv, set_seed, setup_logging


CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "config.yaml")
log = logging.getLogger(__name__)


def build_model_from_ckpt(ckpt, num_classes, max_seq_len, device):
    cfg = ckpt["cfg"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    max_len = max_seq_len + model_cfg["max_len_extra"]

    backbone = CausalBackbone(
        input_dim=data_cfg["input_dim"],
        d_model=model_cfg["d_model"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        ffn_dim=model_cfg["ffn_dim"],
        dropout=model_cfg["dropout"],
        max_len=max_len,
    )
    cls_head = ClsHead(model_cfg["d_model"], num_classes, dropout=model_cfg["dropout"])

    state = ckpt["model_state"]
    if any(k.startswith("ntp_head.") for k in state.keys()):
        ntp_head = NTPHead(model_cfg["d_model"], data_cfg["input_dim"], dropout=model_cfg["dropout"])
        log.info("[Inference] Checkpoint contains aux NTP head")
    else:
        ntp_head = None
        log.info("[Inference] Checkpoint has no aux NTP head")

    model = FinetuneModel(backbone, cls_head, ntp_head).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, data_cfg["input_dim"]


def save_channel_attention(attention_dir, file_path, attention, length):
    if not attention_dir:
        return
    out_dir = Path(attention_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(file_path).stem}_channel_attention.npy"
    np.save(out_path, attention[:length].astype(np.float32))


@torch.no_grad()
def save_timepoint_predictions(
    output_dir,
    file_path,
    model,
    gt_label,
    label2idx,
    idx2label,
    device,
    max_seq_len,
    normalization_eps,
):
    if not output_dir:
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(file_path).stem}.csv"

    arr = np.load(file_path).astype(np.float32)
    length = min(arr.shape[0], max_seq_len)
    arr = arr[:length].reshape(length, -1)

    rows = []
    for timepoint in range(1, length + 1):
        prefix = normalize_array(arr[:timepoint], normalization_eps)
        x = torch.from_numpy(prefix).unsqueeze(0).to(device)
        mask = torch.zeros(1, timepoint, dtype=torch.bool, device=device)
        lengths = torch.tensor([timepoint], dtype=torch.long, device=device)

        cls_logits, _ = model(x, src_key_padding_mask=mask, lengths=lengths)
        probs = torch.softmax(cls_logits, dim=-1).squeeze(0)
        pred_idx = int(probs.argmax().item())
        gt_idx = label2idx.get(gt_label)
        rows.append(
            {
                "timepoint": timepoint,
                "pred": idx2label[pred_idx],
                "pred_confidence": float(probs[pred_idx].item()),
                "gt": gt_label,
                "gt_confidence": float(probs[gt_idx].item()) if gt_idx is not None else "",
            }
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timepoint", "pred", "pred_confidence", "gt", "gt_confidence"],
        )
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def run_inference(
    model,
    loader,
    label2idx,
    idx2label,
    device,
    attention_dir=None,
    timepoint_predictions_dir=None,
    max_seq_len=None,
    normalization_eps=None,
):
    model.eval()
    results = []
    for x, y, mask, lengths, raw_labels, file_paths in loader:
        x, mask, lengths = x.to(device), mask.to(device), lengths.to(device)
        cls_logits, _, channel_attn = model(
            x,
            src_key_padding_mask=mask,
            lengths=lengths,
            return_channel_attn=True,
        )
        probs = torch.softmax(cls_logits, dim=-1).cpu().numpy()
        preds = cls_logits.argmax(1).cpu().numpy()
        channel_attn = channel_attn.cpu().numpy()
        lengths_np = lengths.cpu().numpy()

        for i, raw_label in enumerate(raw_labels):
            save_channel_attention(attention_dir, file_paths[i], channel_attn[i], int(lengths_np[i]))
            if timepoint_predictions_dir:
                save_timepoint_predictions(
                    timepoint_predictions_dir,
                    file_paths[i],
                    model,
                    raw_label,
                    label2idx,
                    idx2label,
                    device,
                    max_seq_len,
                    normalization_eps,
                )
            pred = idx2label[int(preds[i])]
            results.append(
                {
                    "file": file_paths[i],
                    "true": raw_label,
                    "pred": pred,
                    "is_known": y[i].item() != -1,
                    "confidence": float(probs[i][preds[i]]),
                    "probs": probs[i],
                }
            )
    return results


def print_report(results):
    known = [r for r in results if r["is_known"]]
    unknown = [r for r in results if not r["is_known"]]

    known_correct = sum(1 for r in known if r["true"] == r["pred"])
    log.info("=" * 75)
    log.info(
        "Known-class accuracy: %d/%d = %.2f%%",
        known_correct,
        len(known),
        known_correct / max(1, len(known)) * 100,
    )

    if known:
        correct_conf = np.mean([r["confidence"] for r in known if r["true"] == r["pred"]])
        wrong_conf = np.mean([r["confidence"] for r in known if r["true"] != r["pred"]])
        correct_conf = 0.0 if np.isnan(correct_conf) else correct_conf
        wrong_conf = 0.0 if np.isnan(wrong_conf) else wrong_conf
        log.info("Mean confidence: correct=%.3f wrong=%.3f", correct_conf, wrong_conf)

    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)
    for r in known:
        per_class_total[r["true"]] += 1
        if r["true"] == r["pred"]:
            per_class_correct[r["true"]] += 1

    classes = sorted(per_class_total.keys())
    log.info("-" * 75)
    log.info("%-22s %7s %7s %8s  %s", "Class", "Correct", "Total", "Acc", "Misclassified as")
    for cls in classes:
        total = per_class_total[cls]
        correct = per_class_correct[cls]
        acc = correct / max(1, total) * 100
        wrong = [r["pred"] for r in known if r["true"] == cls and r["pred"] != cls]
        wrong_text = ", ".join(f"{k}x{v}" for k, v in Counter(wrong).most_common(3)) or "-"
        log.info("%-22s %7d %7d %7.1f%%  %s", cls, correct, total, acc, wrong_text)

    if classes:
        macro = np.mean([per_class_correct[c] / per_class_total[c] for c in classes])
        log.info("-" * 75)
        log.info("Macro-avg accuracy: %.2f%%", macro * 100)

    if unknown:
        log.info("-" * 75)
        log.info("Unknown-class samples: %d", len(unknown))
        for cls, count in Counter(r["pred"] for r in unknown).most_common():
            log.info("  %-22s %5d (%.1f%%)", cls, count, count / len(unknown) * 100)


def main():
    cfg = load_config(CONFIG_PATH)
    task = cfg.get("task")
    if task != "fate":
        raise SystemExit(
            f"[Inference] config task is {task!r}, expected 'fate'. "
            "Set task: fate in config.yaml to run fate inference."
        )
    setup_logging(cfg["logging"])
    set_seed(cfg["project"]["seed"], cfg["project"]["deterministic"])

    data_cfg = cfg["data"]
    inference_cfg = cfg["inference4cellfate"]
    device = get_device(cfg["project"]["device"])
    log.info("[Inference] Device: %s", device)

    ckpt = torch.load(inference_cfg["checkpoint_path"], map_location=device)
    label2idx = ckpt["label2idx"]
    idx2label = ckpt["idx2label"]
    num_classes = len(label2idx)
    train_cfg = ckpt["cfg"]
    max_seq_len = inference_cfg["max_seq_len"] or train_cfg["data"]["max_seq_len"]

    log.info(
        "[Inference] Loaded checkpoint epoch=%s best_val_acc=%s",
        ckpt.get("epoch", "?"),
        ckpt.get("best_val_acc", "?"),
    )
    log.info("[Inference] max_seq_len=%s classes=%d", max_seq_len, num_classes)

    test_dir = inference_cfg["test_dir"]
    all_files = list_npy_files(test_dir)
    if not all_files:
        raise FileNotFoundError(f"No .npy files found under {test_dir}")
    labels = [
        get_label_from_path(f, data_cfg["label_delimiter"], data_cfg["label_index"])
        for f in all_files
    ]
    log.info("[Inference] Test files: %d from %s", len(all_files), test_dir)
    for label, count in Counter(labels).most_common():
        status = "known" if label in label2idx else "unknown"
        log.info("  %-22s %5d %s", label, count, status)

    dataset = InferenceDataset(
        all_files,
        labels,
        label2idx,
        max_seq_len,
        data_cfg["normalization_eps"],
    )
    loader = DataLoader(
        dataset,
        batch_size=inference_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_inference,
        num_workers=inference_cfg["num_workers"],
        pin_memory=inference_cfg["pin_memory"],
        drop_last=False,
    )

    model, input_dim = build_model_from_ckpt(ckpt, num_classes, max_seq_len, device)
    sample = np.load(all_files[0]).astype(np.float32).reshape(np.load(all_files[0]).shape[0], -1)
    if sample.shape[-1] != input_dim:
        log.warning("[Inference] input_dim mismatch: ckpt=%d test_file=%d", input_dim, sample.shape[-1])

    results = run_inference(
        model,
        loader,
        label2idx,
        idx2label,
        device,
        inference_cfg.get("attention_output_dir"),
        inference_cfg.get("timepoint_predictions_output_dir")
        if inference_cfg.get("timepoint_predictions_enabled", False)
        else None,
        max_seq_len,
        data_cfg["normalization_eps"],
    )
    print_report(results)
    save_predictions_csv(
        inference_cfg["output_csv"],
        results,
        idx2label,
        inference_cfg["top_k"],
    )
    log.info("[Inference] Predictions saved to: %s", inference_cfg["output_csv"])


if __name__ == "__main__":
    main()
