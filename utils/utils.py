import csv
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml


log = logging.getLogger(__name__)


def setup_logging(log_cfg):
    logging.basicConfig(
        level=getattr(logging, log_cfg["level"].upper()),
        format=log_cfg["format"],
        datefmt=log_cfg["datefmt"],
    )


def _deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, dict) and cfg.get("inherits"):
        base_path = path.parent / cfg["inherits"]
        with open(base_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_scheduler(optimizer, epochs, warmup_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def encode_regression_target(y, use_log_target):
    return torch.log1p(y) if use_log_target else y


def decode_regression_prediction(y_pred, use_log_target):
    return torch.expm1(y_pred).clamp(min=0.0) if use_log_target else y_pred.clamp(min=0.0)


def compute_regression_metrics(pred, target, tolerances):
    err = pred - target
    mae = err.abs().mean().item()
    rmse = torch.sqrt((err ** 2).mean()).item()
    within = {}
    for tolerance in tolerances:
        within[f"within_{int(tolerance)}"] = (err.abs() <= float(tolerance)).float().mean().item()
    return mae, rmse, within


def format_within(within):
    keys = sorted(within.keys(), key=lambda x: int(x.split("_")[-1]))
    return " ".join(f"{key}={within[key]:.4f}" for key in keys)


def dataloader_kwargs(loader_cfg, shuffle):
    return {
        "batch_size": loader_cfg["batch_size"],
        "shuffle": shuffle,
        "num_workers": loader_cfg["num_workers"],
        "pin_memory": loader_cfg["pin_memory"],
        "drop_last": loader_cfg["drop_last"],
    }


def save_predictions_csv(path, results, idx2label, top_k):
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    classes = [idx2label[i] for i in sorted(idx2label)]
    fieldnames = ["file", "true", "pred", "is_known", "confidence"]
    fieldnames += [f"top{i + 1}_label" for i in range(top_k)]
    fieldnames += [f"top{i + 1}_prob" for i in range(top_k)]
    fieldnames += [f"prob_{cls}" for cls in classes]

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            probs = r["probs"]
            top_idx = np.argsort(probs)[::-1][:top_k]
            row = {
                "file": r["file"],
                "true": r["true"],
                "pred": r["pred"],
                "is_known": r["is_known"],
                "confidence": r["confidence"],
            }
            for i, idx in enumerate(top_idx):
                row[f"top{i + 1}_label"] = idx2label[int(idx)]
                row[f"top{i + 1}_prob"] = float(probs[idx])
            for i, cls in enumerate(classes):
                row[f"prob_{cls}"] = float(probs[i])
            writer.writerow(row)
