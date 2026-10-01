import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.sequence_dataset import CellCycleInferenceDataset, collate_cellcycle_inference, list_npy_files
from models.transformer import CausalBackbone, FinetuneRegressionModel, NTPHead, RegressionHead
from utils.utils import decode_regression_prediction, get_device, load_config, set_seed, setup_logging


CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "config.yaml")
log = logging.getLogger(__name__)


def build_model_from_ckpt(ckpt, max_seq_len, device):
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
    reg_head = RegressionHead(model_cfg["d_model"], dropout=model_cfg["dropout"])
    state = ckpt["model_state"]
    if any(k.startswith("ntp_head.") for k in state.keys()):
        ntp_head = NTPHead(model_cfg["d_model"], data_cfg["input_dim"], dropout=model_cfg["dropout"])
    else:
        ntp_head = None

    model = FinetuneRegressionModel(backbone, reg_head, ntp_head).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, data_cfg["input_dim"]


def save_channel_attention(attention_dir, file_name, attention, length):
    if not attention_dir:
        return
    out_dir = Path(attention_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(file_name).stem}_channel_attention.npy"
    np.save(out_path, attention[:length].astype(np.float32))


@torch.no_grad()
def run_inference(model, loader, device, use_log_target, attention_dir=None):
    model.eval()
    rows = []
    for x, mask, obs_lens, total_lens, remain_gts, names in loader:
        x, mask, obs_lens = x.to(device), mask.to(device), obs_lens.to(device)
        pred_raw, _, channel_attn = model(
            x,
            src_key_padding_mask=mask,
            lengths=obs_lens,
            return_channel_attn=True,
        )
        pred_remain = decode_regression_prediction(pred_raw, use_log_target).cpu().numpy()
        channel_attn = channel_attn.cpu().numpy()
        obs_lens_np = obs_lens.cpu().numpy()

        for i, name in enumerate(names):
            save_channel_attention(attention_dir, name, channel_attn[i], int(obs_lens_np[i]))
            gt_remain = float(remain_gts[i].item())
            pred = float(pred_remain[i])
            rows.append(
                {
                    "file_name": name,
                    "gt_total_len": int(total_lens[i].item()),
                    "input_len": int(obs_lens[i].item()),
                    "gt_remain_len": gt_remain,
                    "pred_remain_len": pred,
                    "diff": pred - gt_remain,
                }
            )
    return rows


def write_csv(rows, output_csv):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_name",
        "gt_total_len",
        "input_len",
        "gt_remain_len",
        "pred_remain_len",
        "diff",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "file_name": row["file_name"],
                    "gt_total_len": row["gt_total_len"],
                    "input_len": row["input_len"],
                    "gt_remain_len": f"{row['gt_remain_len']:.6f}",
                    "pred_remain_len": f"{row['pred_remain_len']:.6f}",
                    "diff": f"{row['diff']:.6f}",
                }
            )


def print_stats(rows):
    if not rows:
        log.warning("[CellCycleInference] No rows to summarize")
        return
    gt = np.array([r["gt_remain_len"] for r in rows], dtype=np.float64)
    pred = np.array([r["pred_remain_len"] for r in rows], dtype=np.float64)
    diff = np.array([r["diff"] for r in rows], dtype=np.float64)
    abs_diff = np.abs(diff)
    log.info("=" * 70)
    log.info("GT remain    : mean=%.4f min=%.4f max=%.4f", gt.mean(), gt.min(), gt.max())
    log.info("Pred remain  : mean=%.4f min=%.4f max=%.4f", pred.mean(), pred.min(), pred.max())
    log.info("Diff(pred-gt): mean=%.4f min=%.4f max=%.4f", diff.mean(), diff.min(), diff.max())
    log.info("Abs diff     : mean=%.4f min=%.4f max=%.4f", abs_diff.mean(), abs_diff.min(), abs_diff.max())
    log.info("=" * 70)


def main():
    cfg = load_config(CONFIG_PATH)
    task = cfg.get("task")
    if task != "cycle":
        raise SystemExit(
            f"[Inference] config task is {task!r}, expected 'cycle'. "
            "Set task: cycle in config.yaml to run cell-cycle inference."
        )
    setup_logging(cfg["logging"])
    set_seed(cfg["project"]["seed"], cfg["project"]["deterministic"])

    data_cfg = cfg["data"]
    inference_cfg = cfg["inference4cellcycle"]
    input_ratio_min = inference_cfg["input_ratio_min"]
    input_ratio_max = inference_cfg["input_ratio_max"]
    if not 0.0 <= input_ratio_min < input_ratio_max <= 1.0:
        raise ValueError(
            f"Require 0 <= input_ratio_min < input_ratio_max <= 1, got "
            f"[{input_ratio_min}, {input_ratio_max}]"
        )

    device = get_device(cfg["project"]["device"])
    log.info("[CellCycleInference] Device: %s", device)

    ckpt = torch.load(inference_cfg["checkpoint_path"], map_location=device)
    train_cfg = ckpt["cfg"]
    max_seq_len = inference_cfg["max_seq_len"] or train_cfg["data"]["max_seq_len"]
    use_log_target = train_cfg["finetune4cellcycle"]["use_log_target"]

    model, input_dim = build_model_from_ckpt(ckpt, max_seq_len, device)
    file_paths = list_npy_files(inference_cfg["test_dir"])
    if not file_paths:
        raise FileNotFoundError(f"No .npy files found under {inference_cfg['test_dir']}")

    sample_raw = np.load(file_paths[0]).astype(np.float32)
    sample_n = min(sample_raw.shape[0], max_seq_len)
    sample = sample_raw[:sample_n].reshape(sample_n, -1)
    if sample.shape[-1] != input_dim:
        log.warning("[CellCycleInference] input_dim mismatch: ckpt=%d file=%d", input_dim, sample.shape[-1])

    dataset = CellCycleInferenceDataset(
        file_paths=file_paths,
        min_input_ratio=input_ratio_min,
        max_input_ratio=input_ratio_max,
        max_seq_len=max_seq_len,
        normalization_eps=data_cfg["normalization_eps"],
    )
    loader = DataLoader(
        dataset,
        batch_size=inference_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_cellcycle_inference,
        num_workers=inference_cfg["num_workers"],
        pin_memory=inference_cfg["pin_memory"],
        drop_last=False,
    )

    rows = run_inference(
        model,
        loader,
        device,
        use_log_target,
        inference_cfg.get("attention_output_dir"),
    )
    write_csv(rows, inference_cfg["output_csv"])
    log.info("[CellCycleInference] CSV saved to: %s", inference_cfg["output_csv"])
    print_stats(rows)


if __name__ == "__main__":
    main()
