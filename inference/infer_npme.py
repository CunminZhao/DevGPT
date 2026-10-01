"""NPME encoder (step 1/2): per-time-point ternary voxel maps -> per-frame Cell Token embeddings.

Input : <input_folder>/<embryo>/<lineage>_<fate>/<...>_<CellID>_<embryo>_<t>_segCell.npz  (scanned recursively)
Output: <output_folder>/<same relative directory>/<stem>.npy
        each file is a plain (32, 4, 4, 4) float32 array -- the Cell Token of that frame.

Sequence assembly is done separately by inference/embed2sequence.py.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.npme_losses import to_one_hot
from models.geometry import compute_geometry_features
from models.npme import ResNetAutoEncoder
from utils.utils import get_device, load_config


DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "config.yaml")


def _strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _load_model_state(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model_state = checkpoint["state_dict"]
    else:
        model_state = checkpoint
    return _strip_module_prefix(model_state)


def _load_volume(fpath):
    """Load a 3D label volume from .npy or .npz (key 'arr' preferred)."""
    loaded = np.load(fpath)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "arr" in loaded.files:
            data = loaded["arr"]
        else:
            data = loaded[loaded.files[0]]
        loaded.close()
    else:
        data = loaded
    return np.round(data).astype(np.uint8)


def _crop_or_pad(data, target_shape):
    d, h, w = data.shape
    td, th, tw = target_shape

    start_d = (d - td) // 2 if d > td else 0
    start_h = (h - th) // 2 if h > th else 0
    start_w = (w - tw) // 2 if w > tw else 0
    data = data[start_d : start_d + td, start_h : start_h + th, start_w : start_w + tw]

    cd, ch, cw = data.shape
    pad_d, pad_h, pad_w = td - cd, th - ch, tw - cw
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        pd_bef, ph_bef, pw_bef = pad_d // 2, pad_h // 2, pad_w // 2
        data = np.pad(
            data,
            (
                (pd_bef, pad_d - pd_bef),
                (ph_bef, pad_h - ph_bef),
                (pw_bef, pad_w - pw_bef),
            ),
            mode="constant",
            constant_values=0,
        )

    return data


def _iter_frames(input_root):
    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError(f"input folder does not exist: {root}")
    return sorted(
        p
        for p in root.rglob("*")
        if p.suffix.lower() in {".npz", ".npy"}
        and ".tmp." not in p.name
        and not p.name.endswith(".tmp.npz")
    )


def _build_model(np_cfg, model_path, device):
    model = ResNetAutoEncoder(
        num_classes=np_cfg["num_classes_seg"],
        base_ch=np_cfg["cnn_base_channels"],
        latent_channels=np_cfg["latent_channels"],
        input_shape=tuple(np_cfg["input_shape"]),
        geometry_dim=np_cfg["geometry_dim"],
        latent_spatial_shape=tuple(np_cfg["latent_spatial_shape"]),
        geometry_conditioner_hidden_dims=tuple(np_cfg["geometry_conditioner_hidden_dims"]),
        geometry_conditioner_channels=np_cfg["geometry_conditioner_channels"],
        geometry_conditioner_dropout=np_cfg["geometry_conditioner_dropout"],
        geometry_readout_output_dims=tuple(np_cfg["geometry_readout_output_dims"]),
        geometry_readout_dropout=np_cfg["geometry_readout_dropout"],
    )
    model.load_state_dict(_load_model_state(model_path))
    model = model.to(device)
    model.eval()
    return model


def run(args):
    cfg = load_config(args.config or DEFAULT_CONFIG_PATH)
    np_cfg = cfg["npme"]
    e2s = cfg.get("embed2seq", {})

    input_folder = args.input_folder or e2s.get("raw_dir", "")
    output_folder = args.output_folder or e2s.get("embedding_dir", "")
    if not input_folder or not output_folder:
        raise SystemExit("input/output folders required: pass --input_folder/--output_folder or set embed2seq.raw_dir/embedding_dir")
    model_path = args.model_path or os.path.join(np_cfg["checkpoint_dir"], "checkpoint_final.pth")

    device = get_device(cfg["project"]["device"])
    print(f"[InferNPME] device: {device}")
    print(f"[InferNPME] input_folder: {input_folder}")
    print(f"[InferNPME] output_folder: {output_folder}")
    print(f"[InferNPME] model: {model_path}")

    input_shape = tuple(np_cfg["input_shape"])
    frame_batch = int(np_cfg["batch_size_per_gpu"])

    model = _build_model(np_cfg, model_path, device)
    frames = _iter_frames(input_folder)
    print(f"[InferNPME] frames found: {len(frames)}")

    input_root = Path(input_folder)
    written, skipped_existing, failed = 0, 0, 0

    for i in range(0, len(frames), frame_batch):
        chunk = frames[i : i + frame_batch]
        vols, geoms, out_paths = [], [], []
        for f in chunk:
            out_path = Path(output_folder) / f.relative_to(input_root).parent / (f.stem + ".npy")
            if out_path.exists() and not args.no_skip_existing:
                skipped_existing += 1
                continue
            try:
                v = _crop_or_pad(_load_volume(str(f)), input_shape)
                g = compute_geometry_features(
                    np.ascontiguousarray(v),
                    np_cfg["voxel_size"],
                    np_cfg["surface_level"],
                    np_cfg["boundary_step_size"],
                )
                vols.append(v)
                geoms.append(g)
                out_paths.append(out_path)
            except Exception as exc:
                failed += 1
                print(f"[WARN] skip unreadable frame {f.name}: {exc}")
        if not vols:
            continue

        x = torch.from_numpy(np.stack(vols)).long().to(device)
        geom = torch.from_numpy(np.stack(geoms)).float().to(device)
        x_oh = to_one_hot(x, np_cfg["num_classes_seg"])
        with torch.no_grad():
            _, z, _ = model(x_oh, geom)
        z_np = z.cpu().float().numpy()

        for j, out_path in enumerate(out_paths):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, z_np[j])
            written += 1

    print("---- summary ----")
    print(f"  encoded : {written}")
    print(f"  existing: {skipped_existing}")
    print(f"  failed  : {failed}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="NPME encoder: per-frame ternary voxels -> per-frame (32,4,4,4) Cell Token .npy."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="",
        help="path to config.yaml (default: <repo>/config.yaml)",
    )
    parser.add_argument("--input_folder", default="", help="override embed2seq.raw_dir")
    parser.add_argument("--output_folder", default="", help="override embed2seq.embedding_dir")
    parser.add_argument(
        "--model_path",
        default="",
        help="NPME checkpoint; empty = <npme.checkpoint_dir>/checkpoint_final.pth",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Recompute even if the output .npy already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(_parse_args())
