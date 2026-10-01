"""Sequence builder (step 2/2): per-frame NPME embeddings -> per-cell morphodynamic sequences.

Input : <embedding_dir>/<embryo>/<lineage>_<fate>/<stem>.npy   (each (32,4,4,4), from infer_npme.py)
Output: <output_dir>/<embryo>/<embryo>_<lineage>_<fate>.npy   ((T, 32, 4, 4, 4))

Pure post-processing (numpy only): group frames by cell directory, order by the
time index encoded in the file name, stack into one sequence per cell.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.utils import load_config


DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "config.yaml")

TASK_DEFAULT_EXCLUDE_FATES = {
    "fate": ["Unspecified", "Other"],
    "cycle": [],
}

TIME_RE = re.compile(r"_(\d+)_segCell\.(npy|npz)$", re.IGNORECASE)


def _frame_sort_key(fpath):
    m = TIME_RE.search(os.path.basename(str(fpath)))
    return (int(m.group(1)) if m else -1, str(fpath))


def _iter_cells(embedding_dir):
    """Yield (embryo, cell_key, frame_paths); one directory of frames = one cell."""
    root = Path(embedding_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"embedding_dir does not exist: {root}")
    for embryo_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cell_dir in sorted(p for p in embryo_dir.iterdir() if p.is_dir()):
            frames = [
                p
                for p in cell_dir.iterdir()
                if p.suffix.lower() == ".npy"
                and ".tmp." not in p.name
                and not p.name.endswith(".tmp.npy")
            ]
            if frames:
                yield embryo_dir.name, cell_dir.name, frames


def _load_embedding(fpath):
    arr = np.load(fpath, allow_pickle=True)
    if arr.dtype == object:
        arr = arr.item()["z"]
    return np.asarray(arr, dtype=np.float32)


def run(args):
    cfg = load_config(args.config or DEFAULT_CONFIG_PATH)
    e2s = cfg["embed2seq"]

    embedding_dir = args.embedding_dir or e2s["embedding_dir"]
    output_dir = args.output_dir or e2s["output_dir"]
    task = cfg.get("task", "fate")
    if task not in TASK_DEFAULT_EXCLUDE_FATES:
        raise SystemExit(f"config 'task' must be one of {sorted(TASK_DEFAULT_EXCLUDE_FATES)}, got: {task!r}")
    exclude_fates = set(e2s.get("exclude_fates", TASK_DEFAULT_EXCLUDE_FATES[task]))

    print(f"[Embed2Seq] embedding_dir: {embedding_dir}")
    print(f"[Embed2Seq] output_dir: {output_dir}")
    print(f"[Embed2Seq] exclude_fates: {sorted(exclude_fates)} (task={task})")

    stats = {}
    total = 0
    for embryo, cell_key, frames in _iter_cells(embedding_dir):
        fate = cell_key.rsplit("_", 1)[-1]
        if fate in exclude_fates:
            continue

        out_dir = Path(output_dir) / embryo
        out_path = out_dir / f"{embryo}_{cell_key}.npy"
        if out_path.exists() and not args.no_skip_existing:
            continue

        frames = sorted(frames, key=_frame_sort_key)
        zs = []
        for f in frames:
            try:
                zs.append(_load_embedding(f))
            except Exception as exc:
                print(f"[WARN] skip unreadable embedding {f.name}: {exc}")
        if not zs:
            print(f"[WARN] no valid embeddings for {cell_key}; sequence skipped")
            continue

        seq = np.stack(zs, axis=0)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_path, seq)

        stats[fate] = stats.get(fate, 0) + 1
        total += 1
        print(f"[OK] {out_path.name}  T={seq.shape[0]}  z={tuple(seq.shape[1:])}")

    print("---- summary ----")
    for fate in sorted(stats):
        print(f"  {fate:15s} {stats[fate]}")
    print(f"  total sequences: {total}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Build per-cell (T,32,4,4,4) sequences from per-frame embeddings."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="",
        help="path to config.yaml (default: <repo>/config.yaml)",
    )
    parser.add_argument("--embedding_dir", default="", help="override embed2seq.embedding_dir")
    parser.add_argument("--output_dir", default="", help="override embed2seq.output_dir")
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Recompute even if the output sequence .npy already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(_parse_args())
