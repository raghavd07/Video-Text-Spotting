"""Glyph pseudo-masks (logo.glyph.pseudo_labels.kmeans_mask) for every LSC positive crop.

Reads data/lsc/<split>/labels.tsv (rows with a non-empty label), writes uint8 PNG masks {0,255} to
data/lsc/<split>/masks/<ds>/<video>/<frame>_pos<k>.png (images/ prefix replaced by masks/) and
data/lsc/<split>/masks.tsv ("<image rel>\t<mask rel>\t<fg_frac>"). k-means seed = config seed.
Existing masks are kept unless --overwrite. Writes runs/<exp_id>/{config.yaml,log.txt,
metrics.json meta.glyph_masks_<split>}.

Usage: python tools/make_glyph_masks.py --config configs/base.yaml [--splits train]
           [--root data/lsc] [--n_init 3] [--no-morph] [--overwrite] [--final]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from logo.detector.infer_paddle import write_run_files
from logo.glyph.pseudo_labels import kmeans_mask
from logo.lsc.dataset import read_labels
from logo.utils.io import get_logger, load_yaml, set_seed
from logo.utils.paths import check_split, project_root


def mask_rel(image_rel: str) -> str:
    """images/<ds>/<video>/<name>.png -> masks/<ds>/<video>/<name>.png"""
    parts = Path(image_rel).parts
    if not parts or parts[0] != "images":
        raise ValueError(f"unexpected crop path {image_rel!r}")
    return Path("masks", *parts[1:]).as_posix()


def read_rgb(p: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR) if p.is_file() else None
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_png(p: Path, img: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise OSError(f"cannot encode {p}")
    buf.tofile(str(p))


def make_split(split_dir: Path, seed: int, n_init: int, morph: bool, overwrite: bool, log) -> dict:
    tsv = split_dir / "labels.tsv"
    if not tsv.is_file():
        raise FileNotFoundError(f"{tsv} (run tools/build_lsc_dataset.py first)")
    rows = [(rel, lab) for rel, lab in read_labels(tsv) if lab]
    t0 = time.perf_counter()
    out, n_new, n_kept, n_empty = [], 0, 0, 0
    for i, (rel, _) in enumerate(rows):
        mrel = mask_rel(rel)
        mp = split_dir / mrel
        if mp.is_file() and not overwrite:
            m = cv2.imdecode(np.fromfile(str(mp), np.uint8), cv2.IMREAD_GRAYSCALE)
            n_kept += 1
        else:
            m = (kmeans_mask(read_rgb(split_dir / rel), seed=seed, n_init=n_init, morph=morph) * 255
                 ).astype(np.uint8)
            write_png(mp, m)
            n_new += 1
        fg = float((m > 127).mean())
        n_empty += fg == 0.0
        out.append((rel, mrel, fg))
        if (i + 1) % 5000 == 0:
            log.info("%s: %d/%d masks", split_dir.name, i + 1, len(rows))
    (split_dir / "masks.tsv").write_text("".join(f"{r}\t{m}\t{f:.4f}\n" for r, m, f in out),
                                         encoding="utf-8")
    fgs = np.array([f for _, _, f in out], np.float64)
    stats = {"split": split_dir.name, "positives": len(rows), "written": n_new, "kept": n_kept,
             "empty_masks": int(n_empty), "seed": seed, "n_init": n_init, "morph": morph,
             "fg_frac_mean": float(fgs.mean()) if len(fgs) else 0.0,
             "fg_frac_p05_p50_p95": [float(v) for v in np.percentile(fgs, [5, 50, 95])] if len(fgs) else [],
             "seconds": round(time.perf_counter() - t0, 2)}
    log.info("%s: %d masks (%d new, %d kept, %d empty) fg_mean=%.3f", split_dir.name, len(rows), n_new,
             n_kept, n_empty, stats["fg_frac_mean"])
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train"], choices=("train", "val", "test"))
    ap.add_argument("--root", type=Path, default=None, help="default: <root>/data/lsc")
    ap.add_argument("--n_init", type=int, default=3)
    ap.add_argument("--no-morph", action="store_true", help="skip the 3x3 cleanup")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--final", action="store_true", help="allow the test split (rule 5)")
    args = ap.parse_args(argv)

    for s in args.splits:
        check_split(s, args.final)
    root = project_root()
    cfg = load_yaml(args.config)
    lsc_root = args.root if args.root is not None else root / "data" / "lsc"
    seed = int(cfg["seed"])
    set_seed(seed)
    run_dir = root / "runs" / cfg["exp_id"]
    log = get_logger("make_glyph_masks", run_dir / "log.txt")
    log.info("exp=%s splits=%s root=%s n_init=%d morph=%s", cfg["exp_id"], args.splits, lsc_root,
             args.n_init, not args.no_morph)
    for split in args.splits:
        stats = make_split(lsc_root / split, seed, args.n_init, not args.no_morph, args.overwrite, log)
        write_run_files(cfg, run_dir, stats, key=f"glyph_masks_{split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
