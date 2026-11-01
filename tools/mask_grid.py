"""PNG grid of LSC crop / glyph-mask pairs (side by side) from data/lsc/<split>/masks.tsv.

Pairs are sampled evenly over masks.tsv (deterministic, no RNG). Each cell: crop | mask, caption
"<label> fg=<fraction>". The test split needs --final.

Usage: python tools/mask_grid.py --out docs/figs/glyph_masks.png [--split train] [--root data/lsc]
           [--n 50] [--cols 5] [--final]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from logo.lsc.dataset import read_labels
from logo.utils.io import get_logger
from logo.utils.paths import check_split, project_root

CAPTION_H = 14
PAD = 4
GAP = 2


def read_pairs(split_dir: Path) -> list[tuple[str, str, float, str]]:
    """(image rel, mask rel, fg_frac, label) in masks.tsv order."""
    labels = dict(read_labels(split_dir / "labels.tsv"))
    out = []
    for ln in (split_dir / "masks.tsv").read_text(encoding="utf-8").splitlines():
        if ln:
            rel, mrel, fg = ln.split("\t")
            out.append((rel, mrel, float(fg), labels.get(rel, "")))
    return out


def sample(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    idx = np.unique(np.round(np.linspace(0, len(items) - 1, n)).astype(int))
    return [items[i] for i in idx]


def _read(p: Path, flag) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), flag) if p.is_file() else None
    if img is None:
        raise FileNotFoundError(p)
    return img


def build_grid(crops: list[np.ndarray], masks: list[np.ndarray], captions: list[str], cols: int) -> np.ndarray:
    h, w = crops[0].shape[:2]
    cell_h, cell_w = h + CAPTION_H + PAD, 2 * w + GAP + PAD
    rows = (len(crops) + cols - 1) // cols
    grid = np.full((rows * cell_h + PAD, cols * cell_w + PAD, 3), 40, np.uint8)
    for i, (c, m, cap) in enumerate(zip(crops, masks, captions)):
        y, x = PAD + (i // cols) * cell_h, PAD + (i % cols) * cell_w
        grid[y:y + h, x:x + w] = cv2.resize(c, (w, h))
        grid[y:y + h, x + w + GAP:x + 2 * w + GAP] = cv2.cvtColor(cv2.resize(m, (w, h)), cv2.COLOR_GRAY2BGR)
        cv2.putText(grid, cap[:40], (x, y + h + CAPTION_H - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return grid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--root", type=Path, default=None, help="default: <root>/data/lsc")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--final", action="store_true", help="required for the test split")
    args = ap.parse_args(argv)
    log = get_logger("mask_grid", project_root() / "logs" / "mask_grid.log")
    try:
        check_split(args.split, args.final)
    except PermissionError as e:
        log.error("%s", e)
        return 1
    split_dir = (args.root if args.root is not None else project_root() / "data" / "lsc") / args.split
    if not (split_dir / "masks.tsv").is_file():
        log.error("%s missing (run tools/make_glyph_masks.py first)", split_dir / "masks.tsv")
        return 1
    items = sample(read_pairs(split_dir), args.n)
    if not items:
        log.error("no masks in %s", split_dir)
        return 1
    crops = [_read(split_dir / r, cv2.IMREAD_COLOR) for r, _, _, _ in items]
    masks = [_read(split_dir / m, cv2.IMREAD_GRAYSCALE) for _, m, _, _ in items]
    caps = [f"{lab.encode('ascii', 'replace').decode('ascii')} fg={fg:.2f}" for _, _, fg, lab in items]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", build_grid(crops, masks, caps, max(1, args.cols)))
    if not ok:
        raise OSError(f"cannot encode {args.out}")
    buf.tofile(str(args.out))
    log.info("%s: %d crop/mask pairs -> %s", args.split, len(items), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
