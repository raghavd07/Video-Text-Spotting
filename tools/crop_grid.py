"""Save a PNG grid of rectified crops (logo.lsc.roi_rotate.crop_quad) for one video.

Source is the GT json (default) or a det/lsc/fused json given by --pred or by --exp-id (reads
runs/<exp_id>/<stage>/<ds>/<video>.json). Objects are sampled evenly over the video in frame order
(deterministic, no RNG); GT don't-care boxes are skipped unless --dont-care.
Captions: GT text, or score (+ recognised text if present). Test-split videos need --final.

Usage: python tools/crop_grid.py --ds icdar2015v --video Video_2_1_2 --out docs/figs/crops.png
           [--pred runs/e1/det/icdar2015v/Video_2_1_2.json | --exp-id e1 --stage det]
           [--n 64] [--cols 8] [--min-score 0.3] [--dont-care] [--final]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from logo.data.schema import load_gt, read_split
from logo.lsc.roi_rotate import crop_many
from logo.utils.io import get_logger, load_json
from logo.utils.paths import DATASETS, frame_path, gt_path, project_root, split_file, stage_dir

CAPTION_H = 14
PAD = 4


def collect(doc: dict, is_gt: bool, dont_care: bool, min_score: float) -> list[tuple[int, dict]]:
    """(frame_id, object) for every usable object, in file order."""
    out = []
    for fid, objs in doc["frames"].items():
        for o in objs:
            if is_gt and o.get("dont_care") and not dont_care:
                continue
            if not is_gt and float(o.get("score", 0.0)) < min_score:
                continue
            out.append((int(fid), o))
    out.sort(key=lambda t: t[0])
    return out


def sample(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    idx = np.unique(np.round(np.linspace(0, len(items) - 1, n)).astype(int))
    return [items[i] for i in idx]


def caption(o: dict, is_gt: bool) -> str:
    if is_gt:
        s = ("#" if o.get("dont_care") else "") + str(o.get("text", ""))
    else:
        s = f'{float(o.get("score", 0.0)):.2f}'
        if "text" in o:
            s += f' {o["text"]}'
    return s.encode("ascii", "replace").decode("ascii")


def read_frame(ds: str, video: str, fid: int) -> np.ndarray | None:
    p = frame_path(ds, video, fid)
    if not p.is_file():
        return None
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def build_grid(crops: list[np.ndarray], labels: list[str], cols: int) -> np.ndarray:
    h, w = crops[0].shape[:2]
    cell_h, cell_w = h + CAPTION_H + PAD, w + PAD
    rows = (len(crops) + cols - 1) // cols
    grid = np.full((rows * cell_h + PAD, cols * cell_w + PAD, 3), 40, np.uint8)
    for i, (c, lab) in enumerate(zip(crops, labels)):
        y, x = PAD + (i // cols) * cell_h, PAD + (i % cols) * cell_w
        grid[y:y + h, x:x + w] = c
        cv2.putText(grid, lab[:24], (x, y + h + CAPTION_H - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return grid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ds", required=True, choices=DATASETS)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, type=Path, help="png file, or a directory")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--pred", type=Path, help="det/lsc/fused json of this video")
    src.add_argument("--exp-id", help="read runs/<exp_id>/<stage>/<ds>/<video>.json")
    ap.add_argument("--stage", default="det", help="stage dir used with --exp-id")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--out-h", type=int, default=32)
    ap.add_argument("--out-w", type=int, default=128)
    ap.add_argument("--min-score", type=float, default=0.0, help="predictions only")
    ap.add_argument("--dont-care", action="store_true", help="include GT don't-care boxes")
    ap.add_argument("--final", action="store_true", help="required for test-split videos")
    args = ap.parse_args(argv)
    log = get_logger("crop_grid", project_root() / "logs" / "crop_grid.log")

    test_file = split_file(args.ds, "test")
    if not args.final and test_file.exists() and args.video in read_split(test_file):
        log.error("%s is in the %s test split; pass --final", args.video, args.ds)
        return 1
    if args.pred is not None:
        doc, is_gt = load_json(args.pred), False
    elif args.exp_id:
        doc, is_gt = load_json(stage_dir(args.exp_id, args.stage, args.ds) / f"{args.video}.json"), False
    else:
        doc, is_gt = load_gt(gt_path(args.ds, args.video)), True

    items = sample(collect(doc, is_gt, args.dont_care, args.min_score), args.n)
    crops, labels = [], []
    by_frame: dict[int, list[dict]] = {}
    for fid, o in items:
        by_frame.setdefault(fid, []).append(o)
    missing = 0
    for fid in sorted(by_frame):
        img = read_frame(args.ds, args.video, fid)
        if img is None:
            missing += 1
            continue
        objs = by_frame[fid]
        crops.extend(crop_many(img, [o["quad"] for o in objs], (args.out_h, args.out_w)))
        labels.extend(f"{fid}:{caption(o, is_gt)}" for o in objs)
    if missing:
        log.warning("%d sampled frames not found under data/frames/%s/%s", missing, args.ds, args.video)
    if not crops:
        log.error("no crops for %s/%s", args.ds, args.video)
        return 1

    out = args.out if args.out.suffix.lower() == ".png" else (
        args.out / f"{args.ds}_{args.video}_{'gt' if is_gt else 'pred'}_crops.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", build_grid(crops, labels, max(1, args.cols)))
    if not ok:
        raise OSError(f"cannot encode {out}")
    buf.tofile(str(out))
    log.info("%s/%s: %d crops (%s) -> %s", args.ds, args.video, len(crops),
             "gt" if is_gt else "pred", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
