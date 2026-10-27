"""Before/after re-scoring panel: one frame drawn twice, boxes coloured by det score (left) and by
score_fused (right), blue = 0 -> red = 1, with a colour bar. Boxes with score <= --hide_below are
drawn thin and grey on both sides.

Input is a fused json (runs/<exp>/fuse/<ds>/<video>.json); the frame is read from
data/frames/<ds>/<video>/%06d.jpg. Default frame: the one with the most boxes.

Usage: python tools/rescore_panel.py --fused runs/E3-lsc/fuse/icdar2015v/Video_2_1_2.json
           --out docs/figs/rescore_Video_2_1_2.png [--frame 12] [--hide_below 0.0] [--final]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from logo.data.schema import read_split
from logo.utils.io import get_logger, load_json
from logo.utils.paths import frame_path, split_file

HIDDEN_BGR = (150, 150, 150)
BAR_H, GAP, TITLE_H = 24, 8, 28


def score_color(s: float) -> tuple[int, int, int]:
    """BGR on a linear blue (0) -> red (1) ramp; s is clipped to [0, 1]."""
    s = min(max(float(s), 0.0), 1.0)
    return (int(round(255 * (1 - s))), 0, int(round(255 * s)))


def draw_scores(img: np.ndarray, objs: list[dict], key: str, hide_below: float = 0.0) -> np.ndarray:
    """Draw quads of `objs` coloured by o[key] onto a copy of img, with the score as label."""
    out = img.copy()
    for o in sorted(objs, key=lambda o: float(o[key])):  # high scores drawn last (on top)
        s = float(o[key])
        pts = np.round(np.asarray(o["quad"], np.float64).reshape(4, 2)).astype(np.int32)
        if s <= hide_below:
            cv2.polylines(out, [pts], True, HIDDEN_BGR, 1, cv2.LINE_AA)
            continue
        color = score_color(s)
        cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)
        x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
        cv2.putText(out, f"{s:.2f}", (x, max(y - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                    cv2.LINE_AA)
    return out


def _titled(img: np.ndarray, title: str) -> np.ndarray:
    bar = np.full((TITLE_H, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (6, TITLE_H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def color_bar(width: int) -> np.ndarray:
    ramp = np.array([score_color(s) for s in np.linspace(0, 1, width)], np.uint8)
    bar = np.repeat(ramp[None], BAR_H, axis=0)
    for s, x in ((0.0, 2), (0.5, width // 2 - 10), (1.0, width - 26)):
        cv2.putText(bar, f"{s:.1f}", (x, BAR_H - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return bar


def make_panel(img: np.ndarray, objs: list[dict], titles=("det score", "fused score"),
               keys=("score", "score_fused"), hide_below: float = 0.0) -> np.ndarray:
    """[TITLE_H + H + BAR_H, 2W + GAP, 3] uint8: left = keys[0], right = keys[1]."""
    left = _titled(draw_scores(img, objs, keys[0], hide_below), titles[0])
    right = _titled(draw_scores(img, objs, keys[1], hide_below), titles[1])
    gap = np.full((left.shape[0], GAP, 3), 255, np.uint8)
    top = np.hstack([left, gap, right])
    return np.vstack([top, color_bar(top.shape[1])])


def busiest_frame(frames: dict) -> int:
    return int(max(frames, key=lambda k: (len(frames[k]), -int(k))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fused", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help=".png / .jpg")
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--hide_below", type=float, default=0.0)
    ap.add_argument("--frames_dir", type=Path, default=None, help="default: data/frames/<ds>/<video>")
    ap.add_argument("--final", action="store_true", help="allow videos of the test split (rule 5)")
    args = ap.parse_args(argv)
    log = get_logger("rescore_panel")

    fused = load_json(args.fused)
    ds, video = fused["dataset"], fused["video"]
    test = split_file(ds, "test")
    if not args.final and test.is_file() and video in read_split(test):
        raise PermissionError(f"{video} is in the test split; pass --final")
    fid = args.frame if args.frame is not None else busiest_frame(fused["frames"])
    objs = fused["frames"].get(str(fid), [])
    src = (args.frames_dir / f"{fid:06d}.jpg") if args.frames_dir else frame_path(ds, video, fid)
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read frame {src}")
    panel = make_panel(img, objs, hide_below=args.hide_below)
    ok, buf = cv2.imencode(args.out.suffix or ".png", panel)
    if not ok:
        raise RuntimeError(f"cannot encode {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(buf.tobytes())  # cv2.imwrite cannot write non-ASCII paths on Windows
    log.info("%s/%s frame %d: %d boxes -> %s", ds, video, fid, len(objs), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
