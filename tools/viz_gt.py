"""Draw GT quads, track ids and text on every frame of one video and write an mp4.

Frames are read from data/frames/<ds>/<video>/ when extracted, else decoded from the raw video
(streamed, never held in memory). Don't-care boxes are grey. Videos of the test split need --final.

Usage: python tools/viz_gt.py --ds icdar2015v --video Video_2_1_2 --out docs/figs/Video_2_1_2.mp4
           [--fps 0] [--max-frames 0] [--final]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from logo.data.convert_gt import find_video
from logo.data.schema import load_gt, read_split
from logo.utils.io import get_logger
from logo.utils.paths import DATASETS, frames_dir, gt_path, project_root, split_file

DONT_CARE_BGR = (160, 160, 160)


def track_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic saturated BGR colour per track id."""
    hue = (int(track_id) * 47) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw_objects(img: np.ndarray, objs: list[dict]) -> np.ndarray:
    """Draw in place and return img. cv2 fonts are ASCII only: other characters become '?'."""
    for o in objs:
        pts = np.round(np.asarray(o["quad"], np.float64).reshape(4, 2)).astype(np.int32)
        color = DONT_CARE_BGR if o["dont_care"] else track_color(o["track_id"])
        cv2.polylines(img, [pts], True, color, 1 if o["dont_care"] else 2, cv2.LINE_AA)
        label = f'{o["track_id"]}:{o["text"]}'.encode("ascii", "replace").decode("ascii")
        x, y = int(pts[:, 0].min()), int(pts[:, 1].min()) - 4
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y = max(y, th + 2)
        cv2.rectangle(img, (x, y - th - 2), (x + tw, y + 2), (0, 0, 0), -1)
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def iter_frames(ds: str, video: str, num_frames: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_id, BGR image) for 1..num_frames; missing extracted frames are skipped."""
    fdir = frames_dir(ds, video)
    if (fdir / "000001.jpg").is_file():
        for fid in range(1, num_frames + 1):
            p = fdir / f"{fid:06d}.jpg"
            if p.is_file():
                img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    yield fid, img
        return
    cap = cv2.VideoCapture(str(find_video(ds, video)))
    try:
        fid = 0
        while fid < num_frames:
            ok, img = cap.read()
            if not ok:
                break
            fid += 1
            yield fid, img
    finally:
        cap.release()


def render(gt: dict, frames: Iterator[tuple[int, np.ndarray]], out: Path, fps: float) -> int:
    """Write the annotated mp4; returns the number of frames written."""
    out.parent.mkdir(parents=True, exist_ok=True)
    writer, n = None, 0
    try:
        for fid, img in frames:
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise OSError(f"cannot open video writer for {out}")
            draw_objects(img, gt["frames"].get(str(fid), []))
            cv2.putText(img, f"#{fid}", (5, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(img)
            n += 1
    finally:
        if writer is not None:
            writer.release()
    return n


def source_fps(ds: str, video: str, default: float = 25.0) -> float:
    try:
        cap = cv2.VideoCapture(str(find_video(ds, video)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    except FileNotFoundError:
        fps = 0.0
    return fps if 1.0 <= fps <= 120.0 else default


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ds", required=True, choices=DATASETS)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, type=Path, help="mp4 file, or a directory")
    ap.add_argument("--fps", type=float, default=0.0, help="0 = fps of the raw video")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--final", action="store_true", help="required for test-split videos")
    args = ap.parse_args(argv)
    log = get_logger("viz_gt", project_root() / "logs" / "viz_gt.log")
    test_file = split_file(args.ds, "test")
    if not args.final and test_file.exists() and args.video in read_split(test_file):
        log.error("%s is in the %s test split; pass --final", args.video, args.ds)
        return 1
    gt = load_gt(gt_path(args.ds, args.video))
    out = args.out if args.out.suffix.lower() == ".mp4" else args.out / f"{args.ds}_{args.video}.mp4"
    n_max = gt["num_frames"] if args.max_frames <= 0 else min(args.max_frames, gt["num_frames"])
    fps = args.fps if args.fps > 0 else source_fps(args.ds, args.video)
    n = render(gt, iter_frames(args.ds, args.video, n_max), out, fps)
    log.info("%s/%s: %d frames -> %s", args.ds, args.video, n, out)
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
