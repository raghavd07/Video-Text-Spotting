"""Decode videos to data/frames/<ds>/<video>/%06d.jpg (OpenCV, JPEG quality 90, 1-based).

With --stride s only frames 1, 1+s, 1+2s, ... are written, named by their original frame index.
Videos already extracted with the same stride are skipped (resumable). Videos come from the split
files; the test split needs --final (rule 5).

Usage: python tools/extract_frames.py --ds icdar2015v [--split train val] [--stride 1]
           [--videos Video_2_1_2 ...] [--quality 90] [--overwrite] [--final]
"""
from __future__ import annotations

import argparse
import sys

from tqdm import tqdm

from logo.data.convert_gt import find_video
from logo.data.frames import extract_frames
from logo.data.schema import list_videos
from logo.utils.io import get_logger
from logo.utils.paths import DATASETS, SPLITS, check_split, frames_dir, project_root


def select_videos(ds: str, splits: list[str], final: bool, only: list[str] | None) -> list[str]:
    vids: list[str] = []
    for s in splits:
        check_split(s, final)
        vids += [v for v in list_videos(ds, s) if v not in vids]
    if only:
        unknown = sorted(set(only) - set(vids))
        if unknown:
            raise ValueError(f"videos not in {ds} splits {splits}: {unknown}")
        vids = [v for v in vids if v in set(only)]
    return vids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ds", nargs="+", required=True, choices=DATASETS)
    ap.add_argument("--split", nargs="+", default=["train", "val"], choices=SPLITS)
    ap.add_argument("--videos", nargs="+", help="restrict to these video ids")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--final", action="store_true", help="required for --split test")
    args = ap.parse_args(argv)
    log = get_logger("extract_frames", project_root() / "logs" / "extract_frames.log")
    for ds in args.ds:
        vids = select_videos(ds, args.split, args.final, args.videos)
        for vid in tqdm(vids, desc=ds):
            n = extract_frames(find_video(ds, vid), frames_dir(ds, vid), args.quality,
                               args.overwrite, args.stride)
            log.info("%s/%s: %d frames (stride %d)", ds, vid, n, args.stride)
    return 0


if __name__ == "__main__":
    sys.exit(main())
