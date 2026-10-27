"""Decode the videos of one split to data/frames/<ds>/<video>/%06d.jpg (resumable)."""
import argparse
import sys

from tqdm import tqdm

from logo.data.convert_gt import find_video
from logo.data.frames import extract_frames
from logo.data.schema import list_videos
from logo.utils.io import get_logger
from logo.utils.paths import DATASETS, SPLITS, check_split, frames_dir, project_root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=DATASETS)
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--final", action="store_true", help="required for --split test")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    check_split(args.split, args.final)
    log = get_logger("extract_frames", project_root() / "logs" / f"extract_frames_{args.ds}.log")
    for vid in tqdm(list_videos(args.ds, args.split), desc=f"{args.ds}/{args.split}"):
        n = extract_frames(find_video(args.ds, vid), frames_dir(args.ds, vid), args.quality, args.overwrite)
        log.info("%s/%s: %d frames", args.ds, vid, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
