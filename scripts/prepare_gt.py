"""Convert raw XML annotations to data/processed/<ds>/<video>.json.

Raw train videos are always converted; raw test videos (if they have GT) only with --final.
"""
import argparse
import sys

from logo.data.convert_gt import convert_video, find_raw_videos
from logo.data.schema import validate_gt
from logo.utils.io import get_logger, save_json
from logo.utils.paths import DATASETS, gt_path, project_root, raw_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=DATASETS)
    ap.add_argument("--final", action="store_true", help="also convert raw test GT")
    args = ap.parse_args(argv)
    log = get_logger("prepare_gt", project_root() / "logs" / f"prepare_gt_{args.ds}.log")
    raw_splits = ("train", "test") if args.final else ("train",)
    n_ok = n_nogt = 0
    for rs in raw_splits:
        for vid, vpath, xpath in find_raw_videos(raw_dir(args.ds), rs):
            if xpath is None:
                n_nogt += 1
                log.warning("no GT xml for %s/%s", rs, vid)
                continue
            gt = validate_gt(convert_video(args.ds, vid, vpath, xpath))
            save_json(gt, gt_path(args.ds, vid))
            n_obj = sum(len(o) for o in gt["frames"].values())
            n_dc = sum(o["dont_care"] for objs in gt["frames"].values() for o in objs)
            log.info("%s/%s: %dx%d frames=%d objects=%d dont_care=%d", rs, vid, gt["width"],
                     gt["height"], gt["num_frames"], n_obj, n_dc)
            n_ok += 1
    log.info("done: %d converted, %d without GT", n_ok, n_nogt)
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
