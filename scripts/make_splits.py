"""Write splits/<ds>_{train,val,test}.txt. val = seeded, scenario-stratified subset of raw train.

Existing split files are never overwritten unless --force.
"""
import argparse
import sys

from logo.data.convert_gt import find_raw_videos
from logo.data.schema import write_split
from logo.data.splits import train_val_split
from logo.utils.io import get_logger, load_yaml
from logo.utils.paths import DATASETS, project_root, raw_dir, split_file


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=DATASETS)
    ap.add_argument("--config", default="configs/base.yaml", help="seed is read from here")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    log = get_logger("make_splits", project_root() / "logs" / "make_splits.log")
    seed = int(load_yaml(args.config)["seed"])
    outs = {s: split_file(args.ds, s) for s in ("train", "val", "test")}
    if not args.force and any(p.exists() for p in outs.values()):
        log.error("split files exist for %s; use --force to overwrite", args.ds)
        return 1
    train_raw = find_raw_videos(raw_dir(args.ds), "train")
    missing = [v for v, _, x in train_raw if x is None]
    if missing:
        log.warning("%d train videos without GT are excluded: %s", len(missing), missing)
    groups = {v: p.parent.name for v, p, x in train_raw if x is not None}
    train, val = train_val_split(groups, args.val_frac, seed)
    test = [v for v, _, _ in find_raw_videos(raw_dir(args.ds), "test")]
    for name, vids in (("train", train), ("val", val), ("test", test)):
        write_split(outs[name], vids)
        log.info("%s %s: %d videos -> %s", args.ds, name, len(vids), outs[name])
    return 0


if __name__ == "__main__":
    sys.exit(main())
