"""Write splits/<ds>_{train,val,test}.txt.

train/val = seeded 80/20 split *by video* of the official train videos (stratified by the raw
sub-folder, i.e. DSText scenario; ICDAR has one group). test = official test videos, written even
when their annotations are missing (a warning then says val acts as held-out test).
Existing different split files are kept unless --force. Seed comes from the config.

Usage: python tools/make_splits.py [--ds icdar2013v icdar2015v dstext] [--config configs/base.yaml]
           [--val-frac 0.2] [--raw-root data/raw] [--out splits] [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from logo.data.convert_gt import find_raw_videos
from logo.data.schema import read_split, write_split
from logo.data.splits import make_splits
from logo.utils.io import get_logger, load_yaml
from logo.utils.paths import DATASETS, SPLITS, project_root


def collect(raw_ds: Path, log) -> tuple[dict[str, str], list[str], int]:
    """({train video: group}, test videos, number of test videos without GT xml)."""
    train_raw = find_raw_videos(raw_ds, "train")
    no_gt = [v for v, _, x in train_raw if x is None]
    if no_gt:
        log.warning("%s: %d train videos without GT excluded: %s", raw_ds.name, len(no_gt), no_gt)
    groups = {v: p.parent.relative_to(raw_ds / "train").as_posix() for v, p, x in train_raw
              if x is not None}
    test_raw = find_raw_videos(raw_ds, "test")
    return groups, [v for v, _, _ in test_raw], sum(x is None for _, _, x in test_raw)


def write_dataset(ds: str, raw_root: Path, out_dir: Path, val_frac: float, seed: int, force: bool,
                  log) -> dict[str, list[str]] | None:
    groups, test, test_no_gt = collect(raw_root / ds, log)
    if not groups:
        log.error("%s: no annotated train videos under %s", ds, raw_root / ds / "train")
        return None
    splits = make_splits(groups, test, val_frac, seed)
    paths = {s: out_dir / f"{ds}_{s}.txt" for s in SPLITS}
    changed = [s for s in SPLITS if not paths[s].exists() or read_split(paths[s]) != splits[s]]
    if changed and not force and any(paths[s].exists() for s in changed):
        log.error("%s: existing split files differ (%s); use --force to overwrite", ds, changed)
        return None
    for s in SPLITS:
        write_split(paths[s], splits[s])
    log.info("%s: train %d / val %d / test %d (seed %d, val_frac %.2f) %s", ds,
             len(splits["train"]), len(splits["val"]), len(splits["test"]), seed, val_frac,
             "written" if changed else "unchanged")
    if test_no_gt:
        log.warning("%s: %d/%d test videos have no annotations; val acts as the held-out test set",
                    ds, test_no_gt, len(test))
    return splits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ds", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--config", default="configs/base.yaml", help="seed is read from here")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--raw-root", type=Path, default=project_root() / "data" / "raw")
    ap.add_argument("--out", type=Path, default=project_root() / "splits")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    log = get_logger("make_splits", project_root() / "logs" / "make_splits.log")
    seed = int(load_yaml(args.config)["seed"])
    results = [write_dataset(ds, args.raw_root, args.out, args.val_frac, seed, args.force, log)
               for ds in args.ds]
    return 0 if all(r is not None for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
