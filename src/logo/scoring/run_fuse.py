"""Fusion stage: lsc json -> fused json (lsc objects plus score_fused = fuse(score, lang_hard)).

Reads runs/<lsc_exp>/lsc/<ds>/<video>.json, writes runs/<exp_id>/fuse/<ds>/<video>.json and
runs/<exp_id>/{config.yaml,log.txt,metrics.json meta.fuse_<ds>_<split>}. feat_idx still refers to
runs/<lsc_exp>/lsc/<ds>/<video>_feats.npy (features are not copied).

Usage: python -m logo.scoring.run_fuse --config configs/E3-lsc.yaml [--lsc_exp EXP]
           [--datasets icdar2015v] [--splits val] [--final]
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from logo.detector.infer_paddle import read_videos, write_run_files
from logo.scoring.fusion import fuse
from logo.utils.io import get_logger, load_json, load_yaml, save_json, set_seed
from logo.utils.paths import DATASETS, check_split, project_root

STAGE = "fuse"


def fuse_video(lsc: dict, mode: str, w: float) -> dict:
    frames = {fid: [{**o, "score_fused": fuse(o["score"], o["lang_hard"], mode, w)} for o in objs]
              for fid, objs in lsc["frames"].items()}
    return {**{k: v for k, v in lsc.items() if k != "frames"}, "frames": frames}


def run(cfg: dict, ds: str, videos: list[str], root: Path, log, lsc_exp: str) -> dict:
    mode, w = cfg["fusion"]["mode"], float(cfg["fusion"]["w"])
    lsc_dir = root / "runs" / lsc_exp / "lsc" / ds
    out_dir = root / "runs" / cfg["exp_id"] / STAGE / ds
    n = 0
    for video in videos:
        fused = fuse_video(load_json(lsc_dir / f"{video}.json"), mode, w)
        save_json(fused, out_dir / f"{video}.json")
        k = sum(len(v) for v in fused["frames"].values())
        n += k
        log.info("%s: %d boxes fused", video, k)
    return {"videos": len(videos), "boxes": n, "mode": mode, "w": w, "lsc_exp": lsc_exp}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--lsc_exp", default=None, help="exp id holding lsc/ outputs; default config exp_id")
    ap.add_argument("--datasets", nargs="+", default=None, choices=DATASETS, help="default: config dataset")
    ap.add_argument("--splits", nargs="+", default=None, choices=("train", "val", "test"),
                    help="default: config split")
    ap.add_argument("--final", action="store_true", help="allow the test split (rule 5)")
    args = ap.parse_args(argv)

    root = project_root()
    cfg = copy.deepcopy(load_yaml(args.config))
    datasets = args.datasets or [cfg["dataset"]]
    splits = [check_split(s, args.final) for s in (args.splits or [cfg["split"]])]
    lsc_exp = args.lsc_exp or cfg["exp_id"]
    set_seed(cfg["seed"])
    run_dir = root / "runs" / cfg["exp_id"]
    log = get_logger("run_fuse", run_dir / "log.txt")
    log.info("exp=%s lsc_exp=%s fusion=%s", cfg["exp_id"], lsc_exp, cfg["fusion"])
    for ds in datasets:
        for split in splits:
            stats = run(cfg, ds, read_videos(ds, split, root, final=args.final), root, log, lsc_exp)
            write_run_files(cfg, run_dir, stats, key=f"fuse_{ds}_{split}")
            log.info("%s/%s: %s", ds, split, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
