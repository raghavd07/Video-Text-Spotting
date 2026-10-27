"""Exports det json for all videos of train/val (+test with --final) via logo.detector.infer_paddle, save_thr 0.1.

The predictor is built once; infer_paddle.run is called per (dataset, split) and writes
runs/<exp_id>/det/<ds>/<video>.json. Per-split timing: runs/<exp_id>/det/timing_<ds>_<split>.json and
metrics.json meta.det_infer_<ds>_<split> (det/timing.json and meta.det_infer hold the last split).
Existing video jsons are skipped unless --overwrite, so an interrupted export resumes.

Usage: python tools/export_all_dets.py --config configs/det_ppyoloe_r_text_s.yaml
           [--datasets icdar2015v dstext] [--splits train val] [--final] [--weights P] [--save_thr 0.1]
           [--max_frames N] [--device gpu|cpu] [--overwrite]
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Callable, Optional

from logo.detector.infer_paddle import (PaddleDetector, Predictor, read_videos, resolve, run,
                                        write_run_files)
from logo.utils.io import get_logger, load_yaml, save_json, set_seed
from logo.utils.paths import DATASETS, check_split, project_root


def plan_splits(splits: Optional[list[str]], final: bool) -> list[str]:
    if splits is None:
        splits = ["train", "val"] + (["test"] if final else [])
    return [check_split(s, final) for s in splits]


def export(cfg: dict, datasets: list[str], splits: list[str], predictor: Predictor, root: Path, log,
           final: bool = False, max_frames: Optional[int] = None, overwrite: bool = False,
           peak_mem_fn: Optional[Callable[[], Optional[float]]] = None) -> dict:
    run_dir = root / "runs" / cfg["exp_id"]
    write_run_files(cfg, run_dir, {"datasets": datasets, "splits": splits,
                                   "save_thr": cfg["detector"]["save_thr"],
                                   "weights": str(cfg["detector"]["weights"])}, key="det_export")
    out = {}
    for ds in datasets:
        for split in splits:
            videos = read_videos(ds, split, root, final=final)
            log.info("=== %s/%s: %d videos", ds, split, len(videos))
            timing = run({**cfg, "dataset": ds, "split": split}, videos, predictor, root, log,
                         max_frames, overwrite, peak_mem_fn)
            key = f"{ds}_{split}"
            save_json(timing, run_dir / "det" / f"timing_{key}.json")
            write_run_files(cfg, run_dir, timing, key=f"det_infer_{key}")
            out[key] = timing
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=None, choices=DATASETS,
                    help="default: config dataset")
    ap.add_argument("--splits", nargs="+", default=None, choices=("train", "val", "test"),
                    help="default: train val (+test with --final)")
    ap.add_argument("--final", action="store_true", help="allow the test split (rule 5)")
    ap.add_argument("--weights", type=Path, default=None, help="default: config detector.weights")
    ap.add_argument("--save_thr", type=float, default=0.1)
    ap.add_argument("--max_frames", type=int, default=None, help="per video (smoke runs)")
    ap.add_argument("--device", choices=("gpu", "cpu"), default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    root = project_root()
    cfg = copy.deepcopy(load_yaml(args.config))
    det = cfg["detector"]
    det["save_thr"] = args.save_thr
    if args.weights is not None:
        det["weights"] = str(args.weights)
    datasets = args.datasets or [cfg["dataset"]]
    splits = plan_splits(args.splits, args.final)

    set_seed(cfg["seed"])
    log = get_logger("export_all_dets", root / "runs" / cfg["exp_id"] / "log.txt")
    device = args.device or det.get("device", "gpu")
    log.info("exp=%s datasets=%s splits=%s save_thr=%s weights=%s device=%s", cfg["exp_id"], datasets,
             splits, det["save_thr"], det["weights"], device)
    predictor = PaddleDetector(resolve(det["config"], root), resolve(det["weights"], root),
                               det.get("input_hw"), device, int(cfg["seed"]))
    export(cfg, datasets, splits, predictor, root, log, args.final, args.max_frames, args.overwrite,
           predictor.peak_memory_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
