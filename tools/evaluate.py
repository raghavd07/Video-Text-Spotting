"""Evaluate detections or tracks against GT and write/update metrics.json (CONTRACT layout).

det  : <pred_dir>/<video>.json (det/lsc/fused; falls back to .jsonl) -> metrics["det"]
track: <pred_dir>/<video>.jsonl -> metrics["track"], metrics["per_video"]
spot : <pred_dir>/<video>.jsonl -> metrics["spot"]; its per-video counts go to meta["spot"]["per_video"]
An existing --out file is updated, so det/track/spot can be run one after another into one file.
det mode only: --score_thr T keeps predictions with o[--score_key] > T (filtered copies are written to a
temporary dir; the threshold is recorded in meta.det).
log.txt and (if absent) config.yaml are written next to --out. A *_test.txt split needs --final.

Usage: python tools/evaluate.py --gt_dir data/processed/icdar2015v --pred_dir runs/<exp>/track/icdar2015v
           --videos splits/icdar2015v_val.txt --mode track --out runs/<exp>/metrics.json
           [--iou 0.5] [--config configs/base.yaml] [--final] [--score_key score_fused --score_thr 0.3]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

from logo.data.schema import read_split
from logo.eval.det_eval import evaluate_detection
from logo.eval.mot_eval import evaluate_mot
from logo.utils.io import get_logger, load_json, load_yaml, save_json
from logo.utils.paths import SPLITS, check_split

MODES = ("det", "track", "spot")
MOT_KEYS = ("MOTA", "MOTP", "IDF1", "MT", "ML")


def empty_metrics() -> dict:
    return {"det": {}, "track": {}, "spot": {}, "per_video": {}, "meta": {}}


def filter_preds(pred_dir: Path, videos: list[str], out_dir: Path, key: str, thr: float) -> Path:
    """Copies <video>.json from pred_dir into out_dir keeping objects with o[key] > thr."""
    for video in videos:
        d = load_json(Path(pred_dir) / f"{video}.json")
        d["frames"] = {fid: [o for o in objs if float(o[key]) > thr] for fid, objs in d["frames"].items()}
        save_json(d, out_dir / f"{video}.json")
    return out_dir


def run(gt_dir: Path, pred_dir: Path, videos: list[str], mode: str, iou: float) -> tuple[dict, dict]:
    """(section, meta) for one mode; section is what goes under metrics[mode]."""
    if mode == "det":
        r = evaluate_detection(gt_dir, pred_dir, videos, iou=iou)
        return ({k: r[k] for k in ("precision", "recall", "f")},
                {"counts": r["counts"], "per_video": r["per_video"]})
    r = evaluate_mot(gt_dir, pred_dir, videos, spotting=(mode == "spot"), iou=iou)
    return {k: r[k] for k in MOT_KEYS}, {"counts": r["counts"], "per_video": r["per_video"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt_dir", type=Path, required=True)
    ap.add_argument("--pred_dir", type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True, help="split file, one video id per line")
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iou", type=float, default=None, help="default: config eval.iou, else 0.5")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--final", action="store_true", help="allow the test split (rule 5)")
    ap.add_argument("--score_key", default="score", help="det mode: score field used by --score_thr")
    ap.add_argument("--score_thr", type=float, default=None, help="det mode: keep o[score_key] > thr")
    args = ap.parse_args(argv)

    if args.score_thr is not None and args.mode != "det":
        ap.error("--score_thr is only supported with --mode det")
    split = args.videos.stem.rsplit("_", 1)[-1]
    if split in SPLITS:
        check_split(split, final=args.final)
    cfg = load_yaml(args.config) if args.config else {}
    iou = args.iou if args.iou is not None else float(cfg.get("eval", {}).get("iou", 0.5))

    out_dir = args.out.parent
    log = get_logger("evaluate", out_dir / "log.txt")
    cfg_out = out_dir / "config.yaml"
    if not cfg_out.exists():
        dump = cfg or {"exp_id": out_dir.name, "eval": {"iou": iou, "spotting": args.mode == "spot"}}
        cfg_out.write_text(yaml.safe_dump(dump, sort_keys=False), encoding="utf-8")

    videos = read_split(args.videos)
    log.info("mode=%s iou=%.2f videos=%d gt=%s pred=%s", args.mode, iou, len(videos), args.gt_dir,
             args.pred_dir)
    if args.score_thr is None:
        section, meta = run(args.gt_dir, args.pred_dir, videos, args.mode, iou)
    else:
        log.info("keeping predictions with %s > %.3f", args.score_key, args.score_thr)
        with tempfile.TemporaryDirectory() as tmp:
            pred = filter_preds(args.pred_dir, videos, Path(tmp), args.score_key, args.score_thr)
            section, meta = run(args.gt_dir, pred, videos, args.mode, iou)
        meta.update(score_key=args.score_key, score_thr=args.score_thr)

    metrics = empty_metrics()
    if args.out.is_file():
        metrics.update(load_json(args.out))
    metrics[args.mode] = section
    if args.mode == "track":
        metrics["per_video"] = meta.pop("per_video")
    meta.update({"gt_dir": str(args.gt_dir), "pred_dir": str(args.pred_dir),
                 "videos": str(args.videos), "num_videos": len(videos), "iou": iou})
    metrics.setdefault("meta", {})[args.mode] = meta
    save_json(metrics, args.out)
    log.info("%s: %s", args.mode, " ".join(f"{k}={v:.2f}" for k, v in section.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
