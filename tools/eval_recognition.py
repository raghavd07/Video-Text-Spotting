"""Word accuracy of the ABINet recogniser on rectified GT crops (non-dont_care) of one split.

Crops come from logo.lsc.roi_rotate.crop_many on data/frames; GT words are compared case-insensitively
on alphanumerics only ([a-z0-9]); GT words with no alphanumeric character are skipped (not counted).
Writes runs/<exp_id>/recog/<ds>/<split>_predictions.csv, and metrics.json -> meta.recog.<ds>.<split>
(word_acc in percent), plus log.txt and (if absent) config.yaml. The test split needs --final.

Usage: python tools/eval_recognition.py --config configs/base.yaml --ds icdar2015v
           [--split val] [--exp-id recog_abinet] [--ckpt third_party/ABINet/workdir/train-abinet/best-train-abinet.pth]
           [--device cuda] [--no-fp16] [--batch 64] [--stride 1] [--videos V1 V2 ...] [--final]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from logo.data.schema import list_videos, load_gt
from logo.lsc.roi_rotate import crop_many
from logo.utils.io import get_logger, load_json, load_yaml, save_json, set_seed
from logo.utils.paths import DATASETS, SPLITS, check_split, frame_path, gt_path, project_root, stage_dir

STAGE = "recog"
CSV_FIELDS = ("video", "frame", "track_id", "quality", "gt", "gt_norm", "pred", "correct", "text_conf",
              "p_eos")
_NON_ALNUM = re.compile(r"[^0-9a-z]")


def normalize(s: str) -> str:
    return _NON_ALNUM.sub("", str(s).lower())


def make_recognizer(ckpt, device, fp16):
    from logo.lsc.abinet_wrapper import ABINetRecognizer
    return ABINetRecognizer(ckpt, device, fp16=fp16)


def read_frame(ds: str, video: str, fid: int) -> np.ndarray | None:
    p = frame_path(ds, video, fid)
    if not p.is_file():
        return None
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def iter_gt_crops(ds: str, video: str, stride: int, counts: dict):
    """Yields (row, crop) for usable GT objects, frame by frame (one frame in memory at a time)."""
    gt = load_gt(gt_path(ds, video))
    for fid in sorted(gt["frames"], key=int):
        if (int(fid) - 1) % stride:
            continue
        objs = []
        for o in gt["frames"][fid]:
            if o.get("dont_care"):
                counts["dont_care"] += 1
            elif not normalize(o.get("text", "")):
                counts["non_alnum"] += 1
            else:
                objs.append(o)
        if not objs:
            continue
        img = read_frame(ds, video, int(fid))
        if img is None:
            counts["missing_frames"] += 1
            continue
        for o, crop in zip(objs, crop_many(img, [o["quad"] for o in objs])):
            row = {"video": video, "frame": int(fid), "track_id": o.get("track_id"),
                   "quality": o.get("quality"), "gt": o.get("text", ""), "gt_norm": normalize(o["text"])}
            yield row, crop


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--ds", choices=DATASETS, default=None, help="default: config dataset")
    ap.add_argument("--split", choices=SPLITS, default="val")
    ap.add_argument("--exp-id", default=None, help="default: <config exp_id>_recog")
    ap.add_argument("--ckpt", type=Path, default=None, help="default: config lsc.ckpt, else ABINet default")
    ap.add_argument("--device", default="cuda" if _cuda() else "cpu")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1, help="use every k-th frame")
    ap.add_argument("--videos", nargs="*", default=None, help="subset of the split's videos")
    ap.add_argument("--final", action="store_true", help="allow the test split (rule 5)")
    args = ap.parse_args(argv)

    check_split(args.split, final=args.final)
    cfg = load_yaml(args.config) if args.config.is_file() else {}
    ds = args.ds or cfg.get("dataset")
    if ds not in DATASETS:
        ap.error(f"--ds required (config dataset is {ds!r})")
    exp_id = args.exp_id or f"{cfg.get('exp_id', 'base')}_recog"
    ckpt = args.ckpt or (cfg.get("lsc") or {}).get("ckpt")
    set_seed(int(cfg.get("seed", 0)))

    run_dir = project_root() / "runs" / exp_id
    log = get_logger("eval_recognition", run_dir / "log.txt")
    if not (run_dir / "config.yaml").exists():
        dump = dict(cfg, exp_id=exp_id, dataset=ds, split=args.split)
        dump["lsc"] = dict(cfg.get("lsc") or {}, ckpt=None if ckpt is None else str(ckpt))
        (run_dir / "config.yaml").write_text(yaml.safe_dump(dump, sort_keys=False), encoding="utf-8")

    videos = list_videos(ds, args.split)
    if args.videos:
        unknown = sorted(set(args.videos) - set(videos))
        if unknown:
            log.error("videos not in %s_%s: %s", ds, args.split, unknown)
            return 1
        videos = [v for v in videos if v in set(args.videos)]

    rec = make_recognizer(ckpt, args.device, not args.no_fp16)
    rec.batch_size = args.batch
    log.info("ds=%s split=%s videos=%d device=%s fp16=%s c=%s d=%s ckpt=%s", ds, args.split,
             len(videos), args.device, not args.no_fp16, getattr(rec, "c", "?"), getattr(rec, "d", "?"),
             ckpt or "default")

    counts = {"dont_care": 0, "non_alnum": 0, "missing_frames": 0}
    n = correct = 0
    conf_sum = 0.0
    per_video: dict[str, dict] = {}
    out_csv = stage_dir(exp_id, STAGE, ds) / f"{args.split}_predictions.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        def flush(rows, crops):
            nonlocal n, correct, conf_sum
            for row, p in zip(rows, rec.predict(crops)):
                ok = normalize(p["text"]) == row["gt_norm"]
                writer.writerow(dict(row, pred=p["text"], correct=int(ok),
                                     text_conf=f'{p["text_conf"]:.4f}', p_eos=f'{p["p_eos"]:.4f}'))
                pv = per_video.setdefault(row["video"], {"n": 0, "correct": 0})
                pv["n"] += 1
                pv["correct"] += int(ok)
                n += 1
                correct += int(ok)
                conf_sum += p["text_conf"]

        for video in videos:
            rows, crops = [], []
            for row, crop in iter_gt_crops(ds, video, max(1, args.stride), counts):
                rows.append(row)
                crops.append(crop)
                if len(crops) >= rec.batch_size:
                    flush(rows, crops)
                    rows, crops = [], []
            if crops:
                flush(rows, crops)
            pv = per_video.get(video, {"n": 0, "correct": 0})
            log.info("%s: %d words, acc %.2f%%", video, pv["n"], 100.0 * pv["correct"] / max(pv["n"], 1))

    result = {"word_acc": 100.0 * correct / n if n else 0.0, "num_words": n, "num_correct": correct,
              "mean_text_conf": conf_sum / n if n else 0.0, "skipped": counts, "stride": args.stride,
              "videos": len(videos), "csv": str(out_csv),
              "per_video": {v: dict(d, word_acc=100.0 * d["correct"] / max(d["n"], 1))
                            for v, d in per_video.items()}}
    metrics_path = run_dir / "metrics.json"
    metrics = {"det": {}, "track": {}, "spot": {}, "per_video": {}, "meta": {}}
    if metrics_path.is_file():
        metrics.update(load_json(metrics_path))
    metrics["meta"].setdefault("recog", {}).setdefault(ds, {})[args.split] = result
    save_json(metrics, metrics_path)
    log.info("%s/%s: word_acc %.2f%% (%d/%d), skipped %s -> %s", ds, args.split, result["word_acc"],
             correct, n, counts, out_csv)
    return 0


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    sys.exit(main())
