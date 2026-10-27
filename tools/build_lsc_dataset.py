"""Builds the LSC (language synergy classifier) crop dataset from GT and detector outputs.

Positives: GT instances with dont_care false; label = lowercase text restricted to the ABINet charset
(instances with nothing left are dropped). Negatives: det boxes (runs/<det_exp>/det/<ds>/<video>.json)
with score > lsc.det_thr whose IoU with EVERY GT quad of the frame (dont_care included) is
< lsc.iou_neg; label "". Per split, positives:negatives = 1:1 (both capped at the smaller count,
sampled with the config seed). Frames 1, 1+s, ... with s = lsc.stride (default 1).

Writes data/lsc/<split>/{images/<ds>/<video>/<frame>_<pos|neg><k>.png, labels.tsv, stats.json} and
runs/<exp_id>/{config.yaml,log.txt,metrics.json meta.lsc_dataset_<split>}.

Usage: python tools/build_lsc_dataset.py --config configs/base.yaml [--det_exp EXP]
           [--datasets icdar2015v] [--splits train val] [--final] [--out data/lsc]
"""
from __future__ import annotations

import argparse
import copy
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from logo.data.schema import list_videos, load_gt
from logo.detector.infer_paddle import write_run_files
from logo.lsc.dataset import clean_label, load_charset
from logo.lsc.roi_rotate import crop_many
from logo.utils.geometry import quad_iou
from logo.utils.io import get_logger, load_json, load_yaml, save_json, set_seed
from logo.utils.paths import DATASETS, check_split, frame_path, gt_path, project_root, stage_dir

OUT_HW = (32, 128)


def is_negative(quad, gt_quads, iou_neg: float) -> bool:
    """True if `quad` overlaps no GT quad: IoU < iou_neg with all of them (vacuously true if none)."""
    return all(quad_iou(quad, g) < iou_neg for g in gt_quads)


def select_negatives(dets: list[dict], gt_quads: list, det_thr: float, iou_neg: float,
                     counts: Optional[Counter] = None) -> list[dict]:
    """Det objects with score > det_thr that are negatives w.r.t. all `gt_quads`."""
    out = []
    for d in dets:
        if float(d["score"]) <= det_thr:
            if counts is not None:
                counts["neg_low_score"] += 1
        elif not is_negative(d["quad"], gt_quads, iou_neg):
            if counts is not None:
                counts["neg_overlap"] += 1
        else:
            out.append(d)
    return out


def select_positives(objs: list[dict], charset: str, counts: Optional[Counter] = None) -> list[tuple]:
    """(quad, label) for non-dont_care GT objects whose cleaned label is non-empty."""
    out = []
    for o in objs:
        if o["dont_care"]:
            if counts is not None:
                counts["pos_dont_care"] += 1
            continue
        label = clean_label(o["text"], charset)
        if not label:
            if counts is not None:
                counts["pos_empty_label"] += 1
            continue
        out.append((o["quad"], label))
    return out


def balance(pos: list, neg: list, rng: np.random.Generator) -> tuple[list, list]:
    """Random subsets of equal size min(len(pos), len(neg)); original order kept."""
    n = min(len(pos), len(neg))
    pick = lambda xs: [xs[i] for i in sorted(rng.choice(len(xs), n, replace=False))] if n else []
    return pick(pos), pick(neg)


def collect(ds: str, videos: list[str], det_exp: str, stride: int, det_thr: float,
            iou_neg: float, charset: str, counts: Counter, log) -> tuple[list, list]:
    """Candidate records (ds, video, frame, quad, label) without image I/O."""
    pos, neg = [], []
    for video in videos:
        gp = gt_path(ds, video)
        if not gp.is_file():
            log.warning("%s/%s: GT missing, skipped", ds, video)
            counts["videos_missing_gt"] += 1
            continue
        gt = load_gt(gp)
        dp = stage_dir(det_exp, "det", ds) / f"{video}.json"
        det_frames = load_json(dp)["frames"] if dp.is_file() else None
        if det_frames is None:
            log.warning("%s/%s: no det json at %s, no negatives", ds, video, dp)
            counts["videos_missing_det"] += 1
        counts["videos"] += 1
        for fid in range(1, int(gt["num_frames"]) + 1, stride):
            if not frame_path(ds, video, fid).is_file():
                counts["frames_missing"] += 1
                continue
            counts["frames"] += 1
            objs = gt["frames"].get(str(fid), [])
            for quad, label in select_positives(objs, charset, counts):
                pos.append((ds, video, fid, quad, label))
            if det_frames is not None:
                gq = [o["quad"] for o in objs]
                for d in select_negatives(det_frames.get(str(fid), []), gq, det_thr, iou_neg, counts):
                    neg.append((ds, video, fid, d["quad"], ""))
    return pos, neg


def write_crops(root_split: Path, records: list[tuple]) -> list[tuple[str, str]]:
    """Crops records (grouped by frame, each frame read once) to PNGs; returns (rel_path, label)."""
    rows = []
    groups: dict[tuple, list] = {}
    for r in records:
        groups.setdefault(r[:3], []).append(r)
    for (ds, video, fid), recs in groups.items():
        img = cv2.imread(str(frame_path(ds, video, fid)), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(frame_path(ds, video, fid))
        crops = crop_many(img, [r[3] for r in recs], OUT_HW)
        d = root_split / "images" / ds / video
        d.mkdir(parents=True, exist_ok=True)
        k = Counter()
        for r, c in zip(recs, crops):
            kind = "neg" if r[4] == "" else "pos"
            name = f"{fid:06d}_{kind}{k[kind]}.png"
            k[kind] += 1
            if not cv2.imwrite(str(d / name), c):
                raise OSError(f"cannot write {d / name}")
            rows.append(((d / name).relative_to(root_split).as_posix(), r[4]))
    return rows


def build_split(cfg: dict, split: str, datasets: list[str], det_exp: str, out: Path, log,
                final: bool = False, charset: Optional[str] = None) -> dict:
    check_split(split, final)
    lsc = cfg["lsc"]
    stride = int(lsc.get("stride", 1))
    if stride < 1:
        raise ValueError("lsc.stride must be >= 1")
    det_thr, iou_neg = float(lsc["det_thr"]), float(lsc["iou_neg"])
    charset = load_charset() if charset is None else charset
    counts: Counter = Counter()
    pos, neg = [], []
    for ds in datasets:
        p, n = collect(ds, list_videos(ds, split), det_exp, stride, det_thr, iou_neg, charset,
                       counts, log)
        pos += p
        neg += n
    rng = np.random.default_rng(int(cfg["seed"]))
    kp, kn = balance(pos, neg, rng)
    if not kp:
        log.warning("%s: 0 samples after balancing (pos=%d neg=%d)", split, len(pos), len(neg))

    split_dir = out / split
    if (split_dir / "images").exists():  # a rerun must not leave stale crops
        shutil.rmtree(split_dir / "images")
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = write_crops(split_dir, kp + kn)
    tsv = split_dir / "labels.tsv"
    tsv.write_text("".join(f"{p}\t{l}\n" for p, l in rows), encoding="utf-8")

    hist = Counter(len(l) for _, l in rows)
    stats = {"split": split, "datasets": datasets, "det_exp": det_exp, "stride": stride,
             "det_thr": det_thr, "iou_neg": iou_neg, "seed": int(cfg["seed"]), "charset": charset,
             "candidates": {"pos": len(pos), "neg": len(neg)},
             "kept": {"pos": len(kp), "neg": len(kn), "total": len(rows)},
             "counts": dict(sorted(counts.items())),
             "len_hist": {str(k): hist[k] for k in sorted(hist)}}
    save_json(stats, split_dir / "stats.json")
    log.info("%s: candidates pos=%d neg=%d -> kept %d+%d (%s)", split, len(pos), len(neg), len(kp),
             len(kn), split_dir)
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--det_exp", default=None, help="exp id holding det/ outputs; default config exp_id")
    ap.add_argument("--datasets", nargs="+", default=None, choices=DATASETS, help="default: config dataset")
    ap.add_argument("--splits", nargs="+", default=None, choices=("train", "val", "test"),
                    help="default: train val (+test with --final)")
    ap.add_argument("--final", action="store_true", help="allow the test split (rule 5)")
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/data/lsc")
    args = ap.parse_args(argv)

    root = project_root()
    cfg = copy.deepcopy(load_yaml(args.config))
    datasets = args.datasets or [cfg["dataset"]]
    splits = args.splits or (["train", "val"] + (["test"] if args.final else []))
    for s in splits:
        check_split(s, args.final)
    det_exp = args.det_exp or cfg["exp_id"]
    out = args.out if args.out is not None else root / "data" / "lsc"

    set_seed(cfg["seed"])
    run_dir = root / "runs" / cfg["exp_id"]
    log = get_logger("build_lsc_dataset", run_dir / "log.txt")
    log.info("exp=%s det_exp=%s datasets=%s splits=%s out=%s", cfg["exp_id"], det_exp, datasets, splits, out)
    charset = load_charset()
    for split in splits:
        stats = build_split(cfg, split, datasets, det_exp, out, log, args.final, charset)
        write_run_files(cfg, run_dir, {k: v for k, v in stats.items() if k != "charset"},
                        key=f"lsc_dataset_{split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
