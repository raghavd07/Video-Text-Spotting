"""Sweep tracker.thr (LST input score threshold) -> CSV + PNG of MOTA / IDF1 for tracking and spotting.

VPMM vectors are computed once per video; for every thr the LST association is rerun on the fused json and
post-processed as in logo.tracking.run_track (mean-score filter tracker.traj_thr for both; the
empty/single-character text filter only for the spotting tracks). Tracks are written to
<out_dir>/thr_<t>/{track,spot}/<ds>/<video>.jsonl and scored with logo.eval.mot_eval.evaluate_mot.
Outputs <out_dir>/sweep.csv, <out_dir>/sweep.png (default out_dir runs/<exp_id>/sweep_thr/<ds>_<split>) and
runs/<exp_id>/{config.yaml,log.txt,metrics.json meta.sweep_thr_<ds>_<split>}. The test split is refused.

Usage: python tools/sweep_threshold.py --config configs/E5-lst.yaml [--fuse_exp EXP] [--lsc_exp EXP]
           [--dataset icdar2015v] [--split val] [--thrs 0.3 0.4 ... 0.9] [--device cuda] [--out_dir DIR]
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import torch

from logo.detector.infer_paddle import read_videos, write_run_files
from logo.eval.mot_eval import evaluate_mot
from logo.tracking.run_track import (embed_ckpt, filter_trajectories, load_lst, run_lst, save_jsonl, traj_thr,
                                     video_inputs, vpmm_features)
from logo.tracking.simple_tracker import tracker_settings
from logo.utils.io import get_logger, load_json, load_yaml, set_seed
from logo.utils.paths import DATASETS, project_root

THRS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
KEYS = ("MOTA", "MOTP", "IDF1", "MT", "ML")
COLORS = {"MOTA": "#2a78d6", "IDF1": "#eb6834"}  # dataviz default categorical slots 1, 2


def tag(thr: float) -> str:
    return f"thr_{thr:.2f}"


def sweep(cfg: dict, ds: str, split: str, thrs, root: Path, out_dir: Path, log, fuse_exp: str,
          lsc_exp: str, device) -> list[dict]:
    """One table row per thr: {"thr", "track_<K>", "spot_<K>"} for K in KEYS."""
    if split == "test":
        raise PermissionError("sweep_threshold never uses the test split")
    videos = read_videos(ds, split, root)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vpmm, lst = load_lst(embed_ckpt(cfg, root), cfg, device)
    min_score, window = traj_thr(cfg), tracker_settings(cfg)["text_window"]
    for video in videos:
        fused = load_json(root / "runs" / fuse_exp / "fuse" / ds / f"{video}.json")
        feats, wh = video_inputs(root, ds, video, lsc_exp)
        emb = vpmm_features(fused, feats, vpmm, wh, device, device.type == "cuda")
        del feats
        for thr in thrs:
            rows = run_lst(fused, emb, lst, wh, thr, window)
            for mode, spotting in (("track", False), ("spot", True)):
                save_jsonl(filter_trajectories(rows, min_score, spotting),
                           out_dir / tag(thr) / mode / ds / f"{video}.jsonl")
        log.info("%s: tracked at %d thresholds", video, len(thrs))
    gt_dir = root / "data" / "processed" / ds
    iou = float((cfg.get("eval") or {}).get("iou", 0.5))
    table = []
    for thr in thrs:
        row = {"thr": round(float(thr), 4)}
        for mode in ("track", "spot"):
            r = evaluate_mot(gt_dir, out_dir / tag(thr) / mode / ds, videos, spotting=(mode == "spot"), iou=iou)
            row.update({f"{mode}_{k}": round(float(r[k]), 4) for k in KEYS})
        table.append(row)
        log.info("thr=%.2f track MOTA=%.2f IDF1=%.2f | spot MOTA=%.2f IDF1=%.2f", thr, row["track_MOTA"],
                 row["track_IDF1"], row["spot_MOTA"], row["spot_IDF1"])
    return table


def write_csv(table: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


def plot(table: list[dict], p: Path, title: str) -> None:
    """Two panels (tracking, spotting), MOTA and IDF1 vs thr on one percent axis; best IDF1 labelled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["thr"] for r in table]
    vals = [r[f"{m}_{k}"] for r in table for m in ("track", "spot") for k in COLORS]
    lo, hi = min(0.0, min(vals)), max(100.0, max(vals))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for ax, (mode, name) in zip(axes, (("track", "Tracking"), ("spot", "Spotting"))):
        for k, c in COLORS.items():
            y = [r[f"{mode}_{k}"] for r in table]
            ax.plot(x, y, color=c, lw=2, marker="o", ms=5, label=k)
        best = max(table, key=lambda r: r[f"{mode}_IDF1"])
        ax.annotate(f"IDF1 {best[f'{mode}_IDF1']:.1f} @ {best['thr']:.1f}", (best["thr"], best[f"{mode}_IDF1"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color="#52514e")
        ax.set_title(name, fontsize=11, loc="left")
        ax.set_xlabel("tracker.thr")
        ax.set_xticks(x)
        ax.set_ylim(lo - 5, hi + 5)
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("%")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(title, fontsize=11)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--fuse_exp", default=None, help="exp id holding fuse/ outputs; default config exp_id")
    ap.add_argument("--lsc_exp", default=None, help="default config pipeline.lsc_exp, else fuse_exp")
    ap.add_argument("--dataset", default=None, choices=DATASETS, help="default: config dataset")
    ap.add_argument("--split", default=None, help="train or val (default: config split); test is refused")
    ap.add_argument("--thrs", nargs="+", type=float, default=list(THRS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = copy.deepcopy(load_yaml(args.config))
    ds, split = args.dataset or cfg["dataset"], args.split or cfg["split"]
    if split not in ("train", "val"):
        ap.error(f"split {split!r} refused: threshold sweeps run on train/val only (never test)")
    if (cfg.get("tracker") or {}).get("type") != "lst":
        ap.error("tracker.thr sweeps need tracker.type lst")
    root = project_root()
    fuse_exp = args.fuse_exp or cfg["exp_id"]
    lsc_exp = args.lsc_exp or (cfg.get("pipeline") or {}).get("lsc_exp") or fuse_exp
    run_dir = root / "runs" / cfg["exp_id"]
    out_dir = args.out_dir or run_dir / "sweep_thr" / f"{ds}_{split}"
    set_seed(cfg["seed"])
    log = get_logger("sweep_threshold", run_dir / "log.txt")
    log.info("exp=%s ds=%s split=%s fuse_exp=%s lsc_exp=%s thrs=%s", cfg["exp_id"], ds, split, fuse_exp,
             lsc_exp, args.thrs)
    table = sweep(cfg, ds, split, args.thrs, root, out_dir, log, fuse_exp, lsc_exp, args.device)
    write_csv(table, out_dir / "sweep.csv")
    plot(table, out_dir / "sweep.png", f"{cfg['exp_id']} {ds}/{split}: MOTA, IDF1 vs tracker.thr")
    best = max(table, key=lambda r: (r["track_IDF1"], r["track_MOTA"]))
    write_run_files(cfg, run_dir, {"fuse_exp": fuse_exp, "lsc_exp": lsc_exp, "traj_thr": traj_thr(cfg),
                                   "csv": str(out_dir / "sweep.csv"), "png": str(out_dir / "sweep.png"),
                                   "best_track_thr": best["thr"], "table": table},
                    key=f"sweep_thr_{ds}_{split}")
    log.info("best tracking thr=%.2f (IDF1 %.2f, MOTA %.2f)", best["thr"], best["track_IDF1"], best["track_MOTA"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
