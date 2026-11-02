"""Sweep fusion.mode x fusion.w x tracker.thr on cached LSC outputs -> CSV + PNGs of MOTA / IDF1 vs thr.

Only fuse, track and eval run: runs/<lsc_exp>/lsc/<ds>/<video>.json is fused in memory with
logo.scoring.run_fuse.fuse_video (fusion.soft_source from the config or --soft_source) for every (mode, w);
detections with score_fused < thr are dropped and the config's tracker (tracker.type simple | lst) is run as in
logo.tracking.run_track (lst: VPMM vectors once per video, trajectory mean-score filter tracker.traj_thr, text
filter for the spotting tracks; simple: thr is applied as this input filter, the tracker itself unchanged).
Tracks go to <out_dir>/<mode>_w<w>/thr_<t>/{track,spot}/<ds>/<video>.jsonl and are scored with
logo.eval.mot_eval.evaluate_mot (voting is not applied). Outputs <out_dir>/sweep_fusion.csv,
<out_dir>/sweep_fusion_mota.png, <out_dir>/sweep_fusion_idf1.png (default out_dir
runs/<exp_id>/sweep_fusion/<ds>_<split>) and runs/<exp_id>/{config.yaml,log.txt,metrics.json
meta.sweep_fusion_<ds>_<split>}. The test split is refused.

Usage: python tools/sweep_fusion.py --config configs/E5-lst.yaml [--lsc_exp EXP] [--dataset icdar2015v]
           [--split val] [--modes hard soft] [--ws 0.3 0.5 0.7] [--thrs 0.4 ... 0.9]
           [--soft_source p_eos|text_conf] [--device cuda] [--out_dir DIR]
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
from logo.scoring.fusion import MODES, SOFT_SOURCES
from logo.scoring.run_fuse import fuse_video, soft_source
from logo.tracking.run_track import (embed_ckpt, filter_trajectories, load_lst, run_lst, save_jsonl, traj_thr,
                                     video_inputs, vpmm_features)
from logo.tracking.simple_tracker import run_tracker, tracker_settings
from logo.utils.io import get_logger, load_json, load_yaml, set_seed
from logo.utils.paths import DATASETS, project_root

WS = (0.3, 0.5, 0.7)
THRS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
KEYS = ("MOTA", "MOTP", "IDF1", "MT", "ML")
TRACKERS = ("simple", "lst")
COLORS = {"hard": "#2a78d6", "soft": "#eb6834"}  # dataviz default categorical slots 1, 2
STYLES = ((0, (1, 1.5)), (0, (4, 2)), "-")      # w ascending: dotted, dashed, solid (cycled)


def tag(mode: str, w: float, thr: float) -> str:
    return f"{mode}_w{w:.2f}/thr_{thr:.2f}"


def filter_input(fused: dict, thr: float) -> dict:
    """fused json without detections whose score_fused < thr."""
    return {**fused, "frames": {k: [o for o in v if float(o["score_fused"]) >= thr]
                                for k, v in fused["frames"].items()}}


class _Tracker:
    """tracker.type simple | lst with the run_track settings; __call__(fused, thr) -> (track rows, spot rows)."""

    def __init__(self, cfg: dict, root: Path, lsc_exp: str, device):
        self.kind = (cfg.get("tracker") or {}).get("type", "simple")
        if self.kind not in TRACKERS:
            raise ValueError(f"sweep_fusion supports tracker.type {TRACKERS}, got {self.kind!r}")
        self.cfg, self.root, self.lsc_exp = cfg, root, lsc_exp
        if self.kind == "lst":
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.vpmm, self.lst = load_lst(embed_ckpt(cfg, root), cfg, self.device)
            self.min_score, self.window = traj_thr(cfg), tracker_settings(cfg)["text_window"]

    def start_video(self, ds: str, video: str, lsc: dict) -> None:
        """lst: VPMM vectors depend on quads and LSC feats only, so they are shared by every (mode, w, thr)."""
        if self.kind == "lst":
            feats, self.wh = video_inputs(self.root, ds, video, self.lsc_exp)
            self.emb = vpmm_features(lsc, feats, self.vpmm, self.wh, self.device, self.device.type == "cuda")
            del feats

    def __call__(self, fused: dict, thr: float) -> tuple[list[dict], list[dict]]:
        if self.kind == "simple":
            rows = run_tracker(filter_input(fused, thr), self.cfg)
            return rows, rows
        rows = run_lst(fused, self.emb, self.lst, self.wh, thr, self.window)
        return filter_trajectories(rows, self.min_score, False), filter_trajectories(rows, self.min_score, True)


def sweep(cfg: dict, ds: str, split: str, modes, ws, thrs, root: Path, out_dir: Path, log, lsc_exp: str,
          source: str = "p_eos", device=None) -> list[dict]:
    """One table row per (mode, w, thr): {"mode", "w", "thr", "soft_source", "track_<K>", "spot_<K>"}."""
    if split == "test":
        raise PermissionError("sweep_fusion never uses the test split")
    if any(m not in MODES for m in modes) or source not in SOFT_SOURCES:
        raise ValueError(f"modes must be in {MODES} and soft_source in {SOFT_SOURCES}")
    videos = read_videos(ds, split, root)
    tracker = _Tracker(cfg, root, lsc_exp, device)
    lsc_dir = root / "runs" / lsc_exp / "lsc" / ds
    for video in videos:
        lsc = load_json(lsc_dir / f"{video}.json")
        tracker.start_video(ds, video, lsc)
        for mode in modes:
            for w in ws:
                fused = fuse_video(lsc, mode, w, source)
                for thr in thrs:
                    for kind, rows in zip(("track", "spot"), tracker(fused, thr)):
                        save_jsonl(rows, out_dir / tag(mode, w, thr) / kind / ds / f"{video}.jsonl")
        log.info("%s: tracked %d settings", video, len(modes) * len(ws) * len(thrs))
    gt_dir = root / "data" / "processed" / ds
    iou = float((cfg.get("eval") or {}).get("iou", 0.5))
    table = []
    for mode in modes:
        for w in ws:
            for thr in thrs:
                row = {"mode": mode, "w": round(float(w), 4), "thr": round(float(thr), 4),
                       "soft_source": source if mode == "soft" else ""}
                for kind in ("track", "spot"):
                    r = evaluate_mot(gt_dir, out_dir / tag(mode, w, thr) / kind / ds, videos,
                                     spotting=(kind == "spot"), iou=iou)
                    row.update({f"{kind}_{k}": round(float(r[k]), 4) for k in KEYS})
                table.append(row)
                log.info("%s w=%.2f thr=%.2f track MOTA=%.2f IDF1=%.2f | spot MOTA=%.2f IDF1=%.2f", mode, w, thr,
                         row["track_MOTA"], row["track_IDF1"], row["spot_MOTA"], row["spot_IDF1"])
    return table


def write_csv(table: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


def plot(table: list[dict], metric: str, p: Path, title: str) -> None:
    """Two panels (tracking, spotting): metric vs thr, colour = mode (hard / soft), line style = w."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ws = sorted({r["w"] for r in table})
    styles = {w: STYLES[i % len(STYLES)] for i, w in enumerate(ws)}
    vals = [r[f"{k}_{metric}"] for r in table for k in ("track", "spot")]
    lo, hi = min(0.0, min(vals)), max(100.0, max(vals))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for ax, (kind, name) in zip(axes, (("track", "Tracking"), ("spot", "Spotting"))):
        for mode in [m for m in COLORS if any(r["mode"] == m for r in table)]:
            for w in ws:
                rs = sorted((r for r in table if r["mode"] == mode and r["w"] == w), key=lambda r: r["thr"])
                if rs:
                    ax.plot([r["thr"] for r in rs], [r[f"{kind}_{metric}"] for r in rs], color=COLORS[mode],
                            ls=styles[w], lw=2, marker="o", ms=4, label=f"{mode} w={w:g}")
        best = max(table, key=lambda r: r[f"{kind}_{metric}"])
        ax.annotate(f"{best[f'{kind}_{metric}']:.1f}: {best['mode']} w={best['w']:g} @ {best['thr']:.1f}",
                    (best["thr"], best[f"{kind}_{metric}"]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color="#52514e")
        ax.set_title(name, fontsize=11, loc="left")
        ax.set_xlabel("tracker.thr")
        ax.set_xticks(sorted({r["thr"] for r in table}))
        ax.set_ylim(lo - 5, hi + 5)
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(f"{metric} (%)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8, ncol=2)
    fig.suptitle(title, fontsize=11)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--lsc_exp", default=None, help="exp id holding lsc/ outputs; default config "
                                                    "pipeline.lsc_exp, else exp_id")
    ap.add_argument("--dataset", default=None, choices=DATASETS, help="default: config dataset")
    ap.add_argument("--split", default=None, help="train or val (default: config split); test is refused")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--ws", nargs="+", type=float, default=list(WS))
    ap.add_argument("--thrs", nargs="+", type=float, default=list(THRS))
    ap.add_argument("--soft_source", default=None, choices=SOFT_SOURCES, help="default: config fusion.soft_source")
    ap.add_argument("--device", default=None, help="tracker.type lst: default cuda if available")
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = copy.deepcopy(load_yaml(args.config))
    ds, split = args.dataset or cfg["dataset"], args.split or cfg["split"]
    if split not in ("train", "val"):
        ap.error(f"split {split!r} refused: fusion sweeps run on train/val only (never test)")
    if (cfg.get("tracker") or {}).get("type", "simple") not in TRACKERS:
        ap.error(f"sweep_fusion needs tracker.type in {TRACKERS}")
    if any(not 0.0 <= w <= 1.0 for w in args.ws):
        ap.error("--ws values must be in [0, 1]")
    root = project_root()
    lsc_exp = args.lsc_exp or (cfg.get("pipeline") or {}).get("lsc_exp") or cfg["exp_id"]
    source = args.soft_source or soft_source(cfg)
    run_dir = root / "runs" / cfg["exp_id"]
    out_dir = args.out_dir or run_dir / "sweep_fusion" / f"{ds}_{split}"
    set_seed(cfg["seed"])
    log = get_logger("sweep_fusion", run_dir / "log.txt")
    log.info("exp=%s ds=%s split=%s lsc_exp=%s tracker=%s modes=%s ws=%s thrs=%s soft_source=%s", cfg["exp_id"],
             ds, split, lsc_exp, cfg["tracker"].get("type", "simple"), args.modes, args.ws, args.thrs, source)
    table = sweep(cfg, ds, split, args.modes, args.ws, args.thrs, root, out_dir, log, lsc_exp, source, args.device)
    write_csv(table, out_dir / "sweep_fusion.csv")
    pngs = {}
    for metric in ("MOTA", "IDF1"):
        pngs[metric] = out_dir / f"sweep_fusion_{metric.lower()}.png"
        plot(table, metric, pngs[metric], f"{cfg['exp_id']} {ds}/{split}: {metric} vs tracker.thr, hard vs soft "
                                          f"(soft = {source})")
    best = max(table, key=lambda r: (r["spot_IDF1"], r["spot_MOTA"]))
    write_run_files(cfg, run_dir, {"lsc_exp": lsc_exp, "tracker": cfg["tracker"].get("type", "simple"),
                                   "soft_source": source, "csv": str(out_dir / "sweep_fusion.csv"),
                                   "png": {k.lower(): str(v) for k, v in pngs.items()},
                                   "best_spot": {k: best[k] for k in ("mode", "w", "thr", "spot_IDF1", "spot_MOTA")},
                                   "table": table},
                    key=f"sweep_fusion_{ds}_{split}")
    log.info("best spotting: %s w=%.2f thr=%.2f (IDF1 %.2f, MOTA %.2f)", best["mode"], best["w"], best["thr"],
             best["spot_IDF1"], best["spot_MOTA"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
