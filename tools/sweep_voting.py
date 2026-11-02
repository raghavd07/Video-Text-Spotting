"""Sweep voting.tau x voting.min_len on existing track outputs -> CSV (+ metrics.json meta).

Reads runs/<track_exp>/track/<ds>/<video>.jsonl and runs/<track_exp>/fuse/<ds>/<video>.json once per video
(text_conf / lang attached as in logo.voting.track_vote.load_video), votes in memory with vote_tracks for every
(tau, min_len) and writes <out_dir>/tau_<t>/min_len_<n>/<ds>/<video>.jsonl; a first "off" row scores the
un-voted tracks. Every setting is scored for tracking and spotting with logo.eval.mot_eval.evaluate_mot.
Outputs <out_dir>/sweep_voting.csv (default out_dir runs/<exp_id>/sweep_voting/<ds>_<split>) and
runs/<exp_id>/{config.yaml,log.txt,metrics.json meta.sweep_voting_<ds>_<split>}. The test split is refused.

Usage: python tools/sweep_voting.py --config configs/E5-lst.yaml [--track_exp EXP] [--dataset icdar2015v]
           [--split val] [--taus 0.0 0.3 0.5 0.7] [--min_lens 1 3 5 10] [--out_dir DIR]
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

from logo.detector.infer_paddle import read_videos, write_run_files
from logo.eval.mot_eval import evaluate_mot
from logo.tracking.run_track import save_jsonl
from logo.utils.io import get_logger, load_yaml, set_seed
from logo.utils.paths import DATASETS, project_root
from logo.voting.track_vote import EXTRA_KEYS, load_video, vote_tracks

TAUS = (0.0, 0.3, 0.5, 0.7)
MIN_LENS = (1, 3, 5, 10)
KEYS = ("MOTA", "MOTP", "IDF1", "MT", "ML")


def tag(tau, min_len) -> str:
    return "off" if tau is None else f"tau_{tau:.2f}/min_len_{min_len}"


def sweep(cfg: dict, ds: str, split: str, taus, min_lens, root: Path, out_dir: Path, log,
          track_exp: str) -> list[dict]:
    """One row per setting: {"voted", "tau", "min_len", "tracks", "track_<K>", "spot_<K>"}; first row = off."""
    if split == "test":
        raise PermissionError("sweep_voting never uses the test split")
    if any(n < 0 for n in min_lens):
        raise ValueError("min_len values must be >= 0")
    videos = read_videos(ds, split, root)
    settings = [(None, None)] + [(float(t), int(n)) for t in taus for n in min_lens]
    n_tracks = {s: 0 for s in settings}
    for video in videos:
        rows = load_video(root, track_exp, ds, video, cfg)
        for tau, n in settings:
            if tau is None:
                out = [{k: v for k, v in r.items() if k not in EXTRA_KEYS} for r in rows]
            else:
                out = vote_tracks(rows, {**cfg, "voting": {"enabled": True, "tau": tau, "min_len": n}})
            n_tracks[tau, n] += len({r["track_id"] for r in out})
            save_jsonl(out, out_dir / tag(tau, n) / ds / f"{video}.jsonl")
        log.info("%s: voted %d settings", video, len(settings) - 1)
    gt_dir = root / "data" / "processed" / ds
    iou = float((cfg.get("eval") or {}).get("iou", 0.5))
    table = []
    for tau, n in settings:
        row = {"voted": tau is not None, "tau": "" if tau is None else tau, "min_len": "" if n is None else n,
               "tracks": n_tracks[tau, n]}
        for kind in ("track", "spot"):
            r = evaluate_mot(gt_dir, out_dir / tag(tau, n) / ds, videos, spotting=(kind == "spot"), iou=iou)
            row.update({f"{kind}_{k}": round(float(r[k]), 4) for k in KEYS})
        table.append(row)
        log.info("%s: tracks=%d track MOTA=%.2f IDF1=%.2f | spot MOTA=%.2f IDF1=%.2f", tag(tau, n),
                 row["tracks"], row["track_MOTA"], row["track_IDF1"], row["spot_MOTA"], row["spot_IDF1"])
    return table


def write_csv(table: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--track_exp", default=None, help="exp id holding track/ and fuse/ outputs; default exp_id")
    ap.add_argument("--dataset", default=None, choices=DATASETS, help="default: config dataset")
    ap.add_argument("--split", default=None, help="train or val (default: config split); test is refused")
    ap.add_argument("--taus", nargs="+", type=float, default=list(TAUS))
    ap.add_argument("--min_lens", nargs="+", type=int, default=list(MIN_LENS))
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = copy.deepcopy(load_yaml(args.config))
    ds, split = args.dataset or cfg["dataset"], args.split or cfg["split"]
    if split not in ("train", "val"):
        ap.error(f"split {split!r} refused: voting sweeps run on train/val only (never test)")
    if any(n < 0 for n in args.min_lens):
        ap.error("--min_lens values must be >= 0")
    root = project_root()
    track_exp = args.track_exp or cfg["exp_id"]
    run_dir = root / "runs" / cfg["exp_id"]
    out_dir = args.out_dir or run_dir / "sweep_voting" / f"{ds}_{split}"
    set_seed(cfg["seed"])
    log = get_logger("sweep_voting", run_dir / "log.txt")
    log.info("exp=%s ds=%s split=%s track_exp=%s taus=%s min_lens=%s", cfg["exp_id"], ds, split, track_exp,
             args.taus, args.min_lens)
    table = sweep(cfg, ds, split, args.taus, args.min_lens, root, out_dir, log, track_exp)
    write_csv(table, out_dir / "sweep_voting.csv")
    best = max(table[1:], key=lambda r: (r["spot_IDF1"], r["spot_MOTA"]))
    write_run_files(cfg, run_dir, {"track_exp": track_exp, "csv": str(out_dir / "sweep_voting.csv"),
                                   "off": {k: table[0][k] for k in ("spot_IDF1", "spot_MOTA")},
                                   "best_spot": {k: best[k] for k in ("tau", "min_len", "spot_IDF1", "spot_MOTA")},
                                   "table": table},
                    key=f"sweep_voting_{ds}_{split}")
    log.info("best spotting: tau=%.2f min_len=%d (IDF1 %.2f, MOTA %.2f); off IDF1 %.2f", best["tau"],
             best["min_len"], best["spot_IDF1"], best["spot_MOTA"], table[0]["spot_IDF1"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
