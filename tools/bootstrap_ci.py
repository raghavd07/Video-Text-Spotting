"""P27: paired video-level bootstrap 95% intervals for MOTA / IDF1 of two runs and their difference (B - A).

Uses metrics.json["per_video"] counts (tracking, from scripts/run_pipeline eval). Each of n resamples draws
the video list with replacement (np.random.default_rng(seed); the same draw for both runs), sums the counts
and computes MOTA = 1 - (misses + fp + switches) / num_objects and IDF1 = 2 idtp / (2 idtp + idfp + idfn)
(percent). Intervals are percentiles; a difference whose interval excludes zero is flagged "*". Both runs
must cover the same videos. Seed from --config (default configs/ablation.yaml); writes
runs/<config exp_id>/{config.yaml,log.txt,metrics.json meta.bootstrap_<A>_vs_<B>}.

Usage: python tools/bootstrap_ci.py E5-lst_yolo A-all_yolo [--config configs/ablation.yaml] [--n 1000]
           [--alpha 0.05] [--runs_dir runs]
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

from logo.detector.infer_paddle import resolve, write_run_files
from logo.eval.mot_eval import PER_VIDEO_KEYS
from logo.utils.io import get_logger, load_json, load_yaml, set_seed
from logo.utils.paths import project_root

COUNT_KEYS = ("num_objects", "misses", "fp", "switches", "idtp", "idfp", "idfn")
METRICS = ("MOTA", "IDF1")


def load_counts(metrics: dict) -> tuple[list[str], np.ndarray]:
    """(sorted videos, int64 [V, len(COUNT_KEYS)]) from metrics["per_video"]."""
    pv = metrics.get("per_video") or {}
    if not pv:
        raise ValueError("metrics.json has no per_video counts (run the eval stage)")
    videos = sorted(pv)
    missing = [k for k in COUNT_KEYS if k not in PER_VIDEO_KEYS or any(k not in pv[v] for v in videos)]
    if missing:
        raise ValueError(f"per_video lacks {missing}")
    return videos, np.array([[pv[v][k] for k in COUNT_KEYS] for v in videos], dtype=np.int64)


def score(counts: np.ndarray) -> dict:
    """Summed counts [..., len(COUNT_KEYS)] -> {"MOTA": [...], "IDF1": [...]} in percent (0 when undefined)."""
    c = {k: counts[..., i].astype(np.float64) for i, k in enumerate(COUNT_KEYS)}
    nobj = c["num_objects"]
    den = 2 * c["idtp"] + c["idfp"] + c["idfn"]
    with np.errstate(divide="ignore", invalid="ignore"):
        mota = np.where(nobj > 0, 100.0 * (1.0 - (c["misses"] + c["fp"] + c["switches"]) / nobj), 0.0)
        idf1 = np.where(den > 0, 100.0 * 2 * c["idtp"] / den, 0.0)
    return {"MOTA": mota, "IDF1": idf1}


def bootstrap(counts_a: np.ndarray, counts_b: np.ndarray, n: int = 1000, seed: int = 0,
              alpha: float = 0.05) -> dict:
    """{metric: {"a"|"b"|"diff": {"value", "lo", "hi"}}}; diff = b - a, plus diff["significant"]."""
    if counts_a.shape != counts_b.shape:
        raise ValueError(f"count shapes differ: {counts_a.shape} vs {counts_b.shape}")
    v = counts_a.shape[0]
    idx = np.random.default_rng(seed).integers(0, v, size=(n, v))
    pa, pb = score(counts_a.sum(0)), score(counts_b.sum(0))
    ba, bb = score(counts_a[idx].sum(1)), score(counts_b[idx].sum(1))
    q = (100 * alpha / 2, 100 * (1 - alpha / 2))
    out = {}
    for m in METRICS:
        r = {}
        for name, point, samples in (("a", pa[m], ba[m]), ("b", pb[m], bb[m]),
                                     ("diff", pb[m] - pa[m], bb[m] - ba[m])):
            lo, hi = np.percentile(samples, q)
            r[name] = {"value": float(point), "lo": float(lo), "hi": float(hi)}
        r["diff"]["significant"] = bool(r["diff"]["lo"] > 0 or r["diff"]["hi"] < 0)
        out[m] = r
    return out


def to_markdown(res: dict, name_a: str, name_b: str, n: int, alpha: float, num_videos: int) -> str:
    pct = round(100 * (1 - alpha))
    def ci(d):
        return f"{d['value']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"
    lines = [f"Paired bootstrap over {num_videos} videos, {n} resamples, {pct}% percentile intervals "
             f"(* = difference interval excludes 0)", "",
             f"| metric | {name_a} | {name_b} | {name_b} - {name_a} | sig |", "|---|---:|---:|---:|:---:|"]
    for m, r in res.items():
        lines.append(f"| {m} | {ci(r['a'])} | {ci(r['b'])} | {ci(r['diff'])} | "
                     f"{'*' if r['diff']['significant'] else ''} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_a", help="exp id (runs/<id>/metrics.json) or a metrics.json path")
    ap.add_argument("run_b")
    ap.add_argument("--config", type=Path, default=Path("configs/ablation.yaml"), help="seed + exp_id for outputs")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--runs_dir", type=Path, default=None, help="default: <project root>/runs")
    args = ap.parse_args(argv)
    if args.n < 1 or not 0 < args.alpha < 1:
        ap.error("--n must be >= 1 and 0 < --alpha < 1")

    root = project_root()
    runs = args.runs_dir or root / "runs"
    cfg = copy.deepcopy(load_yaml(resolve(args.config, root)))
    paths = [Path(r) if r.endswith(".json") else runs / r / "metrics.json" for r in (args.run_a, args.run_b)]
    (va, ca), (vb, cb) = (load_counts(load_json(p)) for p in paths)
    if va != vb:
        ap.error(f"runs cover different videos: only in A {sorted(set(va) - set(vb))}, "
                 f"only in B {sorted(set(vb) - set(va))}")
    set_seed(cfg["seed"])
    run_dir = root / "runs" / cfg["exp_id"]
    log = get_logger("bootstrap_ci", run_dir / "log.txt")
    res = bootstrap(ca, cb, n=args.n, seed=int(cfg["seed"]), alpha=args.alpha)
    md = to_markdown(res, args.run_a, args.run_b, args.n, args.alpha, len(va))
    print(md, end="")
    log.info("bootstrap %s vs %s: %s", args.run_a, args.run_b,
             " ".join(f"d{m}={r['diff']['value']:.2f}[{r['diff']['lo']:.2f},{r['diff']['hi']:.2f}]"
                      for m, r in res.items()))
    na, nb = (p.parent.name for p in paths)
    write_run_files(cfg, run_dir, {"a": str(paths[0]), "b": str(paths[1]), "n": args.n, "alpha": args.alpha,
                                   "seed": int(cfg["seed"]), "videos": va, "results": res},
                    key=f"bootstrap_{na}_vs_{nb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
