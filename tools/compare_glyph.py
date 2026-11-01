"""Glyph-branch ablation: word accuracy of LSC checkpoints on data/lsc/<split>, full and noisy subset.

Each --ckpt NAME=PATH is run with ABINetRecognizer (alignment head) over every crop; the first one is
the reference. Noisy subset = the `--noisy_frac` (0.3) of crops with the lowest mean Sobel gradient
magnitude (blur proxy, on the stored 32x128 grey crop). word_acc is over positives (label != ""),
tb_acc over all crops. Each non-reference checkpoint gets a paired comparison vs the reference on
positives: delta word_acc, exact McNemar p-value and a paired bootstrap 95% CI (seed = config seed);
verdict "within noise" when p >= 0.05 or the CI contains 0.
Writes runs/<exp_id>/{config.yaml, log.txt, metrics.json meta.glyph_ablation}; exp_id default
<config exp_id>_compare.

Usage: python tools/compare_glyph.py --config configs/E3-lsc-glyph.yaml \\
           --ckpt base=runs/E3-lsc/checkpoints/best.pth --ckpt glyph=runs/E3-lsc-glyph/checkpoints/best.pth
           [--split val] [--root data/lsc] [--noisy_frac 0.3] [--pos_only] [--device cuda] [--final]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from logo.detector.infer_paddle import write_run_files
from logo.lsc.abinet_wrapper import ABINetRecognizer
from logo.lsc.dataset import LSCDataset
from logo.lsc.train_lsc import score_predictions
from logo.utils.io import get_logger, load_yaml, set_seed
from logo.utils.paths import check_split, project_root

CHUNK = 512  # crops held in RAM at once


def grad_magnitude(img_bgr: np.ndarray) -> float:
    """Mean Sobel gradient magnitude of the grey crop (low = blurry / flat)."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(dx * dx + dy * dy).mean())


def noisy_subset(scores: np.ndarray, frac: float, pool=None) -> np.ndarray:
    """bool mask of the ceil(frac * |pool|) lowest scores within `pool` (bool mask, default all)."""
    scores = np.asarray(scores, np.float64)
    pool = np.ones(len(scores), bool) if pool is None else np.asarray(pool, bool)
    idx = np.flatnonzero(pool)
    k = int(math.ceil(float(frac) * len(idx)))
    out = np.zeros(len(scores), bool)
    out[idx[np.argsort(scores[idx], kind="stable")[:k]]] = True
    return out


def mcnemar_p(a_ok, b_ok) -> float:
    """Exact two-sided McNemar test on paired correctness vectors."""
    a_ok, b_ok = np.asarray(a_ok, bool), np.asarray(b_ok, bool)
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0 ** n
    return float(min(1.0, 2.0 * tail))


def paired_bootstrap(a_ok, b_ok, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI (percent) of mean(b_ok) - mean(a_ok) under paired resampling."""
    d = np.asarray(b_ok, np.float64) - np.asarray(a_ok, np.float64)
    if len(d) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(int(seed))
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = d[rng.integers(0, len(d), len(d))].mean()
    lo, hi = np.percentile(means, [2.5, 97.5]) * 100.0
    return float(lo), float(hi)


def paired(ref_preds, preds, labels, sel, seed) -> dict:
    """Reference vs candidate on the positives inside `sel`."""
    pos = [i for i in np.flatnonzero(sel) if labels[i]]
    a = np.array([ref_preds[i] == labels[i] for i in pos], bool)
    b = np.array([preds[i] == labels[i] for i in pos], bool)
    p = mcnemar_p(a, b)
    lo, hi = paired_bootstrap(a, b, seed=seed)
    delta = 100.0 * (b.mean() - a.mean()) if len(pos) else 0.0
    noise = p >= 0.05 or lo <= 0.0 <= hi
    verdict = "within noise" if noise else ("better" if delta > 0 else "worse")
    return {"num_pos": len(pos), "delta_word_acc": float(delta), "mcnemar_p": p, "ci95": [lo, hi],
            "ref_only_correct": int((a & ~b).sum()), "cand_only_correct": int((~a & b).sum()),
            "verdict": verdict}


def subset_scores(preds, labels, sel) -> dict:
    idx = np.flatnonzero(sel)
    return score_predictions([preds[i] for i in idx], [labels[i] for i in idx])


def predict_split(rec: ABINetRecognizer, ds: LSCDataset) -> list[str]:
    out = []
    for s in range(0, len(ds), CHUNK):
        out += [r["text"] for r in rec.predict([ds.load(i) for i in range(s, min(len(ds), s + CHUNK))])]
    return out


def blur_scores(ds: LSCDataset) -> np.ndarray:
    return np.array([grad_magnitude(ds.load(i)) for i in range(len(ds))], np.float64)


def compare(preds: dict, labels: list[str], scores: np.ndarray, frac: float, pos_only: bool,
            seed: int) -> dict:
    """preds {name: [text]} (first = reference) -> results for 'full' and 'noisy' subsets."""
    pool = np.array([bool(l) for l in labels]) if pos_only else None
    subsets = {"full": np.ones(len(labels), bool), "noisy": noisy_subset(scores, frac, pool)}
    names = list(preds)
    res = {"reference": names[0], "noisy_frac": float(frac), "noisy_pool": "pos" if pos_only else "all",
           "noisy_grad_max": float(scores[subsets["noisy"]].max()) if subsets["noisy"].any() else None}
    for sub, sel in subsets.items():
        res[sub] = {"num_crops": int(sel.sum()),
                    "models": {n: subset_scores(preds[n], labels, sel) for n in names},
                    "vs_reference": {n: paired(preds[names[0]], preds[n], labels, sel, seed)
                                     for n in names[1:]}}
    return res


def _parse_ckpts(items) -> dict:
    out = {}
    for it in items:
        name, sep, path = it.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--ckpt expects NAME=PATH, got {it!r}")
        if name in out:
            raise ValueError(f"duplicate --ckpt name {name!r}")
        out[name] = Path(path)
    if len(out) < 2:
        raise ValueError("need at least two --ckpt (reference first)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="NAME=PATH; first = reference")
    ap.add_argument("--split", default="val")
    ap.add_argument("--root", type=Path, default=None, help="default: <root>/data/lsc")
    ap.add_argument("--noisy_frac", type=float, default=0.3)
    ap.add_argument("--pos_only", action="store_true", help="rank only positives for the noisy subset")
    ap.add_argument("--exp_id", default=None)
    ap.add_argument("--device", default=None, help="default: cuda if available")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--final", action="store_true", help="allow the test split (rule 5)")
    args = ap.parse_args(argv)

    check_split(args.split, args.final)
    if not 0.0 < args.noisy_frac <= 1.0:
        raise ValueError("--noisy_frac must be in (0, 1]")
    ckpts = _parse_ckpts(args.ckpt)
    root = project_root()
    cfg = load_yaml(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    exp_id = args.exp_id or f"{cfg['exp_id']}_compare"
    run_dir = root / "runs" / exp_id
    log = get_logger("compare_glyph", run_dir / "log.txt")
    lsc_root = args.root if args.root is not None else root / "data" / "lsc"
    ds = LSCDataset(lsc_root / args.split, train=False)
    if len(ds) == 0:
        raise ValueError(f"empty split {lsc_root / args.split}")
    labels = [lab for _, lab in ds.items]
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("exp=%s split=%s crops=%d pos=%d ckpts=%s", exp_id, args.split, len(ds),
             sum(1 for l in labels if l), {k: str(v) for k, v in ckpts.items()})

    t0 = time.perf_counter()
    scores = blur_scores(ds)
    preds = {}
    for name, ck in ckpts.items():
        ck = ck if ck.is_absolute() else root / ck
        rec = ABINetRecognizer(ck, args.device, fp16=not args.no_fp16)
        preds[name] = predict_split(rec, ds)
        del rec
        log.info("%s: predicted %d crops", name, len(preds[name]))
    res = compare(preds, labels, scores, args.noisy_frac, args.pos_only, seed)
    res.update(split=args.split, ckpts={k: str(v) for k, v in ckpts.items()},
               seconds=round(time.perf_counter() - t0, 2))
    for sub in ("full", "noisy"):
        for name, m in res[sub]["models"].items():
            log.info("[%s] %-10s word_acc %.2f%% (pos %d) tb_acc %.2f%%", sub, name, m["word_acc"],
                     m["num_pos"], m["tb_acc"])
        for name, c in res[sub]["vs_reference"].items():
            log.info("[%s] %s - %s: %+.2f pts, McNemar p=%.3g, CI95 [%+.2f, %+.2f] -> %s", sub, name,
                     res["reference"], c["delta_word_acc"], c["mcnemar_p"], *c["ci95"], c["verdict"])
    write_run_files(dict(cfg, exp_id=exp_id), run_dir, res, key="glyph_ablation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
