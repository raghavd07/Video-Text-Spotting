"""P29: paper figures (vector PDF) from runs/* -> paper/figs/<name>.pdf.

Figures (numbers and boxes are read from run outputs; frames from data/frames, embedded as raster under vector
overlays):
  threshold_<exp>_<ds>_<split>.pdf   meta.sweep_fusion_*: Track/Spot MOTA and IDF1 vs score threshold, hard
                                     (blue, solid, circles) vs soft (orange, dashed, squares) at threshold.w
  ci_runs.pdf                        meta.bootstrap_*: per-run value with bootstrap interval (MOTA, IDF1)
  ci_diff.pdf                        meta.bootstrap_*: difference B - A with interval; filled = significant
  rescore_<exp>_<video>_f<frame>.pdf runs/<exp>/fuse/<ds>/<video>.json: boxes coloured by det score (left) and
                                     score_fused (right), one sequential blue scale
  overlay_<exp>_<video>.pdf          runs/<exp>/<stage>/<ds>/<video>.jsonl: `frames` frames evenly spaced over the
                                     frames with tracks; colour = track id (first 8 ids in order of appearance,
                                     the rest grey), label "id text"
  failures_<exp>_<ds>_<split>.pdf    errors of runs/<exp>/<stage> vs GT (matching of tools/error_analysis.py),
                                     per_cat examples per category sampled with the config seed; GT blue solid,
                                     prediction orange dashed
Settings in <config>.figures (see configs/paper.yaml). Dataset/split of a run from runs/<exp>/config.yaml;
a run or meta entry on the test split is skipped unless --final (rule 5). Writes
runs/<exp_id>/{config.yaml,log.txt,metrics.json meta.figures}.

Usage: python tools/make_figures.py [--config configs/paper.yaml] [--final] [--out_dir paper/figs]
           [--only threshold ci rescore overlay failures]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from logo.data.schema import load_gt, read_split
from logo.detector.infer_paddle import resolve, write_run_files
from logo.eval.match import load_pred_frames
from logo.utils.io import get_logger, load_json, load_yaml, set_seed
from logo.utils.paths import project_root

_REPO = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m  # dataclasses in make_tables resolve their module through sys.modules
    spec.loader.exec_module(m)
    return m


mt = _load(_REPO / "tools" / "make_tables.py", "make_tables_tool")
ea = _load(_REPO / "tools" / "error_analysis.py", "error_analysis_tool")

KINDS = ("threshold", "ci", "rescore", "overlay", "failures")
# dataviz reference palette: categorical slots 1-8 (light), sequential blue 100..700, chrome
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SEQ_BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e4e3dd"
OTHER_ID = MUTED
MODE_STYLE = {"hard": dict(color=SERIES[0], ls="-", marker="o"), "soft": dict(color=SERIES[1], ls="--", marker="s")}
GT_STYLE, PRED_STYLE = dict(color=SERIES[0], ls="-", lw=1.8), dict(color=SERIES[1], ls="--", lw=1.4)
MAX_W = 960       # frames are downscaled to at most this width before embedding
CROP_MAX = 360    # failure crops: longest side in px


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8, "axes.titlesize": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "axes.titlecolor": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "lines.linewidth": 2.0, "lines.markersize": 5})
    return plt


def _style_axes(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save_pdf(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    _plt().close(fig)
    return str(path)


def _ascii(s, n: int = 18) -> str:
    s = str(s).encode("ascii", "replace").decode("ascii")
    return s if len(s) <= n else s[:n - 1] + "~"


def run_ds_split(root: Path, exp: str) -> tuple[str, str]:
    rc = mt.run_config(root / "runs", exp)
    if "dataset" not in rc or "split" not in rc:
        raise ValueError(f"runs/{exp}/config.yaml lacks dataset/split")
    return rc["dataset"], rc["split"]


def read_rgb(root: Path, ds: str, video: str, fid: int, wh: tuple[int, int], log=None) -> np.ndarray:
    p = root / "data" / "frames" / ds / video / f"{int(fid):06d}.jpg"
    img = ea.read_image(p)
    if img is None:
        if log:
            log.warning("missing frame %s: grey canvas", p)
        w, h = wh
        return np.full((max(h, 1), max(w, 1), 3), 200, np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def downscale(img: np.ndarray, max_w: int) -> tuple[np.ndarray, float]:
    s = min(1.0, max_w / img.shape[1])
    if s < 1.0:
        img = cv2.resize(img, (int(round(img.shape[1] * s)), int(round(img.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
    return img, s


def draw_quad(ax, quad, scale: float = 1.0, offset=(0.0, 0.0), ring: bool = True, **style) -> None:
    from matplotlib.patches import Polygon
    pts = (np.asarray(quad, np.float64).reshape(4, 2) - np.asarray(offset)) * scale
    lw = style.pop("lw", 1.4)
    if ring:  # dark ring keeps light strokes visible on bright frames
        ax.add_patch(Polygon(pts, closed=True, fill=False, ec=INK, lw=lw + 1.0, alpha=0.6))
    ax.add_patch(Polygon(pts, closed=True, fill=False, ec=style.pop("color"), lw=lw, ls=style.pop("ls", "-")))


def _image_axes(ax, img) -> None:
    ax.imshow(img, interpolation="antialiased")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ----------------------------------------------------------------------------------------------- charts
def fig_threshold(cfg: dict, runs: dict, out_dir: Path, final: bool, log=None) -> list[str]:
    plt = _plt()
    w_want = float((cfg.get("threshold") or {}).get("w", 0.5))
    out = []
    panels = (("track_MOTA", "Tracking MOTA"), ("track_IDF1", "Tracking IDF1"),
              ("spot_MOTA", "Spotting MOTA"), ("spot_IDF1", "Spotting IDF1"))
    for exp, ds, split, e in mt.iter_meta(runs, "sweep_fusion", final, log):
        table = e.get("table") or []
        if not table:
            continue
        w = mt.pick_w(table, w_want)
        fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.9), sharex=True)
        for ax, (key, title) in zip(axes, panels):
            for mode, style in MODE_STYLE.items():
                pts = sorted((float(r["thr"]), float(r[key])) for r in table
                             if r.get("mode") == mode and mt.is_num(r.get("w")) and abs(float(r["w"]) - w) < 1e-9
                             and mt.is_num(r.get(key)))
                if pts:
                    x, y = zip(*pts)
                    ax.plot(x, y, label=mode, **style)
            ax.set_title(title)
            ax.set_xlabel("score threshold")
            _style_axes(ax)
        axes[0].set_ylabel("%")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, [f"{l} fusion" for l in labels], loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 1.04))
        fig.suptitle(f"{exp}, {ds} {split}, w={w:.2f}", y=1.1, fontsize=8, color=MUTED)
        fig.tight_layout()
        out.append(save_pdf(fig, out_dir / f"threshold_{exp}_{ds}_{split}.pdf"))
    return out


def _bootstrap_entries(runs: dict) -> list[dict]:
    seen, out = set(), []
    for m in runs.values():
        for key, e in (m.get("meta") or {}).items():
            if key.startswith("bootstrap_") and key not in seen and isinstance(e, dict) and e.get("results"):
                seen.add(key)
                out.append(e)
    return out


def fig_ci(runs: dict, out_dir: Path, log=None) -> list[str]:
    entries = _bootstrap_entries(runs)
    if not entries:
        return []
    plt = _plt()
    metrics = sorted({k for e in entries for k in e["results"]}, key=lambda k: (k != "MOTA", k))
    per_run: dict[str, dict] = {}
    pairs = []
    for e in entries:
        a, b = mt._exp_name(e["a"]), mt._exp_name(e["b"])
        for side, exp in (("a", a), ("b", b)):
            for k, r in e["results"].items():
                per_run.setdefault(exp, {}).setdefault(k, r[side])
        pairs.append((f"{a} vs {b}", e["results"]))
    level = f"{100 * (1 - float(entries[0].get('alpha', 0.05))):g}%"
    out = []

    exps = list(per_run)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.9, 0.45 * len(exps) + 0.9), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(exps))
    for ax, k in zip(axes, metrics):
        v = np.array([per_run[x].get(k, {}).get("value", np.nan) for x in exps], np.float64)
        lo = np.array([per_run[x].get(k, {}).get("lo", np.nan) for x in exps], np.float64)
        hi = np.array([per_run[x].get(k, {}).get("hi", np.nan) for x in exps], np.float64)
        ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="o", color=SERIES[0], ecolor=SERIES[0], elinewidth=2,
                    capsize=3)
        for vi, yi in zip(v, y):
            if np.isfinite(vi):
                ax.annotate(f"{vi:.2f}", (vi, yi), textcoords="offset points", xytext=(0, 6), ha="center",
                            fontsize=6.5, color=INK)
        ax.set_title(f"{k} ({level} bootstrap interval)")
        ax.set_xlabel("%")
        _style_axes(ax)
    axes[0].set_yticks(y, exps)
    axes[0].set_ylim(len(exps) - 0.5, -0.7)
    fig.tight_layout()
    out.append(save_pdf(fig, out_dir / "ci_runs.pdf"))

    fig, axes = plt.subplots(1, len(metrics), figsize=(6.9, 0.45 * len(pairs) + 0.9), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(pairs))
    for ax, k in zip(axes, metrics):
        ax.axvline(0.0, color=MUTED, lw=1.0)
        for yi, (_, res) in zip(y, pairs):
            d = res.get(k, {}).get("diff")
            if not d:
                continue
            sig = bool(d.get("significant"))
            ax.errorbar([d["value"]], [yi], xerr=[[d["value"] - d["lo"]], [d["hi"] - d["value"]]], fmt="o",
                        color=SERIES[0], elinewidth=2, capsize=3, mfc=SERIES[0] if sig else SURFACE, mew=1.5)
        ax.set_title(f"{k}: B - A (filled = interval excludes 0)")
        ax.set_xlabel("percentage points")
        _style_axes(ax)
    axes[0].set_yticks(y, [p for p, _ in pairs])
    axes[0].set_ylim(len(pairs) - 0.5, -0.7)
    fig.tight_layout()
    out.append(save_pdf(fig, out_dir / "ci_diff.pdf"))
    return out


# ----------------------------------------------------------------------------------------------- frames
def _check_run(root: Path, exp: str, final: bool, log=None) -> Optional[tuple[str, str]]:
    ds, split = run_ds_split(root, exp)
    if split == "test" and not final:
        if log:
            log.info("skip figures of %s (test split needs --final)", exp)
        return None
    return ds, split


def _video(root: Path, ds: str, split: str, video: Optional[str]) -> str:
    videos = read_split(root / "splits" / f"{ds}_{split}.txt")
    if video is None:
        return videos[0]
    if video not in videos:
        raise ValueError(f"{video} is not in splits/{ds}_{split}.txt")
    return video


def fig_rescore(fcfg: dict, root: Path, out_dir: Path, final: bool, log=None) -> list[str]:
    exp = fcfg["exp_id"]
    chk = _check_run(root, exp, final, log)
    if chk is None:
        return []
    ds, split = chk
    video = _video(root, ds, split, fcfg.get("video"))
    fused = load_json(root / "runs" / exp / "fuse" / ds / f"{video}.json")
    frames = {int(k): v for k, v in fused["frames"].items() if v}
    if not frames:
        raise ValueError(f"runs/{exp}/fuse/{ds}/{video}.json has no boxes")
    fid = int(fcfg["frame"]) if fcfg.get("frame") is not None else min(frames, key=lambda f: (-len(frames[f]), f))
    objs = frames.get(fid, [])
    gt = load_gt(root / "data" / "processed" / ds / f"{video}.json")
    img, s = downscale(read_rgb(root, ds, video, fid, (gt["width"], gt["height"]), log), MAX_W)
    rc = mt.run_config(root / "runs", exp).get("fusion") or {}

    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    norm = Normalize(0.0, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 6.9 / 2 * img.shape[0] / img.shape[1] + 0.8))
    for ax, key, title in ((axes[0], "score", "before: detector score"),
                           (axes[1], "score_fused", f"after: fused score ({rc.get('mode', '-')}, "
                                                    f"w={rc.get('w', '-')})")):
        _image_axes(ax, img)
        for o in sorted(objs, key=lambda o: float(o[key])):
            draw_quad(ax, o["quad"], s, color=cmap(norm(float(o[key]))), lw=1.6)
        ax.set_title(title)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=list(axes), orientation="horizontal", fraction=0.05, pad=0.04, aspect=50)
    cb.set_label("score")
    cb.outline.set_edgecolor(MUTED)
    fig.suptitle(f"{exp}: {video} frame {fid} ({len(objs)} boxes)", fontsize=8, color=MUTED)
    return [save_pdf(fig, out_dir / f"rescore_{exp}_{video}_f{fid}.pdf")]


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fig_overlay(fcfg: dict, root: Path, out_dir: Path, final: bool, log=None) -> list[str]:
    exp, stage = fcfg["exp_id"], fcfg.get("stage", "vote")
    chk = _check_run(root, exp, final, log)
    if chk is None:
        return []
    ds, split = chk
    video = _video(root, ds, split, fcfg.get("video"))
    rows = _read_jsonl(root / "runs" / exp / stage / ds / f"{video}.jsonl")
    by_frame: dict[int, list[dict]] = {}
    for r in rows:
        by_frame.setdefault(int(r["frame"]), []).append(r)
    if not by_frame:
        raise ValueError(f"runs/{exp}/{stage}/{ds}/{video}.jsonl is empty")
    fids = sorted(by_frame)
    k = max(1, min(int(fcfg.get("frames", 4)), len(fids)))
    chosen = sorted({fids[i] for i in np.linspace(0, len(fids) - 1, k).round().astype(int)})
    colors: dict[int, str] = {}
    for f in chosen:
        for r in sorted(by_frame[f], key=lambda r: int(r["track_id"])):
            tid = int(r["track_id"])
            if tid not in colors:
                colors[tid] = SERIES[len(colors)] if len(colors) < len(SERIES) else OTHER_ID
    gt = load_gt(root / "data" / "processed" / ds / f"{video}.json")
    plt = _plt()
    ncol = 2 if len(chosen) > 1 else 1
    nrow = -(-len(chosen) // ncol)
    aspect = gt["height"] / max(gt["width"], 1)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.9, 6.9 / ncol * aspect * nrow + 0.3 * nrow), squeeze=False)
    for ax in axes.ravel()[len(chosen):]:
        ax.axis("off")
    for ax, f in zip(axes.ravel(), chosen):
        img, s = downscale(read_rgb(root, ds, video, f, (gt["width"], gt["height"]), log), MAX_W)
        _image_axes(ax, img)
        for r in by_frame[f]:
            tid = int(r["track_id"])
            draw_quad(ax, r["quad"], s, color=colors[tid], lw=1.6)
            q = np.asarray(r["quad"], np.float64).reshape(4, 2) * s
            text = f"{tid} {_ascii(r.get('text', ''), 14)}".strip()
            ax.text(q[:, 0].min(), q[:, 1].min() - 2, text, fontsize=5.5, color=INK, va="bottom", ha="left",
                    bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec=colors[tid], lw=0.8, alpha=0.85))
        ax.set_title(f"frame {f}")
    fig.suptitle(f"{exp} ({stage}): {video}", fontsize=8, color=MUTED)
    fig.tight_layout()
    return [save_pdf(fig, out_dir / f"overlay_{exp}_{video}.pdf")]


def _crop(img: np.ndarray, quads: list) -> tuple[np.ndarray, tuple[float, float], float]:
    pts = np.concatenate([np.asarray(q, np.float64).reshape(4, 2) for q in quads])
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    m = max(16.0, 0.6 * max(x1 - x0, y1 - y0))
    H, W = img.shape[:2]
    x0, y0 = int(max(0, x0 - m)), int(max(0, y0 - m))
    x1, y1 = int(min(W, x1 + m + 1)), int(min(H, y1 + m + 1))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        crop = np.full((32, 32, 3), 200, np.uint8)
    s = min(1.0, CROP_MAX / max(crop.shape[:2]))
    if s < 1.0:
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                          interpolation=cv2.INTER_AREA)
    return crop, (float(x0), float(y0)), s


def fig_failures(fcfg: dict, seed: int, root: Path, out_dir: Path, final: bool, log=None) -> list[str]:
    exp, stage = fcfg["exp_id"], fcfg.get("stage", "vote")
    chk = _check_run(root, exp, final, log)
    if chk is None:
        return []
    ds, split = chk
    rc = mt.run_config(root / "runs", exp)
    iou = float(fcfg["iou"] if fcfg.get("iou") is not None else (rc.get("eval") or {}).get("iou", 0.5))
    per_cat = int(fcfg.get("per_cat", 3))
    track_dir = root / "runs" / exp / stage / ds
    counts = ea.empty_counts()
    gallery = ea.Gallery(per_cat, np.random.default_rng(seed))
    sizes = {}
    for video in read_split(root / "splits" / f"{ds}_{split}.txt"):
        g = load_gt(root / "data" / "processed" / ds / f"{video}.json")
        sizes[video] = (int(g["width"]), int(g["height"]))
        gt = {int(k): v for k, v in g["frames"].items()}
        ea.analyze_video(gt, load_pred_frames(track_dir, video, tracks=True), int(g["num_frames"]), video, iou,
                         counts, gallery)
        del g, gt
    cats = [c for c in ea.CATEGORIES if gallery.items[c]]
    if not cats:
        if log:
            log.info("failures: no errors in %s", exp)
        return []
    plt = _plt()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(len(cats), per_cat, figsize=(6.9, 1.75 * len(cats) + 0.4), squeeze=False)
    for ri, c in enumerate(cats):
        total = sum(counts[c].values())
        for ci in range(per_cat):
            ax = axes[ri, ci]
            if ci >= len(gallery.items[c]):
                ax.axis("off")
                continue
            ex = gallery.items[c][ci]
            img = read_rgb(root, ds, ex["video"], ex["frame"], sizes[ex["video"]], log)
            quads = [q for q in (ex["gt_quad"], ex["pred_quad"]) if q is not None]
            crop, off, s = _crop(img, quads)
            _image_axes(ax, crop)
            if ex["gt_quad"] is not None:
                draw_quad(ax, ex["gt_quad"], s, off, **GT_STYLE)
            if ex["pred_quad"] is not None:
                draw_quad(ax, ex["pred_quad"], s, off, **PRED_STYLE)
            parts = []
            if ex["gt_text"] is not None:
                parts.append(f"GT '{_ascii(ex['gt_text'], 12)}'")
            if ex["pred_text"] is not None:
                parts.append(f"pred '{_ascii(ex['pred_text'], 12)}'")
            ax.set_xlabel(f"{_ascii(ex['video'], 16)} f{ex['frame']}, h={ex['height']:g}px\n" + " | ".join(parts),
                          fontsize=5.5, color=INK)
            if ci == 0:
                ax.set_ylabel(f"{ea.LABELS[c]}\n({total})", fontsize=7, color=INK)
    handles = [Line2D([], [], **GT_STYLE, label="ground truth"), Line2D([], [], **PRED_STYLE, label="prediction")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return [save_pdf(fig, out_dir / f"failures_{exp}_{ds}_{split}.pdf")]


# ----------------------------------------------------------------------------------------------- main
def make_all(cfg: dict, root: Path, out_dir: Path, kinds=KINDS, final: bool = False, log=None) -> dict:
    runs = mt.load_runs(root / "runs")
    figs = cfg.get("figures") or {}
    out: dict[str, list[str]] = {}
    if "threshold" in kinds:
        out["threshold"] = fig_threshold(cfg, runs, out_dir, final, log)
    if "ci" in kinds:
        out["ci"] = fig_ci(runs, out_dir, log)
    if "rescore" in kinds and figs.get("rescore"):
        out["rescore"] = fig_rescore(figs["rescore"], root, out_dir, final, log)
    if "overlay" in kinds and figs.get("overlay"):
        out["overlay"] = fig_overlay(figs["overlay"], root, out_dir, final, log)
    if "failures" in kinds and figs.get("failures"):
        out["failures"] = fig_failures(figs["failures"], int(cfg["seed"]), root, out_dir, final, log)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    ap.add_argument("--final", action="store_true", help="include test-split runs (rule 5)")
    ap.add_argument("--out_dir", type=Path, default=None, help="default: <project root>/paper/figs")
    ap.add_argument("--only", nargs="+", choices=KINDS, default=list(KINDS))
    args = ap.parse_args(argv)
    root = project_root()
    cfg_path = resolve(args.config, root)
    if not cfg_path.is_file():
        ap.error(f"config not found: {cfg_path}")
    cfg = copy.deepcopy(load_yaml(cfg_path))
    set_seed(int(cfg["seed"]))
    run_dir = root / "runs" / cfg["exp_id"]
    log = get_logger("make_figures", run_dir / "log.txt")
    out = make_all(cfg, root, args.out_dir or root / "paper" / "figs", args.only, args.final, log)
    for kind, paths in out.items():
        for p in paths:
            log.info("figure %s -> %s", kind, p)
    write_run_files(cfg, run_dir, {"final": bool(args.final), "figures": out}, key="figures")
    log.info("%d figures written", sum(len(v) for v in out.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
