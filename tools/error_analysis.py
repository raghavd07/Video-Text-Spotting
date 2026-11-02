"""Error analysis of a run's tracks against GT: counts per category and text-height bin, bar chart, gallery.

Matching is the evaluator's (logo.eval.mot_eval, tracking mode): per frame, dont_care GT dropped and
predictions > 50% inside a dont_care quad ignored, IoU >= --iou, py-motmetrics accumulator. Categories:
  missed_gt       care GT with no match (MISS)                        bin = GT height
  false_positive  kept prediction with no match (FP)                   bin = prediction height
  wrong_text      matched (MATCH/SWITCH), GT normalized text non-empty and != prediction's
                  (logo.eval.match.normalize_text)                     bin = GT height
  id_switch       SWITCH events                                        bin = GT height
  fragmentation   tracked -> missed transitions inside a GT track's tracked span (motmetrics
                  num_fragmentations definition); event frame = first missed frame   bin = GT height
Height = shorter side of the quad's minimum-area rectangle (as tools/dataset_stats.py); bins <20, 20-50, >50 px.
missed_gt / false_positive / id_switch totals equal per_video misses / fp / switches of metrics.json "track".

Writes <out_dir>/errors.csv (category,size_bin,count), <out_dir>/errors_bar.png, <out_dir>/gallery_<cat>.png
(--per_cat examples per category, reservoir-sampled with cfg seed; GT green, prediction red; for
fragmentation the red box is the last matched prediction before the gap) and
runs/<exp_id>/{config.yaml (if absent),log.txt,metrics.json meta.errors_<ds>_<split>}.
Tracks: runs/<exp_id>/<--stage>/<ds>/<video>.jsonl (default vote = what run_pipeline evaluates).

Usage: python tools/error_analysis.py --config configs/E5-lst.yaml [--stage vote|track] [--track_dir DIR]
           [--iou 0.5] [--per_cat 6] [--out_dir docs/figs/errors] [--final]
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Optional

import cv2
import motmetrics as mm
import numpy as np
import yaml

from logo.data.schema import load_gt, read_split
from logo.eval.match import frame_ids, load_pred_frames, normalize_text, prepare_frame
from logo.utils.io import get_logger, load_json, load_yaml, save_json, set_seed
from logo.utils.paths import check_dataset, check_split, project_root

CATEGORIES = ("missed_gt", "false_positive", "wrong_text", "id_switch", "fragmentation")
LABELS = {"missed_gt": "missed GT", "false_positive": "false positives", "wrong_text": "wrong transcription",
          "id_switch": "ID switches", "fragmentation": "fragmentations"}
SIZE_BINS = ("<20", "20-50", ">50")
SMALL_PX, LARGE_PX = 20.0, 50.0
GT_BGR, PRED_BGR = (0, 200, 0), (0, 0, 255)
TILE_W, TILE_H, CAP_H, TITLE_H, COLS = 320, 180, 36, 30, 3
MISSING_BGR = (128, 128, 128)
# ordinal blue ramp (dataviz reference palette steps 250/450/650, validated --ordinal on light) + chrome
BIN_COLORS = ("#86b6ef", "#2a78d6", "#104281")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e4e3dd"


def quad_height(quad) -> float:
    """Text height = shorter side of the minimum-area rectangle around the quad."""
    (_, _), (w, h), _ = cv2.minAreaRect(np.asarray(quad, np.float32).reshape(4, 2))
    return float(min(w, h))


def size_bin(h: float) -> str:
    return SIZE_BINS[0] if h < SMALL_PX else SIZE_BINS[1] if h <= LARGE_PX else SIZE_BINS[2]


def empty_counts() -> dict:
    return {c: {b: 0 for b in SIZE_BINS} for c in CATEGORIES}


class Gallery:
    """Per-category reservoir sample of k examples (deterministic for a given rng and event order)."""

    def __init__(self, k: int, rng: np.random.Generator):
        self.k, self.rng = int(k), rng
        self.seen = {c: 0 for c in CATEGORIES}
        self.items: dict[str, list[dict]] = {c: [] for c in CATEGORIES}

    def offer(self, cat: str, ex: dict) -> None:
        self.seen[cat] += 1
        n, items = self.seen[cat], self.items[cat]
        if len(items) < self.k:
            items.append(ex)
        elif self.k:
            j = int(self.rng.integers(n))
            if j < self.k:
                items[j] = ex


def _int_id(x) -> Optional[int]:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else int(x)


def _accumulate(gt: dict, pred: dict, num_frames: int, video: str, iou: float):
    """Tracking-mode accumulator (as mot_eval) plus {frame: ({gt_tid: obj}, {pred_tid: obj})} of kept objects."""
    acc = mm.MOTAccumulator(auto_id=False)
    lookup = {}
    for fid in frame_ids(num_frames, pred):
        fm = prepare_frame(gt.get(fid, []), pred.get(fid, []), spotting=False)
        dist = 1.0 - fm.iou
        dist[fm.iou < iou] = np.nan
        g_ids = [int(o["track_id"]) for o in fm.gt]
        p_ids = [int(o["track_id"]) for o in fm.pred]
        if len(set(g_ids)) != len(g_ids) or len(set(p_ids)) != len(p_ids):
            raise ValueError(f"{video} frame {fid}: duplicate track_id")
        acc.update(g_ids, p_ids, dist, frameid=fid)
        lookup[fid] = (dict(zip(g_ids, fm.gt)), dict(zip(p_ids, fm.pred)))
    return acc, lookup


def _example(video: str, fid: int, g: Optional[dict], p: Optional[dict], h: float, note: str = "") -> dict:
    return {"video": video, "frame": int(fid), "gt_quad": g["quad"] if g else None,
            "pred_quad": p["quad"] if p else None, "gt_text": g.get("text", "") if g else None,
            "pred_text": p.get("text", "") if p else None, "height": round(float(h), 1), "note": note}


def analyze_video(gt: dict, pred: dict, num_frames: int, video: str, iou: float, counts: dict,
                  gallery: Optional[Gallery] = None) -> dict:
    """Adds one video's errors to `counts`; returns its per-category totals."""
    acc, lookup = _accumulate(gt, pred, num_frames, video, iou)
    ev = acc.mot_events
    ev = ev[ev.Type.isin(["MATCH", "SWITCH", "MISS", "FP"])]
    tot = {c: 0 for c in CATEGORIES}
    last_hid: dict[int, int] = {}
    last_pred: dict[int, dict] = {}
    span: dict[int, list] = {}  # oid -> [(frame, missed, gt_obj, last matched pred obj)]

    def add(cat, h, ex):
        counts[cat][size_bin(h)] += 1
        tot[cat] += 1
        if gallery is not None:
            gallery.offer(cat, ex)

    for (fid, _), typ, oid, hid in zip(ev.index, ev.Type, ev.OId, ev.HId):
        fid, oid, hid = int(fid), _int_id(oid), _int_id(hid)
        g_map, p_map = lookup[fid]
        if typ == "FP":
            p = p_map[hid]
            h = quad_height(p["quad"])
            add("false_positive", h, _example(video, fid, None, p, h, f"pred id {hid}"))
            continue
        g = g_map[oid]
        h = quad_height(g["quad"])
        if typ == "MISS":
            add("missed_gt", h, _example(video, fid, g, None, h, f"GT id {oid}"))
            span.setdefault(oid, []).append((fid, True, g, last_pred.get(oid)))
            continue
        p = p_map[hid]
        gt_t = normalize_text(g.get("text", ""))
        if gt_t and gt_t != normalize_text(p.get("text", "")):
            add("wrong_text", h, _example(video, fid, g, p, h))
        if typ == "SWITCH":
            add("id_switch", h, _example(video, fid, g, p, h, f"GT {oid}: pred {last_hid.get(oid)}->{hid}"))
        last_hid[oid], last_pred[oid] = hid, p
        span.setdefault(oid, []).append((fid, False, g, p))

    for oid, seq in span.items():
        tracked = [i for i, s in enumerate(seq) if not s[1]]
        if not tracked:
            continue
        for i in range(tracked[0] + 1, tracked[-1] + 1):
            fid, missed, g, p = seq[i]
            if missed and not seq[i - 1][1]:
                h = quad_height(g["quad"])
                add("fragmentation", h, _example(video, fid, g, p, h, f"GT id {oid} lost"))
    return tot


# ----------------------------------------------------------------------------------------------- outputs
def write_csv(counts: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "size_bin", "count"])
        for c in CATEGORIES:
            for b in SIZE_BINS:
                w.writerow([c, b, counts[c][b]])
            w.writerow([c, "all", sum(counts[c].values())])


def plot_bars(counts: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    x = np.arange(len(CATEGORIES))
    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bottom = np.zeros(len(CATEGORIES))
    for b, color in zip(SIZE_BINS, BIN_COLORS):
        v = np.array([counts[c][b] for c in CATEGORIES], np.float64)
        ax.bar(x, v, bottom=bottom, width=0.6, color=color, edgecolor=SURFACE, linewidth=2,
               label=f"{b} px")
        bottom += v
    top = max(float(bottom.max()), 1.0)
    for xi, t in zip(x, bottom):
        ax.text(xi, t + 0.015 * top, f"{int(t)}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_ylim(0, top * 1.12)
    ax.set_xticks(x, [LABELS[c] for c in CATEGORIES], fontsize=8, color=INK)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("count", fontsize=8, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(title="text height", fontsize=8, title_fontsize=8, frameon=False, labelcolor=INK)
    ax.set_title(title, fontsize=10, color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _ascii(s, n: int = 20) -> str:
    s = str(s).encode("ascii", "replace").decode("ascii")
    return s if len(s) <= n else s[:n - 1] + "~"


def read_image(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)  # non-ASCII paths on Windows


def render_tile(img: np.ndarray, ex: dict) -> np.ndarray:
    """[TILE_H + CAP_H, TILE_W, 3]: crop around GT/prediction quads (drawn on the frame), letterboxed."""
    canvas = img.copy()
    quads = []
    for key, color, thick in (("gt_quad", GT_BGR, 4), ("pred_quad", PRED_BGR, 1)):  # GT stays visible under pred
        if ex[key] is not None:
            pts = np.round(np.asarray(ex[key], np.float64).reshape(4, 2)).astype(np.int32)
            cv2.polylines(canvas, [pts], True, color, thick, cv2.LINE_AA)
            quads.append(pts)
    pts = np.concatenate(quads)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    m = int(max(16, 0.6 * max(x1 - x0, y1 - y0)))
    H, W = canvas.shape[:2]
    x0, y0 = max(0, int(x0) - m), max(0, int(y0) - m)
    x1, y1 = min(W, int(x1) + m + 1), min(H, int(y1) + m + 1)
    crop = canvas[y0:y1, x0:x1]
    tile = np.full((TILE_H, TILE_W, 3), 255, np.uint8)
    if crop.size:
        s = min(TILE_W / crop.shape[1], TILE_H / crop.shape[0])
        cw, ch = max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))
        r = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        ox, oy = (TILE_W - cw) // 2, (TILE_H - ch) // 2
        tile[oy:oy + ch, ox:ox + cw] = r
    cap = np.full((CAP_H, TILE_W, 3), 255, np.uint8)
    line1 = f"{_ascii(ex['video'], 22)} f{ex['frame']} h={ex['height']:g}px {_ascii(ex['note'], 22)}"
    parts = []
    if ex["gt_text"] is not None:
        parts.append(f"GT '{_ascii(ex['gt_text'], 14)}'")
    if ex["pred_text"] is not None:
        parts.append(f"pred '{_ascii(ex['pred_text'], 14)}'")
    for i, (txt, color) in enumerate(((line1, (60, 60, 60)), (" | ".join(parts), (0, 0, 0)))):
        cv2.putText(cap, txt, (4, 14 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return np.vstack([tile, cap])


def render_gallery(examples: list[dict], title: str, frame_of) -> np.ndarray:
    """Grid of COLS columns with a title bar; frame_of(video, frame) -> BGR image."""
    rows = max(1, -(-len(examples) // COLS))
    th = TILE_H + CAP_H
    out = np.full((TITLE_H + rows * (th + 4), COLS * (TILE_W + 4), 3), 255, np.uint8)
    cv2.putText(out, _ascii(title, 80), (6, TITLE_H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                cv2.LINE_AA)
    for i, ex in enumerate(examples):
        r, c = divmod(i, COLS)
        y, x = TITLE_H + r * (th + 4), c * (TILE_W + 4)
        out[y:y + th, x:x + TILE_W] = render_tile(frame_of(ex["video"], ex["frame"]), ex)
    return out


def write_image(img: np.ndarray, path: Path) -> None:
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise RuntimeError(f"cannot encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())


def top_two_sentence(totals: dict) -> str:
    total = sum(totals.values())
    if total == 0:
        return "No errors were found."
    order = sorted(CATEGORIES, key=lambda c: (-totals[c], CATEGORIES.index(c)))[:2]
    a, b = order
    return (f"The two largest error categories are {LABELS[a]} ({totals[a]}, {100 * totals[a] / total:.1f}% of "
            f"all errors) and {LABELS[b]} ({totals[b]}, {100 * totals[b] / total:.1f}%).")


# ----------------------------------------------------------------------------------------------- main
def analyze(cfg: dict, root: Path, stage: str = "vote", track_dir: Optional[Path] = None,
            iou: Optional[float] = None, per_cat: int = 6, out_dir: Optional[Path] = None,
            final: bool = False, log=None) -> dict:
    """Runs the analysis for cfg dataset/split and writes all outputs; returns the summary dict."""
    ds, split, exp = check_dataset(cfg["dataset"]), check_split(cfg["split"], final), cfg["exp_id"]
    run_dir = root / "runs" / exp
    log = log or get_logger("error_analysis", run_dir / "log.txt")
    set_seed(cfg["seed"])
    iou = float(iou if iou is not None else (cfg.get("eval") or {}).get("iou", 0.5))
    track_dir = Path(track_dir) if track_dir else run_dir / stage / ds
    out_dir = Path(out_dir) if out_dir else root / "docs" / "figs" / "errors"
    gt_dir = root / "data" / "processed" / ds
    videos = read_split(root / "splits" / f"{ds}_{split}.txt")
    log.info("error_analysis exp=%s ds=%s split=%s videos=%d iou=%.2f tracks=%s", exp, ds, split,
             len(videos), iou, track_dir)

    counts = empty_counts()
    gallery = Gallery(per_cat, np.random.default_rng(int(cfg["seed"])))
    sizes = {}
    per_video = {}
    for video in videos:
        g = load_gt(gt_dir / f"{video}.json")
        sizes[video] = (int(g["width"]), int(g["height"]))
        gt = {int(k): v for k, v in g["frames"].items()}
        pred = load_pred_frames(track_dir, video, tracks=True)
        per_video[video] = analyze_video(gt, pred, int(g["num_frames"]), video, iou, counts, gallery)
        del gt, pred, g
    totals = {c: sum(counts[c].values()) for c in CATEGORIES}

    write_csv(counts, out_dir / "errors.csv")
    plot_bars(counts, out_dir / "errors_bar.png", f"Errors by category and text height - {exp} {ds} {split}")

    def frame_of(video: str, fid: int) -> np.ndarray:
        p = root / "data" / "frames" / ds / video / f"{fid:06d}.jpg"
        img = read_image(p)
        if img is None:
            log.warning("missing frame %s: drawing on a grey canvas", p)
            w, h = sizes[video]
            img = np.full((max(h, 1), max(w, 1), 3), MISSING_BGR, np.uint8)
        return img

    galleries = {}
    for c in CATEGORIES:
        ex = gallery.items[c]
        if not ex:
            continue
        p = out_dir / f"gallery_{c}.png"
        write_image(render_gallery(ex, f"{LABELS[c]} ({totals[c]}) - {exp} {ds} {split}", frame_of), p)
        galleries[c] = str(p)

    sentence = top_two_sentence(totals)
    res = {"stage": stage, "track_dir": str(track_dir), "iou": iou, "num_videos": len(videos),
           "counts": counts, "totals": totals, "per_video": per_video, "out_dir": str(out_dir),
           "galleries": galleries, "summary": sentence}
    write_run_files(cfg, run_dir, res, key=f"errors_{ds}_{split}")
    log.info("totals: %s", " ".join(f"{c}={totals[c]}" for c in CATEGORIES))
    log.info("%s", sentence)
    return res


def write_run_files(cfg: dict, run_dir: Path, res: dict, key: str) -> None:
    """config.yaml if absent; metrics.json updated with meta[key] (other sections untouched)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_out = run_dir / "config.yaml"
    if not cfg_out.exists():
        cfg_out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    m_out = run_dir / "metrics.json"
    metrics = {"det": {}, "track": {}, "spot": {}, "per_video": {}, "meta": {}}
    if m_out.is_file():
        metrics.update(load_json(m_out))
    metrics.setdefault("meta", {})[key] = res
    save_json(metrics, m_out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--stage", choices=("vote", "track"), default="vote")
    ap.add_argument("--track_dir", type=Path, default=None, help="default: runs/<exp_id>/<stage>/<ds>")
    ap.add_argument("--iou", type=float, default=None, help="default: config eval.iou, else 0.5")
    ap.add_argument("--per_cat", type=int, default=6, help="gallery examples per category")
    ap.add_argument("--out_dir", type=Path, default=None, help="default: docs/figs/errors")
    ap.add_argument("--final", action="store_true", help="allow split=test (rule 5)")
    args = ap.parse_args(argv)
    cfg = copy.deepcopy(load_yaml(args.config))
    if cfg.get("split") == "test" and not args.final:
        ap.error("config split is test: pass --final")
    res = analyze(cfg, project_root(), args.stage, args.track_dir, args.iou, args.per_cat, args.out_dir,
                  args.final)
    print(res["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
