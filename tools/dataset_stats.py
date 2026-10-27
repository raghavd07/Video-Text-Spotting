"""Dataset statistics from data/processed/<ds>/<video>.json for the videos in the split files.

Per dataset and split: videos, frames, instances, tracks, dont_care share, text height
(min side of the quad's minimum-area rectangle, cared-for instances only) median and share below
20 px. Writes docs/figs/dataset_stats.csv, docs/figs/text_height_hist.csv,
docs/figs/text_height_hist.png, docs/figs/small_text_share.png and a section in docs/data_report.md.
Test-split GT is read only with --final (rule 5). GT files are loaded one at a time.

Usage: python tools/dataset_stats.py [--ds icdar2013v icdar2015v dstext] [--final]
           [--figs docs/figs] [--report docs/data_report.md]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from logo.data.schema import load_gt, read_split
from logo.utils.io import get_logger
from logo.utils.paths import DATASETS, gt_path, project_root, split_file

SMALL_PX = 20.0
BIN_EDGES = (0, 5, 10, 15, 20, 30, 40, 60, 80, 120, 160, np.inf)
COLS = ("dataset", "split", "videos", "frames", "instances", "tracks", "dont_care_pct",
        "height_median_px", "small_pct")
MARK_START, MARK_END = "<!-- dataset_stats:start -->", "<!-- dataset_stats:end -->"
# validated categorical slots 1-2 (dataviz reference palette, light mode) and text inks
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64"


def quad_height(quad) -> float:
    """Text height = shorter side of the minimum-area rectangle around the quad."""
    (_, _), (w, h), _ = cv2.minAreaRect(np.asarray(quad, np.float32).reshape(4, 2))
    return float(min(w, h))


def video_stats(gt: dict) -> dict:
    objs = [o for fo in gt["frames"].values() for o in fo]
    heights = np.array([quad_height(o["quad"]) for o in objs if not o["dont_care"]], np.float32)
    return {"frames": int(gt["num_frames"]), "instances": len(objs),
            "tracks": len({o["track_id"] for o in objs}),
            "dont_care": sum(bool(o["dont_care"]) for o in objs), "heights": heights}


def aggregate(ds: str, split: str, per_video: list[dict]) -> dict:
    h = np.concatenate([v["heights"] for v in per_video]) if per_video else np.zeros(0, np.float32)
    inst = sum(v["instances"] for v in per_video)
    return {"dataset": ds, "split": split, "videos": len(per_video),
            "frames": sum(v["frames"] for v in per_video), "instances": inst,
            "tracks": sum(v["tracks"] for v in per_video),
            "dont_care_pct": round(100.0 * sum(v["dont_care"] for v in per_video) / inst, 2)
            if inst else 0.0,
            "height_median_px": round(float(np.median(h)), 1) if h.size else 0.0,
            "small_pct": round(100.0 * float((h < SMALL_PX).mean()), 2) if h.size else 0.0,
            "_hist": np.histogram(h, bins=np.asarray(BIN_EDGES, np.float64))[0]}


def bin_labels() -> list[str]:
    e = BIN_EDGES
    return [f"{e[i]:g}-{e[i + 1]:g}" if np.isfinite(e[i + 1]) else f">={e[i]:g}"
            for i in range(len(e) - 1)]


def collect(datasets, final: bool, log) -> list[dict]:
    rows = []
    for ds in datasets:
        per_split: dict[str, list[dict]] = {}
        for split in (("train", "val", "test") if final else ("train", "val")):
            sf = split_file(ds, split)
            if not sf.exists():
                log.warning("%s: %s missing, skipped", ds, sf)
                continue
            vids = [v for v in read_split(sf) if gt_path(ds, v).is_file()]
            missing = len(read_split(sf)) - len(vids)
            if missing:
                log.warning("%s/%s: %d videos without GT json skipped", ds, split, missing)
            per_split[split] = [video_stats(load_gt(gt_path(ds, v))) for v in vids]
            rows.append(aggregate(ds, split, per_split[split]))
        rows.append(aggregate(ds, "all", [v for vs in per_split.values() for v in vs]))
    return rows


def write_csvs(rows: list[dict], figs: Path) -> None:
    figs.mkdir(parents=True, exist_ok=True)
    with open(figs / "dataset_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(figs / "text_height_hist.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "split", *bin_labels()])
        for r in rows:
            w.writerow([r["dataset"], r["split"], *r["_hist"].tolist()])


def _style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color="#e4e3dd", linewidth=0.8)
    ax.set_axisbelow(True)


def plot(rows: list[dict], figs: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    totals = [r for r in rows if r["split"] == "all" and r["instances"]]
    if not totals:
        return
    labels = bin_labels()
    x = np.arange(len(labels))
    cut = BIN_EDGES.index(SMALL_PX) - 0.5
    fig, axes = plt.subplots(1, len(totals), figsize=(4.2 * len(totals), 3.4), sharey=True,
                             squeeze=False)
    for ax, r in zip(axes[0], totals):
        share = 100.0 * r["_hist"] / max(1, r["_hist"].sum())
        ax.bar(x, share, width=0.8, color=BLUE)
        ax.axvline(cut, color=MUTED, linestyle="--", linewidth=1)
        ax.text(0.98, 0.95, f"<{SMALL_PX:g} px (left of dashed line): {r['small_pct']:.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK)
        ax.set_title(r["dataset"], fontsize=10, color=INK)
        ax.set_xticks(x, labels, rotation=60, ha="right")
        ax.set_xlabel("text height (px)", fontsize=8, color=MUTED)
        _style(ax)
    axes[0][0].set_ylabel("share of instances (%)", fontsize=8, color=MUTED)
    fig.suptitle("Text height distribution (train+val, cared-for instances)", fontsize=11,
                 color=INK)
    fig.tight_layout()
    fig.savefig(figs / "text_height_hist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    x = np.arange(len(totals))
    for off, key, color, name in ((-0.2, "small_pct", BLUE, f"small text (<{SMALL_PX:g} px)"),
                                  (0.2, "dont_care_pct", ORANGE, "don't care")):
        vals = [r[key] for r in totals]
        bars = ax.bar(x + off, vals, width=0.38, color=color, label=name)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in vals], fontsize=8, color=INK, padding=2)
    ax.set_xticks(x, [r["dataset"] for r in totals])
    ax.set_ylabel("share of instances (%)", fontsize=8, color=MUTED)
    ax.set_title("Small and don't-care text per dataset (train+val)", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(figs / "small_text_share.png", dpi=150)
    plt.close(fig)


def markdown(rows: list[dict], figs: Path, report: Path, final: bool) -> str:
    try:
        rel = Path(os.path.relpath(figs, report.parent))
    except ValueError:  # different drives on Windows
        rel = figs
    lines = [MARK_START, "## Statistics", "",
             f"### {date.today().isoformat()} - `tools/dataset_stats.py{' --final' if final else ''}`",
             "", "| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    lines += ["| " + " | ".join(str(r[c]) for c in COLS) + " |" for r in rows]
    lines += ["", f"- height = shorter side of the min-area rectangle of the quad, cared-for "
              f"instances only; small = height < {SMALL_PX:g} px; dont_care_pct over all instances",
              "- test rows only with --final; datasets without test GT have no test row",
              f"- CSV: `{(rel / 'dataset_stats.csv').as_posix()}`, "
              f"`{(rel / 'text_height_hist.csv').as_posix()}`", "",
              f"![text height]({(rel / 'text_height_hist.png').as_posix()})", "",
              f"![small text share]({(rel / 'small_text_share.png').as_posix()})", MARK_END, ""]
    return "\n".join(lines)


def update_report(report: Path, section: str) -> None:
    text = report.read_text(encoding="utf-8") if report.exists() else "# Data report\n\n"
    if MARK_START in text and MARK_END in text:
        head, rest = text.split(MARK_START, 1)
        tail = rest.split(MARK_END, 1)[1].lstrip("\n")
        text = head + section + tail
    else:
        text = text.rstrip("\n") + "\n\n" + section
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ds", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--final", action="store_true", help="also read test-split GT")
    ap.add_argument("--figs", type=Path, default=project_root() / "docs" / "figs")
    ap.add_argument("--report", type=Path, default=project_root() / "docs" / "data_report.md")
    args = ap.parse_args(argv)
    log = get_logger("dataset_stats", project_root() / "logs" / "dataset_stats.log")
    rows = collect(args.ds, args.final, log)
    write_csvs(rows, args.figs)
    plot(rows, args.figs)
    update_report(args.report, markdown(rows, args.figs.resolve(), args.report.resolve(),
                                        args.final))
    for r in rows:
        log.info(" ".join(f"{c}={r[c]}" for c in COLS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
