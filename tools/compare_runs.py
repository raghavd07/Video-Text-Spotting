"""Print a Markdown table of selected metrics (from runs/<exp_id>/metrics.json) for several exp ids.

Metrics are <section>.<key> paths into metrics.json (e.g. det.f, track.IDF1, spot.MOTA); missing -> "-".

Usage: python tools/compare_runs.py E4-base E4-lsc [--metrics det.f track.MOTA ...] [--runs_dir runs]
           [--out docs/compare_E4.md]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from logo.utils.io import load_json
from logo.utils.paths import project_root

DEFAULT_METRICS = ("det.precision", "det.recall", "det.f", "track.MOTA", "track.MOTP", "track.IDF1",
                   "track.MT", "track.ML", "spot.MOTA", "spot.MOTP", "spot.IDF1")


def lookup(metrics: dict, path: str):
    v = metrics
    for k in path.split("."):
        if not isinstance(v, dict) or k not in v:
            return None
        v = v[k]
    return v


def fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.2f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def compare(exp_ids: list[str], metrics: list[str], runs_dir: Path) -> str:
    lines = ["| exp_id | " + " | ".join(metrics) + " |", "|---|" + "---:|" * len(metrics)]
    for exp in exp_ids:
        p = Path(runs_dir) / exp / "metrics.json"
        m = load_json(p) if p.is_file() else {}
        lines.append(f"| {exp} | " + " | ".join(fmt(lookup(m, k)) for k in metrics) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("exp_ids", nargs="+")
    ap.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    ap.add_argument("--runs_dir", type=Path, default=None, help="default: <project root>/runs")
    ap.add_argument("--out", type=Path, default=None, help="also write the table here")
    args = ap.parse_args(argv)
    md = compare(args.exp_ids, args.metrics, args.runs_dir or project_root() / "runs")
    print(md, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
