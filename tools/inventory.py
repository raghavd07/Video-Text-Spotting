"""Inventory of data/raw/<ds>: videos vs. annotation files, matched by file stem.

Raw naming (checked 2026-10-26):
  <split>/[<category>/]<Video>.mp4            video (DSText adds a category folder)
  <Video>_GT.xml                              box-level annotation  -> counted as "annotation"
  <Video>_GT.txt                              ICDAR2015 train: object-id -> transcription (auxiliary)
  <Video>_GT_voc.txt                          ICDAR2015: per-video vocabulary (auxiliary, NOT a box GT)
  {train,test}setVocabulary.txt               ICDAR2015 split vocabulary (ignored)
Stems are matched case-insensitively (ICDAR2013 train has 'video_51_7_4_GT.xml').

Usage: python tools/inventory.py --root data/raw [--report docs/data_report.md] [--no-write]
Stdlib only, so it also runs in logo-paddle.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

DATASETS = ("icdar2013v", "icdar2015v", "dstext")
EXPECTED = {"icdar2013v": {"total": 28},
            "icdar2015v": {"total": 49, "train": 25, "test": 24},
            "dstext": {"total": 100, "train": 50, "test": 50}}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
ANN_SUFFIX = "_gt.xml"
AUX_SUFFIXES = ("_gt_voc.txt", "_gt.txt")  # longest first
SECTION = "## Inventory"


def _split_of(path: Path, ds_dir: Path) -> str:
    parts = path.relative_to(ds_dir).parts
    return parts[0] if len(parts) > 1 else "."


def _ann_stem(name: str) -> str | None:
    low = name.lower()
    return name[: -len(ANN_SUFFIX)] if low.endswith(ANN_SUFFIX) else None


def _aux_stem(name: str) -> str | None:
    low = name.lower()
    for suf in AUX_SUFFIXES:
        if low.endswith(suf):
            return name[: -len(suf)]
    return None


def scan_dataset(ds_dir: Path) -> dict:
    """Return {"splits": {split: row}, "total": row} for one dataset folder."""
    ds_dir = Path(ds_dir)
    splits: dict[str, dict] = {}

    def row(split: str) -> dict:
        return splits.setdefault(split, {"videos": {}, "anns": {}, "aux": {}})

    if ds_dir.is_dir():
        for p in sorted(ds_dir.rglob("*")):
            if not p.is_file():
                continue
            r = row(_split_of(p, ds_dir))
            rel = p.relative_to(ds_dir).as_posix()
            if p.suffix.lower() in VIDEO_EXTS:
                r["videos"].setdefault(p.stem.lower(), []).append(rel)
            elif (s := _ann_stem(p.name)) is not None:
                r["anns"].setdefault(s.lower(), []).append(rel)
            elif (s := _aux_stem(p.name)) is not None:
                r["aux"].setdefault(s.lower(), []).append(rel)

    out = {}
    for split, r in sorted(splits.items()):
        v, a, x = r["videos"], r["anns"], r["aux"]
        case_mismatch = sorted(
            f"{v[k][0]} <-> {a[k][0]}" for k in set(v) & set(a)
            if Path(v[k][0]).stem != Path(a[k][0]).name[: -len(ANN_SUFFIX)]
        )
        out[split] = {
            "num_videos": sum(len(l) for l in v.values()),
            "num_annotations": sum(len(l) for l in a.values()),
            "num_aux": sum(len(l) for l in x.values()),
            "matched": len(set(v) & set(a)),
            "videos_without_ann": sorted(v[k][0] for k in set(v) - set(a)),
            "anns_without_video": sorted(a[k][0] for k in set(a) - set(v)),
            "aux_only": sorted(v[k][0] for k in (set(v) - set(a)) & set(x)),
            "duplicate_stems": sorted(k for d in (v, a) for k, l in d.items() if len(l) > 1),
            "case_mismatch": case_mismatch,
        }
    total = {k: sum(s[k] for s in out.values())
             for k in ("num_videos", "num_annotations", "num_aux", "matched")}
    total.update({k: [i for s in out.values() for i in s[k]]
                  for k in ("videos_without_ann", "anns_without_video", "aux_only",
                            "duplicate_stems", "case_mismatch")})
    return {"splits": out, "total": total}


def scan_root(root: Path) -> dict:
    return {ds: scan_dataset(Path(root) / ds) for ds in DATASETS}


def _expected(ds: str, split: str) -> str:
    n = EXPECTED[ds].get(split)
    return "-" if n is None else str(n)


def render_table(results: dict) -> str:
    head = "| dataset | split | videos | expected | GT xml | matched | video w/o GT | GT w/o video | aux txt |"
    lines = [head, "|" + "---|" * 9]
    for ds, res in results.items():
        rows = list(res["splits"].items()) + [("total", res["total"])]
        if not res["splits"]:
            rows = [("total", res["total"])]
        for split, s in rows:
            lines.append(f"| {ds} | {split} | {s['num_videos']} | {_expected(ds, split)} | "
                         f"{s['num_annotations']} | {s['matched']} | {len(s['videos_without_ann'])} | "
                         f"{len(s['anns_without_video'])} | {s['num_aux']} |")
    return "\n".join(lines)


def render_details(results: dict) -> str:
    lines = []
    for ds, res in results.items():
        t = res["total"]
        exp = EXPECTED[ds]["total"]
        flag = "OK" if t["num_videos"] == exp else f"MISMATCH (expected {exp})"
        lines.append(f"- **{ds}**: {t['num_videos']} videos, {t['num_annotations']} GT xml -> {flag}")
        for key, label in (("videos_without_ann", "videos without GT xml"),
                           ("aux_only", "  of which only aux txt (vocabulary/transcripts, no boxes)"),
                           ("anns_without_video", "GT xml without video"),
                           ("case_mismatch", "matched only case-insensitively"),
                           ("duplicate_stems", "duplicate stems")):
            items = t[key]
            if key == "aux_only" and items and items == t["videos_without_ann"]:
                lines.append(f"  - all {len(items)} have only aux txt (vocabulary/transcripts, no boxes)")
            elif items:
                shown = ", ".join(f"`{i}`" for i in items[:60])
                more = f" ... (+{len(items) - 60})" if len(items) > 60 else ""
                lines.append(f"  - {label} ({len(items)}): {shown}{more}")
    return "\n".join(lines)


def render_markdown(results: dict, root: Path) -> str:
    return (f"### {date.today().isoformat()} - `tools/inventory.py --root {Path(root).as_posix()}`\n\n"
            f"{render_table(results)}\n\n{render_details(results)}\n")


def append_report(block: str, report: Path) -> None:
    """Append `block` at the end of the '## Inventory' section (created if missing)."""
    report = Path(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    text = report.read_text(encoding="utf-8") if report.exists() else "# Data report\n"
    lines = text.rstrip("\n").split("\n")
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == SECTION)
    except StopIteration:
        lines += ["", SECTION]
        start = len(lines) - 1
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    new = lines[:end] + [""] + block.rstrip("\n").split("\n") + ([""] if end < len(lines) else []) + lines[end:]
    report.write_text("\n".join(new) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--report", type=Path, default=Path("docs/data_report.md"))
    ap.add_argument("--no-write", action="store_true", help="print only, do not touch the report")
    args = ap.parse_args(argv)
    if not args.root.is_dir():
        print(f"root not found: {args.root}", file=sys.stderr)
        return 1
    results = scan_root(args.root)
    print(render_table(results))
    print()
    print(render_details(results))
    if not args.no_write:
        append_report(render_markdown(results, args.root), args.report)
        print(f"\nappended to {args.report} under '{SECTION}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
