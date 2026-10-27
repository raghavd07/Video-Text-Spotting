"""Convert raw GT xml to data/processed/<ds>/<video>.json (CONTRACT GT schema) and print a summary.

Raw train annotations are always converted; raw test annotations only with --final (rule 5).
width/height: first frame image data/frames/<ds>/<video>/000001.jpg if present, else the raw video
header, else 0 (flagged in the log and the summary). num_frames = max(frame count, last xml frame id).

Usage: python tools/convert_annotations.py [--ds icdar2013v icdar2015v dstext] [--final]
           [--raw-root data/raw] [--frames-root data/frames] [--out data/processed]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from logo.data.convert_gt import build_gt, probe_video
from logo.data.parsers import PARSERS, STAT_KEYS, find_annotations
from logo.data.schema import validate_gt
from logo.utils.io import get_logger, save_json
from logo.utils.paths import DATASETS, project_root

COLS = ("dataset", "videos", "frames", "instances", "dont_care", *STAT_KEYS, "size_flagged")


def resolve_size(frames_dir: Path, video_path: Path | None) -> tuple[int, int, int, str]:
    """(width, height, num_frames, source); source in {'frames', 'video', 'missing'}."""
    first = frames_dir / "000001.jpg"
    if first.is_file():
        img = cv2.imread(str(first))
        if img is not None:
            n = sum(1 for p in frames_dir.glob("[0-9]" * 6 + ".jpg"))
            return img.shape[1], img.shape[0], n, "frames"
    if video_path is not None:
        try:
            w, h, n = probe_video(video_path)
            if w > 0 and h > 0:
                return w, h, n, "video"
        except OSError:
            pass
    return 0, 0, 0, "missing"


def convert_dataset(ds: str, raw_root: Path, frames_root: Path, out_root: Path, final: bool,
                    log) -> dict:
    row = {c: 0 for c in COLS}
    row["dataset"] = ds
    for raw_split in (("train", "test") if final else ("train",)):
        for vid, xml_path, video_path in find_annotations(raw_root / ds, raw_split):
            stats: dict = {}
            frames = PARSERS[ds](xml_path, stats)
            w, h, n, src = resolve_size(frames_root / ds / vid, video_path)
            gt = validate_gt(build_gt(vid, ds, frames, w, h, n))
            save_json(gt, out_root / ds / f"{vid}.json")
            objs = [o for fo in gt["frames"].values() for o in fo]
            row["videos"] += 1
            row["frames"] += gt["num_frames"]
            row["instances"] += len(objs)
            row["dont_care"] += sum(o["dont_care"] for o in objs)
            for k in STAT_KEYS:
                row[k] += stats.get(k, 0)
            if src == "missing":
                row["size_flagged"] += 1
                log.warning("%s/%s/%s: width/height unknown (no frames, no readable video) -> 0",
                            ds, raw_split, vid)
            log.info("%s/%s/%s: %dx%d (%s) frames=%d objects=%d %s", ds, raw_split, vid, w, h, src,
                     gt["num_frames"], len(objs), stats or "")
    return row


def render_table(rows: list[dict]) -> str:
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in COLS]
    line = lambda vals: "  ".join(str(v).rjust(wd) for v, wd in zip(vals, widths))  # noqa: E731
    return "\n".join([line(COLS), line("-" * wd for wd in widths)]
                     + [line(r[c] for c in COLS) for r in rows])


def main(argv=None) -> int:
    root = project_root()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ds", nargs="+", choices=DATASETS, default=list(DATASETS))
    ap.add_argument("--final", action="store_true", help="also convert raw test GT")
    ap.add_argument("--raw-root", type=Path, default=root / "data" / "raw")
    ap.add_argument("--frames-root", type=Path, default=root / "data" / "frames")
    ap.add_argument("--out", type=Path, default=root / "data" / "processed")
    ap.add_argument("--log", type=Path, default=root / "logs" / "convert_annotations.log")
    args = ap.parse_args(argv)
    log = get_logger("convert_annotations", args.log)
    rows = [convert_dataset(ds, args.raw_root, args.frames_root, args.out, args.final, log)
            for ds in args.ds]
    print(render_table(rows))
    return 0 if all(r["videos"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
