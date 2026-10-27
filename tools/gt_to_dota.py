"""data/processed GT -> DOTA polygons + COCO json for PP-YOLOE-R text fine-tuning (one class 'text').

dont_care boxes are OMITTED, not written as difficult=1: PaddleDetection's DOTA->COCO converter
(configs/rotate/tools/convert.py) keeps difficult 0 and 1 alike as targets (iscrowd=0) and only drops
difficult=2. Frames whose only boxes are dont_care are skipped as well, so unreadable text is never
taught as background. Every written line therefore has difficult=0.

Output (<out> = data/det_dota):
  <out>/<split>/labelTxt/<ds>__<video>__<fid:06d>.txt  DOTA lines 'x1 y1 x2 y2 x3 y3 x4 y4 text 0'
  <out>/<split>/images.txt   dataset file listing: one frame per line, relative to data/frames
  <out>/<split>.json         COCO json with the fields convert.py writes; file_name relative to data/frames
  <out>/<split>_smoke.json   smoke_n evenly spaced images of <split>.json
  <out>/<split>_stats.json
ppdet reads frames in place: dataset_dir 'data', image_dir 'frames', anno_path 'det_dota/<split>.json'.

Usage: python tools/gt_to_dota.py [--datasets icdar2013v icdar2015v dstext] [--splits train val]
           [--stride 2] [--out data/det_dota] [--smoke_n 200] [--min_area 4] [--final]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from logo.data.schema import load_gt, read_split
from logo.utils.geometry import signed_area
from logo.utils.io import get_logger, save_json
from logo.utils.paths import check_dataset, check_split, project_root

DATASETS = ("icdar2013v", "icdar2015v", "dstext")
CATEGORY = "text"


def frame_ids(num_frames: int, stride: int) -> range:
    """1-based ids 1, 1+stride, ... <= num_frames."""
    return range(1, int(num_frames) + 1, int(stride))


def clean_quad(quad, width: int, height: int, min_area: float) -> Optional[list[float]]:
    """Clips the quad to the image; None if its clipped area is below min_area (degenerate)."""
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    q[:, 0] = q[:, 0].clip(0, width - 1)
    q[:, 1] = q[:, 1].clip(0, height - 1)
    if abs(signed_area(q.reshape(-1))) < min_area:
        return None
    return [round(float(v), 1) for v in q.reshape(-1)]


def dota_line(poly: list[float]) -> str:
    return " ".join(f"{v:.1f}" for v in poly) + f" {CATEGORY} 0"


def coco_annotation(poly: list[float], image_id: int) -> dict:
    """Same fields and bbox/area convention as PaddleDetection convert.process_single_sample."""
    xs, ys = poly[0::2], poly[1::2]
    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
    w, h = xmax - xmin, ymax - ymin
    return {"category_id": 1, "segmentation": [poly], "iscrowd": 0,
            "bbox": [xmin, ymin, w, h], "area": w * h, "image_id": image_id}


def smoke_subset(coco: dict, n: int) -> dict:
    """n evenly spaced images (deterministic, spans many videos) with their annotations."""
    images = coco["images"]
    if n <= 0 or not images:
        keep = []
    else:
        idx = np.unique(np.linspace(0, len(images) - 1, min(n, len(images))).round().astype(int))
        keep = [images[i] for i in idx]
    ids = {im["id"] for im in keep}
    return {"categories": coco["categories"], "images": keep,
            "annotations": [a for a in coco["annotations"] if a["image_id"] in ids]}


def convert_split(root: Path, datasets, split: str, stride: int, out: Path, smoke_n: int = 200,
                  min_area: float = 4.0, final: bool = False, log=None) -> dict:
    check_split(split, final)
    label_dir = out / split / "labelTxt"
    label_dir.mkdir(parents=True, exist_ok=True)
    for old in label_dir.glob("*.txt"):  # a rerun with another stride must not leave stale frames
        old.unlink()

    images, annotations, listing = [], [], []
    st = {"split": split, "stride": stride, "datasets": list(datasets), "videos": 0, "images": 0,
          "boxes": 0, "dont_care_omitted": 0, "degenerate_dropped": 0,
          "frames_only_dont_care": 0, "frames_no_boxes": 0, "frames_missing": 0}
    for ds in datasets:
        check_dataset(ds)
        for video in read_split(root / "splits" / f"{ds}_{split}.txt"):
            gt = load_gt(root / "data" / "processed" / ds / f"{video}.json")
            w, h = int(gt["width"]), int(gt["height"])
            st["videos"] += 1
            missing = 0
            for fid in frame_ids(gt["num_frames"], stride):
                rel = f"{ds}/{video}/{fid:06d}.jpg"
                if not (root / "data" / "frames" / rel).is_file():
                    missing += 1
                    continue
                objs = gt["frames"].get(str(fid), [])
                polys = []
                for o in objs:
                    if o["dont_care"]:
                        st["dont_care_omitted"] += 1
                        continue
                    p = clean_quad(o["quad"], w, h, min_area)
                    if p is None:
                        st["degenerate_dropped"] += 1
                    else:
                        polys.append(p)
                if not polys:
                    st["frames_only_dont_care" if objs else "frames_no_boxes"] += 1
                    continue
                image_id = len(images) + 1
                images.append({"file_name": rel, "id": image_id, "width": w, "height": h})
                annotations.extend(coco_annotation(p, image_id) for p in polys)
                listing.append(rel)
                (label_dir / f"{ds}__{video}__{fid:06d}.txt").write_text(
                    "\n".join(dota_line(p) for p in polys) + "\n", encoding="utf-8")
            st["frames_missing"] += missing
            if missing and log:
                log.warning("%s/%s: %d strided frames missing on disk (not extracted?)", ds, video,
                            missing)

    for i, a in enumerate(annotations):
        a["id"] = i + 1
    coco = {"categories": [{"id": 1, "name": CATEGORY, "supercategory": CATEGORY}],
            "images": images, "annotations": annotations}
    save_json(coco, out / f"{split}.json")
    save_json(smoke_subset(coco, smoke_n), out / f"{split}_smoke.json")
    (out / split / "images.txt").write_text("".join(f"{r}\n" for r in listing), encoding="utf-8")
    st["images"], st["boxes"] = len(images), len(annotations)
    save_json(st, out / f"{split}_stats.json")
    if log:
        log.info("%s: %s", split, st)
    return st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--splits", nargs="+", default=["train"], choices=("train", "val", "test"))
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/data/det_dota")
    ap.add_argument("--smoke_n", type=int, default=200)
    ap.add_argument("--min_area", type=float, default=4.0, help="px^2 after clipping")
    ap.add_argument("--root", type=Path, default=None, help="project root (default: auto)")
    ap.add_argument("--final", action="store_true", help="allow the test split (rule 5)")
    args = ap.parse_args(argv)
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    root = (args.root or project_root()).resolve()
    out = args.out if args.out is not None else root / "data" / "det_dota"
    log = get_logger("gt_to_dota", out / "gt_to_dota_log.txt")
    for split in args.splits:
        convert_split(root, args.datasets, split, args.stride, out, args.smoke_n, args.min_area,
                      args.final, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
