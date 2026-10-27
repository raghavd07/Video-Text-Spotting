"""data/processed GT -> Ultralytics OBB dataset at <detector.obb_data.root>/<ds>/ (train/val splits only).

Layout (Ultralytics derives label paths by replacing the last /images/ with /labels/):
  images/{train,val}/<video>_<frame:06d>.jpg   hard link to data/frames (copy if linking fails), or a
                                               re-encoded copy with dont_care quads filled grey (114)
                                               when obb_data.mask_dont_care is true
  labels/{train,val}/<video>_<frame:06d>.txt   "0 x1 y1 x2 y2 x3 y3 x4 y4" per non-dont_care object,
                                               corners normalized by width/height and clipped to [0, 1]
  <ds>.yaml                                    path (absolute), train, val, names {0: text}
Frames 1, 1+s, 1+2s, ... with s = obb_data.stride (train) / obb_data.val_stride (val); frames without
objects get an empty label file (background). Stale Ultralytics *.cache files are removed.

Usage: python -m logo.detector.prepare_obb_data --config configs/det_yolo11s_obb.yaml
           [--splits train val] [--overwrite]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from logo.detector.infer_paddle import project_root, read_videos, resolve, write_run_files
from logo.utils.io import get_logger, load_json, load_yaml, save_json, set_seed

MASK_VALUE = (114, 114, 114)
MIN_AREA_PX = 4.0  # clipped quads smaller than this (pixels^2) are dropped


def _area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def quad_to_label(quad, width: int, height: int) -> Optional[list[float]]:
    """Pixel quad -> 8 normalized coords clipped to [0, 1], or None if the clipped quad is degenerate."""
    pts = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    pts = np.clip(pts / np.array([width, height], np.float64), 0.0, 1.0)
    if _area(pts * np.array([width, height])) < MIN_AREA_PX:
        return None
    return [float(v) for v in pts.reshape(-1)]


def frame_label_lines(objs: list[dict], width: int, height: int) -> list[str]:
    lines = []
    for o in objs:
        if o["dont_care"]:
            continue
        lab = quad_to_label(o["quad"], width, height)
        if lab is not None:
            lines.append("0 " + " ".join(f"{v:.6f}" for v in lab))
    return lines


def place_image(src: Path, dst: Path, mask_quads: list) -> str:
    """Writes dst; returns 'masked', 'link' or 'copy'."""
    if mask_quads:
        import cv2

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot read {src}")
        polys = [np.round(np.asarray(q, np.float64).reshape(4, 2)).astype(np.int32) for q in mask_quads]
        cv2.fillPoly(img, polys, MASK_VALUE)
        if not cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"cannot write {dst}")
        return "masked"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def prepare_split(ds: str, split: str, videos: list[str], root: Path, out_root: Path, stride: int,
                  mask_dont_care: bool, overwrite: bool, log) -> dict:
    img_dir, lab_dir = out_root / "images" / split, out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    stats = {"videos": 0, "images": 0, "boxes": 0, "dont_care": 0, "empty": 0, "missing_frames": 0,
             "skipped_existing": 0, "link": 0, "copy": 0, "masked": 0}
    for video in videos:
        gt = load_json(root / "data" / "processed" / ds / f"{video}.json")
        w, h, n = int(gt["width"]), int(gt["height"]), int(gt["num_frames"])
        frame_dir = root / "data" / "frames" / ds / video
        for fid in range(1, n + 1, stride):
            src = frame_dir / f"{fid:06d}.jpg"
            if not src.is_file():
                stats["missing_frames"] += 1
                continue
            name = f"{video}_{fid:06d}"
            dst, lab = img_dir / f"{name}.jpg", lab_dir / f"{name}.txt"
            objs = gt["frames"].get(str(fid), [])
            lines = frame_label_lines(objs, w, h)
            dc = [o["quad"] for o in objs if o["dont_care"]]
            if dst.exists() and lab.exists() and not overwrite:
                stats["skipped_existing"] += 1
            else:
                if dst.exists():
                    dst.unlink()
                stats[place_image(src, dst, dc if mask_dont_care else [])] += 1
                lab.write_text("".join(f"{ln}\n" for ln in lines), encoding="utf-8")
            stats["images"] += 1
            stats["boxes"] += len(lines)
            stats["dont_care"] += len(dc)
            stats["empty"] += not lines
        stats["videos"] += 1
        log.info("%s/%s: %d frames (stride %d)", split, video, len(range(1, n + 1, stride)), stride)
    for c in (out_root / "labels").glob("*.cache"):
        c.unlink()
    return stats


def write_dataset_yaml(out_root: Path, ds: str, splits) -> Path:
    data = {"path": out_root.resolve().as_posix(), "names": {0: "text"}}
    data.update({s: f"images/{s}" for s in splits})
    p = out_root / f"{ds}.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return p


def prepare(cfg: dict, root: Path, splits=("train", "val"), overwrite: bool = False, log=None) -> dict:
    ds = cfg["dataset"]
    od = cfg["detector"].get("obb_data", {})
    out_root = resolve(od.get("root", "data/obb"), root) / ds
    strides = {"train": int(od.get("stride", 5)), "val": int(od.get("val_stride", od.get("stride", 5)))}
    mask = bool(od.get("mask_dont_care", True))
    log = log or get_logger("prepare_obb_data", root / "runs" / cfg["exp_id"] / "log.txt")
    result = {"out_root": out_root.as_posix(), "mask_dont_care": mask, "splits": {}}
    for split in splits:
        if split not in strides:
            raise ValueError(f"split {split!r} not allowed; OBB training data uses train/val only")
        videos = read_videos(ds, split, root)
        st = prepare_split(ds, split, videos, root, out_root, strides[split], mask, overwrite, log)
        result["splits"][split] = {"stride": strides[split], **st}
        log.info("%s: %s", split, st)
    result["yaml"] = write_dataset_yaml(out_root, ds, splits).as_posix()
    save_json(result, out_root / "stats.json")
    write_run_files(cfg, root / "runs" / cfg["exp_id"], result, key="obb_data")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=("train", "val"))
    ap.add_argument("--overwrite", action="store_true", help="rewrite existing images/labels")
    args = ap.parse_args(argv)
    root = project_root()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    res = prepare(cfg, root, args.splits, args.overwrite)
    print(f"dataset yaml: {res['yaml']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
