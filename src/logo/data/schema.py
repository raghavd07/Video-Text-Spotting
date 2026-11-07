"""GT json schema (contract) and split lists."""
from __future__ import annotations

from pathlib import Path

from logo.utils.io import load_json
from logo.utils.paths import DATASETS, check_dataset, split_file

GT_KEYS = ("video", "dataset", "width", "height", "num_frames", "frames")
OBJ_KEYS = ("track_id", "quad", "text", "dont_care", "quality")


def validate_gt(gt: dict) -> dict:
    missing = [k for k in GT_KEYS if k not in gt]
    if missing:
        raise ValueError(f"GT missing keys {missing}")
    if gt["dataset"] not in DATASETS:
        raise ValueError(f"bad dataset {gt['dataset']!r}")
    n = int(gt["num_frames"])
    for fid, objs in gt["frames"].items():
        if not 1 <= int(fid) <= n:
            raise ValueError(f"frame id {fid} outside 1..{n}")
        for o in objs:
            if not all(k in o for k in OBJ_KEYS):
                raise ValueError(f"frame {fid}: object missing keys {[k for k in OBJ_KEYS if k not in o]}")
            if len(o["quad"]) != 8:
                raise ValueError(f"frame {fid}: quad must have 8 values")
            if not isinstance(o["dont_care"], bool):
                raise ValueError(f"frame {fid}: dont_care must be bool")
    return gt


def load_gt(path) -> dict:
    return validate_gt(load_json(path))


def read_split(path) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def write_split(path, videos) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{v}\n" for v in videos), encoding="utf-8")


def list_videos(ds: str, split: str) -> list[str]:
    check_dataset(ds)
    p = split_file(ds, split)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run scripts/make_splits.py --ds {ds}")
    return read_split(p)
