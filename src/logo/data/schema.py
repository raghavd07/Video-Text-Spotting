"""CONTRACT GT schema (data/processed/<ds>/<video>.json) and split-file helpers.

    {"video": str, "dataset": one of DATASETS, "width": int, "height": int, "num_frames": int,
     "frames": {"1": [obj, ...], ..., str(num_frames): [...]}}
    obj = {"track_id": int, "quad": [8 floats, clockwise], "text": str, "dont_care": bool,
           "quality": "LOW" | "MODERATE" | "HIGH" | other upper-case str | None}

Frame keys are 1-based ids <= num_frames. Converted GT (build_gt) lists every frame, empty ones included;
readers must also accept sparse frames (a missing key means no annotations). Track ids are unique within
a frame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from logo.utils.io import load_json
from logo.utils.paths import DATASETS, split_file

PathLike = Union[str, Path]

TOP_KEYS = ("video", "dataset", "width", "height", "num_frames", "frames")
OBJ_KEYS = ("track_id", "quad", "text", "dont_care", "quality")


def _fail(gt: dict, msg: str) -> None:
    raise ValueError(f"invalid GT ({gt.get('dataset')}/{gt.get('video')}): {msg}")


def validate_gt(gt: dict) -> dict:
    """Raises ValueError if gt breaks the schema; returns gt unchanged."""
    if not isinstance(gt, dict):
        raise ValueError(f"GT must be a dict, got {type(gt).__name__}")
    missing = [k for k in TOP_KEYS if k not in gt]
    if missing:
        _fail(gt, f"missing keys {missing}")
    if gt["dataset"] not in DATASETS:
        _fail(gt, f"unknown dataset {gt['dataset']!r}")
    for k in ("width", "height", "num_frames"):
        if not isinstance(gt[k], int) or isinstance(gt[k], bool) or gt[k] < 0:
            _fail(gt, f"{k} must be a non-negative int, got {gt[k]!r}")
    frames = gt["frames"]
    if not isinstance(frames, dict):
        _fail(gt, "frames must be a dict")
    for fid in frames:
        if not (isinstance(fid, str) and fid.isdigit() and 1 <= int(fid) <= gt["num_frames"]):
            _fail(gt, f"frame key {fid!r} is not a 1-based frame id <= num_frames ({gt['num_frames']})")
    for fid, objs in frames.items():
        if not isinstance(objs, list):
            _fail(gt, f"frame {fid} must hold a list")
        seen = set()
        for o in objs:
            miss = [k for k in OBJ_KEYS if k not in o]
            if miss:
                _fail(gt, f"frame {fid}: object missing {miss}")
            if not isinstance(o["track_id"], int) or isinstance(o["track_id"], bool):
                _fail(gt, f"frame {fid}: track_id must be int")
            if o["track_id"] in seen:
                _fail(gt, f"frame {fid}: duplicate track_id {o['track_id']}")
            seen.add(o["track_id"])
            q = o["quad"]
            if not isinstance(q, list) or len(q) != 8 or not all(isinstance(v, (int, float)) for v in q):
                _fail(gt, f"frame {fid}: quad must be 8 numbers")
            if not isinstance(o["text"], str):
                _fail(gt, f"frame {fid}: text must be str")
            if not isinstance(o["dont_care"], bool):
                _fail(gt, f"frame {fid}: dont_care must be bool")
            if o["quality"] is not None and not isinstance(o["quality"], str):
                _fail(gt, f"frame {fid}: quality must be str or null")
    return gt


def load_gt(p: PathLike) -> dict:
    return validate_gt(load_json(p))


def read_split(p: PathLike) -> list[str]:
    """Video ids, one per line; blank lines and surrounding whitespace ignored."""
    return [ln.strip() for ln in Path(p).read_text(encoding="utf-8").splitlines() if ln.strip()]


def write_split(p: PathLike, videos: list[str]) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f"{v}\n" for v in videos), encoding="utf-8")


def list_videos(ds: str, split: str) -> list[str]:
    """Videos of splits/<ds>_<split>.txt. ValueError for an unknown dataset, FileNotFoundError if missing."""
    p = split_file(ds, split)
    if not p.is_file():
        raise FileNotFoundError(f"split file not found: {p} (run scripts/make_splits.py --ds {ds})")
    return read_split(p)
