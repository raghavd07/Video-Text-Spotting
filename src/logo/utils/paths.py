"""Contract paths. Project root = $LOGO_ROOT if set, else the repository root."""
from __future__ import annotations

import os
from pathlib import Path

DATASETS = ("icdar2013v", "icdar2015v", "dstext")
SPLITS = ("train", "val", "test")


def project_root() -> Path:
    env = os.environ.get("LOGO_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[3]


def check_dataset(ds: str) -> str:
    if ds not in DATASETS:
        raise ValueError(f"unknown dataset {ds!r}; expected one of {DATASETS}")
    return ds


def check_split(split: str, final: bool = False) -> str:
    """Rule 5: the test split is only reachable when a script is called with --final."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if split == "test" and not final:
        raise PermissionError("test split requires --final")
    return split


def raw_dir(ds: str) -> Path:
    return project_root() / "data" / "raw" / check_dataset(ds)


def frames_dir(ds: str, video: str) -> Path:
    return project_root() / "data" / "frames" / check_dataset(ds) / video


def frame_path(ds: str, video: str, frame_id: int) -> Path:
    return frames_dir(ds, video) / f"{int(frame_id):06d}.jpg"


def gt_path(ds: str, video: str) -> Path:
    return project_root() / "data" / "processed" / check_dataset(ds) / f"{video}.json"


def split_file(ds: str, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return project_root() / "splits" / f"{check_dataset(ds)}_{split}.txt"


def stage_dir(exp_id: str, stage: str, ds: str) -> Path:
    return project_root() / "runs" / exp_id / stage / check_dataset(ds)
