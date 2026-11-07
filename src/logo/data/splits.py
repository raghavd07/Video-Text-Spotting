"""Seeded, group-stratified train/val split of the raw train videos."""
from __future__ import annotations

import random
from collections import defaultdict


def train_val_split(videos: dict[str, str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """videos: {video_id: group}. Each group with >= 2 videos contributes round(frac*n) (>= 1) to val."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be in (0, 1)")
    groups = defaultdict(list)
    for v, g in videos.items():
        groups[g].append(v)
    rng = random.Random(int(seed))
    train, val = [], []
    for g in sorted(groups):
        vs = sorted(groups[g])
        rng.shuffle(vs)
        k = max(1, round(val_frac * len(vs))) if len(vs) >= 2 else 0
        val += vs[:k]
        train += vs[k:]
    return sorted(train), sorted(val)


def check_disjoint(splits: dict[str, list[str]]) -> None:
    """Raise AssertionError if a video id (case-insensitive) is repeated within or across splits."""
    seen: dict[str, str] = {}
    for name, vids in splits.items():
        for v in vids:
            k = v.lower()
            if k in seen:
                raise AssertionError(f"video {v!r} appears in {seen[k]!r} and {name!r}")
            seen[k] = name


def make_splits(train_videos: dict[str, str], test_videos, val_frac: float, seed: int
                ) -> dict[str, list[str]]:
    """train_videos: {official train video: group}; test_videos: official test ids.
    Returns {"train", "val", "test"} (sorted lists), checked with check_disjoint."""
    train, val = train_val_split(train_videos, val_frac, seed)
    out = {"train": train, "val": val, "test": sorted(test_videos)}
    check_disjoint(out)
    return out
