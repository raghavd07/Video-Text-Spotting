"""Seeded, group-stratified train/val splits by video."""
from __future__ import annotations

import random
from collections import defaultdict


def train_val_split(groups: dict[str, str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """{video: group} -> (train, val), both sorted.

    Per group (in sorted group order), round(n * val_frac) videos go to val, half rounded up; a group of
    one video therefore stays in train for val_frac < 0.5. Deterministic for a given seed.
    """
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    by_group: dict[str, list[str]] = defaultdict(list)
    for v, g in groups.items():
        by_group[g].append(v)
    rng = random.Random(seed)
    train: list[str] = []
    val: list[str] = []
    for g in sorted(by_group):
        vids = sorted(by_group[g])
        rng.shuffle(vids)
        k = int(len(vids) * val_frac + 0.5)
        val += vids[:k]
        train += vids[k:]
    return sorted(train), sorted(val)


def check_disjoint(splits: dict[str, list[str]]) -> None:
    """AssertionError if a split repeats a video or two splits share one (case-insensitive)."""
    owner: dict[str, str] = {}
    for name, vids in splits.items():
        keys = [v.lower() for v in vids]
        assert len(keys) == len(set(keys)), f"split {name} contains duplicate videos"
        for k in keys:
            assert k not in owner, f"video {k!r} is in both {owner.get(k)} and {name}"
            owner[k] = name


def make_splits(groups: dict[str, str], test: list[str], val_frac: float, seed: int) -> dict[str, list[str]]:
    """{train, val, test}: train/val from train_val_split(groups), test = sorted official test videos."""
    train, val = train_val_split(groups, val_frac, seed)
    splits = {"train": train, "val": val, "test": sorted(test)}
    check_disjoint(splits)
    return splits
