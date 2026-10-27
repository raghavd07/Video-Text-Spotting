"""Detection / language score fusion."""
from __future__ import annotations

MODES = ("hard", "soft")


def fuse(det, lang, mode, w=0.5) -> float:
    """mode "hard": (1-w)*det + w*lang_hard (w=0.5 is the paper's average); "soft": task P25."""
    if mode not in MODES:
        raise ValueError(f"unknown fusion mode {mode!r}; expected one of {MODES}")
    w = float(w)
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"fusion weight w must be in [0, 1], got {w}")
    if mode == "soft":
        raise NotImplementedError("soft fusion is implemented in task P25")
    if lang not in (0, 1):
        raise ValueError(f"hard fusion expects lang_hard in {{0, 1}}, got {lang!r}")
    return (1.0 - w) * float(det) + w * float(lang)
