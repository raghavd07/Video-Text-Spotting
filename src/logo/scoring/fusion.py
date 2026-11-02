"""Detection / language score fusion."""
from __future__ import annotations

MODES = ("hard", "soft")
SOFT_SOURCES = ("p_eos", "text_conf")
_EPS = 1e-6


def lang_score(obj: dict, mode: str, soft_source: str = "p_eos") -> float:
    """Language input of fuse() for an lsc object: "hard" -> lang_hard; "soft" -> 1 - p_eos (probability that
    the first decoded token is not the end token) or text_conf (mean character probability)."""
    if mode not in MODES:
        raise ValueError(f"unknown fusion mode {mode!r}; expected one of {MODES}")
    if mode == "hard":
        return obj["lang_hard"]
    if soft_source not in SOFT_SOURCES:
        raise ValueError(f"unknown fusion.soft_source {soft_source!r}; expected one of {SOFT_SOURCES}")
    return 1.0 - float(obj["p_eos"]) if soft_source == "p_eos" else float(obj["text_conf"])


def fuse(det, lang, mode, w=0.5) -> float:
    """(1-w)*det + w*lang. mode "hard": lang = lang_hard in {0, 1} (w=0.5 is the paper's average);
    "soft": lang = lang_soft in [0, 1] (see lang_score)."""
    if mode not in MODES:
        raise ValueError(f"unknown fusion mode {mode!r}; expected one of {MODES}")
    w = float(w)
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"fusion weight w must be in [0, 1], got {w}")
    if mode == "soft":
        lang = float(lang)
        if not -_EPS <= lang <= 1.0 + _EPS:  # float16 round-off from the LSC is tolerated, then clipped
            raise ValueError(f"soft fusion expects lang_soft in [0, 1], got {lang!r}")
        return (1.0 - w) * float(det) + w * min(max(lang, 0.0), 1.0)
    if lang not in (0, 1):
        raise ValueError(f"hard fusion expects lang_hard in {{0, 1}}, got {lang!r}")
    return (1.0 - w) * float(det) + w * float(lang)
