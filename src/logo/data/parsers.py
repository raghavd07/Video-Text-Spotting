"""Per-dataset GT XML parsers -> {frame id: [obj]} (CONTRACT objects, see logo.data.schema).

Every parser takes (xml_path, stats=None) and is deterministic, so track ids are stable across runs.
`stats` (optional dict) counts fixes; a key appears only when its count is non-zero (see STAT_KEYS):
  quality_fixed  misspelt Quality values corrected (ICDAR 2015: "MODERTE" -> "MODERATE")
  dup_dropped    DSText: verbatim repeat of an object in the same frame, dropped
  dup_reid       DSText: same ID reused in one frame for different text/box; given a new unique id
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional, Union

from logo.data.convert_gt import GT_SUFFIX, VIDEO_EXTS, is_dont_care, iter_xml, make_obj, xml_points

PathLike = Union[str, Path]

STAT_KEYS = ("quality_fixed", "dup_dropped", "dup_reid")
QUALITY_FIX = {"MODERTE": "MODERATE", "MODERTATE": "MODERATE", "MODERAT": "MODERATE", "HIHG": "HIGH"}

__all__ = ["PARSERS", "STAT_KEYS", "find_annotations", "is_dont_care", "parse_icdar2013v", "parse_icdar2015v",
           "parse_dstext"]


def _bump(stats: Optional[dict], key: str) -> None:
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1


def _quality(el: ET.Element, stats: Optional[dict]) -> Optional[str]:
    q = (el.get("Quality") or "").strip().upper()
    if not q:
        return None
    if q in QUALITY_FIX:
        _bump(stats, "quality_fixed")
        q = QUALITY_FIX[q]
    return q


def _parse_icdar(xml_path: PathLike, stats: Optional[dict] = None) -> dict[int, list[dict]]:
    frames: dict[int, list[dict]] = {}
    for fid, objs in iter_xml(xml_path):
        frames.setdefault(fid, [])
        for el in objs:
            frames[fid].append(make_obj(el, int(el.get("ID")), _quality(el, stats)))
    return frames


def parse_icdar2013v(xml_path: PathLike, stats: Optional[dict] = None) -> dict[int, list[dict]]:
    return _parse_icdar(xml_path, stats)


def parse_icdar2015v(xml_path: PathLike, stats: Optional[dict] = None) -> dict[int, list[dict]]:
    return _parse_icdar(xml_path, stats)


def parse_dstext(xml_path: PathLike, stats: Optional[dict] = None) -> dict[int, list[dict]]:
    """DSText XML: no Quality attribute (quality None). Within a frame, an exact repeat (same ID, text and
    points) is dropped; the same ID on a different object gets a new id (max ID in the file + 1, + 2, ...),
    reused for the same (ID, text) pair in later frames so the re-identified track stays one track."""
    parsed = list(iter_xml(xml_path))
    next_id = max((int(el.get("ID")) for _, objs in parsed for el in objs), default=0) + 1
    reid: dict[tuple[int, str], int] = {}
    frames: dict[int, list[dict]] = {}
    for fid, objs in parsed:
        out = frames.setdefault(fid, [])
        seen: dict[int, tuple[str, list[float]]] = {}
        for el in objs:
            tid, text, quad = int(el.get("ID")), el.get("Transcription") or "", xml_points(el)
            if tid in seen:
                if seen[tid] == (text, quad):
                    _bump(stats, "dup_dropped")
                    continue
                key = (tid, text)
                if key not in reid:
                    reid[key] = next_id
                    next_id += 1
                _bump(stats, "dup_reid")
                tid = reid[key]
            elif (tid, text) in reid:
                tid = reid[(tid, text)]
            if any(o["track_id"] == tid for o in out):  # re-id'd pair repeated in this frame
                _bump(stats, "dup_dropped")
                continue
            seen.setdefault(int(el.get("ID")), (text, quad))
            out.append(make_obj(el, tid, _quality(el, stats)))
    return frames


PARSERS: dict[str, Callable[..., dict[int, list[dict]]]] = {
    "icdar2013v": parse_icdar2013v,
    "icdar2015v": parse_icdar2015v,
    "dstext": parse_dstext,
}


def find_annotations(raw_ds: PathLike, raw_split: str) -> list[tuple[str, Path, Optional[Path]]]:
    """[(video id, GT xml path, video path or None)] for every *_GT.xml under raw_ds/raw_split (any case),
    sorted by video id. The id takes the video file's spelling when one sits next to the XML."""
    base = Path(raw_ds) / raw_split
    if not base.is_dir():
        return []
    out = []
    for x in base.rglob("*"):
        if not (x.is_file() and x.name.lower().endswith(GT_SUFFIX)):
            continue
        stem = x.name[: -len(GT_SUFFIX)]
        video = next((p for p in sorted(x.parent.iterdir())
                      if p.is_file() and p.suffix.lower() in VIDEO_EXTS and p.stem.lower() == stem.lower()), None)
        out.append((video.stem if video else stem, x, video))
    return sorted(out, key=lambda t: (t[0], str(t[1])))
