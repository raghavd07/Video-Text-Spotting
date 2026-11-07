"""Per-dataset raw annotation parsers: <video>_GT.xml -> {frame_id: [contract GT object]}.

All three datasets share one layout (checked on data/raw 2026-10-26):
  <Frames><frame ID="k"><object ID=.. Transcription=.. [attrs]><Point x= y=/> x4 </object></frame></Frames>
  icdar2013v  us-ascii, attrs Language/Mirrored/Quality; dont-care '#Don't Care#', '#unreadable#';
              non-ASCII letters were lost as '?' in the raw files (kept as is); points in zigzag order
  icdar2015v  UTF-8, same attrs as 13v; dont-care '##DONT#CARE##'; one 'MODERTE' quality typo
  dstext      attrs category/language, no Quality (-> null); coords may be slightly negative;
              3 videos repeat objects verbatim inside a frame
Duplicate IDs within a frame: exact repeats are dropped; a different box reusing an ID gets a fresh
ID (max ID of the video + 1, + 2, ... in file order), so track_id stays unique per frame and stable.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from logo.utils.geometry import order_quad

DONT_CARE_TEXTS = frozenset({"##DONT#CARE##", "#DON'T CARE#", "###", "#UNREADABLE#"})
QUALITY_FIX = {"MODERTE": "MODERATE"}
STAT_KEYS = ("dup_dropped", "dup_reid", "quality_fixed")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
GT_SUFFIX = "_gt.xml"

log = logging.getLogger(__name__)


def is_dont_care(text: str) -> bool:
    t = text.strip().upper()
    return t == "" or t in DONT_CARE_TEXTS


def _bump(stats: dict | None, key: str, n: int = 1) -> None:
    if stats is not None:
        stats[key] = stats.get(key, 0) + n


def parse_frames_xml(xml_path, stats: dict | None = None) -> dict[int, list[dict]]:
    """Shared parser. `stats` (optional) is updated with the STAT_KEYS counters."""
    root = ET.parse(xml_path).getroot()
    frames: dict[int, list[dict]] = {}
    conflicts: list[dict] = []
    max_id = 0
    for fr in root.iter("frame"):
        fid = int(fr.get("ID"))
        if fid < 1:
            raise ValueError(f"{xml_path}: frame id {fid} is not 1-based")
        objs = frames.setdefault(fid, [])
        seen: dict[int, dict] = {o["track_id"]: o for o in objs}
        for ob in fr.findall("object"):
            pts = ob.findall("Point")
            if len(pts) != 4:
                raise ValueError(f"{xml_path}: frame {fid} object {ob.get('ID')} has {len(pts)} points")
            text = ob.get("Transcription", "")
            quality = ob.get("Quality")
            if quality:
                quality = quality.strip().upper()
                if quality in QUALITY_FIX:
                    quality = QUALITY_FIX[quality]
                    _bump(stats, "quality_fixed")
            o = {
                "track_id": int(ob.get("ID")),
                "quad": order_quad([float(v) for p in pts for v in (p.get("x"), p.get("y"))]),
                "text": text,
                "dont_care": is_dont_care(text),
                "quality": quality or None,
            }
            max_id = max(max_id, o["track_id"])
            prev = seen.get(o["track_id"])
            if prev is not None:
                if prev == o:
                    _bump(stats, "dup_dropped")
                    continue
                conflicts.append(o)
            else:
                seen[o["track_id"]] = o
            objs.append(o)
    for o in conflicts:
        max_id += 1
        log.warning("%s: duplicate ID %d in one frame with a different box -> new ID %d",
                    Path(xml_path).name, o["track_id"], max_id)
        o["track_id"] = max_id
        _bump(stats, "dup_reid")
    return frames


def parse_icdar2013v(xml_path, stats: dict | None = None) -> dict[int, list[dict]]:
    return parse_frames_xml(xml_path, stats)


def parse_icdar2015v(xml_path, stats: dict | None = None) -> dict[int, list[dict]]:
    return parse_frames_xml(xml_path, stats)


def parse_dstext(xml_path, stats: dict | None = None) -> dict[int, list[dict]]:
    return parse_frames_xml(xml_path, stats)


PARSERS = {"icdar2013v": parse_icdar2013v, "icdar2015v": parse_icdar2015v, "dstext": parse_dstext}


def find_annotations(raw_root, raw_split: str) -> list[tuple[str, Path, Path | None]]:
    """[(video_id, xml_path, video_path_or_None)] under raw_root/raw_split, sorted by video id.

    Matching is case-insensitive; the video id takes the video file's spelling when it exists.
    """
    folder = Path(raw_root) / raw_split
    if not folder.exists():
        return []
    videos = {p.stem.lower(): p for p in folder.rglob("*")
              if p.is_file() and p.suffix.lower() in VIDEO_EXTS}
    out = []
    for x in folder.rglob("*"):
        if x.is_file() and x.name.lower().endswith(GT_SUFFIX):
            stem = x.name[: -len(GT_SUFFIX)]
            v = videos.get(stem.lower())
            out.append((v.stem if v else stem, x, v))
    ids = [o[0].lower() for o in out]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate video ids under {folder}: {sorted(dup)}")
    return sorted(out, key=lambda o: o[0])
