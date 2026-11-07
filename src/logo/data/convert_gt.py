"""Raw video/XML discovery and generic ICDAR-style XML -> CONTRACT GT conversion (see logo.data.schema).

Raw layout: data/raw/<ds>/<train|test>/[<group>/...]<Video>.<ext> with <Video>_GT.xml next to it
(the XML name may differ in case, e.g. video_1_1_1_GT.xml).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

import cv2

from logo.utils.geometry import order_quad
from logo.utils.paths import check_dataset, raw_dir

PathLike = Union[str, Path]

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv")
GT_SUFFIX = "_gt.xml"


def is_dont_care(text: Optional[str]) -> bool:
    """Empty/whitespace text, or a #...# marker (###, ##DONT#CARE##, #Don't care#, #unreadable#)."""
    t = (text or "").strip()
    return not t or (len(t) >= 2 and t.startswith("#") and t.endswith("#")) or t == "#"


def norm_quality(q: Optional[str]) -> Optional[str]:
    q = (q or "").strip().upper()
    return q or None


def xml_points(obj: ET.Element) -> list[float]:
    pts = [(float(p.get("x")), float(p.get("y"))) for p in obj.iter("Point")]
    if len(pts) != 4:
        raise ValueError(f"object {obj.get('ID')}: expected 4 points, got {len(pts)}")
    return order_quad([v for xy in pts for v in xy])


def iter_xml(xml_path: PathLike):
    """(frame id, [object elements]) per <frame>, in file order."""
    root = ET.parse(str(xml_path)).getroot()
    for fr in root.iter("frame"):
        yield int(fr.get("ID")), list(fr.iter("object"))


def make_obj(el: ET.Element, track_id: int, quality: Optional[str]) -> dict:
    text = el.get("Transcription") or ""
    return {"track_id": int(track_id), "quad": xml_points(el), "text": text, "dont_care": is_dont_care(text),
            "quality": quality}


def parse_xml(xml_path: PathLike) -> dict[int, list[dict]]:
    """Generic parser: {frame id: [obj]} with every <frame> present (possibly empty). Quality upper-cased."""
    frames: dict[int, list[dict]] = {}
    for fid, objs in iter_xml(xml_path):
        frames.setdefault(fid, [])
        for el in objs:
            frames[fid].append(make_obj(el, int(el.get("ID")), norm_quality(el.get("Quality"))))
    return frames


def build_gt(video: str, ds: str, frames: dict[int, list[dict]], width: int, height: int,
             num_frames: int) -> dict:
    """CONTRACT GT dict; num_frames = max(num_frames, last annotated frame id); missing frames are empty."""
    last = max((int(f) for f in frames), default=0)
    n = max(int(num_frames), last)
    return {"video": video, "dataset": check_dataset(ds), "width": int(width), "height": int(height),
            "num_frames": n, "frames": {str(i): list(frames.get(i, frames.get(str(i), [])))  # type: ignore[call-overload]
                                        for i in range(1, n + 1)}}


def probe_video(path: PathLike) -> tuple[int, int, int]:
    """(width, height, frame count from the header). OSError if the video cannot be opened."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise OSError(f"cannot open video {path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return w, h, max(n, 0)


def convert_video(ds: str, video: str, video_path: PathLike, xml_path: PathLike) -> dict:
    w, h, n = probe_video(video_path)
    return build_gt(video, ds, parse_xml(xml_path), w, h, n)


def _gt_xml_for(video_path: Path) -> Optional[Path]:
    want = (video_path.stem + GT_SUFFIX).lower()
    for p in sorted(video_path.parent.iterdir()):
        if p.is_file() and p.name.lower() == want:
            return p
    return None


def find_raw_videos(raw_ds: PathLike, raw_split: str) -> list[tuple[str, Path, Optional[Path]]]:
    """[(video id, video path, GT xml path or None)] under raw_ds/raw_split, sorted by video id."""
    base = Path(raw_ds) / raw_split
    if not base.is_dir():
        return []
    vids = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(((p.stem, p, _gt_xml_for(p)) for p in vids), key=lambda t: (t[0], str(t[1])))


def find_video(ds: str, video: str) -> Path:
    """The raw video file named <video>.<ext> anywhere under data/raw/<ds>/."""
    base = raw_dir(ds)
    if base.is_dir():
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.stem == video and p.suffix.lower() in VIDEO_EXTS:
                return p
    raise FileNotFoundError(f"raw video {video!r} not found under {base}")
