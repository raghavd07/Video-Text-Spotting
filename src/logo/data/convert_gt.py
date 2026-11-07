"""Raw ICDAR13/15-video and DSText XML -> contract GT json.

Raw layout: data/raw/<ds>/{train,test}/[**/]<video>.mp4 with <video>_GT.xml beside it
(filename case may differ, e.g. ICDAR2013 'video_51_7_4_GT.xml').
"""
from __future__ import annotations

from pathlib import Path

import cv2

from logo.data.parsers import DONT_CARE_TEXTS, PARSERS, is_dont_care, parse_frames_xml  # noqa: F401
from logo.utils.paths import check_dataset, raw_dir

RAW_SPLITS = ("train", "test")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def parse_xml(xml_path) -> dict[int, list[dict]]:
    """Return {frame_id: [object, ...]} with contract object keys (see logo.data.parsers)."""
    return parse_frames_xml(xml_path)


def probe_video(path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {path}")
    try:
        return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()


def build_gt(video: str, ds: str, frames: dict[int, list[dict]], width: int, height: int,
             num_frames: int) -> dict:
    """Every frame 1..N gets a key (empty list if unannotated)."""
    n = max([int(num_frames)] + list(frames))
    return {
        "video": video,
        "dataset": check_dataset(ds),
        "width": int(width),
        "height": int(height),
        "num_frames": n,
        "frames": {str(i): frames.get(i, []) for i in range(1, n + 1)},
    }


def find_raw_videos(raw_root, raw_split: str) -> list[tuple[str, Path, Path | None]]:
    """[(video_id, video_path, xml_path_or_None)] sorted by video id."""
    folder = Path(raw_root) / raw_split
    if not folder.exists():
        return []
    xmls = {p.name.lower(): p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".xml"}
    out = []
    for v in folder.rglob("*"):
        if v.is_file() and v.suffix.lower() in VIDEO_EXTS:
            out.append((v.stem, v, xmls.get(f"{v.stem.lower()}_gt.xml")))
    ids = [o[0] for o in out]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate video ids under {folder}: {sorted(dup)}")
    return sorted(out, key=lambda o: o[0])


def find_video(ds: str, video: str) -> Path:
    for split in RAW_SPLITS:
        for vid, path, _ in find_raw_videos(raw_dir(ds), split):
            if vid == video:
                return path
    raise FileNotFoundError(f"video {video} not found under {raw_dir(ds)}")


def convert_video(ds: str, video: str, video_path, xml_path) -> dict:
    w, h, n = probe_video(video_path)
    return build_gt(video, ds, PARSERS[check_dataset(ds)](xml_path), w, h, n)
