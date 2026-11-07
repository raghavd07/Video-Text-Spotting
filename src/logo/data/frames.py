"""Video -> 1-based JPEG frames <out_dir>/%06d.jpg (resumable)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import cv2

PathLike = Union[str, Path]

MARKER = ".extracted.json"


def _read_marker(out_dir: Path) -> dict | None:
    try:
        return json.loads((out_dir / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _imwrite(p: Path, img, quality: int) -> None:
    # imencode + write_bytes instead of cv2.imwrite: cv2 cannot write non-ASCII paths on Windows.
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise OSError(f"cannot encode frame {p.name}")
    p.write_bytes(buf.tobytes())


def extract_frames(video: PathLike, out_dir: PathLike, quality: int = 90, overwrite: bool = False,
                   stride: int = 1) -> int:
    """Decodes `video` to out_dir/%06d.jpg and returns the number of frames written.

    Frame names are the original 1-based frame index; with stride s only frames 1, 1+s, 1+2s, ... are kept.
    A marker file records the stride and count: a later call with the same stride (and no overwrite) is a
    no-op that returns the recorded count. A different stride or overwrite clears old frames first.
    """
    if int(stride) < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    video, out_dir = Path(video), Path(out_dir)
    m = _read_marker(out_dir)
    if not overwrite and m and m.get("stride") == int(stride) and m.get("frames", 0) > 0:
        if len(list(out_dir.glob("[0-9]" * 6 + ".jpg"))) == m["frames"]:
            return int(m["frames"])
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MARKER).unlink(missing_ok=True)
    for p in out_dir.glob("[0-9]" * 6 + ".jpg"):
        p.unlink()

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"cannot open video {video}")
    n = idx = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            idx += 1
            if (idx - 1) % stride == 0:
                _imwrite(out_dir / f"{idx:06d}.jpg", img, quality)
                n += 1
    finally:
        cap.release()
    (out_dir / MARKER).write_text(json.dumps({"video": video.name, "stride": int(stride), "frames": n,
                                              "decoded": idx, "quality": int(quality)}), encoding="utf-8")
    return n
