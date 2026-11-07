"""Video -> data/frames/<ds>/<video>/%06d.jpg (1-based)."""
from __future__ import annotations

from pathlib import Path

import cv2

from logo.utils.io import load_json, save_json

MARKER = ".extracted.json"  # {"stride", "quality", "frames"}; written last, so it means "complete"


def extract_frames(video_path, out_dir, quality: int = 95, overwrite: bool = False,
                   stride: int = 1) -> int:
    """Decode the video and write every `stride`-th frame as a JPEG named by its original 1-based
    frame index (1, 1+stride, ...), so file names stay aligned with GT frame ids.

    Skips the video if the folder holds a completion marker with the same stride (or, for
    stride 1 without marker, the expected number of frames). Returns the number of frames on disk.
    """
    stride = int(stride)
    if stride < 1:
        raise ValueError("stride must be >= 1")
    out_dir = Path(out_dir)
    marker = out_dir / MARKER
    if not overwrite and marker.is_file():
        m = load_json(marker)
        if int(m.get("stride", 0)) == stride:
            return int(m["frames"])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {video_path}")
    try:
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        existing = sorted(out_dir.glob("*.jpg")) if out_dir.exists() else []
        if not overwrite and stride == 1 and expected > 0 and len(existing) == expected:
            save_json({"stride": 1, "quality": None, "frames": expected}, marker)
            return expected
        out_dir.mkdir(parents=True, exist_ok=True)
        if marker.exists():
            marker.unlink()
        for p in existing:
            p.unlink()
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        idx = n = 0
        while True:
            idx += 1
            if (idx - 1) % stride:
                if not cap.grab():  # skip without decoding
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break
            ok, buf = cv2.imencode(".jpg", frame, params)
            if not ok:
                raise OSError(f"jpeg encode failed at frame {idx} of {video_path}")
            # write_bytes instead of cv2.imwrite: cv2 cannot write to non-ASCII paths on Windows
            (out_dir / f"{idx:06d}.jpg").write_bytes(buf.tobytes())
            n += 1
    finally:
        cap.release()
    save_json({"stride": stride, "quality": int(quality), "frames": n}, marker)
    return n
