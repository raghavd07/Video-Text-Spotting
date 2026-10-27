import cv2
import numpy as np

from logo.data.frames import extract_frames


def _video(path, n=5):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 24))
    for i in range(n):
        w.write(np.full((24, 32, 3), i * 40, np.uint8))
    w.release()


def test_extract_and_resume(tmp_path):
    v = tmp_path / "v.avi"
    _video(v)
    out = tmp_path / "frames" / "v"
    assert extract_frames(v, out) == 5
    names = sorted(p.name for p in out.glob("*.jpg"))
    assert names == [f"{i:06d}.jpg" for i in range(1, 6)]
    m1 = (out / "000001.jpg").stat().st_mtime_ns
    assert extract_frames(v, out) == 5
    assert (out / "000001.jpg").stat().st_mtime_ns == m1
    img = cv2.imread(str(out / "000003.jpg"))
    assert img.shape == (24, 32, 3) and abs(int(img.mean()) - 80) < 5
