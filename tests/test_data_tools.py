import csv
import importlib.util
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from logo.data.frames import MARKER, extract_frames
from logo.data.schema import write_split
from logo.utils.io import save_json
from logo.utils.paths import split_file

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_tool", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _video(path, n=7):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 24))
    for i in range(n):
        w.write(np.full((24, 32, 3), i * 30, np.uint8))
    w.release()


def _gt(video="v", n=3, ds="icdar2015v"):
    frames = {str(i): [] for i in range(1, n + 1)}
    frames["1"] = [{"track_id": 1, "quad": [2, 2, 30, 2, 30, 12, 2, 12], "text": "AB",
                    "dont_care": False, "quality": "HIGH"},
                   {"track_id": 2, "quad": [0, 0, 60, 0, 60, 40, 0, 40], "text": "###",
                    "dont_care": True, "quality": None}]
    frames["2"] = [{"track_id": 1, "quad": [2, 2, 52, 2, 52, 32, 2, 32], "text": "AB",
                    "dont_care": False, "quality": "HIGH"}]
    return {"video": video, "dataset": ds, "width": 32, "height": 24, "num_frames": n,
            "frames": frames}


def test_extract_frames_stride_and_marker(tmp_path):
    v = tmp_path / "v.avi"
    _video(v)
    out = tmp_path / "f"
    assert extract_frames(v, out, quality=90, stride=3) == 3
    assert sorted(p.name for p in out.glob("*.jpg")) == ["000001.jpg", "000004.jpg", "000007.jpg"]
    assert abs(int(cv2.imread(str(out / "000004.jpg")).mean()) - 90) < 5
    assert (out / MARKER).is_file()
    m = (out / "000001.jpg").stat().st_mtime_ns
    assert extract_frames(v, out, stride=3) == 3  # skipped
    assert (out / "000001.jpg").stat().st_mtime_ns == m
    assert extract_frames(v, out, stride=1) == 7  # different stride -> re-extract
    assert len(list(out.glob("*.jpg"))) == 7
    with pytest.raises(ValueError):
        extract_frames(v, out, stride=0)


def test_extract_tool_respects_test_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    ef = _load("extract_frames")
    write_split(split_file("dstext", "train"), ["a", "b"])
    write_split(split_file("dstext", "val"), ["c"])
    write_split(split_file("dstext", "test"), ["t"])
    assert ef.select_videos("dstext", ["train", "val"], False, None) == ["a", "b", "c"]
    assert ef.select_videos("dstext", ["train"], False, ["b"]) == ["b"]
    with pytest.raises(PermissionError):
        ef.select_videos("dstext", ["test"], False, None)
    assert ef.select_videos("dstext", ["test"], True, None) == ["t"]
    with pytest.raises(ValueError):
        ef.select_videos("dstext", ["train"], False, ["zz"])


def test_viz_gt_writes_mp4(tmp_path):
    vz = _load("viz_gt")
    gt = _gt(n=4)
    frames = ((i, np.zeros((24, 32, 3), np.uint8)) for i in range(1, 5))
    out = tmp_path / "o" / "v.mp4"
    assert vz.render(gt, frames, out, 5.0) == 4
    cap = cv2.VideoCapture(str(out))
    ok, img = cap.read()
    cap.release()
    assert ok and img.shape == (24, 32, 3) and img.max() > 0  # boxes/labels drawn
    assert vz.track_color(3) == vz.track_color(3) != vz.track_color(4)
    img = vz.draw_objects(np.zeros((50, 80, 3), np.uint8),
                          [{"track_id": 1, "quad": [5, 20, 60, 20, 60, 40, 5, 40],
                            "text": "été", "dont_care": False}])
    assert img.sum() > 0


def test_viz_gt_main_refuses_test_video(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    vz = _load("viz_gt")
    write_split(split_file("icdar2015v", "test"), ["v"])
    assert vz.main(["--ds", "icdar2015v", "--video", "v", "--out", str(tmp_path / "x.mp4")]) == 1


def test_dataset_stats_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    st = _load("dataset_stats")
    assert st.quad_height([2, 2, 30, 2, 30, 12, 2, 12]) == pytest.approx(10.0)
    assert st.quad_height([0, 10, 10, 0, 40, 30, 30, 40]) == pytest.approx(np.hypot(10, 10), 1e-4)
    for v in ("a", "b"):
        save_json(_gt(v), tmp_path / "data" / "processed" / "icdar2015v" / f"{v}.json")
    save_json(_gt("t"), tmp_path / "data" / "processed" / "icdar2015v" / "t.json")
    write_split(split_file("icdar2015v", "train"), ["a"])
    write_split(split_file("icdar2015v", "val"), ["b", "missing"])
    write_split(split_file("icdar2015v", "test"), ["t"])
    rows = st.collect(["icdar2015v"], False, logging.getLogger("t"))
    assert [r["split"] for r in rows] == ["train", "val", "all"]
    tot = rows[-1]
    assert (tot["videos"], tot["frames"], tot["instances"], tot["tracks"]) == (2, 6, 6, 4)
    assert tot["dont_care_pct"] == pytest.approx(100 / 3, abs=0.01)
    assert tot["small_pct"] == 50.0 and tot["_hist"].sum() == 4
    assert len(st.collect(["icdar2015v"], True, logging.getLogger("t"))) == 4

    report = tmp_path / "docs" / "data_report.md"
    report.parent.mkdir(parents=True)
    report.write_text("# Data report\n\n## Inventory\nkeep me\n", encoding="utf-8")
    assert st.main(["--ds", "icdar2015v", "--figs", str(tmp_path / "docs" / "figs"),
                    "--report", str(report)]) == 0
    assert st.main(["--ds", "icdar2015v", "--figs", str(tmp_path / "docs" / "figs"),
                    "--report", str(report)]) == 0
    text = report.read_text(encoding="utf-8")
    assert "keep me" in text and text.count(st.MARK_START) == 1 and "figs/text_height_hist.png" in text
    figs = tmp_path / "docs" / "figs"
    for f in ("dataset_stats.csv", "text_height_hist.csv", "text_height_hist.png",
              "small_text_share.png"):
        assert (figs / f).stat().st_size > 0
    with open(figs / "dataset_stats.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 3
