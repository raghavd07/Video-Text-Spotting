import cv2
import numpy as np
import pytest

from logo.data.convert_gt import convert_video, find_video
from logo.data.schema import load_gt, read_split
from logo.utils.io import save_json
from logo.utils.paths import (check_dataset, check_split, frame_path, gt_path, project_root, raw_dir,
                              stage_dir)

XML = """<Frames><frame ID="2"><object ID="5" Transcription="HI" Quality="high">
<Point x="0" y="0"/><Point x="4" y="0"/><Point x="4" y="2"/><Point x="0" y="2"/></object></frame></Frames>"""


def test_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    r = tmp_path.resolve()
    assert project_root() == r
    assert frame_path("dstext", "V", 7) == r / "data" / "frames" / "dstext" / "V" / "000007.jpg"
    assert gt_path("icdar2015v", "V") == r / "data" / "processed" / "icdar2015v" / "V.json"
    assert stage_dir("e1", "det", "dstext") == r / "runs" / "e1" / "det" / "dstext"
    assert raw_dir("icdar2013v") == r / "data" / "raw" / "icdar2013v"
    with pytest.raises(ValueError):
        check_dataset("bovtext")
    with pytest.raises(ValueError):
        check_split("dev")
    assert check_split("val") == "val"


def test_default_root_is_repo(monkeypatch):
    monkeypatch.delenv("LOGO_ROOT", raising=False)
    assert (project_root() / "src" / "logo" / "utils" / "paths.py").is_file()


def test_read_split_skips_blank(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("a\n\n  b  \n", encoding="utf-8")
    assert read_split(p) == ["a", "b"]


def test_convert_find_load(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    folder = raw_dir("dstext") / "train" / "Game"
    folder.mkdir(parents=True)
    vp = folder / "Video_9.avi"
    w = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 24))
    for _ in range(3):
        w.write(np.zeros((24, 32, 3), np.uint8))
    w.release()
    xp = folder / "Video_9_GT.xml"
    xp.write_text(XML, encoding="utf-8")
    assert find_video("dstext", "Video_9") == vp
    with pytest.raises(FileNotFoundError):
        find_video("dstext", "Video_404")
    gt = convert_video("dstext", "Video_9", vp, xp)
    assert (gt["width"], gt["height"], gt["num_frames"]) == (32, 24, 3)
    assert gt["frames"]["2"][0]["quality"] == "HIGH" and gt["frames"]["1"] == []
    save_json(gt, gt_path("dstext", "Video_9"))
    assert load_gt(gt_path("dstext", "Video_9")) == gt
    save_json({**gt, "dataset": "bad"}, tmp_path / "bad.json")
    with pytest.raises(ValueError):
        load_gt(tmp_path / "bad.json")
