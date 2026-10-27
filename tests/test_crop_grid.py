import importlib.util
from pathlib import Path

import cv2
import numpy as np

from logo.data.schema import write_split
from logo.utils.io import save_json
from logo.utils.paths import frame_path, gt_path, split_file, stage_dir

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_tool", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup(root: Path, video="v1", n_frames=5):
    q = [20, 10, 120, 10, 120, 40, 20, 40]
    gt = {"video": video, "dataset": "icdar2015v", "width": 160, "height": 90, "num_frames": n_frames,
          "frames": {}}
    det = {"video": video, "dataset": "icdar2015v", "frames": {}}
    for fid in range(1, n_frames + 1):
        img = np.full((90, 160, 3), 200, np.uint8)
        cv2.putText(img, "AB", (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        p = frame_path("icdar2015v", video, fid)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img)
        gt["frames"][str(fid)] = [
            {"track_id": 1, "quad": q, "text": "AB", "dont_care": False, "quality": "HIGH"},
            {"track_id": 2, "quad": [0, 60, 10, 60, 10, 70, 0, 70], "text": "###", "dont_care": True,
             "quality": None}]
        det["frames"][str(fid)] = [{"quad": q, "score": 0.9}, {"quad": q, "score": 0.1}]
    save_json(gt, gt_path("icdar2015v", video))
    save_json(det, stage_dir("e0", "det", "icdar2015v") / f"{video}.json")


def test_crop_grid_gt_and_det(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    _setup(tmp_path)
    tool = _load("crop_grid")
    out = tmp_path / "g.png"
    assert tool.main(["--ds", "icdar2015v", "--video", "v1", "--out", str(out), "--n", "3", "--cols", "2"]) == 0
    g = cv2.imread(str(out))
    cell_h, cell_w = 32 + tool.CAPTION_H + tool.PAD, 128 + tool.PAD
    assert g.shape == (2 * cell_h + tool.PAD, 2 * cell_w + tool.PAD, 3)  # 3 crops, dont_care skipped

    assert tool.main(["--ds", "icdar2015v", "--video", "v1", "--out", str(tmp_path), "--exp-id", "e0",
                      "--min-score", "0.5"]) == 0
    g = cv2.imread(str(tmp_path / "icdar2015v_v1_pred_crops.png"))
    assert g.shape[0] == cell_h + tool.PAD and g.shape[1] == 8 * cell_w + tool.PAD  # 5 crops in one row


def test_crop_grid_refuses_test_video(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    _setup(tmp_path)
    write_split(split_file("icdar2015v", "test"), ["v1"])
    tool = _load("crop_grid")
    assert tool.main(["--ds", "icdar2015v", "--video", "v1", "--out", str(tmp_path / "x.png")]) == 1
    assert tool.main(["--ds", "icdar2015v", "--video", "v1", "--out", str(tmp_path / "x.png"), "--final"]) == 0


def test_sample_is_even():
    tool = _load("crop_grid")
    assert tool.sample(list(range(10)), 3) == [0, 4, 9]
    assert tool.sample(list(range(100)), 64) == sorted(set(tool.sample(list(range(100)), 64)))
    assert tool.sample([1, 2], 64) == [1, 2]
