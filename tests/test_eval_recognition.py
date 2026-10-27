import csv
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from logo.data.schema import write_split
from logo.utils.io import load_json, save_json
from logo.utils.paths import frame_path, gt_path, project_root, split_file

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load():
    spec = importlib.util.spec_from_file_location("eval_recognition_tool", TOOLS / "eval_recognition.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeRecognizer:
    batch_size = 64
    c, d = 26, 512

    def __init__(self):
        self.calls = []

    def predict(self, crops):
        self.calls.append(len(crops))
        return [{"text": "ab", "text_conf": 0.5, "p_eos": 0.1, "feat": np.zeros((26, 512), np.float16)}
                for _ in crops]


def _setup(video="v1", n_frames=4):
    q = [20, 10, 120, 10, 120, 40, 20, 40]
    objs = [{"track_id": 1, "quad": q, "text": "AB", "dont_care": False, "quality": "HIGH"},
            {"track_id": 2, "quad": q, "text": "a-b!", "dont_care": False, "quality": None},
            {"track_id": 3, "quad": q, "text": "cd", "dont_care": False, "quality": None},
            {"track_id": 4, "quad": q, "text": "###", "dont_care": True, "quality": None},
            {"track_id": 5, "quad": q, "text": "--", "dont_care": False, "quality": None}]
    gt = {"video": video, "dataset": "icdar2015v", "width": 160, "height": 90, "num_frames": n_frames,
          "frames": {str(f): objs for f in range(1, n_frames + 1)}}
    for f in range(1, n_frames + 1):
        p = frame_path("icdar2015v", video, f)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), np.full((90, 160, 3), 200, np.uint8))
    save_json(gt, gt_path("icdar2015v", video))
    write_split(split_file("icdar2015v", "val"), [video])
    write_split(split_file("icdar2015v", "test"), ["v9"])


def test_normalize():
    tool = _load()
    assert tool.normalize("Hello-World! 42") == "helloworld42"
    assert tool.normalize("###") == ""


def test_eval_recognition_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    _setup()
    tool = _load()
    fake = FakeRecognizer()
    monkeypatch.setattr(tool, "make_recognizer", lambda ckpt, device, fp16: fake)
    rc = tool.main(["--ds", "icdar2015v", "--exp-id", "r1", "--device", "cpu", "--batch", "5",
                    "--config", str(tmp_path / "none.yaml")])
    assert rc == 0
    run = project_root() / "runs" / "r1"
    for f in ("config.yaml", "log.txt", "metrics.json"):
        assert (run / f).is_file()
    r = load_json(run / "metrics.json")["meta"]["recog"]["icdar2015v"]["val"]
    assert (r["num_words"], r["num_correct"]) == (12, 8)
    assert r["word_acc"] == pytest.approx(100 * 8 / 12)
    assert r["skipped"] == {"dont_care": 4, "non_alnum": 4, "missing_frames": 0}
    assert max(fake.calls) <= 5 and sum(fake.calls) == 12
    with open(run / "recog" / "icdar2015v" / "val_predictions.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 12 and {r["gt"] for r in rows} == {"AB", "a-b!", "cd"}
    assert sum(int(r["correct"]) for r in rows) == 8


def test_stride_and_test_split_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    _setup()
    tool = _load()
    monkeypatch.setattr(tool, "make_recognizer", lambda ckpt, device, fp16: FakeRecognizer())
    assert tool.main(["--ds", "icdar2015v", "--exp-id", "r2", "--stride", "2",
                      "--config", str(tmp_path / "none.yaml")]) == 0
    r = load_json(project_root() / "runs" / "r2" / "metrics.json")["meta"]["recog"]["icdar2015v"]["val"]
    assert r["num_words"] == 6  # frames 1 and 3
    with pytest.raises(PermissionError):
        tool.main(["--ds", "icdar2015v", "--split", "test", "--config", str(tmp_path / "none.yaml")])
