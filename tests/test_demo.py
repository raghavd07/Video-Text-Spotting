import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from logo.utils.io import get_logger, load_json, save_json

ROOT = Path(__file__).resolve().parents[1]
C, D, NF, W, H = 3, 4, 8, 160, 96


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


demo = _load(ROOT / "scripts" / "demo.py", "demo")


class FakeRecognizer:
    c, d, batch_size = C, D, 4

    def predict(self, crops):
        return [{"text": "EXIT", "text_conf": 0.9, "p_eos": 0.1, "feat": np.zeros((C, D), np.float16)}
                for _ in crops]


def fake_predictor(dcfg):
    def predict(paths):
        out = []
        for p in paths:
            x = 10 + 4 * int(Path(p).stem)
            out.append(np.array([[0, 0.9, x, 30, x + 40, 30, x + 40, 50, x, 50]], np.float32))
        return out
    return predict


def make_video(p):
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))
    assert vw.isOpened()
    for i in range(NF):
        img = np.full((H, W, 3), 30, np.uint8)
        x = 14 + 4 * (i + 1)
        img[30:50, x:x + 40] = 255
        vw.write(img)
    vw.release()


def cfg():
    return {"exp_id": "E-src", "dataset": "icdar2015v", "split": "val", "seed": 0,
            "detector": {"weights": "w.pt", "batch_size": 2, "save_thr": 0.1},
            "lsc": {"ckpt": None, "det_thr": 0.3, "iou_neg": 0.05},
            "fusion": {"mode": "hard", "w": 0.5}, "tracker": {"type": "simple", "thr": 0.5},
            "voting": {"enabled": True, "tau": 0.5, "min_len": 3}, "eval": {"iou": 0.5, "spotting": True}}


def test_run_demo_end_to_end(tmp_path):
    video, out, tmp_parent = tmp_path / "in.mp4", tmp_path / "res" / "out.mp4", tmp_path / "tmp"
    tmp_parent.mkdir()
    make_video(video)
    log = get_logger("test_demo", tmp_path / "log.txt")
    s = demo.run_demo(video, out, cfg(), tmp_path, log, device="cpu", predictor_factory=fake_predictor,
                      recognizer_factory=lambda *a: FakeRecognizer(), tmp_parent=tmp_parent)
    assert s["frames"] == NF and (s["width"], s["height"]) == (W, H)
    assert s["tracks"] == 1 and s["longest_track"]["boxes"] == NF and s["longest_track"]["text"] == "EXIT"
    assert s["fps_pipeline"] > 0 and set(s["seconds"]) == {"extract", "det", "lsc", "fuse", "track", "vote",
                                                            "render"}
    cap = cv2.VideoCapture(str(out))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n == NF
    assert list(tmp_parent.iterdir()) == []  # temp tree cleaned
    m = load_json(tmp_path / "runs" / "demo" / "metrics.json")
    assert m["meta"]["demo_in"]["tracks"] == 1
    assert (tmp_path / "runs" / "demo" / "config.yaml").is_file()
    assert "longest track: #" in demo.format_summary(s)


def test_detector_lookup_and_plan(tmp_path):
    c = cfg()
    assert demo.detector_config(c, tmp_path)["weights"] == "w.pt"
    c["detector"], c["pipeline"] = {}, {"det_exp": "det_x"}
    with pytest.raises(FileNotFoundError):
        demo.detector_config(c, tmp_path)
    save_json({"detector": {"config": "p.yml", "weights": "m.pdparams"}}, tmp_path / "runs" / "det_x" / "config.yaml")
    det = demo.detector_config(c, tmp_path)
    assert demo.detector_plan(det) == "paddle" and demo.detector_plan({"weights": "a.pt"}) == "obb"


def test_paddle_command(tmp_path, monkeypatch):
    monkeypatch.delenv("PADDLE_PY", raising=False)
    det = {"config": "configs/p.yml", "weights": "runs/x/m.pdparams", "paddle_python": "envs/py.exe"}
    dcfg = demo.det_run_config(cfg(), det, tmp_path, "cuda")
    assert dcfg["detector"]["device"] == "gpu" and Path(dcfg["detector"]["weights"]).is_absolute()
    troot = tmp_path / "t"
    troot.mkdir()
    cmd, env = demo.paddle_command(dcfg, det, troot, tmp_path)
    assert cmd[0] == str(tmp_path / "envs" / "py.exe") and cmd[1:3] == ["-m", "logo.detector.infer_paddle"]
    assert env["LOGO_ROOT"] == str(troot) and str(tmp_path / "src") in env["PYTHONPATH"]
    assert demo.load_yaml(troot / "det_config.yaml")["detector"]["config"] == str(tmp_path / "configs" / "p.yml")
    with pytest.raises(FileNotFoundError):
        demo.paddle_python({}, tmp_path)


def test_draw_and_main_errors(tmp_path):
    img = np.zeros((H, W, 3), np.uint8)
    demo.draw_tracks(img, [{"track_id": 3, "quad": [10, 30, 50, 30, 50, 50, 10, 50], "text": "café"}])
    assert img.any()
    with pytest.raises(SystemExit) as e:
        demo.main(["--video", "x.mp4", "--out", "y.mp4", "--config", str(tmp_path / "missing.yaml")])
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        demo.main(["--config", str(ROOT / "configs" / "base.yaml")])
