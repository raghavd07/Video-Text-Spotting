import logging
from pathlib import Path

import numpy as np
import pytest

from logo.detector import infer_paddle as ip
from logo.utils.io import load_json, load_yaml, save_json

ROOT = Path(__file__).resolve().parents[1]
PPDET_CFG = ROOT / "third_party/PaddleDetection/configs/rotate/ppyoloe_r/ppyoloe_r_crn_s_3x_dota.yml"
# the fake predictor never decodes frames, so a JPEG header is enough
FAKE_JPG = b"\xff\xd8\xff\xd9"


def row(score, x=10.0, cls=0.0):
    return [cls, score, x, 5, x + 20, 5, x + 20, 15, x, 15]


class FakePredictor:
    """Frame k gets one box with score 0.05*k (k from the file name) plus one dummy -1 row."""

    def __init__(self):
        self.calls, self.model_seconds = [], 0.0

    def __call__(self, paths):
        self.calls.append(len(paths))
        self.model_seconds += 0.001 * len(paths)
        return [np.array([row(0.05 * int(Path(p).stem)), row(0.9, cls=-1)], np.float32)
                for p in paths]


def make_tree(root, ds="icdar2015v", videos=("v1", "v2"), n=5):
    (root / "splits").mkdir(parents=True)
    for split, vids in (("val", videos), ("test", ("t1",))):
        (root / "splits" / f"{ds}_{split}.txt").write_text("\n".join(vids) + "\n", encoding="utf-8")
    for v in videos:
        d = root / "data" / "frames" / ds / v
        d.mkdir(parents=True)
        for i in range(1, n + 1):
            (d / f"{i:06d}.jpg").write_bytes(FAKE_JPG)
        (d / "notes.txt").write_text("x")
    return {"exp_id": "e1", "dataset": ds, "split": "val", "seed": 0,
            "detector": {"batch_size": 2, "save_thr": 0.1, "input_hw": [32, 64]}}


def test_rows_to_objects_filters_and_sorts():
    rows = np.array([row(0.3), row(0.1), row(0.8, x=50), row(0.95, cls=-1)], np.float32)
    objs = ip.rows_to_objects(rows, 0.1)
    assert [o["score"] for o in objs] == pytest.approx([0.8, 0.3])
    assert objs[0]["quad"] == [50.0, 5.0, 70.0, 5.0, 70.0, 15.0, 50.0, 15.0]
    assert all(isinstance(v, float) for o in objs for v in o["quad"])
    assert ip.rows_to_objects(np.zeros((0, 10)), 0.1) == []


def test_list_frames_sorted_one_based(tmp_path):
    for i in (10, 2, 1):
        (tmp_path / f"{i:06d}.jpg").write_bytes(FAKE_JPG)
    (tmp_path / "x.png").write_bytes(b"")
    assert [f for f, _ in ip.list_frames(tmp_path)] == [1, 2, 10]
    with pytest.raises(FileNotFoundError):
        ip.list_frames(tmp_path / "missing")


def test_test_split_needs_final(tmp_path):
    make_tree(tmp_path)
    with pytest.raises(PermissionError):
        ip.read_videos("icdar2015v", "test", tmp_path)
    assert ip.read_videos("icdar2015v", "test", tmp_path, final=True) == ["t1"]
    assert ip.read_videos("icdar2015v", "val", tmp_path) == ["v1", "v2"]


def test_run_writes_contract_files(tmp_path):
    cfg = make_tree(tmp_path)
    pred, log = FakePredictor(), logging.getLogger("test_infer_paddle")
    timing = ip.run(cfg, ["v1", "v2"], pred, tmp_path, log, peak_mem_fn=lambda: 123.4)

    assert pred.calls == [2, 2, 1, 2, 2, 1]  # batch_size 2 over 5 frames, per video
    det = load_json(tmp_path / "runs/e1/det/icdar2015v/v1.json")
    assert det["video"] == "v1" and det["dataset"] == "icdar2015v"
    assert sorted(det["frames"], key=int) == ["1", "2", "3", "4", "5"]
    assert det["frames"]["1"] == [] and det["frames"]["2"] == []  # 0.05, 0.10 not > save_thr
    assert det["frames"]["5"][0]["score"] == pytest.approx(0.25)
    assert len(det["frames"]["5"][0]["quad"]) == 8

    assert timing["frames"] == 10 and timing["videos"] == 2 and timing["peak_gpu_mem_mb"] == 123.4
    assert timing["ms_per_frame_model"] == pytest.approx(1.0)
    assert load_json(tmp_path / "runs/e1/det/timing.json") == timing
    metrics = load_json(tmp_path / "runs/e1/metrics.json")
    assert set(metrics) >= {"det", "track", "spot", "per_video", "meta"}
    assert metrics["meta"]["det_infer"] == timing
    assert load_yaml(tmp_path / "runs/e1/config.yaml")["exp_id"] == "e1"


def test_run_skips_existing_and_keeps_metrics(tmp_path):
    cfg = make_tree(tmp_path)
    save_json({"det": {"f": 1.0}, "meta": {"det": {"x": 1}}}, tmp_path / "runs/e1/metrics.json")
    save_json({"video": "v1"}, tmp_path / "runs/e1/det/icdar2015v/v1.json")
    pred = FakePredictor()
    ip.run(cfg, ["v1", "v2"], pred, tmp_path, logging.getLogger("t"), max_frames=3)
    assert pred.calls == [2, 1]
    assert load_json(tmp_path / "runs/e1/det/icdar2015v/v1.json") == {"video": "v1"}
    metrics = load_json(tmp_path / "runs/e1/metrics.json")
    assert metrics["det"] == {"f": 1.0} and metrics["meta"]["det"] == {"x": 1}
    assert metrics["meta"]["det_infer"]["frames"] == 3


def test_paddle_detector_cpu_random_weights(tmp_path):
    pytest.importorskip("paddle")
    pytest.importorskip("ppdet")
    cv2 = pytest.importorskip("cv2")
    if not PPDET_CFG.is_file():
        pytest.skip("PaddleDetection not cloned")
    img = np.random.default_rng(0).integers(0, 255, (48, 80, 3), dtype=np.uint8)
    paths = []
    for i in (1, 2):
        p = tmp_path / f"{i:06d}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    det = ip.PaddleDetector(PPDET_CFG, None, input_hw=[64, 64], device="cpu")
    out = det(paths)
    assert len(out) == 2 and all(o.ndim == 2 and o.shape[1] == 10 for o in out)
    assert det.model_seconds > 0 and det.peak_memory_mb() is None
