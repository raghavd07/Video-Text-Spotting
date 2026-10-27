import logging
from pathlib import Path

import numpy as np
import pytest

from logo.detector import infer_obb as io_
from logo.detector import infer_paddle as ip
from logo.utils.geometry import signed_area
from logo.utils.io import load_json


def test_results_to_rows_orders_and_shapes():
    # counter-clockwise box (in image coords) -> reordered clockwise like GT
    pts = np.array([[[10, 5], [10, 15], [30, 15], [30, 5]]], np.float32)
    rows = io_.results_to_rows(pts, np.array([0.7]), np.array([3.0]))
    assert rows.shape == (1, 10) and rows.dtype == np.float32
    assert rows[0, 0] == 3.0 and rows[0, 1] == pytest.approx(0.7)
    assert signed_area(rows[0, 2:]) > 0
    assert sorted(map(tuple, rows[0, 2:].reshape(4, 2))) == sorted(map(tuple, pts[0]))
    assert io_.results_to_rows(np.zeros((0, 4, 2)), np.zeros(0), np.zeros(0)).shape == (0, 10)
    objs = ip.rows_to_objects(rows, 0.1)
    assert len(objs) == 1 and len(objs[0]["quad"]) == 8


class FakeOBB:
    half = True

    def __init__(self):
        self.model_seconds = 0.0

    def __call__(self, paths):
        self.model_seconds += 0.002 * len(paths)
        pts = np.array([[[10, 5], [30, 5], [30, 15], [10, 15]]], np.float32)
        return [io_.results_to_rows(pts, np.array([0.5]), np.array([0.0])) for _ in paths]


def test_run_with_obb_predictor_records_amp(tmp_path):
    ds = "icdar2015v"
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / f"{ds}_val.txt").write_text("v1\n", encoding="utf-8")
    d = tmp_path / "data" / "frames" / ds / "v1"
    d.mkdir(parents=True)
    for i in (1, 2, 3):
        (d / f"{i:06d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    cfg = {"exp_id": "e1", "dataset": ds, "split": "val", "seed": 0,
           "detector": {"batch_size": 2, "save_thr": 0.1, "input_hw": [32, 64]}}
    pred = FakeOBB()
    timing = ip.run(cfg, ["v1"], pred, tmp_path, logging.getLogger("t"), peak_mem_fn=lambda: 1.0,
                    amp=pred.half)
    assert timing["amp"] is True and timing["frames"] == 3
    assert timing["ms_per_frame_model"] == pytest.approx(2.0)
    det = load_json(tmp_path / "runs/e1/det/icdar2015v/v1.json")
    assert det["frames"]["3"][0]["score"] == pytest.approx(0.5)
    assert load_json(tmp_path / "runs/e1/metrics.json")["meta"]["det_infer"]["amp"] is True


def test_ultralytics_obb_cpu_random_weights(tmp_path):
    pytest.importorskip("ultralytics")
    cv2 = pytest.importorskip("cv2")
    img = np.random.default_rng(0).integers(0, 255, (48, 80, 3), dtype=np.uint8)
    paths = []
    for i in (1, 2):
        p = tmp_path / f"{i:06d}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    # yaml -> randomly initialised model, no download; conf 0 so some boxes come back
    det = io_.UltralyticsOBB(Path("yolo11n-obb.yaml"), input_hw=[64, 64], device="cpu", conf=0.0,
                             half=True, max_det=5)
    assert det.half is False  # FP16 only on GPU
    out = det(paths)
    assert len(out) == 2 and all(o.ndim == 2 and o.shape[1] == 10 for o in out)
    assert all(len(o) <= 5 for o in out)
    assert det.model_seconds > 0 and det.peak_memory_mb() is None
