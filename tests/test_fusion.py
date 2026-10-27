import logging

import pytest

from logo.scoring import run_fuse
from logo.scoring.fusion import fuse
from logo.utils.io import load_json, save_json


def test_hard_fusion():
    assert fuse(0.8, 1, "hard") == pytest.approx(0.9)
    assert fuse(0.8, 0, "hard") == pytest.approx(0.4)
    assert fuse(0.6, 1, "hard", w=0.5) == pytest.approx((0.6 + 1) / 2)  # paper: plain average
    assert fuse(0.6, 1, "hard", w=0.0) == pytest.approx(0.6)
    assert fuse(0.6, 0, "hard", w=1.0) == 0.0
    assert isinstance(fuse(1, 1, "hard"), float)


def test_fusion_errors():
    with pytest.raises(NotImplementedError):
        fuse(0.5, 1, "soft")
    with pytest.raises(ValueError):
        fuse(0.5, 1, "avg")
    with pytest.raises(ValueError):
        fuse(0.5, 1, "hard", w=1.5)
    with pytest.raises(ValueError):
        fuse(0.5, 0.7, "hard")


def test_run_fuse(tmp_path):
    obj = {"quad": [0, 0, 4, 0, 4, 2, 0, 2], "score": 0.6, "text": "a", "text_conf": 0.9, "p_eos": 0.1,
           "lang_hard": 1, "feat_idx": 0}
    lsc = {"video": "v1", "dataset": "icdar2015v",
           "frames": {"1": [obj, {**obj, "lang_hard": 0, "feat_idx": 1}], "2": []}}
    save_json(lsc, tmp_path / "runs/L/lsc/icdar2015v/v1.json")
    cfg = {"exp_id": "E", "fusion": {"mode": "hard", "w": 0.5}}
    stats = run_fuse.run(cfg, "icdar2015v", ["v1"], tmp_path, logging.getLogger("t"), "L")
    out = load_json(tmp_path / "runs/E/fuse/icdar2015v/v1.json")
    a, b = out["frames"]["1"]
    assert a["score_fused"] == pytest.approx(0.8) and b["score_fused"] == pytest.approx(0.3)
    assert {k: v for k, v in a.items() if k != "score_fused"} == obj
    assert out["frames"]["2"] == [] and out["video"] == "v1"
    assert stats["boxes"] == 2
