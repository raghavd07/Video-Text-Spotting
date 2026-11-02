import logging

import pytest

from logo.scoring import run_fuse
from logo.scoring.fusion import fuse, lang_score
from logo.utils.io import load_json, save_json


def test_hard_fusion():
    assert fuse(0.8, 1, "hard") == pytest.approx(0.9)
    assert fuse(0.8, 0, "hard") == pytest.approx(0.4)
    assert fuse(0.6, 1, "hard", w=0.5) == pytest.approx((0.6 + 1) / 2)  # paper: plain average
    assert fuse(0.6, 1, "hard", w=0.0) == pytest.approx(0.6)
    assert fuse(0.6, 0, "hard", w=1.0) == 0.0
    assert isinstance(fuse(1, 1, "hard"), float)


def test_soft_fusion():
    assert fuse(0.8, 0.6, "soft") == pytest.approx(0.7)
    assert fuse(0.8, 0.6, "soft", w=0.3) == pytest.approx(0.7 * 0.8 + 0.3 * 0.6)
    assert fuse(0.8, 0.6, "soft", w=0.7) == pytest.approx(0.3 * 0.8 + 0.7 * 0.6)
    assert fuse(0.6, 1, "soft") == pytest.approx(fuse(0.6, 1, "hard"))  # soft == hard on {0, 1}
    assert fuse(0.6, 0, "soft") == pytest.approx(fuse(0.6, 0, "hard"))
    assert fuse(0.6, 0.2, "soft", w=0.0) == pytest.approx(0.6) and fuse(0.6, 0.2, "soft", w=1.0) == pytest.approx(0.2)
    assert fuse(0.5, 1.0 + 1e-7, "soft", w=1.0) == 1.0 and fuse(0.5, -1e-7, "soft", w=1.0) == 0.0  # clipped
    assert isinstance(fuse(1, 1, "soft"), float)


def test_lang_score():
    o = {"lang_hard": 1, "p_eos": 0.25, "text_conf": 0.9}
    assert lang_score(o, "hard") == 1 and lang_score(o, "hard", "text_conf") == 1
    assert lang_score(o, "soft") == pytest.approx(0.75)  # 1 - p_eos
    assert lang_score(o, "soft", "text_conf") == pytest.approx(0.9)
    assert fuse(0.6, lang_score(o, "soft"), "soft", w=0.5) == pytest.approx(0.675)
    with pytest.raises(ValueError):
        lang_score(o, "soft", "mean")
    with pytest.raises(ValueError):
        lang_score(o, "avg")


def test_fusion_errors():
    with pytest.raises(ValueError):
        fuse(0.5, 1.2, "soft")
    with pytest.raises(ValueError):
        fuse(0.5, -0.1, "soft")
    with pytest.raises(ValueError):
        fuse(0.5, 0.5, "soft", w=-0.1)
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
    assert stats["boxes"] == 2 and "soft_source" not in stats


@pytest.mark.parametrize("source,expect", [(None, (0.75, 0.55)), ("p_eos", (0.75, 0.55)),
                                           ("text_conf", (0.75, 0.45))])
def test_run_fuse_soft(tmp_path, source, expect):
    obj = {"quad": [0, 0, 4, 0, 4, 2, 0, 2], "score": 0.6, "text": "a", "text_conf": 0.9, "p_eos": 0.1,
           "lang_hard": 1, "feat_idx": 0}
    lsc = {"video": "v1", "dataset": "icdar2015v",
           "frames": {"1": [obj, {**obj, "text_conf": 0.3, "p_eos": 0.5, "feat_idx": 1}]}}
    save_json(lsc, tmp_path / "runs/L/lsc/icdar2015v/v1.json")
    cfg = {"exp_id": "E", "fusion": {"mode": "soft", "w": 0.5, **({"soft_source": source} if source else {})}}
    stats = run_fuse.run(cfg, "icdar2015v", ["v1"], tmp_path, logging.getLogger("t"), "L")
    out = load_json(tmp_path / "runs/E/fuse/icdar2015v/v1.json")
    assert [o["score_fused"] for o in out["frames"]["1"]] == pytest.approx(expect)
    assert stats["soft_source"] == (source or "p_eos")
