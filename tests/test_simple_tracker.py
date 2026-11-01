import logging

import numpy as np
import pytest

from logo.tracking import run_track
from logo.tracking.simple_tracker import match, quad_iou_matrix, run_tracker, tracker_settings
from logo.utils.geometry import quad_iou
from logo.utils.io import save_json

CFG = {"tracker": {"type": "simple", "thr": 0.5}}


def box(x, y, w=40, h=20):
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def det(quad, score=0.9, text="", conf=0.0):
    return {"quad": [float(v) for v in quad], "score": score, "text": text, "text_conf": conf,
            "p_eos": 0.0, "lang_hard": int(text != ""), "feat_idx": 0, "score_fused": score}


def fused(frames: dict) -> dict:
    return {"video": "v1", "dataset": "icdar2015v", "frames": {str(k): v for k, v in frames.items()}}


def ids_by_frame(rows):
    out = {}
    for r in rows:
        out.setdefault(r["frame"], []).append(r["track_id"])
    return out


def test_iou_matrix_matches_geometry():
    rng = np.random.default_rng(0)
    a = [box(*rng.uniform(0, 60, 2)) for _ in range(5)] + [[0, 0, 10, 10, 10, 0, 0, 10]]  # bowtie
    b = [box(*rng.uniform(0, 60, 2)) for _ in range(4)] + [box(500, 500)]
    m = quad_iou_matrix(a, b)
    ref = np.array([[quad_iou(p, q) for q in b] for p in a])
    np.testing.assert_allclose(m, ref, atol=1e-6)
    assert m[:, -1].sum() == 0
    assert quad_iou_matrix([], b).shape == (0, 5)


def test_match_gate():
    iou = np.array([[0.9, 0.1], [0.2, 0.25]])
    assert match(iou, 0.3) == [(0, 0)]
    assert match(np.zeros((0, 3)), 0.3) == []


def test_settings():
    s = tracker_settings(CFG)
    assert (s["high_thr"], s["low_thr"], s["max_age"]) == (0.6, 0.3, 15)
    assert tracker_settings({"max_age": 3})["max_age"] == 3
    with pytest.raises(ValueError):
        tracker_settings({"tracker": {"low_thr": 0.7, "high_thr": 0.6}})


def test_id_persists_across_gap():
    frames = {f: ([] if f in (5, 6, 7) else [det(box(10 + 6 * f, 50))]) for f in range(1, 13)}
    rows = run_tracker(fused(frames), CFG)
    assert len(rows) == 9 and {r["track_id"] for r in rows} == {1}
    assert sorted({r["frame"] for r in rows}) == [1, 2, 3, 4, 8, 9, 10, 11, 12]
    assert all(set(r) == {"frame", "track_id", "quad", "score", "text"} for r in rows)


def test_track_dies_after_max_age():
    frames = {1: [det(box(10, 10))], 2: [det(box(10, 10))], 6: [det(box(10, 10))]}
    rows = run_tracker(fused(frames), {"tracker": {"max_age": 3}})
    assert [r["track_id"] for r in rows] == [1, 1, 2]
    assert [r["track_id"] for r in run_tracker(fused(frames), {"tracker": {"max_age": 4}})] == [1, 1, 1]


def test_crossing_boxes_keep_ids():
    frames = {}
    for f in range(1, 31):
        # 12 px/frame, 2 px apart vertically: without velocity prediction the ids swap at f=16
        a, b = det(box(12 * f, 100), text="LEFT", conf=0.9), det(box(372 - 12 * f, 102), text="RIGHT", conf=0.9)
        frames[f] = [b, a] if f % 2 else [a, b]  # shuffle det order
    rows = run_tracker(fused(frames), CFG)
    assert len(rows) == 60
    by_id = {}
    for r in rows:
        by_id.setdefault(r["track_id"], []).append(r)
    assert set(by_id) == {1, 2}
    for tid, rs in by_id.items():
        xs = [r["quad"][0] for r in rs]
        assert np.all(np.diff(xs) > 0) or np.all(np.diff(xs) < 0)  # never swaps direction
        assert len({r["text"] for r in rs}) == 1


def test_two_stage_low_scores():
    frames = {1: [det(box(10, 10), 0.9)], 2: [det(box(14, 10), 0.4), det(box(300, 10), 0.5)],
              3: [det(box(18, 10), 0.2)], 4: [det(box(22, 10), 0.7)]}
    rows = run_tracker(fused(frames), CFG)
    assert [(r["frame"], r["track_id"]) for r in rows] == [(1, 1), (2, 1), (4, 1)]
    assert rows[1]["score"] == pytest.approx(0.4)


def test_text_highest_conf_recent():
    frames = {1: [det(box(0, 0), text="HELL0", conf=0.5)], 2: [det(box(2, 0), text="HELLO", conf=0.9)],
              3: [det(box(4, 0), text="HELI", conf=0.3)], 4: [det(box(6, 0), text="HELP", conf=0.6)]}
    rows = run_tracker(fused(frames), CFG)
    assert [r["text"] for r in rows] == ["HELL0", "HELLO", "HELLO", "HELLO"]
    rows = run_tracker(fused(frames), {"tracker": {"text_window": 2}})
    assert [r["text"] for r in rows] == ["HELL0", "HELLO", "HELLO", "HELP"]


def test_run_track_writes_jsonl(tmp_path):
    frames = {1: [det(box(0, 0))], 2: [det(box(3, 0))], 3: []}
    save_json(fused(frames), tmp_path / "runs/F/fuse/icdar2015v/v1.json")
    cfg = {"exp_id": "T", **CFG}
    stats = run_track.run(cfg, "icdar2015v", ["v1"], tmp_path, logging.getLogger("t"), "F")
    from logo.eval.match import load_pred_frames
    pred = load_pred_frames(tmp_path / "runs/T/track/icdar2015v", "v1", tracks=True)
    assert sorted(pred) == [1, 2] and pred[2][0]["track_id"] == 1
    assert stats["boxes"] == 2 and stats["tracks"] == 1
    with pytest.raises(NotImplementedError):
        run_track.run({"exp_id": "T", "tracker": {"type": "unknown"}}, "icdar2015v", ["v1"], tmp_path,
                      logging.getLogger("t"), "F")
