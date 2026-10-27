import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from logo.eval.det_eval import evaluate_detection
from logo.eval.match import Boxes, inside_fraction, iou_matrix, normalize_text
from logo.eval.mot_eval import PER_VIDEO_KEYS, evaluate_mot
from logo.utils.geometry import quad_iou
from logo.utils.io import load_json, save_json

_spec = importlib.util.spec_from_file_location(
    "evaluate_tool", Path(__file__).resolve().parents[1] / "tools" / "evaluate.py")
ev_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev_tool)

NF, W, H = 20, 40, 20
TEXTS = ["Hello", "World!", "exit-7"]
VIDEOS = ["v1", "v2"]
DC_QUAD = [300, 200, 360, 200, 360, 240, 300, 240]


def box(x, y, w=W, h=H):
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def make_gt(video):
    frames = {}
    for f in range(1, NF + 1):
        objs = [{"track_id": t + 1, "quad": box(20 + 80 * t + 2 * f, 50 + 40 * t), "text": TEXTS[t],
                 "dont_care": False, "quality": None} for t in range(3)]
        if video == "v2":
            objs.append({"track_id": 9, "quad": DC_QUAD, "text": "###", "dont_care": True,
                         "quality": None})
        frames[str(f)] = objs
    return {"video": video, "dataset": "icdar2015v", "width": 640, "height": 360, "num_frames": NF,
            "frames": frames}


def gt_tracks(gt):
    return [{"frame": int(f), "track_id": o["track_id"], "quad": list(o["quad"]), "score": 1.0,
             "text": o["text"]} for f, objs in gt["frames"].items() for o in objs if not o["dont_care"]]


def write_tracks(d: Path, video, rows):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{video}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


@pytest.fixture
def gt_dir(tmp_path):
    d = tmp_path / "gt"
    for v in VIDEOS:
        save_json(make_gt(v), d / f"{v}.json")
    return d


def tracks_dir(tmp_path, gt_dir, name, edit=lambda rows, v: rows):
    d = tmp_path / name
    for v in VIDEOS:
        write_tracks(d, v, edit(gt_tracks(load_json(gt_dir / f"{v}.json")), v))
    return d


def test_normalize_text():
    assert normalize_text("Exit-7!") == "exit7"
    assert normalize_text("###") == "" and normalize_text(None) == ""


def test_iou_matches_geometry_and_prefilter():
    a = [10, 10, 60, 5, 65, 30, 15, 35]
    b = box(30, 12, 50, 30)
    far = box(500, 500)
    m = iou_matrix(Boxes.from_quads([a, far]), Boxes.from_quads([b]))
    assert m[0, 0] == pytest.approx(quad_iou(a, b), abs=1e-9)
    assert m[1, 0] == 0.0
    frac = inside_fraction(Boxes.from_quads([box(0, 0, 10, 10)]), Boxes.from_quads([box(5, 0, 10, 10)]))
    assert frac[0] == pytest.approx(0.5)


def test_gt_as_prediction_is_perfect(tmp_path, gt_dir):
    pd = tracks_dir(tmp_path, gt_dir, "perfect")
    det = evaluate_detection(gt_dir, pd, VIDEOS)
    assert det["precision"] == det["recall"] == det["f"] == pytest.approx(100.0)
    for spotting in (False, True):
        m = evaluate_mot(gt_dir, pd, VIDEOS, spotting=spotting)
        for k in ("MOTA", "MOTP", "IDF1", "MT"):
            assert m[k] == pytest.approx(100.0), k
        assert m["ML"] == 0.0
        pv = m["per_video"]["v1"]
        assert set(pv) == set(PER_VIDEO_KEYS)
        assert pv["frames"] == NF and pv["num_objects"] == 3 * NF and pv["idtp"] == 3 * NF


def test_half_width_shift_reduces_mota(tmp_path, gt_dir):
    def shift(rows, v):
        for r in rows:
            r["quad"] = [c + W / 2 if i % 2 == 0 else c for i, c in enumerate(r["quad"])]
        return rows
    m = evaluate_mot(gt_dir, tracks_dir(tmp_path, gt_dir, "shift", shift), VIDEOS, spotting=False)
    assert m["MOTA"] < 100.0
    assert m["per_video"]["v1"]["misses"] == 3 * NF


def test_id_swap_lowers_idf1_not_det(tmp_path, gt_dir):
    def swap(rows, v):
        for r in rows:
            if r["frame"] > NF // 2 and r["track_id"] in (1, 2):
                r["track_id"] = 3 - r["track_id"]
        return rows
    pd = tracks_dir(tmp_path, gt_dir, "swap", swap)
    m = evaluate_mot(gt_dir, pd, VIDEOS, spotting=False)
    assert m["IDF1"] < 100.0
    assert m["per_video"]["v1"]["switches"] == 2
    assert evaluate_detection(gt_dir, pd, VIDEOS)["f"] == pytest.approx(100.0)


def test_text_shuffle_only_hurts_spot(tmp_path, gt_dir):
    def shuffle(rows, v):
        for r in rows:
            r["text"] = TEXTS[r["track_id"] % 3]
        return rows
    pd = tracks_dir(tmp_path, gt_dir, "shuffle", shuffle)
    assert evaluate_detection(gt_dir, pd, VIDEOS)["f"] == pytest.approx(100.0)
    assert evaluate_mot(gt_dir, pd, VIDEOS, spotting=False)["MOTA"] == pytest.approx(100.0)
    spot = evaluate_mot(gt_dir, pd, VIDEOS, spotting=True)
    assert spot["MOTA"] < 100.0 and spot["IDF1"] < 100.0


def test_prediction_inside_dont_care_is_ignored(tmp_path, gt_dir):
    def add_dc(rows, v):
        if v == "v2":
            rows += [{"frame": f, "track_id": 50, "quad": box(305, 205, 40, 30), "score": 0.9,
                      "text": "x"} for f in range(1, NF + 1)]
        return rows
    pd = tracks_dir(tmp_path, gt_dir, "dc", add_dc)
    det = evaluate_detection(gt_dir, pd, VIDEOS)
    assert det["precision"] == det["recall"] == pytest.approx(100.0)
    assert det["per_video"]["v2"]["ignored"] == NF
    m = evaluate_mot(gt_dir, pd, VIDEOS, spotting=False)
    assert m["per_video"]["v2"]["fp"] == 0 and m["MOTA"] == pytest.approx(100.0)

    def half_out(rows, v):  # only 25% inside the dont_care quad -> a real FP
        if v == "v2":
            rows += [{"frame": 1, "track_id": 51, "quad": box(345, 200, 60, 40), "score": 0.9,
                      "text": "x"}]
        return rows
    det = evaluate_detection(gt_dir, tracks_dir(tmp_path, gt_dir, "dc2", half_out), VIDEOS)
    assert det["counts"]["fp"] == 1


def test_det_json_input(tmp_path, gt_dir):
    pd = tmp_path / "det"
    for v in VIDEOS:
        gt = load_json(gt_dir / f"{v}.json")
        save_json({"video": v, "dataset": "icdar2015v", "frames": {
            f: [{"quad": o["quad"], "score": 0.9} for o in objs] for f, objs in gt["frames"].items()}},
            pd / f"{v}.json")
    assert evaluate_detection(gt_dir, pd, VIDEOS)["f"] == pytest.approx(100.0)


def test_tool_writes_contract_layout(tmp_path, gt_dir):
    pd = tracks_dir(tmp_path, gt_dir, "tool")
    split = tmp_path / "splits" / "icdar2015v_val.txt"
    split.parent.mkdir()
    split.write_text("v1\nv2\n", encoding="utf-8")
    out = tmp_path / "runs" / "exp" / "metrics.json"
    for mode in ev_tool.MODES:
        assert ev_tool.main(["--gt_dir", str(gt_dir), "--pred_dir", str(pd), "--videos", str(split),
                             "--mode", mode, "--out", str(out)]) == 0
    m = load_json(out)
    assert set(m["det"]) == {"precision", "recall", "f"}
    assert set(m["track"]) == set(m["spot"]) == {"MOTA", "MOTP", "IDF1", "MT", "ML"}
    assert set(m["per_video"]) == set(VIDEOS) and set(m["per_video"]["v1"]) == set(PER_VIDEO_KEYS)
    assert set(m["meta"]["spot"]["per_video"]) == set(VIDEOS)
    assert all(np.isclose(m["spot"][k], 100.0) for k in ("MOTA", "IDF1"))
    assert (out.parent / "log.txt").is_file() and (out.parent / "config.yaml").is_file()
    for h in list(ev_tool.get_logger("evaluate").handlers):
        h.close()
        ev_tool.get_logger("evaluate").removeHandler(h)

    test_split = split.with_name("icdar2015v_test.txt")
    test_split.write_text("v1\n", encoding="utf-8")
    with pytest.raises(PermissionError):
        ev_tool.main(["--gt_dir", str(gt_dir), "--pred_dir", str(pd), "--videos", str(test_split),
                      "--mode", "det", "--out", str(out)])
