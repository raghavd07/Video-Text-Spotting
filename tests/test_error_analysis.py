import csv
import importlib.util
import json
from pathlib import Path

import cv2
import motmetrics as mm
import numpy as np

from logo.eval.mot_eval import evaluate_mot
from logo.utils.io import load_json, save_json

_spec = importlib.util.spec_from_file_location(
    "error_analysis", Path(__file__).resolve().parents[1] / "tools" / "error_analysis.py")
ea = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ea)

DS, V, N, S = "icdar2015v", "v1", 6, 200


def rect(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


G1, G2, G4 = rect(10, 10, 60, 20), rect(100, 40, 160, 70), rect(20, 120, 100, 175)  # h 10, 30, 55
DC, FP = rect(10, 40, 60, 70), rect(150, 150, 190, 175)                             # FP h 25


def gt_obj(tid, quad, text, dc=False):
    return {"track_id": tid, "quad": quad, "text": text, "dont_care": dc, "quality": None}


def build(root: Path):
    frames = {}
    for f in range(1, N + 1):
        objs = [gt_obj(1, G1, "abc"), gt_obj(2, G2, "hello")]
        if f == 1:
            objs.append(gt_obj(3, DC, "###", dc=True))
        if f == 6:
            objs.append(gt_obj(4, G4, "BIG"))
        frames[str(f)] = objs
    save_json({"video": V, "dataset": DS, "width": S, "height": S, "num_frames": N, "frames": frames},
              root / "data" / "processed" / DS / f"{V}.json")
    tracks = []
    for f in range(1, N + 1):
        if f != 3:                                   # GT1 missed at f3 -> 1 miss + 1 fragmentation
            tracks.append({"frame": f, "track_id": 10, "quad": G1, "score": 0.9,
                           "text": "abx" if f == 5 else "abc"})  # wrong text at f5
        tracks.append({"frame": f, "track_id": 20 if f <= 3 else 21, "quad": G2, "score": 0.9,
                       "text": "hello"})             # switch at f4
    tracks.append({"frame": 2, "track_id": 30, "quad": FP, "score": 0.5, "text": "zz"})
    tracks.append({"frame": 1, "track_id": 40, "quad": rect(12, 42, 58, 68), "score": 0.5, "text": "q"})  # in DC
    d = root / "runs" / "EX" / "vote" / DS
    d.mkdir(parents=True)
    (d / f"{V}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tracks), encoding="utf-8")
    (root / "splits").mkdir()
    (root / "splits" / f"{DS}_val.txt").write_text(f"{V}\n", encoding="utf-8")
    fd = root / "data" / "frames" / DS / V
    fd.mkdir(parents=True)
    for f in range(1, N + 1):
        if f != 6:                                   # frame 6 missing -> grey canvas fallback
            cv2.imwrite(str(fd / f"{f:06d}.jpg"), np.full((S, S, 3), 200, np.uint8))
    return {"exp_id": "EX", "dataset": DS, "split": "val", "seed": 0, "eval": {"iou": 0.5}}


def test_size_bins_and_sentence():
    assert [ea.size_bin(h) for h in (5, 19.9, 20, 50, 50.1)] == ["<20", "<20", "20-50", "20-50", ">50"]
    assert abs(ea.quad_height(rect(0, 0, 40, 12)) - 12) < 1e-4
    assert ea.top_two_sentence({c: 0 for c in ea.CATEGORIES}) == "No errors were found."


def test_gallery_reservoir_deterministic():
    def run():
        g = ea.Gallery(3, np.random.default_rng(0))
        for i in range(50):
            g.offer("missed_gt", {"i": i})
        return [e["i"] for e in g.items["missed_gt"]]
    assert run() == run() and len(run()) == 3 and run() != [0, 1, 2]


def test_analyze_end_to_end(tmp_path):
    cfg = build(tmp_path)
    out = tmp_path / "figs"
    res = ea.analyze(cfg, tmp_path, out_dir=out)
    c = res["counts"]
    assert c["missed_gt"] == {"<20": 1, "20-50": 0, ">50": 1}
    assert c["false_positive"] == {"<20": 0, "20-50": 1, ">50": 0}
    assert c["wrong_text"] == {"<20": 1, "20-50": 0, ">50": 0}
    assert c["id_switch"] == {"<20": 0, "20-50": 1, ">50": 0}
    assert c["fragmentation"] == {"<20": 1, "20-50": 0, ">50": 0}
    assert res["summary"].startswith("The two largest error categories are missed GT (2, 33.3%")
    assert "false positives (1, 16.7%)" in res["summary"]

    # consistent with the evaluator and motmetrics' own fragmentation count
    ref = evaluate_mot(tmp_path / "data" / "processed" / DS, tmp_path / "runs" / "EX" / "vote" / DS, [V],
                       spotting=False)["per_video"][V]
    t = res["totals"]
    assert (t["missed_gt"], t["false_positive"], t["id_switch"]) == (ref["misses"], ref["fp"], ref["switches"])
    from logo.eval.match import load_pred_frames
    gt = {int(k): v for k, v in load_json(tmp_path / "data" / "processed" / DS / f"{V}.json")["frames"].items()}
    acc, _ = ea._accumulate(gt, load_pred_frames(tmp_path / "runs" / "EX" / "vote" / DS, V, True), N, V, 0.5)
    assert int(mm.metrics.create().compute(acc, metrics=["num_fragmentations"]).iloc[0, 0]) == t["fragmentation"]

    with open(out / "errors.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(ea.CATEGORIES) * 4
    assert {"category": "missed_gt", "size_bin": "all", "count": "2"} in rows
    bar = cv2.imread(str(out / "errors_bar.png"))
    assert bar is not None and bar.shape[1] > 500
    gal = cv2.imread(str(out / "gallery_missed_gt.png"))
    th = ea.TILE_H + ea.CAP_H
    assert gal.shape == (ea.TITLE_H + th + 4, ea.COLS * (ea.TILE_W + 4), 3)
    tile = gal[ea.TITLE_H:ea.TITLE_H + ea.TILE_H, :ea.TILE_W]
    green = (tile[..., 1] > 150) & (tile[..., 0] < 80) & (tile[..., 2] < 80)
    assert green.any()
    assert set(res["galleries"]) == set(ea.CATEGORIES)
    sw = cv2.imread(str(out / "gallery_id_switch.png"))[ea.TITLE_H:ea.TITLE_H + ea.TILE_H, :ea.TILE_W]
    assert ((sw[..., 2] > 150) & (sw[..., 1] < 80) & (sw[..., 0] < 80)).any()   # red prediction

    m = load_json(tmp_path / "runs" / "EX" / "metrics.json")
    assert m["meta"]["errors_icdar2015v_val"]["totals"] == t
    assert (tmp_path / "runs" / "EX" / "config.yaml").is_file()
    assert (tmp_path / "runs" / "EX" / "log.txt").is_file()


def test_test_split_needs_final(tmp_path):
    cfg = build(tmp_path)
    cfg["split"] = "test"
    try:
        ea.analyze(cfg, tmp_path, out_dir=tmp_path / "figs")
    except PermissionError:
        return
    raise AssertionError("test split must require --final")
