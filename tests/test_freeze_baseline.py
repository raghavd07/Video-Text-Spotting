import importlib.util
from pathlib import Path

import cv2
import motmetrics.lap as mm_lap
import numpy as np
import pytest
import yaml

from logo.utils.io import get_logger, load_json, save_json

TOOLS = Path(__file__).resolve().parents[1] / "tools"
DS, NF, C, D = "icdar2015v", 4, 3, 4


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fb = _load("freeze_baseline")


class FakeRecognizer:
    c, d, batch_size = C, D, 4

    def predict(self, crops):
        return [{"text": "word" if cr.mean() > 100 else "", "text_conf": 0.9, "p_eos": 0.1,
                 "feat": np.zeros((C, D), np.float16)} for cr in crops]


def fake_factory(cfg, root, log, device=None):
    return FakeRecognizer()


def box(x, y, w=40, h=16):
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def make_cfg(exp, w, lsc_exp=None):
    pc = {"det_exp": "D", **({"lsc_exp": lsc_exp} if lsc_exp else {})}
    return {"exp_id": exp, "dataset": DS, "split": "val", "seed": 0, "detector": {}, "pipeline": pc,
            "lsc": {"ckpt": None, "det_thr": 0.3, "iou_neg": 0.05}, "fusion": {"mode": "hard", "w": w},
            "tracker": {"type": "simple", "thr": 0.5}, "voting": {"enabled": False, "tau": 0.5, "min_len": 3},
            "eval": {"iou": 0.5, "spotting": True}}


DECISIONS = """# Decisions

|Date|Task|Decision|Why|Impact|
|-|-|-|-|-|
|2026-10-26|P01|`save\\_json` atomic|safety|none|
|||(example) detector|6 GB GPU|lower det recall|
|2026-11-01|P19|use VPMM `\\[N,2d]` features|keep upstream|deviates from DeepSolo queries|
"""


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    monkeypatch.setattr(mm_lap, "default_solver", "scipy")
    gt_q, fp_q, low_q = [box(10, 10), box(100, 10)], box(10, 80), box(100, 80)
    fdir = tmp_path / "data/frames" / DS / "v1"
    fdir.mkdir(parents=True)
    for f in range(1, NF + 1):
        img = np.zeros((120, 160, 3), np.uint8)
        for q in gt_q:
            img[q[1]:q[5], q[0]:q[2]] = 220
        cv2.imwrite(str(fdir / f"{f:06d}.jpg"), img)
    save_json({"video": "v1", "dataset": DS, "width": 160, "height": 120, "num_frames": NF,
               "frames": {str(f): [{"track_id": i + 1, "quad": q, "text": "word", "dont_care": False,
                                    "quality": None} for i, q in enumerate(gt_q)] for f in range(1, NF + 1)}},
              tmp_path / "data/processed" / DS / "v1.json")
    save_json({"video": "v1", "dataset": DS, "frames": {str(f): (
        [{"quad": q, "score": 0.8} for q in gt_q] + [{"quad": fp_q, "score": 0.7}, {"quad": low_q, "score": 0.2}])
        for f in range(1, NF + 1)}}, tmp_path / "runs/D/det" / DS / "v1.json")
    (tmp_path / "splits").mkdir()
    for s in ("train", "val", "test"):
        (tmp_path / "splits" / f"{DS}_{s}.txt").write_text("v1\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    for name, cfg in (("B", make_cfg("B", 0.0, lsc_exp="L")), ("L", make_cfg("L", 0.5))):
        (tmp_path / "configs" / f"{name}.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/decisions.md").write_text(DECISIONS, encoding="utf-8")
    (tmp_path / "docs/baseline_report.md").write_text("### P13\n\nold text\n", encoding="utf-8")
    yield tmp_path
    for name in ("run_pipeline", "freeze_baseline", "profile_efficiency", "evaluate"):
        lg = get_logger(name)
        for h in list(lg.handlers):
            h.close()
            lg.removeHandler(h)


def plan_of(split="val"):
    return {"exp_id": "FZ", "dataset": DS, "split": split, "seed": 0,
            "rows": [{"label": "Baseline", "config": "configs/B.yaml"}, {"label": "+LSC hard", "config": "configs/L.yaml"}],
            "metrics": ["det.f", "track.MOTA", "spot.IDF1"],
            "paper_table_v": {"+LSC hard": {"det.f": 70.0, "track.MOTA": None, "spot.IDF1": 55.5}},
            "repro": {"row": "+LSC hard", "tol": 0.3}, "profile": {"frames": 3, "warmup": 1}}


def test_helpers(root):
    devs = fb.read_deviations(root / "docs/decisions.md")
    assert devs == ["2026-11-01 P19: use VPMM `[N,2d]` features (why: keep upstream; "
                    "impact: deviates from DeepSolo queries)"]
    assert fb.metric_diffs({"det": {"f": 1.0}, "track": {"MOTA": 2.0}}, {"det": {"f": 1.2}, "track": {}}) == {
        "det.f": pytest.approx(0.2), "track.MOTA": float("inf")}
    rows = fb.load_rows(plan_of(), root)
    assert [rows[i]["cfg"]["exp_id"] for i in fb.run_order(rows)] == ["L", "B"]  # B reuses L's lsc
    p = root / "r.md"
    fb.write_report(p, f"{fb.FREEZE_BEGIN}\nA\n{fb.FREEZE_END}\n")
    p.write_text("head\n" + p.read_text(encoding="utf-8") + "tail\n", encoding="utf-8")
    fb.write_report(p, f"{fb.FREEZE_BEGIN}\nB\n{fb.FREEZE_END}\n")
    assert p.read_text(encoding="utf-8") == f"head\n{fb.FREEZE_BEGIN}\nB\n{fb.FREEZE_END}\ntail\n"


def test_freeze_end_to_end(root):
    log = get_logger("freeze_baseline", root / "runs/FZ/log.txt")
    res = fb.freeze(plan_of(), root, log, device="cpu", recognizer_factory=fake_factory,
                    profile_kw={"recognizer": FakeRecognizer()})
    assert res["repro"]["ok"] and set(res["repro"]["diffs"]) >= {"det.f", "track.MOTA", "spot.IDF1"}
    assert max(res["repro"]["diffs"].values()) <= 0.3
    md = (root / "docs/baseline_report.md").read_text(encoding="utf-8")
    assert md.startswith("### P13\n\nold text\n") and md.count(fb.FREEZE_BEGIN) == 1
    assert "paper reported (Table V)" in md and "| 70.00 / - / 55.50 |" in md and "| n/r |" in md
    assert "### Efficiency" in md and "| Baseline | lsc |" in md and "**PASS**" in md
    assert "P19: use VPMM" in md and "P01" not in md
    m = load_json(root / "runs/FZ/metrics.json")["meta"]["freeze_icdar2015v_val"]
    assert [r["exp_id"] for r in m["rows"]] == ["B", "L"] and m["repro"]["ok"]
    assert (root / "runs/FZ/config.yaml").is_file() and (root / "runs/FZ/log.txt").is_file()
    assert any("git tag -a v1.0-base" in c for c in res["git"])

    # second call replaces the block instead of appending
    fb.freeze(plan_of(), root, log, device="cpu", skip_profile=True, recognizer_factory=fake_factory)
    md = (root / "docs/baseline_report.md").read_text(encoding="utf-8")
    assert md.count(fb.FREEZE_BEGIN) == 1 and "### Efficiency" not in md


def test_det_override_and_suffix(root):
    t = make_cfg("T", 0.5, lsc_exp="OLD")  # e.g. E5-lst reading E3-lsc
    (root / "configs/T.yaml").write_text(yaml.safe_dump(t), encoding="utf-8")
    plan = {**plan_of(), "det_exp": "D2", "exp_suffix": "_s"}
    plan["rows"] = plan["rows"] + [{"label": "T", "config": "configs/T.yaml"}]
    with pytest.raises(ValueError, match="OLD"):
        fb.load_rows(plan, root)
    plan["rows"][-1]["lsc_exp"] = "L"
    rows = {r["label"]: r["cfg"] for r in fb.load_rows(plan, root)}
    assert {k: (c["exp_id"], c["pipeline"]["det_exp"], c["pipeline"].get("lsc_exp")) for k, c in rows.items()} == {
        "Baseline": ("B_s", "D2", "L_s"), "+LSC hard": ("L_s", "D2", None), "T": ("T_s", "D2", "L_s")}
    assert fb.run_order(fb.load_rows(plan, root))[0] == 1  # L_s provides lsc for B_s and T_s

    # end to end on the override: new exp dirs only, the original ones are never created
    save_json(load_json(root / "runs/D/det" / DS / "v1.json"), root / "runs/D2/det" / DS / "v1.json")
    plan["rows"] = plan["rows"][:2]
    res = fb.freeze(plan, root, get_logger("freeze_baseline", root / "runs/FZ/log.txt"), device="cpu",
                    skip_profile=True, recognizer_factory=fake_factory)
    assert res["repro"]["ok"] and (root / "runs/L_s/lsc" / DS / "v1.json").is_file()
    assert not (root / "runs/L").exists() and not (root / "runs/B").exists()
    assert load_json(root / "runs/L_s/metrics.json")["meta"]["pipeline_icdar2015v_val"]["det_exp"] == "D2"


def test_freeze_refuses_test_split(root):
    with pytest.raises(PermissionError):
        fb.freeze(plan_of("test"), root, get_logger("freeze_baseline"), recognizer_factory=fake_factory)
