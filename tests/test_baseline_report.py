import importlib.util
import logging
from pathlib import Path

import pytest
import yaml

from logo.scoring import run_fuse
from logo.utils.io import get_logger, load_json, save_json

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


br = _load("baseline_report")
NF = 3


def box(x, y, w=40, h=20):
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def _close_logs():
    for name in ("evaluate", "baseline_report"):
        lg = get_logger(name)
        for h in list(lg.handlers):
            h.close()
            lg.removeHandler(h)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Per frame: 2 GT words; dets = both words (0.8, text), one FP (0.7, no text), one low box."""
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    gt_q, fp_q, low_q = [box(10, 10), box(100, 10)], box(10, 100), box(100, 100)
    gt = {"video": "v1", "dataset": "icdar2015v", "width": 320, "height": 240, "num_frames": NF,
          "frames": {str(f): [{"track_id": i + 1, "quad": q, "text": "w", "dont_care": False,
                               "quality": None} for i, q in enumerate(gt_q)] for f in range(1, NF + 1)}}
    save_json(gt, tmp_path / "data/processed/icdar2015v/v1.json")
    det = {"video": "v1", "dataset": "icdar2015v", "frames": {str(f): (
        [{"quad": q, "score": 0.8} for q in gt_q] + [{"quad": fp_q, "score": 0.7},
                                                      {"quad": low_q, "score": 0.25}])
        for f in range(1, NF + 1)}}
    save_json(det, tmp_path / "runs/D/det/icdar2015v/v1.json")
    lsc = {**det, "frames": {f: [{**o, "text": "w" if o["score"] == 0.8 else "", "text_conf": 0.9,
                                  "p_eos": 0.1, "lang_hard": int(o["score"] == 0.8), "feat_idx": 0}
                                 for o in objs if o["score"] > 0.3] for f, objs in det["frames"].items()}}
    save_json(lsc, tmp_path / "runs/E/lsc/icdar2015v/v1.json")
    cfg = {"exp_id": "E", "dataset": "icdar2015v", "split": "val", "seed": 0,
           "lsc": {"det_thr": 0.3}, "fusion": {"mode": "hard", "w": 0.5}, "eval": {"iou": 0.5}}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits/icdar2015v_val.txt").write_text("v1\n", encoding="utf-8")
    run_fuse.run(cfg, "icdar2015v", ["v1"], tmp_path, logging.getLogger("t"), "E")
    yield tmp_path
    _close_logs()


def test_report_rows(root):
    out = root / "docs/baseline_report.md"
    assert br.main(["--config", str(root / "cfg.yaml"), "--det_exp", "D"]) == 0
    rows = load_json(root / "runs/E/metrics.json")["meta"]["baseline_det_val"]["rows"]
    before, f3, f5 = rows
    assert (before["tp"], before["fp"], before["fn"]) == (2 * NF, NF, 0)   # low box dropped by > 0.3
    assert before["precision"] == pytest.approx(200 / 3)
    assert (f3["tp"], f3["fp"]) == (2 * NF, NF)                            # FP fused 0.35 > 0.3
    assert (f5["tp"], f5["fp"], f5["precision"], f5["recall"]) == (2 * NF, 0, 100.0, 100.0)
    md = out.read_text(encoding="utf-8")
    assert md.count("| icdar2015v |") == 3 and "fused score > 0.5" in md
    m = load_json(root / "runs/E/metrics_det_fused_0.5_icdar2015v.json")
    assert m["meta"]["det"]["score_key"] == "score_fused" and m["meta"]["det"]["score_thr"] == 0.5
    assert (root / "runs/E/config.yaml").is_file() and (root / "runs/E/log.txt").is_file()


def test_test_split_needs_final(root):
    with pytest.raises(PermissionError):
        br.main(["--config", str(root / "cfg.yaml"), "--det_exp", "D", "--split", "test"])
