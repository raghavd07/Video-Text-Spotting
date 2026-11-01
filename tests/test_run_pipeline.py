import importlib.util
from pathlib import Path

import cv2
import motmetrics.lap as mm_lap
import numpy as np
import pytest

from logo.utils.io import get_logger, load_json, save_json

ROOT = Path(__file__).resolve().parents[1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rp = _load(ROOT / "scripts" / "run_pipeline.py", "run_pipeline")
cr = _load(ROOT / "tools" / "compare_runs.py", "compare_runs")
DS, NF, C, D = "icdar2015v", 4, 3, 4


class FakeRecognizer:
    """Bright crop -> "word", dark crop -> ""."""

    c, d, batch_size = C, D, 4

    def predict(self, crops):
        return [{"text": "word" if cr.mean() > 100 else "", "text_conf": 0.9, "p_eos": 0.1,
                 "feat": np.zeros((C, D), np.float16)} for cr in crops]


def fake_factory(cfg, root, log, device=None):
    return FakeRecognizer()


def box(x, y, w=40, h=16):
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def make_cfg(exp, w, lsc_exp=None, split="val"):
    pc = {"det_exp": "D"}
    if lsc_exp:
        pc["lsc_exp"] = lsc_exp
    return {"exp_id": exp, "dataset": DS, "split": split, "seed": 0, "detector": {}, "pipeline": pc,
            "lsc": {"ckpt": None, "det_thr": 0.3, "iou_neg": 0.05}, "fusion": {"mode": "hard", "w": w},
            "tracker": {"type": "simple", "thr": 0.5}, "voting": {"enabled": False, "tau": 0.5, "min_len": 3},
            "eval": {"iou": 0.5, "spotting": True}}


def _close_logs():
    lg = get_logger("run_pipeline")
    for h in list(lg.handlers):
        h.close()
        lg.removeHandler(h)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Per frame: 2 bright GT words (det 0.8) + 1 dark FP box (det 0.7) + 1 low box (0.2)."""
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    # other tests in the suite can leave motmetrics picking the uninstalled "lap" solver
    monkeypatch.setattr(mm_lap, "default_solver", "scipy")
    gt_q, fp_q, low_q = [box(10, 10), box(100, 10)], box(10, 80), box(100, 80)
    fdir = tmp_path / "data/frames" / DS / "v1"
    fdir.mkdir(parents=True)
    for f in range(1, NF + 1):
        img = np.zeros((120, 160, 3), np.uint8)
        for q in gt_q:
            img[q[1]:q[5], q[0]:q[2]] = 220
        cv2.imwrite(str(fdir / f"{f:06d}.jpg"), img)
    gt = {"video": "v1", "dataset": DS, "width": 160, "height": 120, "num_frames": NF,
          "frames": {str(f): [{"track_id": i + 1, "quad": q, "text": "word", "dont_care": False,
                               "quality": None} for i, q in enumerate(gt_q)] for f in range(1, NF + 1)}}
    save_json(gt, tmp_path / "data/processed" / DS / "v1.json")
    det = {"video": "v1", "dataset": DS, "frames": {str(f): (
        [{"quad": q, "score": 0.8} for q in gt_q] + [{"quad": fp_q, "score": 0.7},
                                                      {"quad": low_q, "score": 0.2}])
        for f in range(1, NF + 1)}}
    save_json(det, tmp_path / "runs/D/det" / DS / "v1.json")
    (tmp_path / "splits").mkdir()
    for s in ("train", "val", "test"):
        (tmp_path / "splits" / f"{DS}_{s}.txt").write_text("v1\n", encoding="utf-8")
    yield tmp_path
    _close_logs()


def run(root, cfg, stages=rp.STAGES, **kw):
    log = get_logger("run_pipeline", root / "runs" / cfg["exp_id"] / "log.txt")
    return rp.run_pipeline(cfg, list(stages), root, log, recognizer_factory=fake_factory, **kw)


def test_full_pipeline_and_skip(root):
    cfg = make_cfg("L", 0.5)
    rep = run(root, cfg)
    assert [rep[s]["status"] for s in rp.STAGES] == ["checked"] + ["ran"] * 5
    for sub in ("lsc/icdar2015v/v1.json", "lsc/icdar2015v/v1_feats.npy", "fuse/icdar2015v/v1.json",
                "track/icdar2015v/v1.jsonl", "vote/icdar2015v/v1.jsonl", "config.yaml", "log.txt"):
        assert (root / "runs/L" / sub).is_file(), sub
    vote = root / "runs/L/vote" / DS / "v1.jsonl"
    assert vote.read_bytes() == (root / "runs/L/track" / DS / "v1.jsonl").read_bytes()
    m = load_json(root / "runs/L/metrics.json")
    assert set(m["det"]) == {"precision", "recall", "f"} and m["det"]["precision"] == pytest.approx(100.0)
    assert set(m["track"]) == {"MOTA", "MOTP", "IDF1", "MT", "ML"} and set(m["spot"]) == set(m["track"])
    assert "v1" in m["per_video"] and "eval_icdar2015v_val" in m["meta"]
    assert m["meta"]["pipeline_icdar2015v_val"]["stages"]["eval"]["status"] == "ran"

    rep = run(root, cfg)
    assert [rep[s]["status"] for s in rp.STAGES] == ["checked"] + ["skipped"] * 5
    rep = run(root, cfg, stages=("fuse", "track", "eval"), force=True)
    assert [rep[s]["status"] for s in ("fuse", "track", "eval")] == ["ran"] * 3


def test_unrequested_middle_stages_run_implicitly(root):
    run(root, make_cfg("L", 0.5), stages=("det", "lsc"))
    rep = run(root, make_cfg("L", 0.5), stages=("fuse", "track", "eval"), force=True)  # no vote files yet
    assert {s: (r["status"], r.get("implicit", False)) for s, r in rep.items()} == {
        "fuse": ("ran", False), "track": ("ran", False), "vote": ("ran", True), "eval": ("ran", False)}
    track, vote = (root / "runs/L" / st / DS / "v1.jsonl" for st in ("track", "vote"))
    assert vote.read_bytes() == track.read_bytes()

    # re-tracking with other fusion weights must not leave eval scoring the old vote files
    old = vote.read_bytes()
    run(root, make_cfg("L", 0.0), stages=("fuse", "track", "eval"), force=True)
    assert vote.read_bytes() == track.read_bytes() != old

    # implicit stages ignore --force: nothing upstream changed, so only eval re-runs
    rep = run(root, make_cfg("L", 0.0), stages=("det", "eval"), force=True)
    assert [rep[s]["status"] for s in rp.STAGES] == ["checked"] + ["skipped"] * 4 + ["ran"]
    assert rp.plan_stages(["eval", "fuse"]) == ["fuse", "track", "vote", "eval"]


def test_upstream_rerun_marks_downstream_dirty(root):
    cfg = make_cfg("L", 0.5)
    run(root, cfg)
    (root / "runs/L/fuse" / DS / "v1.json").unlink()
    rep = run(root, cfg)
    assert [rep[s]["status"] for s in ("lsc", "fuse", "track", "vote", "eval")] == \
           ["skipped", "ran", "ran", "ran", "ran"]


def test_base_reuses_lsc_exp_and_compare(root, capsys):
    run(root, make_cfg("L", 0.5))
    rep = run(root, make_cfg("B", 0.0, lsc_exp="L"))
    assert rep["lsc"]["status"] == "checked" and not (root / "runs/B/lsc").exists()
    fused = load_json(root / "runs/B/fuse" / DS / "v1.json")
    assert all(o["score_fused"] == pytest.approx(o["score"]) for v in fused["frames"].values() for o in v)
    mb, ml = (load_json(root / f"runs/{e}/metrics.json") for e in "BL")
    assert mb["det"]["precision"] < ml["det"]["precision"]  # dark FP box survives without LSC

    capsys.readouterr()
    assert cr.main(["L", "B", "nope", "--metrics", "det.precision", "track.MOTA", "x.y",
                    "--runs_dir", str(root / "runs")]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "| exp_id | det.precision | track.MOTA | x.y |" and len(lines) == 5
    assert lines[2].startswith("| L | 100.00 |") and lines[2].endswith(" - |")
    assert lines[4] == "| nope | - | - | - |"


def test_missing_inputs_and_test_split(root):
    with pytest.raises(FileNotFoundError, match="lsc outputs of exp NOPE"):
        run(root, make_cfg("B", 0.0, lsc_exp="NOPE"), stages=("lsc",))
    with pytest.raises(FileNotFoundError, match="det outputs"):
        run(root, make_cfg("L", 0.5), stages=("det",), det_exp="NOPE")
    with pytest.raises(PermissionError):
        run(root, make_cfg("T", 0.5, split="test"))
    rep = run(root, make_cfg("T", 0.5, split="test"), stages=("det", "lsc"), final=True)
    assert rep["lsc"]["status"] == "ran"


def test_cli_rejects_test_split_and_bad_stage(root, tmp_path):
    import yaml
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(make_cfg("T", 0.5, split="test")), encoding="utf-8")
    with pytest.raises(SystemExit):
        rp.main(["--config", str(p)])
    with pytest.raises(SystemExit):
        rp.main(["--config", str(p), "--final", "--stage", "det,bogus"])
    assert rp.parse_stages("eval,det, fuse") == ["det", "fuse", "eval"]
