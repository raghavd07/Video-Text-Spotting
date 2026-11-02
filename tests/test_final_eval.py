import importlib.util
from pathlib import Path

import cv2
import motmetrics.lap as mm_lap
import numpy as np
import pytest
import yaml

from logo.utils.io import get_logger, load_json, save_json

ROOT = Path(__file__).resolve().parents[1]
NF, C, D = 4, 3, 4
DSS = ("icdar2015v", "icdar2013v")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fe = _load(ROOT / "scripts" / "final_eval.py", "final_eval")


class FakeRecognizer:
    c, d, batch_size = C, D, 4

    def predict(self, crops):
        return [{"text": "word" if cr.mean() > 100 else "", "text_conf": 0.9, "p_eos": 0.1,
                 "feat": np.zeros((C, D), np.float16)} for cr in crops]


def fake_factory(cfg, root, log, device=None):
    return FakeRecognizer()


def box(x, y, w=40, h=16):
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def best_cfg():
    return {"exp_id": "BEST", "dataset": "icdar2015v", "split": "val", "seed": 7, "detector": {},
            "pipeline": {"det_exp": "D", "lsc_exp": "L"},
            "lsc": {"ckpt": "ck/lsc.pth", "det_thr": 0.3, "iou_neg": 0.05}, "fusion": {"mode": "hard", "w": 0.5},
            "tracker": {"type": "simple", "thr": 0.5}, "voting": {"enabled": True, "tau": 0.5, "min_len": 2},
            "eval": {"iou": 0.5, "spotting": True}}


DECISIONS = """# Decisions

|Date|Task|Decision|Why|Impact|
|-|-|-|-|-|
|2026-11-02|P28|held-out icdar2015v: val|first guess|none|
|2026-11-02|P28|held-out icdar2015v: test|labels available|none|
|2026-11-02|P28|held-out icdar2013v: val|test labels unavailable|val also used for tuning|
"""


@pytest.fixture
def root(tmp_path, monkeypatch):
    """icdar2015v test video t1 and icdar2013v val video v1: 2 bright GT words + 1 dark FP box per frame."""
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    monkeypatch.setattr(mm_lap, "default_solver", "scipy")
    gt_q, fp_q = [box(10, 10), box(100, 10)], box(10, 80)
    (tmp_path / "splits").mkdir()
    for ds, split, v in (("icdar2015v", "test", "t1"), ("icdar2013v", "val", "v1")):
        fdir = tmp_path / "data/frames" / ds / v
        fdir.mkdir(parents=True)
        for f in range(1, NF + 1):
            img = np.zeros((120, 160, 3), np.uint8)
            for q in gt_q:
                img[q[1]:q[5], q[0]:q[2]] = 220
            cv2.imwrite(str(fdir / f"{f:06d}.jpg"), img)
        save_json({"video": v, "dataset": ds, "width": 160, "height": 120, "num_frames": NF,
                   "frames": {str(f): [{"track_id": i + 1, "quad": q, "text": "word", "dont_care": False,
                                        "quality": None} for i, q in enumerate(gt_q)] for f in range(1, NF + 1)}},
                  tmp_path / "data/processed" / ds / f"{v}.json")
        save_json({"video": v, "dataset": ds, "frames": {str(f): [{"quad": q, "score": 0.8} for q in gt_q]
                                                         + [{"quad": fp_q, "score": 0.7}] for f in range(1, NF + 1)}},
                  tmp_path / "runs/D/det" / ds / f"{v}.json")
        for s in ("train", "val", "test"):
            (tmp_path / "splits" / f"{ds}_{s}.txt").write_text(f"{v}\n" if s == split else "x\n", encoding="utf-8")
    for s in ("train", "val", "test"):
        (tmp_path / "splits" / f"dstext_{s}.txt").write_text("d1\n", encoding="utf-8")
    (tmp_path / "ck").mkdir()
    (tmp_path / "ck/lsc.pth").write_bytes(b"0")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/decisions.md").write_text(DECISIONS, encoding="utf-8")
    (tmp_path / "best.yaml").write_text(yaml.safe_dump(best_cfg()), encoding="utf-8")
    yield tmp_path
    for name in ("run_pipeline", "final_eval", "profile_efficiency", "evaluate"):
        lg = get_logger(name)
        for h in list(lg.handlers):
            h.close()
            lg.removeHandler(h)


def test_helpers(root):
    assert fe.held_out_splits(root / "docs/decisions.md") == {"icdar2015v": "test", "icdar2013v": "val"}
    best = best_cfg()
    cfg = fe.final_config(best, "icdar2013v", "val", root)
    assert fe.changed_keys(best, cfg) == ["dataset", "exp_id", "pipeline.lsc_exp"]
    assert cfg["exp_id"] == "FINAL-icdar2013v" and cfg["pipeline"] == {"det_exp": "D", "lsc_exp": "FINAL-icdar2013v"}
    assert fe.config_hash(best) == fe.config_hash(best_cfg()) != fe.config_hash(cfg)
    # lsc.ckpt null: the source exp's own checkpoint would silently be replaced by the ABINet default
    nul = {**best, "lsc": {**best["lsc"], "ckpt": None}}
    (root / "runs/L/checkpoints").mkdir(parents=True)
    (root / "runs/L/checkpoints/best.pth").write_bytes(b"0")
    with pytest.raises(ValueError, match="resolve differently"):
        fe.final_config(nul, "icdar2013v", "val", root)


def test_refusals(root):
    with pytest.raises(SystemExit):
        fe.main(["--config", str(root / "best.yaml")])
    with pytest.raises(SystemExit):  # missing config: argparse error, not a traceback
        fe.main(["--config", str(root / "configs/FINAL.yaml"), "--final"])
    with pytest.raises(PermissionError):
        fe.final_eval(best_cfg(), "best.yaml", root, final=False)
    (root / "data/processed/icdar2015v/t1.json").unlink()  # test labels unavailable
    with pytest.raises(FileNotFoundError, match="held-out icdar2015v: val"):
        fe.final_eval(best_cfg(), "best.yaml", root, final=True, skip_profile=True,
                      recognizer_factory=fake_factory)
    assert not (root / "runs/FINAL-icdar2013v").exists()  # preflight fails before anything runs
    (root / "docs/decisions.md").write_text("# none\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no 'held-out icdar2013v"):
        fe.final_eval(best_cfg(), "best.yaml", root, final=True, datasets=["icdar2013v"])


def test_final_eval_end_to_end(root):
    res = fe.final_eval(best_cfg(), "best.yaml", root, final=True, device="cpu", frames=3, warmup=1,
                        recognizer_factory=fake_factory, profile_kw={"recognizer": FakeRecognizer()})
    assert set(res["results"]) == set(DSS) and "dstext" in res["skipped"]
    for ds, split in (("icdar2015v", "test"), ("icdar2013v", "val")):
        d = root / "runs" / f"FINAL-{ds}"
        m = load_json(d / "metrics.json")
        assert m["det"]["f"] > 0 and m["track"]["MOTA"] > 0 and m["spot"]["IDF1"] > 0
        meta = m["meta"]["final"]
        assert meta["split"] == split and meta["seed"] == 7 and meta["gpu"] == "cpu"
        assert meta["config_hash"] == fe.config_hash(best_cfg()) and "git_commit" in meta
        assert f"eval_{ds}_{split}" in m["meta"]
        assert (d / "lsc" / ds / "t1.json" if split == "test" else d / "lsc" / ds / "v1.json").is_file()
        assert (d / "config.yaml").is_file() and (d / "log.txt").is_file()
        saved = fe.ra.saved_config(d / "config.yaml")
        assert saved["fusion"] == best_cfg()["fusion"] and saved["voting"] == best_cfg()["voting"]
    eff = load_json(root / "runs/FINAL-efficiency.json")
    assert set(eff["datasets"]) == set(DSS) and eff["meta"]["seed"] == 7
    e = eff["datasets"]["icdar2015v"]
    assert e["split"] == "test" and set(e["stages"]) == {"lsc", "fuse", "track"}
    assert set(e["stages"]["lsc"]) == {"params", "peak_mem_mb", "ms_per_frame"} and e["e2e"]["fps"] > 0

    # a new best config is refused while the old FINAL runs exist, unless --force
    other = {**best_cfg(), "fusion": {"mode": "hard", "w": 0.3}}
    with pytest.raises(ValueError, match="--force"):
        fe.final_eval(other, "best.yaml", root, final=True, skip_profile=True, recognizer_factory=fake_factory)
    fe.final_eval(other, "best.yaml", root, final=True, force=True, skip_profile=True, device="cpu",
                  datasets=["icdar2013v"], recognizer_factory=fake_factory)
    assert fe.ra.saved_config(root / "runs/FINAL-icdar2013v/config.yaml")["fusion"]["w"] == 0.3
