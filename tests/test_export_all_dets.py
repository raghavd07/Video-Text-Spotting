import importlib.util
import logging
from pathlib import Path

import numpy as np
import pytest

from logo.utils.io import load_json

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_tool", ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = _load("export_all_dets")


class FakePredictor:
    """One box per frame with score 0.05 * frame id."""

    model_seconds = 0.0

    def __call__(self, paths):
        return [np.array([[0, 0.05 * int(Path(p).stem), 1, 1, 9, 1, 9, 5, 1, 5]], np.float32)
                for p in paths]


def make_tree(root, ds="icdar2015v"):
    (root / "splits").mkdir(parents=True)
    for split, vids in (("train", ["a"]), ("val", ["b"]), ("test", ["c"])):
        (root / "splits" / f"{ds}_{split}.txt").write_text("\n".join(vids) + "\n", encoding="utf-8")
        for v in vids:
            d = root / "data" / "frames" / ds / v
            d.mkdir(parents=True)
            for i in range(1, 5):
                (d / f"{i:06d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")


def cfg():
    return {"exp_id": "exp", "dataset": "icdar2015v", "split": "val", "seed": 0,
            "detector": {"config": "c.yml", "weights": "w.pdparams", "batch_size": 3, "save_thr": 0.1}}


def test_plan_splits():
    assert ex.plan_splits(None, False) == ["train", "val"]
    assert ex.plan_splits(None, True) == ["train", "val", "test"]
    with pytest.raises(PermissionError):
        ex.plan_splits(["val", "test"], False)


def test_export(tmp_path):
    make_tree(tmp_path)
    log = logging.getLogger("test_export_all_dets")
    res = ex.export(cfg(), ["icdar2015v"], ["train", "val"], FakePredictor(), tmp_path, log)
    assert set(res) == {"icdar2015v_train", "icdar2015v_val"}
    det_dir = tmp_path / "runs" / "exp" / "det"
    a = load_json(det_dir / "icdar2015v" / "a.json")
    # scores 0.05,0.10,0.15,0.20: > 0.1 keeps frames 3, 4 (float32 0.1 is not > 0.1)
    assert [len(a["frames"][str(i)]) for i in range(1, 5)] == [0, 0, 1, 1]
    assert (det_dir / "icdar2015v" / "b.json").is_file()
    assert not (det_dir / "icdar2015v" / "c.json").exists()
    assert load_json(det_dir / "timing_icdar2015v_val.json")["frames"] == 4
    meta = load_json(tmp_path / "runs" / "exp" / "metrics.json")["meta"]
    assert {"det_export", "det_infer_icdar2015v_train", "det_infer_icdar2015v_val"} <= set(meta)
    assert meta["det_export"]["save_thr"] == 0.1

    with pytest.raises(PermissionError):
        ex.export(cfg(), ["icdar2015v"], ["test"], FakePredictor(), tmp_path, log)
    ex.export(cfg(), ["icdar2015v"], ["test"], FakePredictor(), tmp_path, log, final=True)
    assert (det_dir / "icdar2015v" / "c.json").is_file()
