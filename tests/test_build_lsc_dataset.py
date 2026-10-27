import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from logo.data.schema import write_split
from logo.lsc.dataset import FALLBACK_CHARSET, LSCDataset, augment, clean_label, read_labels
from logo.utils.io import load_json, save_json, set_seed
from logo.utils.paths import frame_path, gt_path, split_file, stage_dir

TOOLS = Path(__file__).resolve().parents[1] / "tools"
GT_Q = [20, 10, 120, 10, 120, 40, 20, 40]
FAR_Q = [10, 60, 60, 60, 60, 85, 10, 85]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_tool", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tool = _load("build_lsc_dataset")


def test_iou_filter_keeps_far_rejects_overlap():
    gts = [GT_Q]
    overlap = [25, 12, 115, 12, 115, 38, 25, 38]
    assert tool.is_negative(FAR_Q, gts, 0.05)
    assert not tool.is_negative(overlap, gts, 0.05)
    dets = [{"quad": FAR_Q, "score": 0.9}, {"quad": overlap, "score": 0.9},
            {"quad": FAR_Q, "score": 0.3}]  # not strictly above det_thr
    assert tool.select_negatives(dets, gts, 0.3, 0.05) == [dets[0]]
    # dont_care GT quads count too
    assert not tool.is_negative(FAR_Q, [GT_Q, FAR_Q], 0.05)
    assert tool.is_negative(FAR_Q, [], 0.05)


def test_clean_label_and_positives():
    assert clean_label("Hello, World-2!", FALLBACK_CHARSET) == "helloworld2"
    objs = [{"quad": GT_Q, "text": "AB", "dont_care": False},
            {"quad": GT_Q, "text": "###", "dont_care": True},
            {"quad": GT_Q, "text": "!!", "dont_care": False}]
    assert tool.select_positives(objs, FALLBACK_CHARSET) == [(GT_Q, "ab")]


def test_balance_is_seeded_and_equal():
    pos, neg = list(range(10)), list(range(100, 104))
    a = tool.balance(pos, neg, np.random.default_rng(0))
    b = tool.balance(pos, neg, np.random.default_rng(0))
    assert a == b and len(a[0]) == len(a[1]) == 4 and a[1] == neg


def _setup(root: Path, n_frames=4):
    for split, vids in (("train", ["v1"]), ("val", ["v2"])):
        write_split(split_file("icdar2015v", split), vids)
        for v in vids:
            gt = {"video": v, "dataset": "icdar2015v", "width": 160, "height": 90,
                  "num_frames": n_frames, "frames": {}}
            det = {"video": v, "dataset": "icdar2015v", "frames": {}}
            for fid in range(1, n_frames + 1):
                img = np.full((90, 160, 3), 200, np.uint8)
                cv2.putText(img, "AB", (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                p = frame_path("icdar2015v", v, fid)
                p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), img)
                gt["frames"][str(fid)] = [
                    {"track_id": 1, "quad": GT_Q, "text": "A-B", "dont_care": False, "quality": None},
                    {"track_id": 2, "quad": [130, 50, 150, 50, 150, 60, 130, 60], "text": "###",
                     "dont_care": True, "quality": None}]
                det["frames"][str(fid)] = [{"quad": GT_Q, "score": 0.95},          # overlaps GT
                                           {"quad": FAR_Q, "score": 0.8},          # negative
                                           {"quad": [5, 5, 15, 5, 15, 9, 5, 9], "score": 0.8},
                                           {"quad": [132, 51, 149, 51, 149, 59, 132, 59],
                                            "score": 0.9}]                         # dont_care hit
            save_json(gt, gt_path("icdar2015v", v))
            save_json(det, stage_dir("d0", "det", "icdar2015v") / f"{v}.json")


def test_build_and_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    _setup(tmp_path)
    cfg = {"exp_id": "e0", "dataset": "icdar2015v", "split": "val", "seed": 3,
           "lsc": {"ckpt": None, "det_thr": 0.3, "iou_neg": 0.05, "stride": 2}}
    cfg_p = tmp_path / "cfg.yaml"
    save_json(cfg, cfg_p)  # JSON is valid YAML
    assert tool.main(["--config", str(cfg_p), "--det_exp", "d0"]) == 0
    with pytest.raises(PermissionError):
        tool.main(["--config", str(cfg_p), "--splits", "test"])

    for split in ("train", "val"):
        st = load_json(tmp_path / "data" / "lsc" / split / "stats.json")
        # frames 1,3: 1 pos/frame, 2 neg/frame -> balanced 2+2
        assert st["candidates"] == {"pos": 2, "neg": 4}
        assert st["kept"] == {"pos": 2, "neg": 2, "total": 4}
        assert st["len_hist"] == {"0": 2, "2": 2}
        assert st["counts"]["neg_overlap"] == 4
        rows = read_labels(tmp_path / "data" / "lsc" / split / "labels.tsv")
        assert sorted(l for _, l in rows) == ["", "", "ab", "ab"]
        assert all((tmp_path / "data" / "lsc" / split / r).is_file() for r, _ in rows)
    m = load_json(tmp_path / "runs" / "e0" / "metrics.json")
    assert {"lsc_dataset_train", "lsc_dataset_val"} <= set(m["meta"])
    assert (tmp_path / "runs" / "e0" / "config.yaml").is_file()
    assert (tmp_path / "runs" / "e0" / "log.txt").is_file()

    ds = LSCDataset(tmp_path / "data" / "lsc" / "train", train=False)
    x, label = ds[0]
    assert x.shape == (3, 32, 128) and x.dtype == torch.float32 and label in ("", "ab")
    set_seed(0)
    a = LSCDataset(ds.root, train=True)[0][0]
    set_seed(0)
    b = LSCDataset(ds.root, train=True)[0][0]
    assert torch.equal(a, b)
    loader = torch.utils.data.DataLoader(LSCDataset(ds.root, train=True), batch_size=4)
    xb, lb = next(iter(loader))
    assert xb.shape == (4, 3, 32, 128) and len(lb) == 4


def test_augment_keeps_shape_dtype():
    img = np.random.default_rng(0).integers(0, 255, (32, 128, 3), dtype=np.uint8)
    rng = np.random.default_rng(1)
    for _ in range(50):
        out = augment(img, rng, p_geo=1.0, p_quality=1.0, p_color=1.0)
        assert out.shape == (32, 128, 3) and out.dtype == np.uint8
