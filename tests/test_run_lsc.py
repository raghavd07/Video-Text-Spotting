import logging

import cv2
import numpy as np
import pytest

from logo.lsc import run_lsc
from logo.utils.io import load_json, save_json

C, D = 3, 4
LOG = logging.getLogger("test_run_lsc")


class FakeRecognizer:
    """Deterministic stand-in for ABINetRecognizer: text depends on the crop's mean intensity."""

    c, d = C, D

    def __init__(self, batch_size=2):
        self.batch_size = batch_size
        self.calls = []

    def predict(self, crops):
        self.calls.append(len(crops))
        out = []
        for cr in crops:
            m = float(cr.mean())
            text = "abc" if m > 100 else ""
            out.append({"text": text, "text_conf": 0.9 if text else 0.0, "p_eos": 0.1 if text else 0.8,
                        "feat": np.full((C, D), m / 255.0, np.float16)})
        return out


def box(x, y, w=40, h=16):
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def make_video(root, ds="icdar2015v", video="v1"):
    """Frame 1: bright box (text) + dark box (no text) + low-score box; frame 2: one dark box."""
    fdir = root / "data" / "frames" / ds / video
    fdir.mkdir(parents=True)
    for fid in (1, 2, 3):
        img = np.zeros((96, 160, 3), np.uint8)
        img[10:26, 10:50] = 220
        cv2.imwrite(str(fdir / f"{fid:06d}.jpg"), img)
    det = {"video": video, "dataset": ds, "frames": {
        "2": [{"quad": box(60, 50), "score": 0.9}],
        "1": [{"quad": box(10, 10), "score": 0.8}, {"quad": box(60, 50), "score": 0.31},
              {"quad": box(100, 60), "score": 0.3}],
        "3": []}}
    save_json(det, root / "runs" / "det_exp" / "det" / ds / f"{video}.json")
    return det


def test_run_video_contract(tmp_path):
    det = make_video(tmp_path)
    rec = FakeRecognizer(batch_size=2)
    fo = tmp_path / "out" / "v1_feats.npy"
    lsc = run_lsc.run_video(det, tmp_path / "data/frames/icdar2015v/v1", rec, 0.3, fo)
    assert list(lsc["frames"]) == ["2", "1", "3"] and lsc["video"] == "v1"
    objs = [o for v in lsc["frames"].values() for o in v]
    assert [o["feat_idx"] for o in objs] == [0, 1, 2]           # file order; score 0.3 is not > 0.3
    assert [o["score"] for o in objs] == [0.9, 0.8, 0.31]
    assert [o["lang_hard"] for o in objs] == [0, 1, 0]
    assert objs[1]["text"] == "abc" and objs[0]["text"] == ""
    for o in objs:
        assert set(o) == {"quad", "score", "text", "text_conf", "p_eos", "lang_hard", "feat_idx"}
    f = np.load(fo)
    assert f.shape == (3, C, D) and f.dtype == np.float16
    assert f[1].mean() > 0.6 and f[0].mean() < 0.1
    assert rec.calls == [2, 1]                                   # batches span frames
    assert not list(fo.parent.glob("*.tmp*"))


def test_batch_size_invariant(tmp_path):
    det = make_video(tmp_path)
    fdir = tmp_path / "data/frames/icdar2015v/v1"
    a = run_lsc.run_video(det, fdir, FakeRecognizer(1), 0.3, tmp_path / "a.npy")
    b = run_lsc.run_video(det, fdir, FakeRecognizer(64), 0.3, tmp_path / "b.npy")
    assert a == b
    np.testing.assert_array_equal(np.load(tmp_path / "a.npy"), np.load(tmp_path / "b.npy"))


def test_empty_video(tmp_path):
    det = {"video": "v0", "dataset": "icdar2015v", "frames": {"1": [{"quad": box(0, 0), "score": 0.1}]}}
    lsc = run_lsc.run_video(det, tmp_path, FakeRecognizer(), 0.3, tmp_path / "v0_feats.npy")
    assert lsc["frames"] == {"1": []}
    assert np.load(tmp_path / "v0_feats.npy").shape == (0, C, D)


def test_run_writes_stage_and_resumes(tmp_path):
    make_video(tmp_path)
    cfg = {"exp_id": "E", "lsc": {"det_thr": 0.3}}
    rec = FakeRecognizer()
    stats = run_lsc.run(cfg, "icdar2015v", ["v1"], rec, tmp_path, LOG, "det_exp")
    out = tmp_path / "runs/E/lsc/icdar2015v"
    assert (out / "v1.json").is_file() and (out / "v1_feats.npy").is_file()
    assert stats["boxes"] == 3 and stats["lang_hard_pos"] == 1
    n = len(rec.calls)
    assert run_lsc.run(cfg, "icdar2015v", ["v1"], rec, tmp_path, LOG, "det_exp")["videos"] == 0
    assert len(rec.calls) == n
    assert load_json(out / "v1.json")["frames"]["1"][0]["feat_idx"] == 1


def test_settings_and_ckpt(tmp_path):
    assert run_lsc.infer_settings({}) == {"batch_size": 64, "amp": True}
    with pytest.raises(ValueError):
        run_lsc.infer_settings({"lsc": {"infer": {"batch_size": 0}}})
    cfg = {"exp_id": "E", "lsc": {"ckpt": "w/init.pth"}}
    assert run_lsc.pick_ckpt(cfg, tmp_path) == tmp_path / "w/init.pth"
    best = tmp_path / "runs/E/checkpoints/best.pth"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"x")
    assert run_lsc.pick_ckpt(cfg, tmp_path) == best
    assert run_lsc.pick_ckpt(cfg, tmp_path, tmp_path / "o.pth") == tmp_path / "o.pth"
    assert run_lsc.pick_ckpt({"exp_id": "F", "lsc": {"ckpt": None}}, tmp_path) is None
