import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from logo.utils.io import get_logger, load_json, save_json
from logo.utils.paths import project_root

TOOLS = Path(__file__).resolve().parents[1] / "tools"
DS, NF, C, D = "icdar2015v", 4, 3, 4
HAS_GOMATCHING = project_root().joinpath("third_party", "GoMatching", "gomatching", "modeling", "roi_heads",
                                         "transformer.py").is_file()


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


pe = _load("profile_efficiency")


class FakeRecognizer:
    c, d, batch_size = C, D, 4

    def __init__(self):
        self.model = torch.nn.Linear(3, 2)  # 8 params

    def predict(self, crops):
        return [{"text": "word" if cr.mean() > 100 else "", "text_conf": 0.9, "p_eos": 0.1,
                 "feat": np.ones((C, D), np.float16)} for cr in crops]


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def __call__(self, paths):
        assert len(paths) == 1  # batch 1
        self.calls += 1
        return [np.zeros((0, 10), np.float32)]


def box(x, y, w=40, h=16):
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def cfg_of(exp, tracker=None, split="val"):
    return {"exp_id": exp, "dataset": DS, "split": split, "seed": 0, "detector": {}, "pipeline": {"det_exp": "D"},
            "lsc": {"ckpt": None, "det_thr": 0.3, "iou_neg": 0.05}, "fusion": {"mode": "hard", "w": 0.5},
            "tracker": tracker or {"type": "simple", "thr": 0.5},
            "voting": {"enabled": False, "tau": 0.5, "min_len": 3}, "eval": {"iou": 0.5, "spotting": True}}


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Two videos of NF frames: 2 bright words (0.8) + 1 dark box (0.7) + 1 low box (0.2) per frame."""
    if HAS_GOMATCHING:
        from logo.tracking.lst_adapter import gomatching_transformer_cls
        gomatching_transformer_cls()  # cached before LOGO_ROOT moves project_root()
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    q = [box(10, 10), box(100, 10), box(10, 80), box(100, 80)]
    for v in ("v1", "v2"):
        fdir = tmp_path / "data/frames" / DS / v
        fdir.mkdir(parents=True)
        for f in range(1, NF + 1):
            img = np.zeros((120, 160, 3), np.uint8)
            for b in q[:2]:
                img[b[1]:b[5], b[0]:b[2]] = 220
            cv2.imwrite(str(fdir / f"{f:06d}.jpg"), img)
        save_json({"video": v, "dataset": DS, "width": 160, "height": 120, "num_frames": NF, "frames": {}},
                  tmp_path / "data/processed" / DS / f"{v}.json")
        save_json({"video": v, "dataset": DS, "frames": {str(f): [
            {"quad": q[0], "score": 0.8}, {"quad": q[1], "score": 0.8}, {"quad": q[2], "score": 0.7},
            {"quad": q[3], "score": 0.2}] for f in range(1, NF + 1)}}, tmp_path / "runs/D/det" / DS / f"{v}.json")
    (tmp_path / "splits").mkdir()
    for s in ("train", "val", "test"):
        (tmp_path / "splits" / f"{DS}_{s}.txt").write_text("v1\nv2\n", encoding="utf-8")
    yield tmp_path
    lg = get_logger("profile_efficiency")
    for h in list(lg.handlers):
        h.close()
        lg.removeHandler(h)


def test_take_frames_spans_videos(root):
    seg = pe.take_frames(root, DS, ["v1", "v2"], NF + 2)
    assert [(v, [f for f, _ in fr]) for v, fr in seg] == [("v1", [1, 2, 3, 4]), ("v2", [1, 2])]
    with pytest.raises(ValueError):
        pe.take_frames(root, DS, ["v1", "v2"], 2 * NF + 1)


def test_profile_simple_then_det(root):
    n = NF + 2
    res = pe.profile(cfg_of("L"), root, frames=n, warmup=1, device="cpu", recognizer=FakeRecognizer())
    st = res["stages"]
    assert list(st) == ["lsc", "fuse", "track"]
    assert st["lsc"]["params"] == 8 and st["lsc"]["boxes"] == 3 * n and st["lsc"]["frames"] == n
    assert all(s["ms_per_frame"] >= 0 for s in st.values())
    assert res["e2e"]["det_included"] is False and res["e2e"]["params"] == 8 and res["e2e"]["fps"] > 0
    m = load_json(root / "runs/L/metrics.json")
    assert m["meta"]["efficiency_icdar2015v_val"]["e2e"] == res["e2e"]
    assert (root / "runs/L/config.yaml").is_file() and (root / "runs/L/log.txt").is_file()

    det = FakeDetector()
    rd = pe.profile(cfg_of("D"), root, frames=n, warmup=2, stages=["det"], predictor=det)
    assert det.calls == n + 2 and rd["stages"]["det"]["frames"] == n
    assert (root / "runs/D/profile/efficiency_icdar2015v_val.json").is_file()

    res = pe.profile(cfg_of("L"), root, frames=n, warmup=0, device="cpu", recognizer=FakeRecognizer())
    assert list(res["stages"]) == ["det", "lsc", "fuse", "track"] and res["e2e"]["det_included"]
    total = sum(s["ms_per_frame"] for s in res["stages"].values())
    assert res["e2e"]["total_ms_per_frame"] == pytest.approx(total, abs=1e-2)

    # a det profile on a different frame count is not mixed in
    res = pe.profile(cfg_of("L"), root, frames=n - 1, warmup=0, device="cpu", recognizer=FakeRecognizer())
    assert "det" not in res["stages"]


def test_profile_rules(root):
    with pytest.raises(PermissionError):
        pe.profile(cfg_of("L", split="test"), root, frames=2, device="cpu", recognizer=FakeRecognizer())
    with pytest.raises(ValueError):
        pe.profile(cfg_of("L"), root, frames=2, stages=["det", "lsc"], recognizer=FakeRecognizer())
    with pytest.raises(SystemExit):
        pe.main(["--config", str(TOOLS.parent / "configs" / "E4-lsc.yaml"), "--stages", "nope"])


@pytest.mark.skipif(not HAS_GOMATCHING, reason="third_party/GoMatching not cloned")
def test_profile_lst(root):
    from logo.tracking.lst_adapter import LSTAdapter
    from logo.vpmm.module import VPMM
    torch.manual_seed(0)
    vpmm, lst = VPMM(C, D), LSTAdapter(2 * D, {"tracker": {"type": "lst", "fc_dim": 32, "num_heads": 4,
                                                           "dropout": 0.0}})
    ck = root / "ck" / "best.pth"
    ck.parent.mkdir()
    torch.save({"vpmm": vpmm.state_dict(), "lst": lst.state_dict(), "c": C, "d": D, "tracker": lst.s}, ck)
    cfg = cfg_of("T", {"type": "lst", "thr": 0.4, "traj_thr": 0.6, "ckpt": str(ck)})
    res = pe.profile(cfg, root, frames=NF + 1, warmup=1, device="cpu", recognizer=FakeRecognizer())
    st = res["stages"]
    assert list(st) == ["lsc", "fuse", "vpmm", "track"]
    assert st["vpmm"]["params"] == sum(p.numel() for p in vpmm.parameters())
    assert st["track"]["params"] == sum(p.numel() for p in lst.parameters())
