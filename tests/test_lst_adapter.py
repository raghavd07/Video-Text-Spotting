import numpy as np
import pytest
import torch

from logo.utils.paths import project_root

if not project_root().joinpath("third_party", "GoMatching", "gomatching", "modeling", "roi_heads",
                               "transformer.py").is_file():
    pytest.skip("third_party/GoMatching not cloned", allow_module_level=True)

from logo.tracking.lst_adapter import LSTAdapter, box_iou, label_detections, lst_settings, quads_to_boxes  # noqa: E402

IN, FC = 16, 32
CFG = {"tracker": {"type": "lst", "fc_dim": FC, "num_heads": 4, "dropout": 0.0}}


def _model():
    torch.manual_seed(0)
    return LSTAdapter(IN, CFG)


def _quad(x, y, w=40, h=12):
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def _frame(objs, protos, noise=0.05, gen=None):
    """objs: list of (track_id, x, y, score)."""
    feats = torch.stack([protos[t] + noise * torch.randn(IN, generator=gen) for t, *_ in objs]) if objs \
        else torch.zeros(0, IN)
    return {"quads": np.array([_quad(x, y) for _, x, y, _ in objs], dtype=np.float32).reshape(-1, 8),
            "feats": feats, "scores": np.array([s for *_, s in objs]), "img_wh": (640, 360),
            "track_ids": np.array([t for t, *_ in objs], dtype=np.int64)}


def _protos():
    g = torch.Generator().manual_seed(3)
    return {t: torch.randn(IN, generator=g) for t in range(1, 6)} | {0: torch.zeros(IN)}


def test_settings():
    s = lst_settings({"tracker": {"type": "lst", "thr": 0.4}})
    assert s["overlap_thr"] == 0.2 and s["fc_dim"] == 1024 and s["test_len"] == 6  # thr is not overlap_thr
    assert lst_settings({"thr": 0.4, "overlap_thr": 0.1})["overlap_thr"] == 0.1
    with pytest.raises(ValueError):
        lst_settings({"test_len": 1})


def test_geometry_helpers():
    b = quads_to_boxes([[10, 20, 50, 20, 50, 40, 10, 40]], (100, 100))
    assert torch.allclose(b, torch.tensor([[0.1, 0.2, 0.5, 0.4]]))
    iou = box_iou(torch.tensor([[0.0, 0, 1, 1], [2, 2, 3, 3]]), torch.tensor([[0.0, 0, 1, 1], [0.5, 0, 1.5, 1]]))
    assert torch.allclose(iou, torch.tensor([[1.0, 1 / 3], [0.0, 0.0]]))
    ids = label_detections([_quad(0, 0), _quad(200, 0), _quad(400, 0)], [_quad(201, 0), _quad(1, 1)], [7, 9])
    assert ids.tolist() == [9, 7, 0]


def test_state_dict_names_match_gomatching():
    keys = _model().state_dict().keys()
    for k in ("asso_head.fc1.weight", "asso_head.fc2.weight", "long_term_matcher.encoder.layers.0.linear1.weight",
              "short_term_matcher.decoder.layers.0.multihead_attn.in_proj_weight"):
        assert k in keys


def test_train_step_learns_and_reaches_vpmm():
    m, p = _model().train(), _protos()
    g = torch.Generator().manual_seed(1)
    clip = [_frame([(1, 10 + 5 * t, 10, 0.9), (2, 300, 100, 0.8), (0, 500, 300, 0.7)] + ([(3, 100, 200, 0.9)]
                   if t % 2 else []), p, gen=g) for t in range(4)]
    clip[1]["feats"].requires_grad_(True)
    out = m.train_step(clip)
    assert set(out) == {"loss_long_asso", "loss_short_asso", "loss"} and torch.isfinite(out["loss"])
    out["loss"].backward()
    assert clip[1]["feats"].grad is not None and clip[1]["feats"].grad.abs().sum() > 0
    clip[1]["feats"] = clip[1]["feats"].detach()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    first = m.train_step(clip, opt)["loss"].item()
    for _ in range(25):
        last = m.train_step(clip, opt)["loss"].item()
    assert last < first


def test_train_step_edge_cases():
    m, p = _model().train(), _protos()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    before = [q.clone() for q in m.parameters()]
    no_ids = [_frame([(0, 10, 10, 0.9)], p), _frame([], p)]
    assert m.train_step(no_ids, opt)["loss"].item() == 0.0
    assert all(torch.equal(a, b) for a, b in zip(before, m.parameters()))
    low = [_frame([(1, 10, 10, 0.1)], p), _frame([(1, 10, 10, 0.2)], p)]  # all below asso_thr
    assert m.train_step(low)["loss"].item() == 0.0
    dup = [_frame([(1, 10, 10, 0.9), (1, 200, 10, 0.9)], p), _frame([(1, 10, 10, 0.9)], p)]
    with pytest.raises(ValueError):
        m.train_step(dup)


def test_associate_consistent_ids_gap_and_new_track():
    m, p = _model().eval(), _protos()
    rng = np.random.default_rng(0)
    seen = {}
    for t in range(10):
        objs = [(1, 20 + 2 * t, 30, 0.9), (2, 300, 150, 0.8)]
        if t not in (4, 5):  # object 3 vanishes for two frames, inside the 6-frame window
            objs.append((3, 100, 250, 0.9))
        if t >= 7:
            objs.append((4, 500, 50, 0.9))
        objs.append((0, 550, 300, 0.1))  # below asso_thr -> -1
        order = rng.permutation(len(objs))
        objs = [objs[i] for i in order]
        ids = m.associate(_frame(objs, p))
        assert ids.dtype == np.int64 and len(ids) == len(objs)
        for (gt, *_), tid in zip(objs, ids):
            if gt == 0:
                assert tid == -1
            else:
                assert seen.setdefault(gt, tid) == tid
    assert len(set(seen.values())) == 4
    m.reset()
    assert sorted(m.associate(_frame([(1, 20, 30, 0.9), (2, 300, 150, 0.9)], p)).tolist()) == [1, 2]


def test_associate_empty_frames_and_train_mode_guard():
    m, p = _model(), _protos()
    with pytest.raises(RuntimeError):
        m.associate(_frame([(1, 10, 10, 0.9)], p))
    m.eval()
    assert m.associate(_frame([], p)).tolist() == []
    assert m.associate(_frame([(1, 10, 10, 0.9)], p)).tolist() == [1]
    assert m.associate(_frame([], p)).tolist() == []
    assert m.associate(_frame([(1, 10, 10, 0.9)], p)).tolist() == [1]
