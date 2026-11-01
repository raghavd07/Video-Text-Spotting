import logging

import motmetrics.lap as mm_lap
import numpy as np
import pytest
import torch

from logo.data.schema import write_split
from logo.tracking import run_track
from logo.tracking.cache_dataset import LSTCacheDataset, save_cache
from logo.tracking.simple_tracker import cosine_distance, embed_settings, match_embed, run_tracker
from logo.tracking.train_embed import (EmbedNet, attach_embeddings, clip_labels, embed_train_settings,
                                       positive_clips, supcon_loss, train)
from logo.utils.io import load_json, save_json

C, D, NF, DS = 2, 8, 10, "icdar2015v"
VIDEOS = {"train": ["t1", "t2"], "val": ["v1"]}


def box(x, y, w=40, h=12):
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def det(quad, emb, score=0.9):
    return {"quad": [float(v) for v in quad], "score": score, "text": "", "text_conf": 0.0, "p_eos": 0.0,
            "lang_hard": 0, "feat_idx": 0, "score_fused": score, "emb": np.asarray(emb, dtype=np.float32)}


def ids(rows):
    return [(r["frame"], r["track_id"]) for r in rows]


def test_settings_and_cost():
    assert embed_settings({}) == {"lam": 0.5, "emb_gate": 0.0, "emb_momentum": 0.9}
    with pytest.raises(ValueError):
        embed_settings({"tracker": {"lam": 1.5}})
    cos = cosine_distance([[1, 0], [0, 1], [0, 0]], [[1, 0], [-1, 0]])
    np.testing.assert_allclose(cos, [[0, 1], [1, 1], [1, 1]], atol=1e-9)
    iou = np.array([[0.5, 0.45], [0.0, 0.0]])
    cos = np.array([[0.9, 0.0], [0.05, 0.9]])
    assert match_embed(iou, cos, 0.0, 0.3, 0.0) == [(0, 0)]  # lam 0: plain IoU
    assert match_embed(iou, cos, 0.8, 0.3, 0.0) == [(0, 1)]  # appearance flips the choice
    assert sorted(match_embed(iou, cos, 0.8, 0.3, 0.1)) == [(0, 1), (1, 0)]  # emb_gate admits IoU-0 pair


def test_embed_resolves_crossing_and_lam0_equals_simple():
    # two same-size boxes swap position between frames 2 and 3 (IoU favours the wrong swap)
    a, b = [1.0, 0.0], [0.0, 1.0]
    frames = {1: [det(box(0, 0), a), det(box(30, 0), b)], 2: [det(box(10, 0), a), det(box(20, 0), b)],
              3: [det(box(18, 0), b), det(box(12, 0), a)]}
    fused = {"video": "v", "dataset": DS, "frames": {str(k): v for k, v in frames.items()}}
    simple = run_tracker(fused, {"tracker": {"type": "simple", "iou_gate": 0.1}})
    lam0 = run_tracker(fused, {"tracker": {"type": "embed", "lam": 0.0, "iou_gate": 0.1}})
    assert ids(simple) == ids(lam0)
    emb = run_tracker(fused, {"tracker": {"type": "embed", "lam": 0.7, "iou_gate": 0.1}})
    by = {(r["frame"], r["quad"][0]): r["track_id"] for r in emb}
    assert by[(3, 12.0)] == by[(1, 0.0)] and by[(3, 18.0)] == by[(1, 30.0)]
    del frames[1][0]["emb"]
    with pytest.raises(ValueError):
        run_tracker(fused, {"tracker": {"type": "embed"}})


def test_supcon_loss():
    labels, frames = np.array([1, 2, 1, 2, -5]), np.array([0, 0, 1, 1, 1])
    good = torch.tensor([[1.0, 0], [0, 1], [1, 0], [0, 1], [-0.7, -0.7]])
    bad = torch.tensor([[1.0, 0], [0, 1], [0, 1], [1, 0], [-0.7, -0.7]])
    good, bad = (torch.nn.functional.normalize(x, dim=1) for x in (good, bad))
    assert supcon_loss(good, labels, frames, 0.1) < supcon_loss(bad, labels, frames, 0.1)
    assert supcon_loss(good, np.array([1, 2]), np.array([0, 0]), 0.1) is None
    lab, fr = clip_labels([{"track_ids": np.array([3, -1])}, {"track_ids": np.array([-1, 3])}])
    assert list(fr) == [0, 0, 1, 1] and lab[0] == lab[3] == 3 and len(set(lab)) == 3


def _write_video(root, video, seed):
    """GT with 2 moving tracks; cache instances = GT boxes + one false positive per frame."""
    rng = np.random.default_rng(seed)
    protos = rng.normal(size=(3, C, D))
    gt_frames, rows = {}, []
    for f in range(1, NF + 1):
        objs = [(1, 20 + 3 * f, 30), (2, 300, 100 + 2 * f)]
        gt_frames[str(f)] = [{"track_id": t, "quad": box(x, y), "text": "ab", "dont_care": False,
                              "quality": None} for t, x, y in objs]
        rows += [(f, t, box(x, y), protos[t]) for t, x, y in objs] + [(f, -1, box(500, 300), protos[0])]
    save_json({"video": video, "dataset": DS, "width": 640, "height": 360, "num_frames": NF,
               "frames": gt_frames}, root / "data" / "processed" / DS / f"{video}.json")
    fids = np.array([r[0] for r in rows], dtype=np.int32)
    feats = np.stack([r[3] + 0.05 * rng.normal(size=(C, D)) for r in rows]).astype(np.float16)
    save_cache(root / "data" / "cache" / DS / f"{video}.npz", {
        "frame_ids": fids, "quads": np.array([r[2] for r in rows], dtype=np.float32), "feats": feats,
        "score_fused": np.full(len(rows), 0.9, dtype=np.float32),
        "track_ids": np.array([r[1] for r in rows], dtype=np.int64),
        "frames": np.arange(1, NF + 1, dtype=np.int32),
        "frame_offsets": np.concatenate([[0], np.cumsum(np.bincount(fids, minlength=NF + 1)[1:])]).astype(np.int64),
        "img_wh": np.array([640, 360], dtype=np.int32)})
    return rows, feats


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    monkeypatch.setattr(mm_lap, "default_solver", "scipy")  # "lap" is not installed
    for split, vids in VIDEOS.items():
        write_split(tmp_path / "splits" / f"{DS}_{split}.txt", vids)
        for i, v in enumerate(vids):
            _write_video(tmp_path, v, seed=i)
    return tmp_path


def _cfg(iters):
    return {"exp_id": "emb_t", "dataset": DS, "split": "val", "seed": 0, "eval": {"iou": 0.5},
            "tracker": {"type": "embed", "lam": 0.5,
                        "train": {"iters": iters, "lr": 1e-3, "warmup_iters": 1, "clip_len": 3, "emb_dim": 16,
                                  "hidden": 32, "log_every": 2, "val_every": 3, "amp": False}}}


def test_positive_clips(root):
    ds = LSTCacheDataset(root / "data" / "cache" / DS / "t1.npz", clip_len=3)
    assert positive_clips(ds) == list(range(len(ds)))
    with pytest.raises(ValueError):
        embed_train_settings({"tracker": {"train": {"clip_len": 1}}})


def test_train_validate_and_run_track(root):
    res = train(_cfg(6), device="cpu")
    run = root / "runs" / "emb_t"
    assert (run / "config.yaml").is_file() and (run / "log.txt").is_file()
    assert [h["iter"] for h in res["history"]] == [3, 6] and res["iters_done"] == 6
    assert 0.0 <= res["best"]["IDF1"] <= 100.0 and "IDF1" in res["baseline_iou"]
    assert (run / "val_track" / "embed" / DS / "v1.jsonl").is_file()
    assert (run / "val_track" / "iou" / DS / "v1.jsonl").is_file()
    meta = load_json(run / "metrics.json")["meta"]["embed_train"]
    assert meta["positive_clips"] > 0 and meta["delta_IDF1"] == pytest.approx(
        meta["best"]["IDF1"] - meta["baseline_iou"]["IDF1"], abs=1e-3)
    best = torch.load(run / "checkpoints" / "best.pth", map_location="cpu")
    assert set(best) >= {"vpmm", "head", "c", "d", "emb_dim", "hidden", "iter", "val"}

    # run_track with tracker.type embed: fused json + lsc feats npy, feat_idx = cache row
    rows, feats = _write_video(root, "v1", seed=0)
    frames = {}
    for i, (f, _, q, _) in enumerate(rows):
        frames.setdefault(str(f), []).append({"quad": [float(v) for v in q], "score": 0.9, "text": "ab",
                                              "text_conf": 0.8, "p_eos": 0.1, "lang_hard": 1, "feat_idx": i,
                                              "score_fused": 0.9})
    save_json({"video": "v1", "dataset": DS, "frames": frames}, root / "runs/F/fuse" / DS / "v1.json")
    (root / "runs/L/lsc" / DS).mkdir(parents=True)
    np.save(root / "runs/L/lsc" / DS / "v1_feats.npy", feats)
    cfg = {"exp_id": "T", "pipeline": {"lsc_exp": "L"},
           "tracker": {"type": "embed", "lam": 0.5, "ckpt": str(run / "checkpoints" / "best.pth")}}
    stats = run_track.run(cfg, DS, ["v1"], root, logging.getLogger("t"), "F", device="cpu")
    assert stats["boxes"] == len(rows) and stats["lsc_exp"] == "L" and stats["tracks"] >= 3
    out = (root / "runs/T/track" / DS / "v1.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(out) == len(rows) and '"emb"' not in out[0]
    assert "emb" not in load_json(root / "runs/F/fuse" / DS / "v1.json")["frames"]["1"][0]


def test_attach_embeddings_unit_norm():
    net = EmbedNet(C, D, emb_dim=8, hidden=16).eval()
    feats = np.random.default_rng(0).normal(size=(3, C, D)).astype(np.float16)
    fused = {"frames": {"2": [{"quad": box(0, 0), "feat_idx": 2}], "1": [{"quad": box(5, 5), "feat_idx": 0}]}}
    assert attach_embeddings(fused, feats, net, (640, 360), "cpu", False) == 2
    e = fused["frames"]["2"][0]["emb"]
    assert e.shape == (8,) and np.linalg.norm(e) == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(IndexError):
        attach_embeddings({"frames": {"1": [{"quad": box(0, 0), "feat_idx": 9}]}}, feats, net, (640, 360),
                          "cpu", False)
