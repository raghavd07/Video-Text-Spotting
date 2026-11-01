import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from logo.tracking.cache_dataset import LSTCacheDataset, collate_first, npz_memmap, save_cache
from logo.utils.io import save_json
from logo.utils.paths import project_root

TOOLS = Path(__file__).resolve().parents[1] / "tools"
C, D = 4, 8


def _load():
    spec = importlib.util.spec_from_file_location("build_lst_cache_tool", TOOLS / "build_lst_cache.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


blc = _load()


def _quad(x, y, w=40, h=12):
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def _synthetic(n_frames=8):
    """Two GT tracks moving right, one dont_care; dets: 2 matches, 1 FP, 1 low score, 1 on dont_care."""
    gt = {"video": "v1", "dataset": "icdar2015v", "width": 320, "height": 240, "num_frames": n_frames,
          "frames": {}}
    fused = {"video": "v1", "dataset": "icdar2015v", "frames": {}}
    k = 0
    for f in range(1, n_frames + 1):
        dx = 3 * f
        gt["frames"][str(f)] = [
            {"track_id": 1, "quad": _quad(10 + dx, 20), "text": "AB", "dont_care": False, "quality": None},
            {"track_id": 7, "quad": _quad(100 + dx, 80), "text": "CD", "dont_care": False, "quality": None},
            {"track_id": 9, "quad": _quad(200, 150), "text": "###", "dont_care": True, "quality": None}]
        if f == 3:  # empty frame
            continue
        dets = [(_quad(101 + dx, 81), 0.9), (_quad(11 + dx, 20), 0.8), (_quad(250, 5), 0.6),
                (_quad(12 + dx, 21), 0.2), (_quad(200, 150), 0.7)]
        fused["frames"][str(f)] = []
        for q, s in dets:
            fused["frames"][str(f)].append({"quad": [float(v) for v in q], "score": s, "text": "x",
                                            "text_conf": 0.5, "p_eos": 0.1, "lang_hard": 1, "feat_idx": k,
                                            "score_fused": s})
            k += 1
    feats = (np.arange(k * C * D, dtype=np.float32).reshape(k, C, D) % 1000).astype(np.float16)
    return fused, gt, feats


def test_label_greedy_highest_iou_first():
    gt = np.asarray([_quad(0, 0)], dtype=np.float64)
    dets = np.asarray([_quad(8, 0), _quad(1, 0), _quad(300, 300)], dtype=np.float32)
    assert blc.label_greedy(dets, gt, [5]).tolist() == [-1, 5, -1]
    assert blc.label_greedy(dets[:0], gt, [5]).shape == (0,)
    assert blc.label_greedy(dets, gt[:0], []).tolist() == [-1, -1, -1]


def test_build_video_arrays():
    fused, gt, feats = _synthetic()
    arr = blc.build_video(fused, gt, feats, score_thr=0.3, iou=0.5)
    assert arr["frames"].tolist() == list(range(1, 9))
    off = arr["frame_offsets"]
    assert off.tolist() == [0, 4, 8, 8, 12, 16, 20, 24, 28]  # low-score det dropped, frame 3 empty
    assert arr["quads"].shape == (28, 8) and arr["feats"].shape == (28, C, D)
    assert arr["feats"].dtype == np.float16 and arr["img_wh"].tolist() == [320, 240]
    assert arr["track_ids"][:4].tolist() == [7, 1, -1, -1]  # FP and dont_care det -> -1
    assert (arr["score_fused"] > 0.3).all()
    # feature rows follow feat_idx: frame-1 kept dets are fused rows 0, 1, 2, 4
    np.testing.assert_array_equal(arr["feats"][:4], feats[[0, 1, 2, 4]])
    assert (arr["frame_ids"][off[3]:off[4]] == 4).all()


def test_build_video_rejects_bad_feat_idx():
    fused, gt, feats = _synthetic()
    with pytest.raises(IndexError):
        blc.build_video(fused, gt, feats[:3])


def test_run_writes_npz_and_dataset_clips(tmp_path):
    fused, gt, feats = _synthetic()
    root = tmp_path
    save_json(fused, root / "runs" / "F" / "fuse" / "icdar2015v" / "v1.json")
    save_json(gt, root / "data" / "processed" / "icdar2015v" / "v1.json")
    (root / "runs" / "L" / "lsc" / "icdar2015v").mkdir(parents=True)
    np.save(root / "runs" / "L" / "lsc" / "icdar2015v" / "v1_feats.npy", feats)
    out_dir = tmp_path / "cache"
    log = blc.get_logger("test_build_lst_cache")
    stats = blc.run("icdar2015v", ["v1"], root, "F", "L", log, out_dir)
    assert stats["videos_built"] == 1 and stats["instances"] == 28 and stats["matched"] == 14
    assert blc.run("icdar2015v", ["v1"], root, "F", "L", log, out_dir)["videos_kept"] == 1

    p = out_dir / "v1.npz"
    with np.load(p) as z:
        full = z["feats"]
    mm = npz_memmap(p, "feats")
    assert isinstance(mm, np.memmap)
    np.testing.assert_array_equal(np.asarray(mm), full)
    del mm

    ds = LSTCacheDataset(out_dir, clip_len=6, stride=1)
    assert len(ds) == 3  # 8 frames -> starts 0, 1, 2
    clip = ds[1]
    assert [f["frame"] for f in clip] == [2, 3, 4, 5, 6, 7]
    assert len(clip[1]["scores"]) == 0 and clip[1]["feats"].shape == (0, C, D)
    f0 = clip[0]
    assert f0["quads"].shape == (4, 8) and f0["feats"].dtype == torch.float16
    assert f0["img_wh"] == (320, 240) and f0["track_ids"].tolist() == [7, 1, -1, -1]
    np.testing.assert_array_equal(f0["feats"].numpy(), full[4:8])
    assert len(LSTCacheDataset(out_dir, clip_len=6, stride=2)) == 2
    assert len(LSTCacheDataset(out_dir, clip_len=9)) == 0
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_first)
    assert len(next(iter(loader))) == 6


def test_clips_never_cross_videos(tmp_path):
    def video(n, tid):
        off = np.arange(n + 1, dtype=np.int64)
        return {"frame_ids": np.arange(1, n + 1, dtype=np.int32),
                "quads": np.tile(np.asarray(_quad(0, 0), np.float32), (n, 1)),
                "feats": np.full((n, C, D), tid, np.float16), "score_fused": np.ones(n, np.float32),
                "track_ids": np.full(n, tid, np.int64), "frames": np.arange(1, n + 1, dtype=np.int32),
                "frame_offsets": off, "img_wh": np.asarray([64, 48], np.int32)}
    save_cache(tmp_path / "a.npz", video(7, 1))
    save_cache(tmp_path / "b.npz", video(6, 2))
    ds = LSTCacheDataset([tmp_path / "a.npz", tmp_path / "b.npz"], clip_len=6)
    assert len(ds) == 3
    for i in range(len(ds)):
        clip = ds[i]
        assert len({f["video"] for f in clip}) == 1
        assert len({int(f["track_ids"][0]) for f in clip}) == 1
    with pytest.raises(ValueError):
        bad = video(3, 1)
        bad["frame_offsets"] = np.asarray([0, 1, 2, 2], np.int64)
        save_cache(tmp_path / "c.npz", bad)


@pytest.mark.skipif(not project_root().joinpath("third_party", "GoMatching", "gomatching", "modeling",
                                                 "roi_heads", "transformer.py").is_file(),
                    reason="third_party/GoMatching not cloned")
def test_clip_trains_adapter_through_vpmm(tmp_path):
    from logo.tracking.lst_adapter import LSTAdapter
    from logo.vpmm.module import VPMM

    fused, gt, feats = _synthetic()
    save_cache(tmp_path / "v1.npz", blc.build_video(fused, gt, feats))
    clip = LSTCacheDataset(tmp_path, clip_len=6)[0]
    torch.manual_seed(0)
    vpmm = VPMM(C, D)
    adapter = LSTAdapter(2 * D, {"tracker": {"fc_dim": 32, "num_heads": 4, "dropout": 0.0}})
    frames = []
    for f in clip:
        n = len(f["scores"])
        wh = torch.tensor([f["img_wh"]], dtype=torch.float32).expand(n, 2)
        q = torch.from_numpy(f["quads"]).view(-1, 4, 2)
        frames.append({**f, "feats": vpmm(q, f["feats"].float(), wh)})
    opt = torch.optim.SGD(list(vpmm.parameters()) + list(adapter.parameters()), lr=1e-3)
    out = adapter.train_step(frames, opt)
    assert torch.isfinite(out["loss"]) and out["loss"] > 0
