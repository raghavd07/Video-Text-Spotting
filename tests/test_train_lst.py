import motmetrics.lap as mm_lap
import numpy as np
import pytest
import torch

from logo.utils.paths import project_root

if not project_root().joinpath("third_party", "GoMatching", "gomatching", "modeling", "roi_heads",
                               "transformer.py").is_file():
    pytest.skip("third_party/GoMatching not cloned", allow_module_level=True)

from logo.data.schema import write_split  # noqa: E402
from logo.tracking.cache_dataset import save_cache  # noqa: E402
from logo.tracking.lst_adapter import gomatching_transformer_cls  # noqa: E402
from logo.tracking.train_lst import lr_factor, train, train_settings  # noqa: E402
from logo.utils.io import load_json, save_json  # noqa: E402

C, D, NF, DS = 2, 8, 10, "icdar2015v"
VIDEOS = {"train": ["t1", "t2"], "val": ["v1"]}


def _quad(x, y, w=40, h=12):
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def _write_video(root, video, seed):
    """GT with 2 moving tracks + a cache whose instances are the GT boxes (one false positive per frame)."""
    rng = np.random.default_rng(seed)
    protos = rng.normal(size=(3, C, D))
    gt_frames, rows = {}, []
    for f in range(1, NF + 1):
        objs = [(1, 20 + 3 * f, 30), (2, 300, 100 + 2 * f)]
        gt_frames[str(f)] = [{"track_id": t, "quad": _quad(x, y), "text": "ab", "dont_care": False,
                              "quality": None} for t, x, y in objs]
        rows += [(f, t, _quad(x, y), protos[t]) for t, x, y in objs] + [(f, -1, _quad(500, 300), protos[0])]
    save_json({"video": video, "dataset": DS, "width": 640, "height": 360, "num_frames": NF,
               "frames": gt_frames}, root / "data" / "processed" / DS / f"{video}.json")
    fids = np.array([r[0] for r in rows], dtype=np.int32)
    save_cache(root / "data" / "cache" / DS / f"{video}.npz", {
        "frame_ids": fids, "quads": np.array([r[2] for r in rows], dtype=np.float32),
        "feats": np.stack([r[3] + 0.05 * rng.normal(size=(C, D)) for r in rows]).astype(np.float16),
        "score_fused": np.full(len(rows), 0.9, dtype=np.float32),
        "track_ids": np.array([r[1] for r in rows], dtype=np.int64),
        "frames": np.arange(1, NF + 1, dtype=np.int32),
        "frame_offsets": np.concatenate([[0], np.cumsum(np.bincount(fids, minlength=NF + 1)[1:])]).astype(np.int64),
        "img_wh": np.array([640, 360], dtype=np.int32)})


@pytest.fixture
def root(tmp_path, monkeypatch):
    gomatching_transformer_cls()  # cached before LOGO_ROOT moves project_root() away from the repo
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    monkeypatch.setattr(mm_lap, "default_solver", "scipy")  # "lap" is not installed
    for split, vids in VIDEOS.items():
        write_split(tmp_path / "splits" / f"{DS}_{split}.txt", vids)
        for i, v in enumerate(vids):
            _write_video(tmp_path, v, seed=i)
    return tmp_path


def _cfg(iters):
    return {"exp_id": "lst_t", "dataset": DS, "split": "val", "seed": 0, "eval": {"iou": 0.5},
            "tracker": {"type": "lst", "thr": 0.2, "fc_dim": 32, "num_heads": 4, "dropout": 0.0,
                        "train": {"iters": iters, "lr": 1e-3, "warmup_iters": 2, "clip_len": 3,
                                  "log_every": 2, "val_every": 3, "amp": False}}}


def test_lr_schedule():
    tc = train_settings({"tracker": {"train": {"iters": 100, "warmup_iters": 10, "min_lr": 5e-6}}})
    f = [lr_factor(i, tc) for i in range(101)]
    assert f[0] == pytest.approx(tc["warmup_factor"]) and f[10] == pytest.approx(1.0)
    assert all(a < b for a, b in zip(f[:10], f[1:11])) and all(a >= b for a, b in zip(f[10:], f[11:]))
    assert f[100] == pytest.approx(0.1)
    smoke = train_settings({}, smoke=True)
    assert smoke["iters"] == 200 and smoke["val_every"] == 100 and smoke["warmup_iters"] == 20
    with pytest.raises(ValueError):
        train_settings({"tracker": {"train": {"clip_len": 1}}})


def test_train_validate_resume(root):
    res = train(_cfg(6), device="cpu")
    run = root / "runs" / "lst_t"
    assert (run / "config.yaml").is_file() and (run / "log.txt").is_file()
    assert (run / "checkpoints" / "best.pth").is_file() and (run / "checkpoints" / "last.pth").is_file()
    assert [h["iter"] for h in res["history"]] == [3, 6] and res["iters_done"] == 6
    assert 0.0 <= res["best"]["IDF1"] <= 100.0
    assert (run / "val_track" / DS / "v1.jsonl").is_file()
    meta = load_json(run / "metrics.json")["meta"]["lst_train"]
    assert meta["best"]["iter"] in (3, 6) and meta["c"] == C and meta["d"] == D
    best = torch.load(run / "checkpoints" / "best.pth", map_location="cpu")
    assert set(best) >= {"vpmm", "lst", "c", "d", "tracker", "iter", "val"}

    res2 = train(_cfg(9), resume=True, device="cpu")
    assert [h["iter"] for h in res2["history"]] == [3, 6, 9] and res2["iters_done"] == 9
    with pytest.raises(PermissionError):
        cfg = _cfg(3)
        cfg["tracker"]["train"]["val_split"] = "test"
        train(cfg, device="cpu")
