import csv
import importlib.util
import json
import logging
from pathlib import Path

import motmetrics.lap as mm_lap
import numpy as np
import pytest
import torch
import yaml

from logo.data.schema import write_split
from logo.tracking import run_track
from logo.tracking.run_track import filter_trajectories, input_thr, traj_thr
from logo.utils.io import load_json, save_json
from logo.utils.paths import project_root

HAS_GOMATCHING = project_root().joinpath("third_party", "GoMatching", "gomatching", "modeling", "roi_heads",
                                         "transformer.py").is_file()
needs_gm = pytest.mark.skipif(not HAS_GOMATCHING, reason="third_party/GoMatching not cloned")
TOOLS = Path(__file__).resolve().parents[1] / "tools"
C, D, NF, DS = 2, 8, 8, "icdar2015v"


def _sweep_tool():
    spec = importlib.util.spec_from_file_location("sweep_threshold_tool", TOOLS / "sweep_threshold.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _quad(x, y, w=40, h=12):
    return [float(v) for v in (x, y, x + w, y, x + w, y + h, x, y + h)]


def _row(f, tid, score, text):
    return {"frame": f, "track_id": tid, "quad": _quad(0, 0), "score": score, "text": text}


def test_thresholds_and_filter():
    assert input_thr({"tracker": {}}, "icdar2015v") == 0.4 and input_thr({"tracker": {}}, "icdar2013v") == 0.4
    assert input_thr({"tracker": {}}, "dstext") == 0.3 and input_thr({"tracker": {"thr": 0.7}}, "dstext") == 0.7
    assert traj_thr({"tracker": {}}) == 0.6 and traj_thr({"tracker": {"traj_thr": 0.5}}) == 0.5
    rows = ([_row(f, 1, 0.9, "hello") for f in (1, 2)] + [_row(f, 2, 0.5, "low") for f in (1, 2)]
            + [_row(1, 3, 0.9, "a"), _row(2, 3, 0.9, "a"), _row(3, 3, 0.9, "ab")]
            + [_row(1, 4, 0.8, " "), _row(1, 5, 0.65, "ok"), _row(2, 5, 0.6, "ok")])
    assert {r["track_id"] for r in filter_trajectories(rows, 0.6, False)} == {1, 3, 4, 5}
    assert {r["track_id"] for r in filter_trajectories(rows, 0.6, True)} == {1, 5}  # 3: "a" most frequent
    assert filter_trajectories([], 0.6, True) == []


def test_sweep_refuses_test_split():
    tool = _sweep_tool()
    with pytest.raises(SystemExit):
        tool.main(["--config", str(TOOLS.parent / "configs" / "E5-lst.yaml"), "--split", "test"])
    with pytest.raises(PermissionError):
        tool.sweep({}, DS, "test", [0.5], Path("."), Path("."), logging.getLogger("t"), "F", "L", "cpu")


@pytest.fixture
def root(tmp_path, monkeypatch):
    from logo.tracking.lst_adapter import LSTAdapter, gomatching_transformer_cls
    from logo.vpmm.module import VPMM
    gomatching_transformer_cls()  # cached before LOGO_ROOT moves project_root() away from the repo
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    monkeypatch.setattr(mm_lap, "default_solver", "scipy")
    torch.manual_seed(0)
    lst_cfg = {"type": "lst", "fc_dim": 32, "num_heads": 4, "dropout": 0.0}
    vpmm, lst = VPMM(C, D), LSTAdapter(2 * D, {"tracker": lst_cfg})
    (tmp_path / "ck").mkdir()
    torch.save({"vpmm": vpmm.state_dict(), "lst": lst.state_dict(), "c": C, "d": D, "tracker": lst.s, "iter": 0,
                "val": None}, tmp_path / "ck" / "best.pth")
    # GT: track 1 moves right, track 2 static ("x" text), track 3 static low score; one 0.35 FP per frame
    gt, fused, n = {}, {}, 0
    rng = np.random.default_rng(0)
    for f in range(1, NF + 1):
        objs = [(1, 20 + 3 * f, 30, 0.9, "hello"), (2, 300, 100, 0.95, "x"), (3, 100, 250, 0.5, "low"),
                (0, 500, 300, 0.35, "fp")]
        gt[str(f)] = [{"track_id": t, "quad": _quad(x, y), "text": s, "dont_care": False, "quality": None}
                      for t, x, y, _, s in objs if t]
        fused[str(f)] = []
        for t, x, y, sc, s in objs:
            fused[str(f)].append({"quad": _quad(x, y), "score": sc, "text": s, "text_conf": 0.9, "p_eos": 0.1,
                                  "lang_hard": 1, "feat_idx": n, "score_fused": sc})
            n += 1
    fused["9"] = []  # empty frame
    save_json({"video": "v1", "dataset": DS, "width": 640, "height": 360, "num_frames": NF + 1, "frames": gt},
              tmp_path / "data" / "processed" / DS / "v1.json")
    save_json({"video": "v1", "dataset": DS, "frames": fused}, tmp_path / "runs/F/fuse" / DS / "v1.json")
    (tmp_path / "runs/L/lsc" / DS).mkdir(parents=True)
    np.save(tmp_path / "runs/L/lsc" / DS / "v1_feats.npy", rng.normal(size=(n, C, D)).astype(np.float16))
    write_split(tmp_path / "splits" / f"{DS}_val.txt", ["v1"])
    return tmp_path


def _cfg(root, **tracker):
    return {"exp_id": "T", "dataset": DS, "split": "val", "seed": 0, "pipeline": {"lsc_exp": "L"},
            "eval": {"iou": 0.5, "spotting": True},
            "tracker": {"type": "lst", "ckpt": str(root / "ck" / "best.pth"), **tracker}}


@needs_gm
def test_run_track_lst(root):
    stats = run_track.run(_cfg(root), DS, ["v1"], root, logging.getLogger("t"), "F", device="cpu")
    assert stats["thr"] == 0.4 and stats["traj_thr"] == 0.6 and stats["text_filter"] and stats["lst_ckpt"]
    rows = [json.loads(s) for s in (root / "runs/T/track" / DS / "v1.jsonl").read_text("utf-8").splitlines()]
    assert set(rows[0]) == {"frame", "track_id", "quad", "score", "text"}
    # FP (0.35 < thr) dropped; track 3 (mean 0.5 < traj_thr) removed; track 2 ("x") removed in spotting mode
    assert {r["text"] for r in rows} == {"hello"} and len(rows) == NF
    assert len({r["track_id"] for r in rows}) == 1 and [r["frame"] for r in rows] == list(range(1, NF + 1))

    cfg = _cfg(root, thr=0.3, traj_thr=0.3)
    cfg["eval"]["spotting"] = False
    run_track.run(cfg, DS, ["v1"], root, logging.getLogger("t"), "F", device="cpu")
    rows = [json.loads(s) for s in (root / "runs/T/track" / DS / "v1.jsonl").read_text("utf-8").splitlines()]
    assert {r["text"] for r in rows} == {"hello", "x", "low", "fp"} and len(rows) == 4 * NF


@needs_gm
def test_sweep_threshold(root):
    tool = _sweep_tool()
    cfg_path = root / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg(root)), encoding="utf-8")
    assert tool.main(["--config", str(cfg_path), "--fuse_exp", "F", "--thrs", "0.3", "0.6", "0.92",
                      "--device", "cpu"]) == 0
    out = root / "runs/T/sweep_thr" / f"{DS}_val"
    with open(out / "sweep.csv", encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    assert [float(r["thr"]) for r in table] == [0.3, 0.6, 0.92]
    assert {"track_MOTA", "track_IDF1", "spot_MOTA", "spot_IDF1"} <= set(table[0])
    assert (out / "sweep.png").stat().st_size > 0
    assert float(table[2]["track_MOTA"]) < float(table[0]["track_MOTA"])  # 0.92 keeps only track 2
    assert float(table[0]["spot_IDF1"]) > 0
    meta = load_json(root / "runs/T/metrics.json")["meta"][f"sweep_thr_{DS}_val"]
    assert len(meta["table"]) == 3 and (root / "runs/T/log.txt").is_file()
    assert (root / "runs/T/config.yaml").is_file()
