import csv
import importlib.util
import logging
from pathlib import Path

import motmetrics.lap as mm_lap
import pytest

from logo.data.schema import write_split
from logo.tracking.run_track import save_jsonl
from logo.utils.io import load_json, save_json

TOOLS = Path(__file__).resolve().parents[1] / "tools"
DS, NF = "icdar2015v", 8


def _tool():
    spec = importlib.util.spec_from_file_location("sweep_voting_tool", TOOLS / "sweep_voting.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _quad(x, y, w=40, h=12):
    return [float(v) for v in (x, y, x + w, y, x + w, y + h, x, y + h)]


@pytest.fixture
def root(tmp_path, monkeypatch):
    """GT track "hello" in every frame. Track 1 follows it with noisy per-frame texts (5x hello, 3 typos);
    track 2 is a 2-box FP with low text_conf / p_eos-based lang."""
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    monkeypatch.setattr(mm_lap, "default_solver", "scipy")
    texts = ["hello", "hel1o", "Hello", "hallo", "HELLO", "hello", "he11o", "hello"]
    gt, fused, rows = {}, {}, []
    for f in range(1, NF + 1):
        gt[str(f)] = [{"track_id": 1, "quad": _quad(20 + f, 30), "text": "hello", "dont_care": False,
                       "quality": None}]
        fused[str(f)] = [{"quad": _quad(20 + f, 30), "score": 0.9, "text": texts[f - 1], "text_conf": 0.9,
                          "p_eos": 0.05, "lang_hard": 1, "feat_idx": 0, "score_fused": 0.9}]
        rows.append({"frame": f, "track_id": 1, "quad": _quad(20 + f, 30), "score": 0.9, "text": texts[f - 1]})
        if f <= 2:
            fused[str(f)].append({"quad": _quad(400, 200), "score": 0.6, "text": "zq", "text_conf": 0.2,
                                  "p_eos": 0.8, "lang_hard": 1, "feat_idx": 0, "score_fused": 0.6})
            rows.append({"frame": f, "track_id": 2, "quad": _quad(400, 200), "score": 0.6, "text": "zq"})
    save_json({"video": "v1", "dataset": DS, "width": 640, "height": 360, "num_frames": NF, "frames": gt},
              tmp_path / "data" / "processed" / DS / "v1.json")
    save_json({"video": "v1", "dataset": DS, "frames": fused}, tmp_path / "runs" / "S" / "fuse" / DS / "v1.json")
    save_jsonl(rows, tmp_path / "runs" / "S" / "track" / DS / "v1.jsonl")
    write_split(tmp_path / "splits" / f"{DS}_val.txt", ["v1"])
    write_split(tmp_path / "splits" / f"{DS}_test.txt", ["v1"])
    return tmp_path


def _cfg():
    return {"exp_id": "S", "dataset": DS, "split": "val", "seed": 0,
            "fusion": {"mode": "soft", "w": 0.5, "soft_source": "p_eos"}, "tracker": {"type": "simple", "thr": 0.5},
            "voting": {"enabled": True, "tau": 0.5, "min_len": 3}, "eval": {"iou": 0.5, "spotting": True}}


def test_sweep_voting(root):
    tool = _tool()
    table = tool.sweep(_cfg(), DS, "val", (0.0, 0.5), (1, 3), root, root / "out", logging.getLogger("t"), "S")
    assert len(table) == 5 and table[0]["voted"] is False
    row = {(r["tau"], r["min_len"]): r for r in table[1:]}
    off = table[0]
    assert off["tracks"] == 2 and row[0.0, 1]["tracks"] == 2
    assert row[0.5, 1]["tracks"] == 1 and row[0.0, 3]["tracks"] == 1  # FP lang 0.2 < 0.5; FP length 2 < 3
    # off: 3 typo frames = 3 misses + 3 FP, plus 2 FP boxes -> spot MOTA 0; voting fixes the typos (75),
    # dropping the FP track gives 100
    assert off["spot_MOTA"] < row[0.0, 1]["spot_MOTA"] < row[0.5, 3]["spot_MOTA"] == pytest.approx(100.0)
    assert off["track_MOTA"] == pytest.approx(75.0) and row[0.5, 3]["track_MOTA"] == pytest.approx(100.0)
    assert (root / "out" / "tau_0.50" / "min_len_3" / DS / "v1.jsonl").is_file()
    tool.write_csv(table, root / "out" / "s.csv")
    with open(root / "out" / "s.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 5


def test_refuses_test_split(root):
    tool = _tool()
    with pytest.raises(PermissionError):
        tool.sweep(_cfg(), DS, "test", (0.5,), (3,), root, root / "o", logging.getLogger("t"), "S")
    cfg_p = root / "c.yaml"
    save_json(_cfg(), cfg_p)
    with pytest.raises(SystemExit):
        tool.main(["--config", str(cfg_p), "--split", "test"])
    with pytest.raises(SystemExit):
        tool.main(["--config", str(cfg_p), "--min_lens", "-1"])


def test_main_writes_outputs(root):
    tool = _tool()
    cfg_p = root / "c.yaml"
    save_json(_cfg(), cfg_p)
    assert tool.main(["--config", str(cfg_p), "--taus", "0.5", "--min_lens", "1", "3"]) == 0
    logging.shutdown()
    assert (root / "runs" / "S" / "sweep_voting" / f"{DS}_val" / "sweep_voting.csv").is_file()
    for name in ("config.yaml", "log.txt", "metrics.json"):
        assert (root / "runs" / "S" / name).is_file()
    meta = load_json(root / "runs" / "S" / "metrics.json")["meta"][f"sweep_voting_{DS}_val"]
    assert len(meta["table"]) == 3 and meta["best_spot"]["tau"] == 0.5 and meta["off"]["spot_MOTA"] == 0.0
