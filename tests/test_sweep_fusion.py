import csv
import importlib.util
import logging
from pathlib import Path

import motmetrics.lap as mm_lap
import pytest

from logo.data.schema import write_split
from logo.utils.io import load_json, save_json

TOOLS = Path(__file__).resolve().parents[1] / "tools"
DS, NF = "icdar2015v", 6


def _tool():
    spec = importlib.util.spec_from_file_location("sweep_fusion_tool", TOOLS / "sweep_fusion.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _quad(x, y, w=40, h=12):
    return [float(v) for v in (x, y, x + w, y, x + w, y + h, x, y + h)]


@pytest.fixture
def root(tmp_path, monkeypatch):
    """GT tracks 1 ("hello", det 0.9, p_eos 0.05) and 2 ("world", det 0.55, p_eos 0.1); one FP per frame
    (det 0.7, lang_hard 1 but p_eos 0.9, text_conf 0.2) that soft fusion pushes below the high threshold."""
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    monkeypatch.setattr(mm_lap, "default_solver", "scipy")
    gt, lsc, n = {}, {}, 0
    for f in range(1, NF + 1):
        gt[str(f)] = [{"track_id": 1, "quad": _quad(20 + 2 * f, 30), "text": "hello", "dont_care": False,
                       "quality": None},
                      {"track_id": 2, "quad": _quad(300, 100), "text": "world", "dont_care": False, "quality": None}]
        dets = [(20 + 2 * f, 30, 0.9, "hello", 0.95, 0.05), (300, 100, 0.55, "world", 0.9, 0.1),
                (500, 300, 0.7, "zz", 0.2, 0.9)]
        lsc[str(f)] = []
        for x, y, s, t, c, pe in dets:
            lsc[str(f)].append({"quad": _quad(x, y), "score": s, "text": t, "text_conf": c, "p_eos": pe,
                                "lang_hard": 1, "feat_idx": n})
            n += 1
    save_json({"video": "v1", "dataset": DS, "width": 640, "height": 360, "num_frames": NF, "frames": gt},
              tmp_path / "data" / "processed" / DS / "v1.json")
    save_json({"video": "v1", "dataset": DS, "frames": lsc}, tmp_path / "runs" / "L" / "lsc" / DS / "v1.json")
    write_split(tmp_path / "splits" / f"{DS}_val.txt", ["v1"])
    write_split(tmp_path / "splits" / f"{DS}_test.txt", ["v1"])
    return tmp_path


def _cfg():
    return {"exp_id": "S", "dataset": DS, "split": "val", "seed": 0, "fusion": {"mode": "hard", "w": 0.5},
            "tracker": {"type": "simple", "thr": 0.5}, "eval": {"iou": 0.5, "spotting": True}}


def test_filter_input():
    tool = _tool()
    fused = {"video": "v", "frames": {"1": [{"score_fused": 0.4}, {"score_fused": 0.5}], "2": []}}
    out = tool.filter_input(fused, 0.5)
    assert out["frames"] == {"1": [{"score_fused": 0.5}], "2": []} and out["video"] == "v"
    assert len(fused["frames"]["1"]) == 2


def test_sweep_simple(root):
    tool = _tool()
    out = root / "out"
    table = tool.sweep(_cfg(), DS, "val", ("hard", "soft"), (0.3, 0.7), (0.4, 0.9), root, out,
                       logging.getLogger("t"), "L")
    assert len(table) == 8 and {(r["mode"], r["w"], r["thr"]) for r in table} == {
        (m, w, t) for m in ("hard", "soft") for w in (0.3, 0.7) for t in (0.4, 0.9)}
    row = {(r["mode"], r["w"], r["thr"]): r for r in table}
    # hard w=0.7: FP fused 0.79 > high_thr 0.6 -> tracked; soft w=0.7: FP 0.3*0.7 + 0.7*0.1 = 0.28 -> dropped
    assert row["hard", 0.7, 0.4]["track_MOTA"] < row["soft", 0.7, 0.4]["track_MOTA"] == pytest.approx(100.0)
    # thr 0.9 keeps only track 1 (soft w=0.3: 0.7*0.9 + 0.3*0.95 = 0.915); track 2 0.655 and the FP are dropped
    assert row["soft", 0.3, 0.9]["track_MOTA"] == pytest.approx(50.0)
    assert row["soft", 0.7, 0.4]["soft_source"] == "p_eos" and row["hard", 0.7, 0.4]["soft_source"] == ""
    assert (out / "soft_w0.70" / "thr_0.40" / "spot" / DS / "v1.jsonl").is_file()

    t2 = tool.sweep(_cfg(), DS, "val", ("soft",), (0.7,), (0.4,), root, root / "out2", logging.getLogger("t"),
                    "L", source="text_conf")
    assert t2[0]["soft_source"] == "text_conf"  # FP text_conf 0.2 -> 0.35, still dropped
    assert t2[0]["track_MOTA"] == pytest.approx(100.0)

    tool.write_csv(table, out / "s.csv")
    with open(out / "s.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8 and "spot_IDF1" in rows[0]
    tool.plot(table, "IDF1", out / "p.png", "t")
    assert (out / "p.png").stat().st_size > 0


def test_refuses_test_split_and_bad_args(root):
    tool = _tool()
    with pytest.raises(PermissionError):
        tool.sweep(_cfg(), DS, "test", ("hard",), (0.5,), (0.5,), root, root / "o", logging.getLogger("t"), "L")
    with pytest.raises(ValueError):
        tool.sweep(_cfg(), DS, "val", ("avg",), (0.5,), (0.5,), root, root / "o", logging.getLogger("t"), "L")
    cfg_p = root / "c.yaml"
    save_json(_cfg(), cfg_p)  # JSON is valid YAML
    with pytest.raises(SystemExit):
        tool.main(["--config", str(cfg_p), "--split", "test"])
    with pytest.raises(SystemExit):
        tool.main(["--config", str(cfg_p), "--ws", "1.5"])
    save_json({**_cfg(), "tracker": {"type": "embed"}}, cfg_p)
    with pytest.raises(SystemExit):
        tool.main(["--config", str(cfg_p)])


def test_main_writes_outputs(root):
    tool = _tool()
    cfg_p = root / "c.yaml"
    save_json(_cfg(), cfg_p)
    assert tool.main(["--config", str(cfg_p), "--lsc_exp", "L", "--ws", "0.5", "--thrs", "0.4", "0.6"]) == 0
    logging.shutdown()
    out = root / "runs" / "S" / "sweep_fusion" / f"{DS}_val"
    for name in ("sweep_fusion.csv", "sweep_fusion_mota.png", "sweep_fusion_idf1.png"):
        assert (out / name).is_file()
    for name in ("config.yaml", "log.txt", "metrics.json"):
        assert (root / "runs" / "S" / name).is_file()
    meta = load_json(root / "runs" / "S" / "metrics.json")["meta"][f"sweep_fusion_{DS}_val"]
    assert len(meta["table"]) == 4 and meta["best_spot"]["mode"] in ("hard", "soft")
