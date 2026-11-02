import importlib.util
import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from logo.utils.io import load_json, save_json

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _tool():
    spec = importlib.util.spec_from_file_location("bootstrap_ci_tool", TOOLS / "bootstrap_ci.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _per_video(rng, n=12, extra_fp=0):
    out = {}
    for i in range(n):
        nobj = int(rng.integers(50, 200))
        miss, fp, sw = int(rng.integers(0, nobj // 3)), int(rng.integers(0, 30)) + extra_fp, int(rng.integers(0, 5))
        idtp = nobj - miss - int(rng.integers(0, 10))
        out[f"v{i}"] = {"frames": 100, "num_objects": nobj, "misses": miss, "fp": fp, "switches": sw,
                        "idtp": idtp, "idfp": fp + 3, "idfn": nobj - idtp}
    return out


def test_score_matches_formulas():
    tool = _tool()
    c = np.array([100, 10, 5, 5, 80, 20, 20])  # num_objects misses fp switches idtp idfp idfn
    s = tool.score(c)
    assert s["MOTA"] == pytest.approx(80.0) and s["IDF1"] == pytest.approx(80.0)
    z = tool.score(np.zeros(7, dtype=np.int64))
    assert z["MOTA"] == 0.0 and z["IDF1"] == 0.0


def test_identical_runs_centered_at_zero():
    tool = _tool()
    _, c = tool.load_counts({"per_video": _per_video(np.random.default_rng(1))})
    res = tool.bootstrap(c, c.copy(), n=1000, seed=7)
    for m in ("MOTA", "IDF1"):
        d = res[m]["diff"]
        assert d["value"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0 and not d["significant"]
        a = res[m]["a"]
        assert a["lo"] < a["value"] < a["hi"] and res[m]["b"] == a
    assert tool.bootstrap(c, c, n=200, seed=7) == tool.bootstrap(c, c, n=200, seed=7)  # seeded


def test_clear_difference_flagged_and_noise_not():
    tool = _tool()
    pa = _per_video(np.random.default_rng(2))
    pb = {v: {**d, "fp": d["fp"] + 40, "idfp": d["idfp"] + 40} for v, d in pa.items()}  # B strictly worse
    _, ca = tool.load_counts({"per_video": pa})
    _, cb = tool.load_counts({"per_video": pb})
    res = tool.bootstrap(ca, cb, n=1000, seed=0)
    assert res["MOTA"]["diff"]["significant"] and res["MOTA"]["diff"]["hi"] < 0
    assert res["IDF1"]["diff"]["significant"] and res["IDF1"]["diff"]["value"] < 0
    # one video slightly better, one slightly worse -> interval straddles 0
    pc = {v: dict(d) for v, d in pa.items()}
    pc["v0"]["misses"] += 3
    pc["v1"]["misses"] = max(0, pc["v1"]["misses"] - 3)
    _, cc = tool.load_counts({"per_video": pc})
    r = tool.bootstrap(ca, cc, n=1000, seed=0)["MOTA"]["diff"]
    assert r["lo"] <= 0 <= r["hi"] and not r["significant"]
    md = tool.to_markdown(res, "A", "B", 1000, 0.05, 12)
    assert "| MOTA |" in md and md.rstrip().endswith("* |")


def test_main(tmp_path, monkeypatch, capsys):
    tool = _tool()
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    pv = _per_video(np.random.default_rng(3), n=5)
    save_json({"per_video": pv}, tmp_path / "runs" / "A" / "metrics.json")
    save_json({"per_video": pv}, tmp_path / "runs" / "B" / "metrics.json")
    save_json({"per_video": {k: pv[k] for k in list(pv)[:4]}}, tmp_path / "runs" / "C" / "metrics.json")
    cfg_p = tmp_path / "abl.yaml"
    cfg_p.write_text(yaml.safe_dump({"exp_id": "abl", "seed": 5}), encoding="utf-8")
    assert tool.main(["A", "B", "--config", str(cfg_p), "--n", "300"]) == 0
    logging.shutdown()
    assert "| IDF1 |" in capsys.readouterr().out
    meta = load_json(tmp_path / "runs" / "abl" / "metrics.json")["meta"]["bootstrap_A_vs_B"]
    assert meta["seed"] == 5 and meta["results"]["MOTA"]["diff"]["value"] == 0.0
    with pytest.raises(SystemExit):
        tool.main(["A", "C", "--config", str(cfg_p)])
    with pytest.raises(ValueError):
        tool.load_counts({"per_video": {}})
