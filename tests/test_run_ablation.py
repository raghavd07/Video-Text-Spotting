import importlib.util
import logging
from pathlib import Path

import pytest
import yaml

from logo.detector.infer_paddle import write_run_files
from logo.utils.io import load_json, save_json

TOOLS = Path(__file__).resolve().parents[1] / "tools"
DS = "icdar2015v"


def _tool():
    spec = importlib.util.spec_from_file_location("run_ablation_tool", TOOLS / "run_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(exp, lsc_exp=None):
    return {"exp_id": exp, "dataset": DS, "split": "val", "seed": 0,
            "pipeline": {"det_exp": "D", **({"lsc_exp": lsc_exp} if lsc_exp else {})},
            "lsc": {"det_thr": 0.3}, "fusion": {"mode": "hard", "w": 0.5}, "tracker": {"type": "lst", "thr": 0.4,
                                                                                       "train": {"lr": 5.0e-5}},
            "voting": {"enabled": False, "tau": 0.5, "min_len": 3}, "eval": {"iou": 0.5, "spotting": True}}


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    (tmp_path / "configs").mkdir()
    for name, cfg in (("base", _cfg("B")), ("lsc", _cfg("L")), ("lst", _cfg("T", lsc_exp="L"))):
        (tmp_path / "configs" / f"{name}.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return tmp_path


def _plan():
    return {"exp_id": "abl", "dataset": DS, "split": "val", "seed": 0, "exp_suffix": "_x",
            "rows": [{"label": "Baseline", "config": "configs/base.yaml"},
                     {"label": "+LSC hard", "config": "configs/lsc.yaml"},
                     {"label": "+VPMM/LST", "config": "configs/lst.yaml"},
                     {"label": "all", "config": "configs/lst.yaml", "exp_id": "A-all",
                      "set": {"fusion.mode": "soft", "voting.enabled": True, "new.key": 1}}],
            "metrics": ["track.MOTA", "spot.IDF1"]}


def _fake_run_row(calls):
    def run_row(row, root, force, final, device, recognizer_factory=None):
        cfg = row["cfg"]
        calls.append((cfg["exp_id"], force))
        run_dir = root / "runs" / cfg["exp_id"]
        write_run_files(cfg, run_dir, {}, key=f"eval_{DS}_val")
        m = load_json(run_dir / "metrics.json")
        m["track"], m["spot"] = {"MOTA": 10.0 * len(calls)}, {"IDF1": 50.0}
        save_json(m, run_dir / "metrics.json")
        return m
    return run_row


def test_load_rows(root):
    tool = _tool()
    rows = tool.load_rows(_plan(), root)
    assert [r["cfg"]["exp_id"] for r in rows] == ["B_x", "L_x", "T_x", "A-all_x"]
    allr = rows[3]["cfg"]
    assert allr["fusion"] == {"mode": "soft", "w": 0.5} and allr["voting"]["enabled"] is True
    assert allr["new"] == {"key": 1} and allr["pipeline"]["lsc_exp"] == "L_x"
    assert rows[2]["cfg"]["fusion"]["mode"] == "hard"  # set does not leak into the shared config
    bad = _plan()
    del bad["rows"][3]["exp_id"]
    with pytest.raises(ValueError, match="duplicate"):
        tool.load_rows(bad, root)
    bad = _plan()
    bad["rows"][3]["set"] = {"split": "test"}
    with pytest.raises(ValueError):
        tool.load_rows(bad, root)


def test_run_all_skips_finished_and_reruns_stale(root, monkeypatch):
    tool = _tool()
    calls = []
    monkeypatch.setattr(tool.fb, "run_row", _fake_run_row(calls))
    log = logging.getLogger("t")
    plan = _plan()
    res = tool.run_all(plan, tool.load_rows(plan, root), root, log)
    assert [c[0] for c in calls][0] == "L_x"  # lsc provider first
    assert sorted(c[0] for c in calls) == ["A-all_x", "B_x", "L_x", "T_x"]
    assert [r["label"] for r in res] == ["Baseline", "+LSC hard", "+VPMM/LST", "all"]
    assert all(r["status"] == "ran" for r in res)

    calls.clear()
    res = tool.run_all(plan, tool.load_rows(plan, root), root, log)
    assert calls == [] and all(r["status"] == "skipped" for r in res)

    plan["rows"][3]["set"]["voting.tau"] = 0.7  # config changed -> stale -> forced re-run
    rows = tool.load_rows(plan, root)
    res = tool.run_all(plan, rows, root, log)
    assert calls == [("A-all_x", True)] and res[3]["status"] == "forced"
    saved = tool.saved_config(root / "runs" / "A-all_x" / "config.yaml")
    assert saved["voting"]["tau"] == 0.7 and saved["tracker"]["train"]["lr"] == 5.0e-5

    calls.clear()
    tool.run_all(plan, rows, root, log, force=True)
    assert len(calls) == 4 and all(f for _, f in calls)
    md = tool.to_markdown(plan, res)
    assert "| all | `A-all_x` |" in md and md.count("\n") == 8


def test_main(root, monkeypatch):
    tool = _tool()
    monkeypatch.setattr(tool.fb, "run_row", _fake_run_row([]))
    plan_p = root / "configs" / "ablation.yaml"
    plan_p.write_text(yaml.safe_dump(_plan()), encoding="utf-8")
    assert tool.main(["--plan", str(plan_p), "--only", "Baseline", "all", "--out", str(root / "a.md")]) == 0
    logging.shutdown()
    meta = load_json(root / "runs" / "abl" / "metrics.json")["meta"][f"ablation_{DS}_val"]
    assert [r["label"] for r in meta["rows"]] == ["Baseline", "all"] and meta["rows"][0]["spot.IDF1"] == 50.0
    assert (root / "a.md").is_file() and (root / "runs" / "abl" / "config.yaml").is_file()
    with pytest.raises(SystemExit):
        tool.main(["--plan", str(plan_p), "--only", "nope"])
    test_plan = {**_plan(), "split": "test"}
    plan_p.write_text(yaml.safe_dump(test_plan), encoding="utf-8")
    with pytest.raises(SystemExit):
        tool.main(["--plan", str(plan_p)])
