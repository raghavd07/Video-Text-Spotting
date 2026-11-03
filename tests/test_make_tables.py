import importlib.util
import json
import sys
from pathlib import Path

import pytest

from logo.utils.io import load_json, save_json

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _tool():
    spec = importlib.util.spec_from_file_location("make_tables_tool", TOOLS / "make_tables.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_tables_tool"] = mod
    spec.loader.exec_module(mod)
    return mod


mt = _tool()


def _metrics(track=None, spot=None, meta=None):
    return {"det": {}, "track": track or {}, "spot": spot or {}, "per_video": {}, "meta": meta or {}}


def _run(root: Path, exp: str, metrics: dict, ds="icdar2015v", split="val"):
    save_json(metrics, root / "runs" / exp / "metrics.json")
    (root / "runs" / exp / "config.yaml").write_text(json.dumps({"exp_id": exp, "dataset": ds, "split": split}),
                                                     encoding="utf-8")


def _eff(stages: dict, det_included: bool):
    total = sum(s["ms_per_frame"] for s in stages.values())
    return {"stages": stages, "e2e": {"total_ms_per_frame": total, "fps": 1000 / total, "params": 4.2e7,
                                      "max_peak_mem_mb": 300.0, "det_included": det_included}}


def build(root: Path):
    rows = [{"label": "Baseline", "exp_id": "B", "det.f": 70.0, "track.MOTA": 40.004, "track.MOTP": 70.0,
             "track.IDF1": 50.0, "spot.MOTA": -6.2, "spot.MOTP": 77.0, "spot.IDF1": 30.0},
            {"label": "+LSC_hard", "exp_id": "L", "det.f": 72.0, "track.MOTA": 39.996, "track.MOTP": 71.0,
             "track.IDF1": 49.0, "spot.MOTA": -4.0, "spot.MOTP": 77.0, "spot.IDF1": 31.0}]
    boot = {"a": "runs/B/metrics.json", "b": "runs/L/metrics.json", "alpha": 0.05,
            "results": {"MOTA": {"a": {"value": 40.0, "lo": 30.0, "hi": 50.0},
                                 "b": {"value": 41.0, "lo": 31.0, "hi": 51.0},
                                 "diff": {"value": 1.0, "lo": 0.2, "hi": 1.8, "significant": True}}}}
    _run(root, "ablation", _metrics(meta={"ablation_icdar2015v_val": {"rows": rows}, "bootstrap_B_vs_L": boot}))
    sweep = [{"mode": m, "w": w, "thr": t, "track_MOTA": 10 * t + (m == "soft"), "track_IDF1": 5.0,
              "spot_MOTA": -t, "spot_IDF1": 3.0 + w} for m in ("hard", "soft") for w in (0.3, 0.5)
             for t in (0.4, 0.6)]
    vote = [{"voted": False, "tau": "", "min_len": "", "tracks": 9, "track_MOTA": 1.0, "track_IDF1": 2.0,
             "spot_MOTA": 3.0, "spot_MOTP": 4.0, "spot_IDF1": 5.0},
            {"voted": True, "tau": 0.5, "min_len": 3, "tracks": 7, "track_MOTA": 1.5, "track_IDF1": 2.0,
             "spot_MOTA": 2.0, "spot_MOTP": 4.0, "spot_IDF1": 6.0}]
    errors = {"stage": "vote", "iou": 0.5,
              "counts": {c: {"<20": 1, "20-50": 2, ">50": 3} for c in mt.ERR_CATS},
              "totals": {c: 6 for c in mt.ERR_CATS}}
    full = _eff({"det": {"ms_per_frame": 20.0}, "lsc": {"ms_per_frame": 30.0}, "track": {"ms_per_frame": 1.0}}, True)
    nodet = _eff({"lsc": {"ms_per_frame": 25.0}, "track": {"ms_per_frame": 1.0}}, False)
    _run(root, "E5", _metrics(
        track={"MOTA": 45.0, "MOTP": 74.0, "IDF1": 52.0, "MT": 40.0, "ML": 30.0},
        spot={"MOTA": -10.0, "MOTP": 77.0, "IDF1": 30.0},
        meta={"sweep_fusion_icdar2015v_val": {"table": sweep, "soft_source": "p_eos", "tracker": "lst"},
              "sweep_voting_icdar2015v_val": {"table": vote}, "errors_icdar2015v_val": errors,
              "efficiency_icdar2015v_val": full}))
    _run(root, "F", _metrics(track={"MOTA": 46.0, "MOTP": 73.0, "IDF1": 53.0, "MT": 41.0, "ML": 39.0},
                             spot={"MOTA": -13.0, "MOTP": 76.0, "IDF1": 32.0},
                             meta={"efficiency_icdar2015v_val": nodet}))
    _run(root, "T", _metrics(track={"MOTA": 99.0}, meta={"efficiency_icdar2015v_test": full}), split="test")
    _run(root, "DETONLY", _metrics(meta={"efficiency_icdar2015v_val": _eff({"det": {"ms_per_frame": 5.0}}, True)}))
    (root / "configs").mkdir()
    (root / "configs" / "rep.yaml").write_text(
        "icdar2015v:\n  - {method: Pub, cite: pub24, source: paper, track: {MOTA: 60.0, ML: null}, spot: {}}\n",
        encoding="utf-8")
    cfg = {"exp_id": "paper", "seed": 0, "threshold": {"w": 0.45},
           "comparison": {"reported": "configs/rep.yaml",
                          "ours": [{"label": "LOGO", "exp_id": "E5"}, {"label": "Final", "exp_id": "F"},
                                   {"label": "Test", "exp_id": "T"}, {"label": "Gone", "exp_id": "NOPE"}]}}
    (root / "configs" / "paper.yaml").write_text(json.dumps(cfg), encoding="utf-8")
    return cfg


def test_format_and_bold_rules():
    assert mt.fmt_value(1.005e6, "mega") == "1.0" and mt.fmt_value(3.7, "int") == "4"
    assert mt.fmt_value(None, "pct") is None and mt.fmt_value(float("nan"), "pct") is None
    t = mt.Table("x", "c", [mt.Col("a", "A", best="max"), mt.Col("b", "B", best="min"), mt.Col("c", "C", best="max")],
                 [{"_label": "r1", "a": 1.004, "b": 2.0, "c": 1.0}, {"_label": "r2", "a": 0.996, "b": 3.0, "c": 1.0},
                  {"_label": "r3", "a": 0.5, "b": 1.0, "c": 1.0, "_nobest": {"b"}}])
    # a: r1, r2 tie at 1.00 after rounding -> both bold; b: min among ranked rows is r1; c: all tie -> none
    assert mt.bold_cells(t) == {(0, "a"), (1, "a"), (0, "b")}
    assert mt.tex_cell("-6.20", True) == r"\textbf{$-$6.20}"
    assert mt.tex_escape("a_b<20 & 5%") == r"a\_b\textless{}20 \& 5\%"
    assert mt.parse_key("sweep_fusion_icdar2015v_val", "sweep_fusion") == ("icdar2015v", "val")
    assert mt.parse_key("sweep_fusion_foo_val", "sweep_fusion") is None
    assert mt.pick_w([{"w": 0.3}, {"w": 0.5}], 0.45) == 0.5


def test_end_to_end(tmp_path, monkeypatch):
    build(tmp_path)
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    assert mt.main(["--config", "configs/paper.yaml"]) == 0
    tex, md = tmp_path / "paper" / "tables", tmp_path / "docs" / "tables"
    names = sorted(p.stem for p in tex.glob("*.tex"))
    assert names == sorted(["ablation_icdar2015v_val", "comparison_icdar2015v", "threshold_E5_icdar2015v_val",
                            "voting_E5_icdar2015v_val", "efficiency_icdar2015v_val", "errors_icdar2015v_val",
                            "errors_bins_E5_icdar2015v_val", "ci"])
    assert sorted(p.stem for p in md.glob("*.md")) == names

    abl = (tex / "ablation_icdar2015v_val.tex").read_text(encoding="utf-8")
    assert r"\toprule" in abl and r"\bottomrule" in abl and r"\cmidrule(lr){3-5}" in abl
    assert r"+LSC\_hard & \textbf{72.00} & 40.00 &" in abl               # 40.004 / 39.996 tie after rounding
    assert r"Baseline & 70.00 & 40.00 &" in abl                          # all-tie columns are not bold
    assert r"\textbf{77.00}" not in abl
    assert r"\textbf{$-$4.00}" in abl

    comp = (tex / "comparison_icdar2015v.tex").read_text(encoding="utf-8")
    assert r"\textit{Reported}" in comp and r"Pub~\cite{pub24} & \textbf{60.00}" in comp
    assert "Test" not in comp and "Gone" not in comp                      # test split skipped, missing run skipped
    assert r"Final [val] & 46.00 & 73.00 & \textbf{53.00} & \textbf{41.00} & 39.00" in comp
    assert r"LOGO [val] & 45.00 & \textbf{74.00} & 52.00 & 40.00 & \textbf{30.00}" in comp  # ML: lower is best

    thr = (md / "threshold_E5_icdar2015v_val.md").read_text(encoding="utf-8")
    assert "hard, w=0.50 Trk MOTA" in thr and "| 0.60 | **6.00** | 5.00 | -0.60 | 3.50 |" in thr   # w=0.45 -> nearest swept w 0.50

    vote = (md / "voting_E5_icdar2015v_val.md").read_text(encoding="utf-8")
    assert "| off | - | - | 9 | 1.00 |" in vote and "| on | 0.50 | 3 | 7 | **1.50** |" in vote

    eff = (md / "efficiency_icdar2015v_val.md").read_text(encoding="utf-8")
    assert "DETONLY" not in eff and "| T" not in eff
    assert "| F\u2020 | - | **25.0** | - | - | 1.0 | 26.0 |" in eff and "| E5 | 20.0 | 30.0 |" in eff
    assert "**25.0**" in eff and "**26.0**" not in eff and "**51.0**" not in eff  # F total not ranked

    ci = (tex / "ci.tex").read_text(encoding="utf-8")
    assert r"B vs L & MOTA & 40.00 & [30.00, 50.00] & \textbf{41.00}" in ci
    assert r"\textbf{1.00} & \textbf{[0.20, 1.80]} & yes" in ci

    m = load_json(tmp_path / "runs" / "paper" / "metrics.json")
    assert len(m["meta"]["tables"]["tables"]) == len(names) and (tmp_path / "runs" / "paper" / "log.txt").is_file()

    assert mt.main(["--config", "configs/paper.yaml", "--final"]) == 0
    eff = (md / "efficiency_icdar2015v_test.md").read_text(encoding="utf-8")
    assert "| T |" in eff


def test_missing_config_is_argparse_error(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    with pytest.raises(SystemExit) as e:
        mt.main(["--config", "configs/none.yaml"])
    assert e.value.code == 2
