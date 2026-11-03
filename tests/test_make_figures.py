import importlib.util
import json
import sys
from pathlib import Path

import cv2
import motmetrics.lap as mm_lap
import numpy as np

from logo.utils.io import load_json, save_json

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _tool():
    spec = importlib.util.spec_from_file_location("make_figures_tool", TOOLS / "make_figures.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_figures_tool"] = mod
    spec.loader.exec_module(mod)
    return mod


mf = _tool()
DS, V, N, W, H = "icdar2015v", "v1", 4, 160, 96


def rect(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


G1, G2, FP = rect(10, 10, 60, 30), rect(80, 50, 150, 80), rect(100, 5, 140, 25)


def build(root: Path, split="val"):
    gt_frames = {str(f): [{"track_id": 1, "quad": G1, "text": "abc", "dont_care": False, "quality": None},
                          {"track_id": 2, "quad": G2, "text": "hello", "dont_care": False, "quality": None}]
                 for f in range(1, N + 1)}
    save_json({"video": V, "dataset": DS, "width": W, "height": H, "num_frames": N, "frames": gt_frames},
              root / "data" / "processed" / DS / f"{V}.json")
    (root / "splits").mkdir(parents=True)
    (root / "splits" / f"{DS}_{split}.txt").write_text(f"{V}\n", encoding="utf-8")
    fd = root / "data" / "frames" / DS / V
    fd.mkdir(parents=True)
    for f in range(1, N):  # last frame missing -> grey canvas
        cv2.imwrite(str(fd / f"{f:06d}.jpg"), np.full((H, W, 3), 60 * f, np.uint8))
    run = root / "runs" / "FIN"
    fused = {"video": V, "dataset": DS, "frames": {str(f): [
        {"quad": G1, "score": 0.9, "text": "abc", "text_conf": 0.9, "p_eos": 0.1, "lang_hard": 1, "feat_idx": 0,
         "score_fused": 0.9},
        {"quad": FP, "score": 0.6, "text": "", "text_conf": 0.1, "p_eos": 0.9, "lang_hard": 0, "feat_idx": 1,
         "score_fused": 0.35}] for f in range(1, N + 1)}}
    save_json(fused, run / "fuse" / DS / f"{V}.json")
    tracks = []
    for f in range(1, N + 1):
        tracks.append({"frame": f, "track_id": 7, "quad": G1, "score": 0.9, "text": "abx" if f == 2 else "abc"})
        if f != 3:
            tracks.append({"frame": f, "track_id": 8 if f < 3 else 9, "quad": G2, "score": 0.8, "text": "hello"})
    tracks.append({"frame": 1, "track_id": 20, "quad": FP, "score": 0.5, "text": "zz"})
    (run / "vote" / DS).mkdir(parents=True)
    (run / "vote" / DS / f"{V}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tracks), encoding="utf-8")
    (run / "config.yaml").write_text(json.dumps({"exp_id": "FIN", "dataset": DS, "split": split,
                                                 "fusion": {"mode": "soft", "w": 0.5}, "eval": {"iou": 0.5}}),
                                     encoding="utf-8")
    sweep = [{"mode": m, "w": 0.5, "thr": t, "track_MOTA": 40 + 10 * t, "track_IDF1": 50 - t,
              "spot_MOTA": -t, "spot_IDF1": 30 + t} for m in ("hard", "soft") for t in (0.4, 0.6, 0.8)]
    boot = {"a": "runs/A/metrics.json", "b": "runs/FIN/metrics.json", "alpha": 0.05,
            "results": {k: {"a": {"value": 40.0, "lo": 30.0, "hi": 50.0}, "b": {"value": 42.0, "lo": 33.0, "hi": 51.0},
                            "diff": {"value": 2.0, "lo": 0.5, "hi": 3.5, "significant": k == "MOTA"}}
                        for k in ("MOTA", "IDF1")}}
    save_json({"det": {}, "track": {}, "spot": {}, "per_video": {},
               "meta": {f"sweep_fusion_{DS}_{split}": {"table": sweep}, "bootstrap_A_vs_FIN": boot}},
              run / "metrics.json")
    (root / "configs").mkdir()
    cfg = {"exp_id": "paper", "seed": 0, "threshold": {"w": 0.5},
           "figures": {"rescore": {"exp_id": "FIN", "video": None, "frame": None},
                       "overlay": {"exp_id": "FIN", "stage": "vote", "video": None, "frames": 3},
                       "failures": {"exp_id": "FIN", "stage": "vote", "per_cat": 2, "iou": None}}}
    (root / "configs" / "paper.yaml").write_text(json.dumps(cfg), encoding="utf-8")


def _is_pdf(p: str) -> bool:
    return Path(p).read_bytes()[:5] == b"%PDF-"


def test_all_figures(tmp_path, monkeypatch):
    build(tmp_path)
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    monkeypatch.setattr(mm_lap, "default_solver", "scipy")  # "lap" is not installed
    assert mf.main(["--config", "configs/paper.yaml"]) == 0
    figs = tmp_path / "paper" / "figs"
    names = sorted(p.name for p in figs.glob("*.pdf"))
    assert names == sorted(["threshold_FIN_icdar2015v_val.pdf", "ci_runs.pdf", "ci_diff.pdf",
                            "rescore_FIN_v1_f1.pdf", "overlay_FIN_v1.pdf", "failures_FIN_icdar2015v_val.pdf"])
    assert all(_is_pdf(str(figs / n)) for n in names)
    m = load_json(tmp_path / "runs" / "paper" / "metrics.json")["meta"]["figures"]["figures"]
    assert set(m) == {"threshold", "ci", "rescore", "overlay", "failures"}


def test_test_split_needs_final(tmp_path, monkeypatch):
    build(tmp_path, split="test")
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    monkeypatch.setattr(mm_lap, "default_solver", "scipy")
    out = tmp_path / "o1"
    assert mf.main(["--config", "configs/paper.yaml", "--out_dir", str(out),
                    "--only", "threshold", "rescore", "overlay", "failures"]) == 0
    assert not list(out.glob("*.pdf"))
    out = tmp_path / "o2"
    assert mf.main(["--config", "configs/paper.yaml", "--out_dir", str(out), "--final", "--only", "rescore",
                    "overlay"]) == 0
    assert sorted(p.name for p in out.glob("*.pdf")) == ["overlay_FIN_v1.pdf", "rescore_FIN_v1_f1.pdf"]
