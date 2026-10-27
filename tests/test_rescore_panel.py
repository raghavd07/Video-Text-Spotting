import importlib.util
from pathlib import Path

import cv2
import numpy as np

from logo.utils.io import save_json

_spec = importlib.util.spec_from_file_location(
    "rescore_panel", Path(__file__).resolve().parents[1] / "tools" / "rescore_panel.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

H, W = 60, 100


def objs():
    return [{"quad": [5, 5, 45, 5, 45, 20, 5, 20], "score": 0.9, "score_fused": 0.95},
            {"quad": [50, 30, 90, 30, 90, 50, 50, 50], "score": 0.4, "score_fused": 0.2}]


def test_score_color_ramp():
    assert rp.score_color(0.0) == (255, 0, 0)   # BGR blue
    assert rp.score_color(1.0) == (0, 0, 255)   # BGR red
    assert rp.score_color(2.0) == rp.score_color(1.0) and rp.score_color(-1) == rp.score_color(0)


def test_panel_layout_and_colours():
    img = np.zeros((H, W, 3), np.uint8)
    p = rp.make_panel(img, objs())
    assert p.shape == (rp.TITLE_H + H + rp.BAR_H, 2 * W + rp.GAP, 3) and p.dtype == np.uint8
    left = p[rp.TITLE_H:rp.TITLE_H + H, :W]
    right = p[rp.TITLE_H:rp.TITLE_H + H, W + rp.GAP:]
    # second box edge (x=50..90, y=50): det 0.4 -> more blue than red; fused 0.2 -> even bluer
    l, r = left[50, 70].astype(int), right[50, 70].astype(int)
    assert l[0] > l[2] and r[0] - r[2] > l[0] - l[2]
    hidden = rp.draw_scores(img, objs(), "score", hide_below=0.5)
    b, g, r = (int(v) for v in hidden[50, 70])
    assert b == g == r > 0                       # grey (anti-aliased) for hidden boxes


def test_main_writes_png(tmp_path):
    fdir = tmp_path / "frames"
    fdir.mkdir()
    cv2.imwrite(str(fdir / "000002.jpg"), np.full((H, W, 3), 128, np.uint8))
    fused = {"video": "vX", "dataset": "icdar2015v", "frames": {"1": [], "2": objs()}}
    save_json(fused, tmp_path / "vX.json")
    out = tmp_path / "figs" / "panel.png"
    assert rp.main(["--fused", str(tmp_path / "vX.json"), "--out", str(out), "--frames_dir", str(fdir)]) == 0
    img = cv2.imread(str(out))
    assert img.shape == (rp.TITLE_H + H + rp.BAR_H, 2 * W + rp.GAP, 3)
