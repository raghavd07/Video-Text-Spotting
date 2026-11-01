import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from logo.glyph.pseudo_labels import border_ring, kmeans_mask, mask_iou
from logo.utils.io import load_json

TOOLS = Path(__file__).resolve().parents[1] / "tools"
BRIGHT, DARK = (235, 220, 70), (25, 35, 80)  # RGB


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_tool", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _word(fg, bg, seed=0, text="HELLO"):
    """32x128 RGB crop with `text` in colour fg on bg plus noise; returns (crop, gt mask)."""
    gt = np.zeros((32, 128), np.uint8)
    cv2.putText(gt, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1, 2, cv2.LINE_AA)
    gt = (gt > 0).astype(np.uint8)
    img = np.where(gt[..., None] > 0, np.array(fg, np.float32), np.array(bg, np.float32))
    img += np.random.default_rng(seed).normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), gt


@pytest.mark.parametrize("morph", [True, False])
def test_bright_word_on_dark(morph):
    crop, gt = _word(BRIGHT, DARK)
    m = kmeans_mask(crop, seed=42, morph=morph)
    assert m.shape == (32, 128) and m.dtype == np.float32 and set(np.unique(m)) <= {0.0, 1.0}
    assert mask_iou(m, gt) > 0.7


@pytest.mark.parametrize("morph", [True, False])
def test_inverted_colours(morph):
    crop, gt = _word(BRIGHT, DARK)
    m = kmeans_mask(255 - crop, seed=42, morph=morph)
    assert mask_iou(m, gt) > 0.7


def test_deterministic_and_uniform():
    crop, _ = _word(DARK, BRIGHT, seed=3)
    assert np.array_equal(kmeans_mask(crop, seed=1), kmeans_mask(crop, seed=1))
    assert kmeans_mask(np.full((32, 128, 3), 90, np.uint8)).sum() == 0
    r = border_ring(32, 128, 3)
    assert r[0].all() and r[:, -1].all() and not r[3:29, 3:125].any()


def _lsc_split(root: Path):
    split = root / "data" / "lsc" / "train"
    rows = []
    for k, (fg, bg) in enumerate([(BRIGHT, DARK), (DARK, BRIGHT)]):
        crop, _ = _word(fg, bg, seed=k)
        rel = f"images/icdar2015v/v1/000001_pos{k}.png"
        (split / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(split / rel), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        rows.append(f"{rel}\thello\n")
    neg = "images/icdar2015v/v1/000001_neg0.png"
    cv2.imwrite(str(split / neg), np.zeros((32, 128, 3), np.uint8))
    rows.append(f"{neg}\t\n")
    (split / "labels.tsv").write_text("".join(rows), encoding="utf-8")
    cfg = root / "cfg.yaml"
    cfg.write_text("exp_id: g0\nseed: 7\n", encoding="utf-8")
    return split, cfg


def test_make_masks_and_grid(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    split, cfg = _lsc_split(tmp_path)
    tool = _load("make_glyph_masks")
    assert tool.main(["--config", str(cfg)]) == 0
    lines = (split / "masks.tsv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # negatives skipped
    for k, ln in enumerate(lines):
        rel, mrel, fg = ln.split("\t")
        assert mrel == f"masks/icdar2015v/v1/000001_pos{k}.png" and 0.05 < float(fg) < 0.6
        m = cv2.imread(str(split / mrel), cv2.IMREAD_GRAYSCALE)
        assert m.shape == (32, 128) and set(np.unique(m)) <= {0, 255}
    meta = load_json(tmp_path / "runs" / "g0" / "metrics.json")["meta"]["glyph_masks_train"]
    assert meta["positives"] == 2 and meta["written"] == 2 and meta["seed"] == 7
    assert (tmp_path / "runs" / "g0" / "config.yaml").is_file()
    assert (tmp_path / "runs" / "g0" / "log.txt").is_file()
    assert tool.main(["--config", str(cfg)]) == 0  # rerun keeps existing masks
    assert load_json(tmp_path / "runs" / "g0" / "metrics.json")["meta"]["glyph_masks_train"]["kept"] == 2
    with pytest.raises(PermissionError):
        tool.main(["--config", str(cfg), "--splits", "test"])

    grid = _load("mask_grid")
    out = tmp_path / "grid.png"
    assert grid.main(["--out", str(out), "--cols", "5"]) == 0
    g = cv2.imread(str(out))
    assert g.shape == (32 + grid.CAPTION_H + 2 * grid.PAD, 5 * (2 * 128 + grid.GAP + grid.PAD) + grid.PAD, 3)
    assert grid.main(["--out", str(out), "--split", "test"]) == 1
