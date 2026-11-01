import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from logo.lsc import abinet_wrapper as aw  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load():
    spec = importlib.util.spec_from_file_location("compare_glyph_tool", TOOLS / "compare_glyph.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cg = _load()


def test_grad_magnitude_blur_proxy():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 128, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    assert cg.grad_magnitude(blurred) < cg.grad_magnitude(img)
    assert cg.grad_magnitude(np.full((32, 128, 3), 7, np.uint8)) == 0.0


def test_noisy_subset():
    s = np.array([5.0, 1.0, 3.0, 2.0, 4.0, 0.5, 9.0, 8.0, 7.0, 6.0])
    sel = cg.noisy_subset(s, 0.3)
    assert sel.sum() == 3 and set(np.flatnonzero(sel)) == {1, 3, 5}
    pool = np.array([True, False] * 5)
    sel = cg.noisy_subset(s, 0.3, pool)
    assert set(np.flatnonzero(sel)) == {2, 4}                       # ceil(0.3 * 5) = 2 lowest positives


def test_mcnemar_and_bootstrap():
    a = np.ones(20, bool)
    assert cg.mcnemar_p(a, a) == 1.0
    b = a.copy()
    b[:10] = False                                                  # 10 discordant, all one way
    assert cg.mcnemar_p(a, b) == pytest.approx(2 / 2 ** 10)
    lo, hi = cg.paired_bootstrap(a, b, n_boot=500, seed=0)
    assert hi < 0
    lo, hi = cg.paired_bootstrap(a, a, n_boot=100, seed=0)
    assert lo == hi == 0.0


def test_compare_verdicts():
    labels = ["ab"] * 40 + [""] * 10
    ref = ["ab"] * 20 + ["x"] * 20 + [""] * 10
    same = list(ref)
    better = ["ab"] * 40 + [""] * 10
    scores = np.arange(50, dtype=np.float64)
    res = cg.compare({"base": ref, "same": same, "better": better}, labels, scores, 0.3, False, 0)
    assert res["full"]["models"]["base"]["word_acc"] == pytest.approx(50.0)
    assert res["full"]["vs_reference"]["same"]["verdict"] == "within noise"
    c = res["full"]["vs_reference"]["better"]
    assert c["verdict"] == "better" and c["delta_word_acc"] == pytest.approx(50.0)
    assert res["noisy"]["num_crops"] == 15                         # lowest 30% of 50 crops
    assert res["noisy"]["models"]["base"]["word_acc"] == pytest.approx(100.0)
    with pytest.raises(ValueError):
        cg._parse_ckpts(["only=one.pth"])


@pytest.mark.skipif(not (aw.ABINET_ROOT / "demo.py").is_file(), reason="third_party/ABINet not cloned")
def test_main_end_to_end(tmp_path, monkeypatch):
    ab = aw._import_abinet()
    paths = []
    for s in (0, 1):
        torch.manual_seed(s)
        p = tmp_path / f"m{s}.pth"
        torch.save({"model": ab["demo"].get_model(aw.make_config()).state_dict(), "opt": None}, p)
        paths.append(p)
    val = tmp_path / "lsc" / "val"
    (val / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    lines = []
    for i, lab in enumerate(["ab", "", "cd", "e1", "", "fg"]):
        cv2.imwrite(str(val / f"images/{i}.png"), rng.integers(0, 256, (32, 128, 3), dtype=np.uint8))
        lines.append(f"images/{i}.png\t{lab}")
    (val / "labels.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("exp_id: t\nseed: 1\n", encoding="utf-8")
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    assert cg.main(["--config", str(cfg), "--ckpt", f"base={paths[0]}", "--ckpt", f"glyph={paths[1]}",
                    "--root", str(tmp_path / "lsc"), "--device", "cpu", "--no-fp16"]) == 0
    run = tmp_path / "runs" / "t_compare"
    assert (run / "config.yaml").is_file() and (run / "log.txt").is_file()
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["meta"]["glyph_ablation"]
    assert m["full"]["num_crops"] == 6 and m["noisy"]["num_crops"] == 2
    assert set(m["full"]["models"]) == {"base", "glyph"} and "glyph" in m["noisy"]["vs_reference"]
    with pytest.raises(PermissionError):
        cg.main(["--config", str(cfg), "--ckpt", f"a={paths[0]}", "--ckpt", f"b={paths[1]}",
                 "--split", "test"])
