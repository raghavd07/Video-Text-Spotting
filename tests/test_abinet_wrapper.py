import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from logo.lsc import abinet_wrapper as aw  # noqa: E402

pytestmark = pytest.mark.skipif(not (aw.ABINET_ROOT / "demo.py").is_file(),
                                reason="third_party/ABINet not cloned")


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    """Recogniser on a randomly initialised checkpoint (no pretrained weights needed)."""
    ab = aw._import_abinet()
    torch.manual_seed(0)
    model = ab["demo"].get_model(aw.make_config())
    ckpt = tmp_path_factory.mktemp("abinet") / "rand.pth"
    torch.save({"model": model.state_dict(), "opt": None}, ckpt)
    return aw.ABINetRecognizer(ckpt, "cpu", fp16=True)


def _crops():
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (32, 128, 3), dtype=np.uint8),
            rng.integers(0, 256, (20, 77), dtype=np.uint8),          # gray, needs resize
            rng.integers(0, 256, (50, 140, 4), dtype=np.uint8),      # BGRA
            rng.integers(0, 256, (31, 90, 3), dtype=np.uint8),
            rng.integers(0, 256, (64, 256, 3), dtype=np.uint8)]


def test_dims_and_outputs(rec):
    assert (rec.c, rec.d, rec.h, rec.w) == (26, 512, 32, 128)
    out = rec.predict(_crops())
    assert len(out) == 5
    for o in out:
        assert set(o) == {"text", "text_conf", "p_eos", "feat"}
        assert isinstance(o["text"], str) and o["text"] == o["text"].lower()
        assert 0.0 <= o["text_conf"] <= 1.0 and 0.0 <= o["p_eos"] <= 1.0
        assert o["feat"].shape == (26, 512) and o["feat"].dtype == np.float16
        assert np.isfinite(o["feat"]).all()
    assert rec.predict([]) == []


def test_batching_invariant(rec):
    crops = _crops()
    rec.batch_size = 64
    full = rec.predict(crops)
    rec.batch_size = 2
    try:
        parts = rec.predict(crops)
    finally:
        rec.batch_size = 64
    for a, b in zip(full, parts):
        assert a["text"] == b["text"]
        assert a["p_eos"] == pytest.approx(b["p_eos"], abs=1e-4)
        np.testing.assert_allclose(a["feat"].astype(np.float32), b["feat"].astype(np.float32), atol=1e-2)


def test_preprocess_matches_demo(rec):
    from PIL import Image
    demo = aw._import_abinet()["demo"]
    bgr = np.random.default_rng(1).integers(0, 256, (40, 100, 3), dtype=np.uint8)
    ref = demo.preprocess(Image.fromarray(bgr[:, :, ::-1].copy()), rec.w, rec.h)
    np.testing.assert_allclose(rec.preprocess([bgr]).numpy(), ref.numpy(), atol=1e-5)


def test_decode_text_conf_p_eos(rec):
    cs = rec.charset  # 0 = null, 1..26 = a..z, 27..36 = 0..9
    p = np.full((2, 26, cs.num_classes), 0.01, np.float32)
    p[0, 0, cs.char_to_label["a"]] = 0.9
    p[0, 1, cs.char_to_label["7"]] = 0.5
    p[0, 2:, 0] = 0.8
    p[0, 4, cs.char_to_label["z"]] = 0.99   # after the end token: ignored
    p[1, :, 0] = 0.7
    (t0, c0, e0), (t1, c1, e1) = aw.decode(p, cs)
    assert (t0, t1) == ("a7", "")
    assert c0 == pytest.approx(0.7) and e0 == pytest.approx(0.01)
    assert c1 == 0.0 and e1 == pytest.approx(0.7)


def test_single_sys_path_insert():
    aw._import_abinet()
    aw._import_abinet()
    assert sys.path.count(str(aw.ABINET_ROOT)) == 1


def test_missing_ckpt(tmp_path):
    with pytest.raises(FileNotFoundError):
        aw.ABINetRecognizer(tmp_path / "nope.pth", "cpu")
