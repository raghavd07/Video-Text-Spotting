import json

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from logo.lsc import abinet_wrapper as aw  # noqa: E402

pytestmark = pytest.mark.skipif(not (aw.ABINET_ROOT / "demo.py").is_file(),
                                reason="third_party/ABINet not cloned")

from logo.lsc import train_lsc as tl  # noqa: E402


@pytest.fixture(scope="module")
def rand_ckpt(tmp_path_factory):
    ab = aw._import_abinet()
    torch.manual_seed(0)
    model = ab["demo"].get_model(aw.make_config())
    p = tmp_path_factory.mktemp("abinet") / "rand.pth"
    torch.save({"model": model.state_dict(), "opt": None}, p)
    return p


@pytest.fixture(scope="module")
def charset():
    ab = aw._import_abinet()
    cfg = aw.make_config()
    return ab["utils"].CharsetMapper(filename=cfg.dataset_charset_path, max_length=cfg.dataset_max_length + 1)


def _write_split(root, labels, seed):
    rng = np.random.default_rng(seed)
    (root / "images").mkdir(parents=True)
    lines = []
    for i, lab in enumerate(labels):
        rel = f"images/{i}.png"
        cv2.imwrite(str(root / rel), rng.integers(0, 256, (32, 128, 3), dtype=np.uint8))
        lines.append(f"{rel}\t{lab}")
    (root / "labels.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_encode_empty_and_word(charset):
    gt, lengths = tl.encode_labels(["", "ab1"], charset)
    assert gt.shape == (2, 26, charset.num_classes) and lengths.tolist() == [1, 4]
    ids = gt.argmax(-1)
    assert (ids[0] == charset.null_label).all()           # end token at step 0, then padding
    assert ids[1, :3].tolist() == [charset.char_to_label[c] for c in "ab1"]
    assert (ids[1, 3:] == charset.null_label).all()
    assert torch.allclose(gt.sum(-1), torch.ones(2, 26))
    with pytest.raises(ValueError):
        tl.encode_labels(["a" * 26], charset)
    with pytest.raises(ValueError):
        tl.encode_labels(["A!"], charset)


def test_lr_schedule_and_settings():
    tc = tl.train_settings({"lsc": {"train": {"epochs": 10}}})
    assert tc["batch_size"] == 48 and tc["accum_steps"] == 1
    assert [tl.lr_at(e, tc) for e in range(10)] == [1e-4] * 6 + [1e-5] * 4


def test_score_predictions():
    r = tl.score_predictions(["abc", "", "ab", "x", ""], ["abc", "", "abd", "", "q"])
    assert (r["num_pos"], r["num_neg"]) == (3, 2)
    assert r["tb_acc"] == pytest.approx(60.0)       # ok: 0,1,2 ; wrong: 3 (text on bg), 4 (bg on text)
    assert r["word_acc"] == pytest.approx(100 / 3)
    assert r["text_recall"] == pytest.approx(200 / 3) and r["bg_recall"] == pytest.approx(50.0)


def test_train_end_to_end(tmp_path, monkeypatch, rand_ckpt):
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    _write_split(tmp_path / "lsc" / "train", ["ab", "", "c1", "", "word", ""], 0)
    _write_split(tmp_path / "lsc" / "val", ["ab", "", "x", ""], 1)
    cfg = {"exp_id": "t_lsc", "seed": 3,
           "lsc": {"ckpt": str(rand_ckpt), "det_thr": 0.3, "iou_neg": 0.05,
                   "train": {"data_root": str(tmp_path / "lsc"), "epochs": 2, "batch_size": 2,
                             "val_batch_size": 2, "accum_steps": 2, "num_workers": 0, "decay_epoch": 1,
                             "log_every": 1, "smoke_samples": 4}}}
    res = tl.train(cfg, smoke=True, device="cpu")
    run = tmp_path / "runs" / "t_lsc_smoke"
    for f in ("config.yaml", "log.txt", "metrics.json", "checkpoints/epoch_01.pth", "checkpoints/best.pth"):
        assert (run / f).is_file(), f
    assert res["num_train"] == 4 and res["num_val"] == 4 and len(res["epochs"]) == 1
    ep = res["epochs"][0]
    assert np.isfinite(ep["train_loss"]) and ep["train_loss"] > 0
    assert set(ep["train_losses"]) == {"vision_loss", "language_loss", "alignment_loss"}
    for k in ("tb_acc", "word_acc", "loss", "num_pos", "num_neg"):
        assert k in ep["val"]
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert m["meta"]["lsc_train"]["best"]["epoch"] == 1
    # weights changed and the checkpoint loads in the inference wrapper
    new = torch.load(run / "checkpoints" / "best.pth", map_location="cpu")["model"]
    old = torch.load(rand_ckpt, map_location="cpu")["model"]
    assert any(not torch.equal(new[k], old[k]) for k in old if old[k].is_floating_point())
    rec = aw.ABINetRecognizer(run / "checkpoints" / "best.pth", "cpu", fp16=False)
    assert len(rec.predict([np.zeros((32, 128, 3), np.uint8)])) == 1
