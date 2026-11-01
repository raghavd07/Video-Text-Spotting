import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from logo.glyph.branch import BackboneTap, GlyphHead, find_resnet, seg_loss  # noqa: E402
from logo.glyph.dataset import GlyphLSCDataset, augment_with_mask, read_masks  # noqa: E402
from logo.lsc import abinet_wrapper as aw  # noqa: E402
from logo.lsc.dataset import augment  # noqa: E402

needs_abinet = pytest.mark.skipif(not (aw.ABINET_ROOT / "demo.py").is_file(),
                                  reason="third_party/ABINet not cloned")


def _feats(n=2, h=32, w=128):
    g = torch.Generator().manual_seed(0)
    return (torch.randn(n, 32, h, w, generator=g), torch.randn(n, 64, h // 2, w // 2, generator=g),
            torch.randn(n, 512, h // 4, w // 4, generator=g))


def test_head_shapes_and_range():
    head = GlyphHead(32, 64, 512)
    sm = head(*_feats())
    assert sm.shape == (2, 1, 32, 128)
    assert float(sm.min()) >= 0.0 and float(sm.max()) <= 1.0
    assert torch.allclose(sm, torch.sigmoid(head.logits(*_feats())))
    assert head(*_feats(1, 31, 127)).shape == (1, 1, 31, 127)  # odd sizes: up2 matches the skip


def test_seg_loss_mse_bce_valid_and_resize():
    logits = torch.zeros(3, 1, 32, 128, requires_grad=True)       # sigmoid = 0.5
    target = torch.zeros(3, 1, 32, 128)
    target[1] = 1.0
    assert seg_loss(logits, target).item() == pytest.approx(0.25)
    v = torch.tensor([1.0, 0.0, 0.0])
    assert seg_loss(logits, target, v).item() == pytest.approx(0.25)
    assert seg_loss(logits, target, v, "bce").item() == pytest.approx(np.log(2), rel=1e-5)
    small = torch.ones(3, 1, 8, 32)                               # resized to 32x128
    assert seg_loss(logits, small).item() == pytest.approx(0.25)
    z = seg_loss(logits, target, torch.zeros(3))
    z.backward()
    assert z.item() == 0.0 and torch.count_nonzero(logits.grad) == 0
    with pytest.raises(ValueError):
        seg_loss(logits, target, kind="l1")


def test_head_learns_a_mask():
    torch.manual_seed(0)
    head = GlyphHead(4, 8, 16, mid=8)
    f = (torch.randn(4, 4, 16, 32), torch.randn(4, 8, 8, 16), torch.randn(4, 16, 4, 8))
    target = (f[0][:, :1] > 0).float()
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        loss = seg_loss(head.logits(*f), target)
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


@needs_abinet
def test_tap_on_abinet_resnet():
    aw._import_abinet()
    from modules.resnet import resnet45
    torch.manual_seed(0)
    net = resnet45().eval()
    wrapper = torch.nn.Sequential(torch.nn.Identity(), net)
    assert find_resnet(wrapper) is net
    x = torch.randn(2, 3, 32, 128)
    with torch.no_grad():
        ref = net(x)
        tap = BackboneTap(wrapper)
        out = net(x)
    assert torch.equal(ref, out)                                   # hooks do not change the output
    assert tap.channels == (32, 64, 512)
    f0, f1, f2 = tap.pop()
    assert f0.shape == (2, 32, 32, 128) and f1.shape == (2, 64, 16, 64) and f2.shape == (2, 512, 8, 32)
    assert torch.equal(f2, out)
    with pytest.raises(RuntimeError):
        tap.pop()
    tap.remove()
    net(x)
    assert tap._feats == {}
    with pytest.raises(ValueError):
        find_resnet(torch.nn.Linear(2, 2))


def test_augment_with_mask_matches_augment():
    rng0 = np.random.default_rng(0)
    img = rng0.integers(0, 256, (32, 128, 3), dtype=np.uint8)
    mask = (rng0.random((32, 128)) > 0.7).astype(np.uint8) * 255
    for s in range(40):
        a = augment(img.copy(), np.random.default_rng(s))
        b, m = augment_with_mask(img.copy(), mask.copy(), np.random.default_rng(s))
        assert np.array_equal(a, b), s
        assert m.shape == (32, 128) and m.dtype == np.uint8
    # geometry only: the mask moves exactly like an image channel
    img2 = np.dstack([mask, mask, mask])
    for s in range(10):
        b, m = augment_with_mask(img2, mask, np.random.default_rng(s), p_geo=1.0, p_quality=0, p_color=0)
        assert np.array_equal(b[..., 0], m)


def _write_glyph_split(root, labels, with_mask, seed):
    rng = np.random.default_rng(seed)
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    lines, mlines = [], []
    for i, lab in enumerate(labels):
        rel = f"images/{i}.png"
        img = np.full((32, 128, 3), 200, np.uint8)
        img[8:24, 10 + 8 * i:40 + 8 * i] = rng.integers(0, 60, 3, dtype=np.uint8)
        cv2.imwrite(str(root / rel), img)
        lines.append(f"{rel}\t{lab}")
        if with_mask[i]:
            m = np.zeros((32, 128), np.uint8)
            m[8:24, 10 + 8 * i:40 + 8 * i] = 255
            cv2.imwrite(str(root / f"masks/{i}.png"), m)
            mlines.append(f"{rel}\tmasks/{i}.png\t{(m > 0).mean():.4f}")
    (root / "labels.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "masks.tsv").write_text("\n".join(mlines) + "\n", encoding="utf-8")


def test_glyph_dataset(tmp_path):
    _write_glyph_split(tmp_path, ["ab", "", "cd"], [True, False, False], 0)
    assert read_masks(tmp_path / "masks.tsv") == {"images/0.png": "masks/0.png"}
    ds = GlyphLSCDataset(tmp_path, train=False)
    x, lab, m, v = ds[0]
    assert x.shape == (3, 32, 128) and lab == "ab" and m.shape == (1, 32, 128) and v.item() == 1.0
    assert m[0, 16, 20].item() == 1.0 and m[0, 0, 0].item() == 0.0
    assert ds[1][3].item() == 0.0 and ds[2][3].item() == 0.0          # no mask -> no L_seg
    assert GlyphLSCDataset(tmp_path, neg_zero=True)[1][3].item() == 1.0
    assert GlyphLSCDataset(tmp_path, neg_zero=True)[2][3].item() == 0.0
    torch.manual_seed(0)
    xt, _, mt, _ = GlyphLSCDataset(tmp_path, train=True)[0]
    assert xt.shape == (3, 32, 128) and mt.shape == (1, 32, 128)
    with pytest.raises(FileNotFoundError):
        GlyphLSCDataset(tmp_path, masks_tsv="missing.tsv")


@needs_abinet
def test_train_with_glyph(tmp_path, monkeypatch):
    from logo.lsc import train_lsc as tl
    ab = aw._import_abinet()
    torch.manual_seed(0)
    ck = tmp_path / "rand.pth"
    torch.save({"model": ab["demo"].get_model(aw.make_config()).state_dict(), "opt": None}, ck)
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    _write_glyph_split(tmp_path / "lsc" / "train", ["ab", "", "c1", "wd"], [True, False, True, True], 0)
    _write_glyph_split(tmp_path / "lsc" / "val", ["ab", ""], [False, False], 1)
    cfg = {"exp_id": "t_glyph", "seed": 3,
           "lsc": {"ckpt": str(ck), "train": {"data_root": str(tmp_path / "lsc"), "epochs": 1,
                                             "batch_size": 2, "val_batch_size": 2, "accum_steps": 1,
                                             "num_workers": 0, "log_every": 1, "smoke_samples": 4}},
           "glyph": {"enabled": True, "weight": 1.0, "loss": "mse"}}
    with pytest.raises(ValueError):
        tl.glyph_settings({"glyph": {"loss": "dice"}})
    assert tl.glyph_settings({})["enabled"] is False
    res = tl.train(cfg, smoke=True, device="cpu")
    ep = res["epochs"][0]
    assert set(ep["train_losses"]) == {"vision_loss", "language_loss", "alignment_loss", "seg_loss"}
    assert np.isfinite(ep["train_loss"]) and ep["train_losses"]["seg_loss"] > 0
    assert res["glyph"]["enabled"] is True
    run = tmp_path / "runs" / "t_glyph_smoke" / "checkpoints"
    g = torch.load(run / "glyph_best.pth", map_location="cpu")
    assert g["channels"] == [32, 64, 512]
    with torch.random.fork_rng(devices=[]):  # same init as train(): seed 3
        torch.manual_seed(3)
        fresh = GlyphHead(32, 64, 512).state_dict()
    assert any(not torch.equal(g["glyph_head"][k], fresh[k]) for k in fresh if fresh[k].is_floating_point())
    rec = aw.ABINetRecognizer(run / "best.pth", "cpu", fp16=False)     # model ckpt format unchanged
    assert len(rec.predict([np.zeros((32, 128, 3), np.uint8)])) == 1
