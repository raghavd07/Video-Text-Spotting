import torch

from logo.vpmm.geometry import centerline
from logo.vpmm.module import VPMM

C, D = 6, 16


def _model():
    torch.manual_seed(0)
    return VPMM(C, D).eval()


def test_output_shape():
    torch.manual_seed(0)
    quads = torch.rand(5, 4, 2) * 100
    feats = torch.randn(5, C, D)
    out = _model()(quads, feats, torch.full((5, 2), 100.0))
    assert out.shape == (5, 2 * D)
    assert _model()(quads, feats, (100, 100)).shape == (5, 2 * D)
    assert _model()(torch.zeros(0, 4, 2), torch.zeros(0, C, D), (10, 10)).shape == (0, 2 * D)


def test_rectangle_midline():
    q = torch.tensor([[[10.0, 20.0], [110.0, 20.0], [110.0, 40.0], [10.0, 40.0]]])
    line = centerline(q, C, (200, 100))
    assert torch.allclose(line[0, :, 1], torch.full((C,), 30.0 / 100))
    assert torch.allclose(line[0, :, 0], torch.linspace(10, 110, C) / 200)


def test_vertex_order_invariance():
    torch.manual_seed(1)
    base = torch.tensor([[12.0, 30.0], [95.0, 22.0], [100.0, 48.0], [15.0, 55.0]])
    square = torch.tensor([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    feats = torch.randn(1, C, D)
    m = _model()
    for q in (base, square):
        ref = m(q[None], feats, (128, 64))
        for shift in range(4):
            for rev in (False, True):
                p = torch.roll(q, shift, dims=0)
                p = p.flip(0) if rev else p
                assert torch.allclose(m(p[None], feats, (128, 64)), ref, atol=1e-6)
