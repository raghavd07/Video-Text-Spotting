"""Glyph segmentation branch (paper Eq. 5): an auxiliary text-mask head on ABINet's ResNet-45.

Features are tapped with forward hooks on the backbone (third_party/ABINet stays unmodified).
For a 32x128 input the ResNet-45 (modules/resnet.py) has three resolutions:
  F0 = resnet.relu   (stem)    32 ch, stride 1, 32x128
  F1 = resnet.layer2           64 ch, stride 2, 16x64
  F2 = resnet.layer5          512 ch, stride 4,  8x32
Head: G2 = phi(F2); G1 = phi(up2(G2) + F1); Sm = tau(up2(G1) + F0), phi = conv3x3 + BN + ReLU whose
conv also maps the channels to the next skip's width (512 -> 64 -> 32), so no extra 1x1 is needed;
tau = conv3x3 + ReLU, conv3x3 + ReLU, conv1x1 -> 1 channel, sigmoid.
L_seg = MSE(Sm, S_pl) (or BCE) with the pseudo-mask resized to Sm, averaged over valid samples.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TAP_NAMES = ("relu", "layer2", "layer5")  # F0, F1, F2 inside the ResNet
SEG_LOSSES = ("mse", "bce")


def find_resnet(model: nn.Module) -> nn.Module:
    """First submodule shaped like ABINet's ResNet (conv1/bn1/relu + layer1..layer5)."""
    attrs = ("conv1", "bn1") + TAP_NAMES + ("layer1", "layer3", "layer4")
    for m in model.modules():
        if all(hasattr(m, a) for a in attrs):
            return m
    raise ValueError("no ABINet ResNet (conv1, bn1, relu, layer1..layer5) found in model")


def _out_channels(layer: nn.Sequential) -> int:
    return int(layer[-1].bn2.num_features)


class BackboneTap:
    """Forward hooks storing (F0, F1, F2) of the last backbone forward; pop() returns and clears them."""

    def __init__(self, model: nn.Module):
        self.resnet = find_resnet(model)
        self._feats: dict = {}
        self.handles = [getattr(self.resnet, n).register_forward_hook(self._hook(n)) for n in TAP_NAMES]

    def _hook(self, name):
        def hook(_module, _inp, out):
            self._feats[name] = out
        return hook

    @property
    def channels(self) -> tuple[int, int, int]:
        r = self.resnet
        return int(r.bn1.num_features), _out_channels(r.layer2), _out_channels(r.layer5)

    def pop(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        missing = [n for n in TAP_NAMES if n not in self._feats]
        if missing:
            raise RuntimeError(f"backbone taps {missing} empty: run the model forward first")
        feats = tuple(self._feats[n] for n in TAP_NAMES)
        self._feats.clear()
        return feats

    def clear(self) -> None:
        self._feats.clear()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        self._feats.clear()


def phi(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


def _up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """up2 (bilinear), sized to the skip feature so odd input sizes still add up."""
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class GlyphHead(nn.Module):
    """Eq. 5 decoder: (F0 [N,c0,H,W], F1 [N,c1,H/2,W/2], F2 [N,c2,H/4,W/4]) -> Sm [N,1,H,W] in (0,1)."""

    def __init__(self, c0: int = 32, c1: int = 64, c2: int = 512, mid: int = 32):
        super().__init__()
        self.channels = (int(c0), int(c1), int(c2))
        self.phi2 = phi(c2, c1)
        self.phi1 = phi(c1, c0)
        self.tau = nn.Sequential(nn.Conv2d(c0, mid, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(mid, mid, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(mid, 1, 1))

    def logits(self, f0, f1, f2) -> torch.Tensor:
        g2 = self.phi2(f2)
        g1 = self.phi1(_up_to(g2, f1) + f1)
        return self.tau(_up_to(g1, f0) + f0)

    def forward(self, f0, f1, f2) -> torch.Tensor:
        return torch.sigmoid(self.logits(f0, f1, f2))


def seg_loss(logits: torch.Tensor, target: torch.Tensor, valid=None, kind: str = "mse") -> torch.Tensor:
    """L_seg on head logits [N,1,h,w] vs pseudo-masks [N,1,H,W] in [0,1] (resized to h,w).

    Per-sample pixel mean, then mean over samples with valid > 0 (valid [N], default all);
    no valid sample -> a zero that keeps the graph. kind "mse" = MSE(sigmoid, S_pl), "bce" = BCE.
    Computed in fp32 (call outside autocast).
    """
    if kind not in SEG_LOSSES:
        raise ValueError(f"glyph loss {kind!r} not in {SEG_LOSSES}")
    logits = logits.float()
    target = target.float().to(logits.device)
    if target.dim() == 3:
        target = target[:, None]
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target, size=logits.shape[-2:], mode="bilinear", align_corners=False)
    if kind == "mse":
        per_px = (torch.sigmoid(logits) - target) ** 2
    else:
        per_px = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    per_sample = per_px.flatten(1).mean(1)
    w = torch.ones_like(per_sample) if valid is None else (valid.float().to(logits.device) > 0).float()
    n = w.sum()
    if n.item() == 0:
        return logits.sum() * 0.0
    return (per_sample * w).sum() / n


class GlyphAux:
    """Tap + head + loss settings used by train_lsc when glyph.enabled."""

    def __init__(self, model: nn.Module, weight: float = 1.0, kind: str = "mse", head=None):
        if kind not in SEG_LOSSES:
            raise ValueError(f"glyph loss {kind!r} not in {SEG_LOSSES}")
        self.tap = BackboneTap(model)
        self.head = GlyphHead(*self.tap.channels) if head is None else head
        self.weight, self.kind = float(weight), kind

    def logits(self) -> torch.Tensor:
        """Head logits for the features of the model forward that just ran."""
        return self.head.logits(*self.tap.pop())

    def loss(self, logits, masks, valid) -> torch.Tensor:
        return seg_loss(logits, masks, valid, self.kind)
