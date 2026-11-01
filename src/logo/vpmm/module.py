"""Visual-Position Mixed Module (paper Eq. 7-8): centerline position embedding + visual features."""
from __future__ import annotations

import torch
from torch import nn

from logo.vpmm.geometry import centerline


class VPMM(nn.Module):
    def __init__(self, c: int, d: int):
        super().__init__()
        self.c, self.d = int(c), int(d)
        # Eq. 7: P = ReLU(W2 ReLU(W1 L)), L = normalized centerline [N,c,2]
        self.pos = nn.Sequential(nn.Linear(2, d), nn.ReLU(inplace=True), nn.Linear(d, d), nn.ReLU(inplace=True))
        # Eq. 8: F = MLP(flatten(P + V)), c*d -> 2d -> 2d
        self.mix = nn.Sequential(
            nn.Linear(self.c * d, 2 * d), nn.ReLU(inplace=True), nn.Linear(2 * d, 2 * d), nn.ReLU(inplace=True)
        )

    def forward(self, quads: torch.Tensor, feats: torch.Tensor, img_wh) -> torch.Tensor:
        if feats.dim() != 3 or feats.shape[1:] != (self.c, self.d):
            raise ValueError(f"feats must be [N,{self.c},{self.d}], got {tuple(feats.shape)}")
        w = self.pos[0].weight
        line = centerline(torch.as_tensor(quads, device=w.device), self.c, img_wh).to(w.dtype)
        if line.shape[0] != feats.shape[0]:
            raise ValueError(f"{line.shape[0]} quads vs {feats.shape[0]} feature rows")
        x = self.pos(line) + feats.to(w.dtype)
        return self.mix(x.flatten(1))
