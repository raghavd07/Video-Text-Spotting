"""Text centerline from quads (paper Eq. 7 input). Image coordinates (y down)."""
from __future__ import annotations

import torch


def _as_wh(img_wh, n: int, ref: torch.Tensor) -> torch.Tensor:
    wh = torch.as_tensor(img_wh, dtype=ref.dtype, device=ref.device)
    if wh.dim() == 1:
        wh = wh.expand(n, 2)
    if wh.shape != (n, 2):
        raise ValueError(f"img_wh must be (w, h) or [N, 2], got {tuple(wh.shape)}")
    return wh


def _stable_order(key: torch.Tensor, idx: torch.Tensor, descending: bool) -> torch.Tensor:
    k = torch.gather(key, 1, idx)
    perm = torch.sort(k, dim=1, descending=descending, stable=True).indices
    return torch.gather(idx, 1, perm)


def _long_sides(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices [N] of the two long edges (edge k = p[k] -> p[k+1]); vertex-order invariant."""
    a, b = p, torch.roll(p, -1, dims=1)
    vec = b - a
    length = vec.norm(dim=2)
    mid = (a + b) / 2
    # lexicographic edge ranking: length desc, |dx| desc, mean y asc, mean x asc (least key sorted first)
    idx = torch.arange(4, device=p.device).expand(p.shape[0], 4).contiguous()
    idx = _stable_order(mid[..., 0], idx, descending=False)
    idx = _stable_order(mid[..., 1], idx, descending=False)
    idx = _stable_order(vec[..., 0].abs(), idx, descending=True)
    idx = _stable_order(length, idx, descending=True)
    i, j = idx[:, 0], idx[:, 1]
    adjacent = (i - j).remainder(2) == 1
    j = torch.where(adjacent, (i + 2).remainder(4), j)  # two longest share a vertex -> use opposite edge
    return i, j


def _edge_points(p: torch.Tensor, k: torch.Tensor, c: int) -> torch.Tensor:
    """c equally spaced points on edge k, oriented left-to-right (ties: top-to-bottom). [N,c,2]."""
    rows = torch.arange(p.shape[0], device=p.device)
    a, b = p[rows, k], p[rows, (k + 1).remainder(4)]
    swap = (a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1]))
    a, b = torch.where(swap[:, None], b, a), torch.where(swap[:, None], a, b)
    t = torch.linspace(0.0, 1.0, c, dtype=p.dtype, device=p.device)[None, :, None]
    return a[:, None] + t * (b - a)[:, None]


def centerline(quads, c: int, img_wh) -> torch.Tensor:
    """quads [N,4,2] (or [N,8]) in pixels -> centerline [N,c,2], x / width, y / height."""
    p = torch.as_tensor(quads)
    if not torch.is_floating_point(p):
        p = p.float()
    p = p.reshape(-1, 4, 2)
    n = p.shape[0]
    if n == 0:
        return p.new_zeros((0, c, 2))
    i, j = _long_sides(p)
    s_i, s_j = _edge_points(p, i, c), _edge_points(p, j, c)
    i_top = (s_i[..., 1].mean(1) <= s_j[..., 1].mean(1))[:, None, None]
    top, bottom = torch.where(i_top, s_i, s_j), torch.where(i_top, s_j, s_i)
    line = (top + bottom) / 2
    return line / _as_wh(img_wh, n, p)[:, None, :]
