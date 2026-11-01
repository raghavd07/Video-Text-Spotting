"""Glyph pseudo-labels: unsupervised text/background masks of rectified text crops.

kmeans_mask clusters the crop's Lab pixels into 2 groups (seeded k-means++, best of n_init runs)
and calls "text" the cluster with fewer pixels on the border ring (ties: the smaller cluster),
which works for both bright-on-dark and dark-on-bright text.
"""
from __future__ import annotations

import cv2
import numpy as np

K = 2


def to_lab(crop_rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB HxWx3 -> float32 Lab HxWx3 (L in [0,100], a/b roughly [-127,127])."""
    if crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {crop_rgb.shape}")
    x = crop_rgb.astype(np.float32)
    if crop_rgb.dtype == np.uint8 or x.max() > 1.0:
        x /= 255.0
    return cv2.cvtColor(x, cv2.COLOR_RGB2Lab)


def border_ring(h: int, w: int, width: int) -> np.ndarray:
    """bool HxW, True on the outer `width` pixels."""
    ring = np.ones((h, w), bool)
    width = max(1, int(width))
    if h > 2 * width and w > 2 * width:
        ring[width:h - width, width:w - width] = False
    return ring


def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int) -> tuple[np.ndarray, float]:
    """One Lloyd run with k-means++ init on x [N,D]; returns (labels [N], inertia)."""
    n = len(x)
    centers = [x[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((x[:, None, :] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1)
        s = float(d2.sum())
        centers.append(x[rng.integers(n)] if s <= 0 else x[rng.choice(n, p=d2 / s)])
    c = np.asarray(centers, np.float64)
    labels = np.zeros(n, np.int64)
    for it in range(iters):
        d2 = ((x[:, None, :] - c[None]) ** 2).sum(-1)
        new = d2.argmin(1)
        if it and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                c[j] = x[m].mean(0)
    inertia = float(((x - c[labels]) ** 2).sum())
    return labels, inertia


def kmeans_mask(crop_rgb: np.ndarray, seed: int = 0, n_init: int = 3, iters: int = 30,
                morph: bool = True, ring_frac: float = 0.08) -> np.ndarray:
    """Text mask of an RGB crop: float32 [H,W] with values in {0,1}.

    ring_frac: border ring width as a fraction of min(H,W) (at least 1 px).
    morph: 3x3 closing (fills stroke gaps) then 3x3 cross opening (drops specks).
    A crop with a single colour gives an all-zero mask.
    """
    h, w = crop_rgb.shape[:2]
    x = to_lab(crop_rgb).reshape(-1, 3).astype(np.float64)
    if np.ptp(x, axis=0).max() < 1e-6:
        return np.zeros((h, w), np.float32)
    rng = np.random.default_rng(int(seed))
    best, best_inertia = None, np.inf
    for _ in range(max(1, int(n_init))):
        labels, inertia = _kmeans(x, K, rng, iters)
        if inertia < best_inertia:
            best, best_inertia = labels, inertia
    labels = best.reshape(h, w)

    ring = border_ring(h, w, round(ring_frac * min(h, w)))
    on_ring = [int((labels[ring] == j).sum()) for j in range(K)]
    sizes = [int((labels == j).sum()) for j in range(K)]
    text = min(range(K), key=lambda j: (on_ring[j], sizes[j]))
    mask = (labels == text).astype(np.uint8)

    if morph:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    return mask.astype(np.float32)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a) > 0.5, np.asarray(b) > 0.5
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 1.0
