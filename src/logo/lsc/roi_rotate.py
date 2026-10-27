"""Rectify text quads into fixed-size crops for the recogniser.

Quad coordinates are continuous image coordinates: pixel (i, j) covers [j, j+1) x [i, i+1).
With that convention an axis-aligned quad with integer corners crops exactly like
cv2.resize(img[y0:y1, x0:x1], (W, H), interpolation=cv2.INTER_LINEAR).
"""
from __future__ import annotations

import cv2
import numpy as np

VERTICAL_RATIO = 1.5  # source height > 1.5 * width -> rotate the crop 90 degrees
_MIN_AREA = 1.0  # px^2; smaller (degenerate) quads fall back to their axis-aligned bbox


def order_vertices(quad) -> np.ndarray:
    """Return [4,2] float64 points as TL, TR, BR, BL for any input order and orientation.

    Points are sorted clockwise around the centroid (y down), then the cyclic start is chosen so
    that the TL->TR edge points closest to +x. Ties (exact 45 degrees) go to the start with the
    smallest x+y. A box rotated by more than 45 degrees therefore gets its short side as "top";
    crop_quad detects that case through the height/width ratio.
    """
    p = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    if p.shape != (4, 2):
        raise ValueError(f"expected 4 points, got {p.shape[0]}")
    c = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]), kind="stable")]
    edges = np.roll(p, -1, axis=0) - p
    cos = edges[:, 0] / np.maximum(np.linalg.norm(edges, axis=1), 1e-12)
    cand = np.flatnonzero(cos >= cos.max() - 1e-9)
    k = int(cand[np.argmin(p[cand].sum(axis=1))])
    return np.roll(p, -k, axis=0)


def _side_lengths(p: np.ndarray) -> tuple[float, float]:
    """(width, height) of an ordered quad: mean of top/bottom and of left/right edge lengths."""
    w = 0.5 * (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3]))
    h = 0.5 * (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1]))
    return float(w), float(h)


def _as_bgr_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.ndim != 3 or img.shape[2] not in (1, 3, 4):
        raise ValueError(f"expected HxW, HxWx1, HxWx3 or HxWx4 image, got {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] == 4:
        img = np.ascontiguousarray(img[:, :, :3])
    return img


def _warp(img: np.ndarray, src: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    # -0.5 shifts move both sides to cv2's pixel-centre convention (see module docstring).
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], np.float32) - 0.5
    m = cv2.getPerspectiveTransform((src - 0.5).astype(np.float32), dst)
    return cv2.warpPerspective(img, m, (out_w, out_h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def crop_quad(img: np.ndarray, quad, out_hw=(32, 128)) -> np.ndarray:
    """Perspective-rectify `quad` (8 floats or [4,2]) from `img` to a uint8 out_h x out_w x 3 crop.

    If the source quad is vertical (height > 1.5 * width) it is warped to out_w x out_h and
    rotated 90 degrees counter-clockwise, so text written top-to-bottom reads left-to-right.
    Channel order of `img` is kept; grayscale is repeated to 3 channels, alpha is dropped.
    """
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"out_hw must be positive, got {out_hw}")
    img = _as_bgr_uint8(img)
    p = order_vertices(quad)
    x, y = p[:, 0], p[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))
    if area < _MIN_AREA:
        x0, y0 = float(x.min()), float(y.min())
        x1, y1 = max(float(x.max()), x0 + 1.0), max(float(y.max()), y0 + 1.0)
        p = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
    w, h = _side_lengths(p)
    if h > VERTICAL_RATIO * w:
        return np.ascontiguousarray(np.rot90(_warp(img, p, out_w, out_h)))
    return _warp(img, p, out_h, out_w)


def crop_many(img: np.ndarray, quads, out_hw=(32, 128)) -> np.ndarray:
    """Crop every quad of one image; returns uint8 [N, out_h, out_w, 3] (N may be 0)."""
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    quads = list(quads)
    out = np.empty((len(quads), out_h, out_w, 3), np.uint8)
    if quads:
        img = _as_bgr_uint8(img)
        for i, q in enumerate(quads):
            out[i] = crop_quad(img, q, (out_h, out_w))
    return out
