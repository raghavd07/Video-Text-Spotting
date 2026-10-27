"""Quadrilateral helpers. Image coordinates (y down); canonical order is visually clockwise."""
from __future__ import annotations

import numpy as np


def _pts(quad) -> np.ndarray:
    return np.asarray(quad, dtype=np.float64).reshape(-1, 2)


def signed_area(quad) -> float:
    """Shoelace area; > 0 means visually clockwise in y-down image coordinates."""
    p = _pts(quad)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_cross(a, b, c, d) -> bool:
    return _cross(c, d, a) * _cross(c, d, b) < 0 and _cross(a, b, c) * _cross(a, b, d) < 0


def is_self_intersecting(quad) -> bool:
    p = _pts(quad)
    return _segments_cross(p[0], p[1], p[2], p[3]) or _segments_cross(p[1], p[2], p[3], p[0])


def order_quad(quad) -> list[float]:
    """Return a simple, clockwise 4-point quad as 8 floats.

    Valid quads keep the annotator's first point (reading direction) and are only reversed if
    counter-clockwise. Self-intersecting ("bowtie") quads, common in ICDAR13/15 XML, are re-sorted
    by angle around the centroid and start at the top-left-most point (min x+y).
    """
    p = _pts(quad)
    if p.shape != (4, 2):
        raise ValueError(f"expected 4 points, got {p.shape[0]}")
    if is_self_intersecting(p):
        c = p.mean(axis=0)
        p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]), kind="stable")]
        p = np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)
    elif signed_area(p) < 0:
        p = p[[0, 3, 2, 1]]
    return [float(v) for v in p.reshape(-1)]


def _clip(subject: list, clipper: np.ndarray) -> list:
    """Sutherland-Hodgman: clip `subject` by convex clockwise `clipper`."""
    out = subject
    n = len(clipper)
    for i in range(n):
        a, b = clipper[i], clipper[(i + 1) % n]
        inp, out = out, []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            e_in = _cross(a, b, e) >= 0
            s_in = _cross(a, b, s) >= 0
            if e_in:
                if not s_in:
                    out.append(_intersect(s, e, a, b))
                out.append(e)
            elif s_in:
                out.append(_intersect(s, e, a, b))
            s = e
    return out


def _intersect(p1, p2, a, b):
    d1 = np.subtract(p2, p1)
    d2 = np.subtract(b, a)
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if den == 0:
        return tuple(p1)
    t = ((a[0] - p1[0]) * d2[1] - (a[1] - p1[1]) * d2[0]) / den
    return (p1[0] + t * d1[0], p1[1] + t * d1[1])


def quad_iou(q1, q2) -> float:
    """IoU of two quads (assumed convex after order_quad)."""
    a = _pts(order_quad(q1))
    b = _pts(order_quad(q2))
    area_a, area_b = abs(signed_area(a)), abs(signed_area(b))
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter_poly = _clip([tuple(p) for p in a], b)
    inter = abs(signed_area(inter_poly)) if len(inter_poly) >= 3 else 0.0
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0
