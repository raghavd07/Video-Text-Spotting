"""Shared evaluation helpers: polygon IoU (shapely, AABB prefilter), dont-care filtering, loaders."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon

from logo.data.schema import load_gt
from logo.utils.geometry import order_quad

DC_INSIDE = 0.5  # a prediction is ignored if > 50% of its area lies inside one dont_care GT quad

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def normalize_text(s) -> str:
    """Lowercase, keep only [a-z0-9]."""
    return _NON_ALNUM.sub("", str(s or "").lower())


def _polygon(quad) -> Polygon:
    poly = Polygon(np.asarray(quad, dtype=np.float64).reshape(4, 2))
    if poly.is_valid:
        return poly
    try:
        poly = Polygon(np.asarray(order_quad(quad)).reshape(4, 2))
    except ValueError:
        pass
    return poly if poly.is_valid else shapely.make_valid(poly)


@dataclass
class Boxes:
    """Polygons of one frame, built once and reused for every IoU / dont-care query."""
    polys: np.ndarray   # object array of shapely geometries
    bounds: np.ndarray  # [N,4] xmin, ymin, xmax, ymax
    areas: np.ndarray   # [N]

    @classmethod
    def from_quads(cls, quads) -> "Boxes":
        polys = np.array([_polygon(q) for q in quads] + [None], dtype=object)[:-1]
        if len(polys) == 0:
            return cls(polys, np.zeros((0, 4)), np.zeros(0))
        return cls(polys, shapely.bounds(polys).reshape(-1, 4), shapely.area(polys))

    def __len__(self) -> int:
        return len(self.polys)

    def subset(self, idx) -> "Boxes":
        idx = np.asarray(idx, dtype=np.int64)
        return Boxes(self.polys[idx], self.bounds[idx], self.areas[idx])


def _intersections(a: Boxes, b: Boxes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(i, j, area) for pairs whose axis-aligned boxes overlap."""
    if len(a) == 0 or len(b) == 0:
        e = np.zeros(0, dtype=np.int64)
        return e, e, np.zeros(0)
    ab, bb = a.bounds[:, None, :], b.bounds[None, :, :]
    ov = ((ab[..., 0] < bb[..., 2]) & (bb[..., 0] < ab[..., 2])
          & (ab[..., 1] < bb[..., 3]) & (bb[..., 1] < ab[..., 3]))
    i, j = np.nonzero(ov)
    if i.size == 0:
        return i, j, np.zeros(0)
    return i, j, shapely.area(shapely.intersection(a.polys[i], b.polys[j]))


def iou_matrix(a: Boxes, b: Boxes) -> np.ndarray:
    out = np.zeros((len(a), len(b)))
    i, j, inter = _intersections(a, b)
    if i.size:
        union = a.areas[i] + b.areas[j] - inter
        out[i, j] = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    return out


def inside_fraction(pred: Boxes, region: Boxes) -> np.ndarray:
    """[P]: max over regions of area(pred & region) / area(pred)."""
    out = np.zeros(len(pred))
    j, i, inter = _intersections(region, pred)
    if i.size:
        frac = np.where(pred.areas[i] > 0, inter / np.maximum(pred.areas[i], 1e-12), 0.0)
        np.maximum.at(out, i, frac)
    return out


@dataclass
class FrameMatch:
    """One frame after dont-care handling: care GT rows x kept prediction columns."""
    gt: list[dict]
    pred: list[dict]
    iou: np.ndarray
    ignored: int


def prepare_frame(gt_objs: list[dict], pred_objs: list[dict], spotting: bool = False) -> FrameMatch:
    """Drops dont_care GT (in spotting mode also GT with empty normalized text) and every
    prediction with more than 50% of its area inside one dont_care GT quad."""
    dc = np.array([bool(o["dont_care"]) or (spotting and not normalize_text(o["text"]))
                   for o in gt_objs], dtype=bool)
    gt_box = Boxes.from_quads([o["quad"] for o in gt_objs])
    pred_box = Boxes.from_quads([o["quad"] for o in pred_objs])
    ignore = np.zeros(len(pred_objs), dtype=bool)
    if dc.any() and pred_objs:
        ignore = inside_fraction(pred_box, gt_box.subset(np.flatnonzero(dc))) > DC_INSIDE
    care, keep = np.flatnonzero(~dc), np.flatnonzero(~ignore)
    return FrameMatch(gt=[gt_objs[k] for k in care], pred=[pred_objs[k] for k in keep],
                      iou=iou_matrix(gt_box.subset(care), pred_box.subset(keep)),
                      ignored=int(ignore.sum()))


def greedy_match(iou: np.ndarray, thr: float) -> list[tuple[int, int]]:
    """One-to-one matching, highest IoU first, pairs with IoU >= thr."""
    i, j = np.nonzero(iou >= thr)
    order = np.argsort(-iou[i, j], kind="stable")
    used_g, used_p, pairs = set(), set(), []
    for k in order:
        g, p = int(i[k]), int(j[k])
        if g not in used_g and p not in used_p:
            used_g.add(g)
            used_p.add(p)
            pairs.append((g, p))
    return pairs


def load_gt_frames(gt_dir, video: str) -> tuple[int, dict[int, list[dict]]]:
    gt = load_gt(Path(gt_dir) / f"{video}.json")
    return int(gt["num_frames"]), {int(k): v for k, v in gt["frames"].items()}


def load_pred_frames(pred_dir, video: str, tracks: bool) -> dict[int, list[dict]]:
    """Predictions grouped by frame from <video>.jsonl (tracks) or <video>.json (det/lsc/fused).

    tracks=True requires the .jsonl file; otherwise .json is preferred and .jsonl is the fallback.
    """
    pred_dir = Path(pred_dir)
    jl, js = pred_dir / f"{video}.jsonl", pred_dir / f"{video}.json"
    if not tracks and js.is_file():
        with open(js, "r", encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f)["frames"].items()}
    if not jl.is_file():
        raise FileNotFoundError(f"no predictions for {video}: {jl}" + ("" if tracks else f" / {js}"))
    out: dict[int, list[dict]] = {}
    with open(jl, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if line.strip():
                o = json.loads(line)
                if tracks and "track_id" not in o:
                    raise ValueError(f"{jl}:{ln}: missing track_id")
                out.setdefault(int(o["frame"]), []).append(o)
    return out


def frame_ids(num_frames: int, pred: dict[int, list]) -> list[int]:
    """GT frames 1..num_frames plus any frame that only has predictions (counted as FP)."""
    return sorted(set(range(1, num_frames + 1)) | set(pred))


def pct(num: float, den: float) -> float:
    return 100.0 * num / den if den > 0 else 0.0
