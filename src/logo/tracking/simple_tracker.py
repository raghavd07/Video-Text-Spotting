"""ByteTrack-style two-stage IoU tracker over fused detections (tracker.type: simple | embed).

Per frame (ascending frame id):
  1. detections with score_fused > high_thr are matched to all live tracks (Hungarian on quad IoU,
     pairs below iou_gate rejected); track quads are predicted by constant centre velocity;
  2. still-unmatched tracks are matched the same way to detections with low_thr < score_fused <= high_thr;
  3. unmatched high-score detections start new tracks; other unmatched detections are dropped;
  4. a track not matched for more than max_age frames is removed.
Track text at each output box = text of the highest text_conf detection among the track's last
text_window matched detections (ties -> most recent).

Config (tracker{}; missing keys use the defaults below): high_thr 0.6, low_thr 0.3, iou_gate 0.3,
max_age 15, text_window 10. tracker.thr is not used by this tracker.

tracker.type embed: every detection carries an in-memory "emb" (L2-normalised vector, attached by
run_track from logo.tracking.train_embed; never written to disk). Stage 1 then uses the cost
(1 - lam) * (1 - IoU) + lam * cos_dist, cos_dist = clip(1 - cos(track_emb, det_emb), 0, 1); a pair is
admissible if IoU >= iou_gate or cos_dist < emb_gate. Stage 2 (low-score detections) stays IoU-only, as in
BoT-SORT. Track embedding = EMA of matched detection embeddings (emb_momentum), re-normalised.
Extra keys (EMBED_DEFAULTS): lam 0.5, emb_gate 0.0 (off), emb_momentum 0.9. lam 0 == tracker.type simple.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.optimize import linear_sum_assignment

DEFAULTS = {"high_thr": 0.6, "low_thr": 0.3, "iou_gate": 0.3, "max_age": 15, "text_window": 10}
EMBED_DEFAULTS = {"lam": 0.5, "emb_gate": 0.0, "emb_momentum": 0.9}


def tracker_settings(cfg: dict) -> dict:
    """Accepts the full config or just its tracker{} section."""
    t = cfg.get("tracker", cfg) or {}
    s = {k: type(v)(t.get(k, v)) for k, v in DEFAULTS.items()}
    if not 0.0 <= s["low_thr"] <= s["high_thr"]:
        raise ValueError(f"need 0 <= low_thr <= high_thr, got {s['low_thr']}, {s['high_thr']}")
    if s["max_age"] < 0 or s["text_window"] < 1:
        raise ValueError("max_age must be >= 0 and text_window >= 1")
    return s


def embed_settings(cfg: dict) -> dict:
    """tracker.type embed extras; accepts the full config or just its tracker{} section."""
    t = cfg.get("tracker", cfg) or {}
    s = {k: float(t.get(k, v)) for k, v in EMBED_DEFAULTS.items()}
    if not 0.0 <= s["lam"] <= 1.0 or not 0.0 <= s["emb_momentum"] < 1.0:
        raise ValueError(f"need 0 <= lam <= 1 and 0 <= emb_momentum < 1, got {s['lam']}, {s['emb_momentum']}")
    return s


def quad_iou_matrix(a, b) -> np.ndarray:
    """IoU [n, m] between quads a [n, 8] and b [m, 8] (shapely convex hulls, bbox prefilter)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4, 2)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4, 2)
    iou = np.zeros((len(a), len(b)))
    if len(a) == 0 or len(b) == 0:
        return iou
    amin, amax = a.min(axis=1), a.max(axis=1)
    bmin, bmax = b.min(axis=1), b.max(axis=1)
    overlap = ((amin[:, None, :] < bmax[None, :, :]) & (bmin[None, :, :] < amax[:, None, :])).all(-1)
    ii, jj = np.nonzero(overlap)
    if len(ii) == 0:
        return iou
    # convex hull fixes bowtie point orders; text quads are convex so valid quads are unchanged
    pa = shapely.convex_hull(shapely.polygons(a))
    pb = shapely.convex_hull(shapely.polygons(b))
    inter = shapely.area(shapely.intersection(pa[ii], pb[jj]))
    union = shapely.area(pa)[ii] + shapely.area(pb)[jj] - inter
    iou[ii, jj] = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    return iou


def match(iou: np.ndarray, gate: float) -> list[tuple[int, int]]:
    """Hungarian assignment maximising IoU; pairs with IoU < gate are discarded."""
    if iou.size == 0:
        return []
    gated = np.where(iou >= gate, iou, 0.0)
    rows, cols = linear_sum_assignment(1.0 - gated)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if iou[r, c] >= gate]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """clip(1 - cos, 0, 1) [n, m] between rows of a [n, k] and b [m, k] (any norm, zero rows -> 1)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return np.clip(1.0 - a @ b.T, 0.0, 1.0)


def match_embed(iou: np.ndarray, cos: np.ndarray, lam: float, iou_gate: float,
                emb_gate: float) -> list[tuple[int, int]]:
    """Hungarian on (1-lam)(1-IoU) + lam*cos; admissible pairs: IoU >= iou_gate or cos < emb_gate."""
    if iou.size == 0:
        return []
    ok = (iou >= iou_gate) | (cos < emb_gate)
    cost = np.where(ok, (1.0 - lam) * (1.0 - iou) + lam * cos, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if ok[r, c]]


@dataclass
class _Track:
    track_id: int
    quad: np.ndarray  # [4, 2] last matched quad
    last_frame: int
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))  # centre px / frame
    texts: deque = field(default_factory=deque)  # (text_conf, frame, text)
    emb: np.ndarray | None = None  # tracker.type embed: unit-norm EMA appearance

    def predict(self, frame: int) -> np.ndarray:
        return self.quad + self.velocity * (frame - self.last_frame)

    def update(self, quad: np.ndarray, frame: int, det: dict) -> None:
        dt = frame - self.last_frame
        if dt > 0:
            self.velocity = (quad.mean(axis=0) - self.quad.mean(axis=0)) / dt
        self.quad, self.last_frame = quad, frame
        self.texts.append((float(det.get("text_conf", 0.0)), frame, str(det.get("text", ""))))

    def update_emb(self, emb, momentum: float) -> None:
        e = np.asarray(emb, dtype=np.float64)
        e = e / max(float(np.linalg.norm(e)), 1e-12)
        if self.emb is not None:
            e = momentum * self.emb + (1.0 - momentum) * e
            e = e / max(float(np.linalg.norm(e)), 1e-12)
        self.emb = e

    def text(self) -> str:
        return max(self.texts, key=lambda t: (t[0], t[1]))[2] if self.texts else ""


def run_tracker(fused: dict, cfg: dict) -> list[dict]:
    """fused json -> track boxes [{"frame","track_id","quad","score","text"}] sorted by (frame, track_id)."""
    s = tracker_settings(cfg)
    kind = (cfg.get("tracker", cfg) or {}).get("type", "simple")
    if kind not in ("simple", "embed"):
        raise ValueError(f"run_tracker supports tracker.type simple|embed, got {kind!r}")
    es = embed_settings(cfg) if kind == "embed" else None
    tracks: list[_Track] = []
    next_id = 1
    out: list[dict] = []
    for frame in sorted(int(k) for k in fused["frames"]):
        dets = fused["frames"][str(frame)]
        tracks = [t for t in tracks if frame - t.last_frame <= s["max_age"]]
        scores = np.array([float(d["score_fused"]) for d in dets])
        quads = np.array([d["quad"] for d in dets], dtype=np.float64).reshape(-1, 4, 2)
        if es is not None and dets:
            if any("emb" not in d for d in dets):
                raise ValueError(f"tracker.type embed: frame {frame} has detections without 'emb'")
            embs = np.stack([np.asarray(d["emb"], dtype=np.float64).reshape(-1) for d in dets])
        high = np.nonzero(scores > s["high_thr"])[0]
        low = np.nonzero((scores > s["low_thr"]) & (scores <= s["high_thr"]))[0]

        assigned: dict[int, int] = {}  # det index -> track index
        free = list(range(len(tracks)))
        for stage, pool in enumerate((high, low)):
            if len(pool) == 0 or not free:
                continue
            pred = np.stack([tracks[i].predict(frame) for i in free])
            iou = quad_iou_matrix(pred, quads[pool])
            if es is not None and stage == 0:
                cos = cosine_distance(np.stack([tracks[i].emb for i in free]), embs[pool])
                pairs = match_embed(iou, cos, es["lam"], s["iou_gate"], es["emb_gate"])
            else:
                pairs = match(iou, s["iou_gate"])
            for r, c in pairs:
                assigned[int(pool[c])] = free[r]
            taken = {free[r] for r, _ in pairs}
            free = [i for i in free if i not in taken]

        for d, ti in assigned.items():
            tracks[ti].update(quads[d], frame, dets[d])
            if es is not None:
                tracks[ti].update_emb(embs[d], es["emb_momentum"])
        for d in high:
            if int(d) not in assigned:
                t = _Track(next_id, quads[d], frame, texts=deque(maxlen=s["text_window"]))
                t.update(quads[d], frame, dets[d])  # dt = 0: velocity stays zero
                if es is not None:
                    t.update_emb(embs[d], es["emb_momentum"])
                tracks.append(t)
                assigned[int(d)] = len(tracks) - 1
                next_id += 1

        rows = []
        for d, ti in assigned.items():
            t = tracks[ti]
            rows.append({"frame": frame, "track_id": t.track_id, "quad": [float(v) for v in dets[d]["quad"]],
                         "score": float(scores[d]), "text": t.text()})
        out.extend(sorted(rows, key=lambda r: r["track_id"]))
    return out
