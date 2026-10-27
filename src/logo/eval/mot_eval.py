"""CLEAR-MOT / ID metrics (percent) with py-motmetrics; spotting adds an exact normalized-text constraint."""
from __future__ import annotations

import motmetrics as mm
import numpy as np

from logo.eval.match import frame_ids, load_gt_frames, load_pred_frames, normalize_text, pct, prepare_frame

PER_VIDEO_KEYS = ("frames", "num_objects", "misses", "fp", "switches", "idtp", "idfp", "idfn")
_MM_METRICS = {"num_frames": "frames", "num_objects": "num_objects", "num_misses": "misses",
               "num_false_positives": "fp", "num_switches": "switches", "idtp": "idtp",
               "idfp": "idfp", "idfn": "idfn", "num_detections": "num_detections", "motp": "motp"}
MT_RATIO, ML_RATIO = 0.8, 0.2


def _ids(objs: list[dict], what: str, video: str, fid: int) -> np.ndarray:
    ids = np.array([int(o["track_id"]) for o in objs], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{video} frame {fid}: duplicate {what} track_id")
    return ids


def _accumulate(gt: dict, pred: dict, num_frames: int, video: str, iou: float, spotting: bool):
    acc = mm.MOTAccumulator(auto_id=False)
    ignored = 0
    for fid in frame_ids(num_frames, pred):
        fm = prepare_frame(gt.get(fid, []), pred.get(fid, []), spotting)
        dist = 1.0 - fm.iou
        dist[fm.iou < iou] = np.nan
        if spotting and dist.size:
            gt_t = np.array([normalize_text(o["text"]) for o in fm.gt], dtype=object)
            pr_t = np.array([normalize_text(o.get("text", "")) for o in fm.pred], dtype=object)
            dist[gt_t[:, None] != pr_t[None, :]] = np.nan
        acc.update(_ids(fm.gt, "GT", video, fid), _ids(fm.pred, "predicted", video, fid), dist,
                   frameid=fid)
        ignored += fm.ignored
    return acc, ignored


def _mt_ml(acc) -> tuple[int, int, int]:
    """(#GT tracks, #covered >= 80% of lifetime, #covered <= 20%); lifetime = care frames."""
    ev = acc.mot_events
    ev = ev[ev.Type.isin(["MATCH", "SWITCH", "MISS"])]
    total = ev["OId"].value_counts()
    if total.empty:
        return 0, 0, 0
    tracked = ev.loc[ev.Type != "MISS", "OId"].value_counts().reindex(total.index, fill_value=0)
    ratio = tracked / total
    return int(len(ratio)), int((ratio >= MT_RATIO).sum()), int((ratio <= ML_RATIO).sum())


def evaluate_mot(gt_dir, track_dir, videos, spotting, iou=0.5) -> dict:
    """{"MOTA","MOTP","IDF1","MT","ML","per_video":{video:{PER_VIDEO_KEYS}},"counts":{...}}.

    Distance = 1 - IoU, NaN below `iou` (and, if spotting, for unequal normalized texts).
    MOTP is reported as mean IoU of matched pairs. Overall values are pooled over all videos.
    """
    mh = mm.metrics.create()
    per_video = {}
    tot = {k: 0 for k in (*PER_VIDEO_KEYS, "num_detections", "gt_tracks", "mt", "ml", "ignored")}
    sum_dist = 0.0
    for video in videos:
        n, gt = load_gt_frames(gt_dir, video)
        pred = load_pred_frames(track_dir, video, tracks=True)
        acc, ignored = _accumulate(gt, pred, n, video, iou, bool(spotting))
        row = mh.compute(acc, metrics=list(_MM_METRICS), name=video).iloc[0]
        c = {v: row[k] for k, v in _MM_METRICS.items()}
        per_video[video] = {k: int(c[k]) for k in PER_VIDEO_KEYS}
        ndet = int(c["num_detections"])
        if ndet:
            sum_dist += float(c["motp"]) * ndet
        n_tracks, mt, ml = _mt_ml(acc)
        for k in PER_VIDEO_KEYS:
            tot[k] += per_video[video][k]
        tot["num_detections"] += ndet
        tot["gt_tracks"] += n_tracks
        tot["mt"] += mt
        tot["ml"] += ml
        tot["ignored"] += ignored
    nobj = tot["num_objects"]
    return {
        "MOTA": 100.0 * (1.0 - (tot["misses"] + tot["fp"] + tot["switches"]) / nobj) if nobj else 0.0,
        "MOTP": 100.0 * (1.0 - sum_dist / tot["num_detections"]) if tot["num_detections"] else 0.0,
        "IDF1": pct(2 * tot["idtp"], 2 * tot["idtp"] + tot["idfp"] + tot["idfn"]),
        "MT": pct(tot["mt"], tot["gt_tracks"]),
        "ML": pct(tot["ml"], tot["gt_tracks"]),
        "per_video": per_video,
        "counts": tot,
    }
