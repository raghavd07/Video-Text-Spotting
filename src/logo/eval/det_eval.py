"""Frame-level detection precision / recall / F (percent), greedy one-to-one IoU matching."""
from __future__ import annotations

from logo.eval.match import frame_ids, greedy_match, load_gt_frames, load_pred_frames, pct, prepare_frame


def evaluate_detection(gt_dir, pred_dir, videos, iou=0.5) -> dict:
    """{"precision","recall","f","per_video":{video:{tp,fp,fn,ignored}},"counts":{tp,fp,fn,ignored}}."""
    per_video, tot = {}, {"tp": 0, "fp": 0, "fn": 0, "ignored": 0}
    for video in videos:
        n, gt = load_gt_frames(gt_dir, video)
        pred = load_pred_frames(pred_dir, video, tracks=False)
        c = {"tp": 0, "fp": 0, "fn": 0, "ignored": 0}
        for fid in frame_ids(n, pred):
            fm = prepare_frame(gt.get(fid, []), pred.get(fid, []))
            tp = len(greedy_match(fm.iou, iou))
            c["tp"] += tp
            c["fp"] += len(fm.pred) - tp
            c["fn"] += len(fm.gt) - tp
            c["ignored"] += fm.ignored
        per_video[video] = c
        for k in tot:
            tot[k] += c[k]
    p = pct(tot["tp"], tot["tp"] + tot["fp"])
    r = pct(tot["tp"], tot["tp"] + tot["fn"])
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"precision": p, "recall": r, "f": f, "per_video": per_video, "counts": tot}
