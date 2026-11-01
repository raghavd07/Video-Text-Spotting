"""LST-Matcher training cache: fused json + LSC feats + GT -> data/cache/<ds>/<video>.npz.

Per video: keeps fused instances with score_fused > --score_thr, gathers their LSC feature rows
(runs/<lsc_exp>/lsc/<ds>/<video>_feats.npy, row = feat_idx, memory-mapped), and labels each with the GT
track_id of its greedy (highest IoU first, one-to-one) match with IoU >= --iou among the frame's
non-dont_care GT objects, else -1. Layout: logo.tracking.cache_dataset. Existing caches are kept unless
--overwrite. Writes runs/<exp_id>/{config.yaml,log.txt,metrics.json meta.lst_cache_<ds>_<split>}.

Usage: python tools/build_lst_cache.py --config configs/E4-lsc.yaml [--fuse_exp EXP] [--lsc_exp EXP]
           [--datasets icdar2015v] [--splits train] [--score_thr 0.3] [--iou 0.5] [--overwrite] [--final]
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

from logo.data.schema import load_gt
from logo.detector.infer_paddle import read_videos, write_run_files
from logo.eval.match import greedy_match
from logo.tracking.cache_dataset import cache_dir, save_cache
from logo.tracking.simple_tracker import quad_iou_matrix
from logo.utils.io import get_logger, load_json, load_yaml, set_seed
from logo.utils.paths import DATASETS, check_split, project_root


def label_greedy(quads, gt_quads, gt_ids, iou: float = 0.5) -> np.ndarray:
    """GT track id per detection (greedy one-to-one, highest IoU first, IoU >= iou), -1 where unmatched."""
    out = np.full(len(quads), -1, dtype=np.int64)
    if len(quads) and len(gt_quads):
        for d, g in greedy_match(quad_iou_matrix(quads, gt_quads), iou):
            out[d] = int(gt_ids[g])
    return out


def build_video(fused: dict, gt: dict, feats: np.ndarray, score_thr: float = 0.3, iou: float = 0.5) -> dict:
    """Cache arrays for one video (see logo.tracking.cache_dataset); feats may be a memmap."""
    n_frames = int(gt["num_frames"])
    det_frames = {int(k): v for k, v in fused["frames"].items()}
    bad = [f for f in det_frames if not 1 <= f <= n_frames]
    if bad:
        raise ValueError(f"{fused['video']}: fused frame ids {bad[:5]} outside 1..{n_frames}")
    frame_ids, quads, scores, tids, rows, offsets = [], [], [], [], [], [0]
    for f in range(1, n_frames + 1):
        objs = [o for o in det_frames.get(f, []) if float(o["score_fused"]) > score_thr]
        q = np.asarray([o["quad"] for o in objs], dtype=np.float32).reshape(-1, 8)
        gt_objs = [o for o in gt["frames"].get(str(f), []) if not o["dont_care"]]
        gq = np.asarray([o["quad"] for o in gt_objs], dtype=np.float64).reshape(-1, 8)
        tids.append(label_greedy(q, gq, [o["track_id"] for o in gt_objs], iou))
        frame_ids.append(np.full(len(objs), f, dtype=np.int32))
        quads.append(q)
        scores.append(np.asarray([o["score_fused"] for o in objs], dtype=np.float32))
        rows.extend(int(o["feat_idx"]) for o in objs)
        offsets.append(offsets[-1] + len(objs))
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) and (rows.min() < 0 or rows.max() >= len(feats)):
        raise IndexError(f"{fused['video']}: feat_idx outside feats rows 0..{len(feats) - 1}")
    order = np.argsort(rows, kind="stable")  # read the memmap in ascending row order
    gathered = np.empty((len(rows),) + tuple(feats.shape[1:]), dtype=np.float16)
    gathered[order] = np.asarray(feats[rows[order]], dtype=np.float16)
    return {"frame_ids": np.concatenate(frame_ids), "quads": np.concatenate(quads),
            "feats": gathered, "score_fused": np.concatenate(scores), "track_ids": np.concatenate(tids),
            "frames": np.arange(1, n_frames + 1, dtype=np.int32),
            "frame_offsets": np.asarray(offsets, dtype=np.int64),
            "img_wh": np.asarray([gt["width"], gt["height"]], dtype=np.int32)}


def run(ds: str, videos: list[str], root: Path, fuse_exp: str, lsc_exp: str, log, out_dir: Path,
        score_thr: float = 0.3, iou: float = 0.5, overwrite: bool = False) -> dict:
    t0 = time.perf_counter()
    n_inst = n_pos = n_built = n_kept = 0
    track_set = set()
    for video in videos:
        out = out_dir / f"{video}.npz"
        if out.is_file() and not overwrite:
            n_kept += 1
            continue
        fused = load_json(root / "runs" / fuse_exp / "fuse" / ds / f"{video}.json")
        gt = load_gt(root / "data" / "processed" / ds / f"{video}.json")
        feats = np.load(root / "runs" / lsc_exp / "lsc" / ds / f"{video}_feats.npy", mmap_mode="r")
        arr = build_video(fused, gt, feats, score_thr, iou)
        del feats
        save_cache(out, arr)
        pos = arr["track_ids"] >= 0
        n_inst += len(pos)
        n_pos += int(pos.sum())
        track_set.update((video, int(t)) for t in arr["track_ids"][pos])
        n_built += 1
        log.info("%s: frames=%d instances=%d matched=%d tracks=%d", video, len(arr["frames"]), len(pos),
                 int(pos.sum()), len(np.unique(arr["track_ids"][pos])))
    return {"videos_built": n_built, "videos_kept": n_kept, "instances": n_inst, "matched": n_pos,
            "matched_frac": round(n_pos / n_inst, 4) if n_inst else 0.0, "gt_tracks_hit": len(track_set),
            "score_thr": score_thr, "iou": iou, "fuse_exp": fuse_exp, "lsc_exp": lsc_exp,
            "seconds": round(time.perf_counter() - t0, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--fuse_exp", default=None, help="exp id holding fuse/ outputs; default config exp_id")
    ap.add_argument("--lsc_exp", default=None,
                    help="exp id holding lsc/ feats; default config pipeline.lsc_exp, else exp_id")
    ap.add_argument("--datasets", nargs="+", default=None, choices=DATASETS, help="default: config dataset")
    ap.add_argument("--splits", nargs="+", default=["train"], choices=("train", "val", "test"))
    ap.add_argument("--score_thr", type=float, default=0.3, help="keep score_fused > this")
    ap.add_argument("--iou", type=float, default=0.5, help="GT match IoU threshold")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--final", action="store_true", help="allow the test split (rule 5)")
    args = ap.parse_args(argv)

    root = project_root()
    cfg = copy.deepcopy(load_yaml(args.config))
    datasets = args.datasets or [cfg["dataset"]]
    splits = [check_split(s, args.final) for s in args.splits]
    fuse_exp = args.fuse_exp or cfg["exp_id"]
    lsc_exp = args.lsc_exp or (cfg.get("pipeline") or {}).get("lsc_exp") or cfg["exp_id"]
    set_seed(cfg["seed"])
    run_dir = root / "runs" / cfg["exp_id"]
    log = get_logger("build_lst_cache", run_dir / "log.txt")
    log.info("exp=%s fuse_exp=%s lsc_exp=%s score_thr=%s iou=%s", cfg["exp_id"], fuse_exp, lsc_exp,
             args.score_thr, args.iou)
    for ds in datasets:
        for split in splits:
            stats = run(ds, read_videos(ds, split, root, final=args.final), root, fuse_exp, lsc_exp, log,
                        cache_dir(ds), args.score_thr, args.iou, args.overwrite)
            write_run_files(cfg, run_dir, stats, key=f"lst_cache_{ds}_{split}")
            log.info("%s/%s: %s", ds, split, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
