"""LSC inference stage: det json -> lsc json + <video>_feats.npy (CONTRACT).

Per video: boxes with score > lsc.det_thr are kept (others dropped), cropped with crop_quad from
data/frames/<ds>/<video>/%06d.jpg and recognised by ABINetRecognizer in batches of
lsc.infer.batch_size (crops are buffered across frames). Each kept object gets
text / text_conf / p_eos / lang_hard / feat_idx; lang_hard = 1 if the first decoded token is not the
end token, i.e. the decoded text is non-empty. feat_idx counts kept boxes across the video in file
order (frames in json key order, boxes in list order) and is the row of <video>_feats.npy
(float16 [N,c,d], streamed to disk through a memmap, so RAM stays at one batch).

Reads runs/<det_exp>/det/<ds>/<video>.json, writes runs/<exp_id>/lsc/<ds>/<video>.{json,_feats.npy}
and runs/<exp_id>/{config.yaml,log.txt,metrics.json meta.lsc_infer_<ds>_<split>}.
Checkpoint: --ckpt, else runs/<exp_id>/checkpoints/best.pth if present, else lsc.ckpt
(null -> ABINet repo weights). Existing outputs are skipped unless --overwrite.

Usage: python -m logo.lsc.run_lsc --config configs/E3-lsc.yaml --det_exp DET_EXP
           [--datasets icdar2015v] [--splits val] [--final] [--ckpt P] [--device cuda] [--overwrite]
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from logo.detector.infer_paddle import read_videos, resolve, write_run_files
from logo.lsc.roi_rotate import crop_quad
from logo.utils.io import get_logger, load_json, load_yaml, save_json, set_seed
from logo.utils.paths import DATASETS, check_split, project_root

STAGE = "lsc"
OUT_HW = (32, 128)
INFER_DEFAULTS = dict(batch_size=64, amp=True)


def infer_settings(cfg: dict) -> dict:
    """lsc.infer merged over INFER_DEFAULTS."""
    ic = dict(INFER_DEFAULTS, **((cfg.get("lsc") or {}).get("infer") or {}))
    ic["batch_size"], ic["amp"] = int(ic["batch_size"]), bool(ic["amp"])
    if ic["batch_size"] < 1:
        raise ValueError("lsc.infer.batch_size must be >= 1")
    return ic


def feats_path(out_dir: Path, video: str) -> Path:
    return out_dir / f"{video}_feats.npy"


def pick_ckpt(cfg: dict, root: Path, ckpt: Optional[Path] = None) -> Optional[Path]:
    """--ckpt > runs/<exp_id>/checkpoints/best.pth > lsc.ckpt (None = ABINet repo default)."""
    if ckpt is not None:
        return resolve(ckpt, root)
    best = root / "runs" / cfg["exp_id"] / "checkpoints" / "best.pth"
    if best.is_file():
        return best
    c = (cfg.get("lsc") or {}).get("ckpt")
    return None if c is None else resolve(c, root)


def lsc_fields(pred: dict, feat_idx: int) -> dict:
    """Recogniser output -> the lsc keys added to a det object."""
    return {"text": pred["text"], "text_conf": float(pred["text_conf"]), "p_eos": float(pred["p_eos"]),
            "lang_hard": int(pred["text"] != ""), "feat_idx": int(feat_idx)}


def run_video(det: dict, frame_dir: Path, recognizer, det_thr: float, feats_out: Path) -> dict:
    """Process one det json; writes feats_out (atomically) and returns the lsc json dict.

    `recognizer` needs predict(list[crop]) -> list[{text,text_conf,p_eos,feat[c,d]}], attributes
    c, d and batch_size.
    """
    kept = [(fid, o) for fid, objs in det["frames"].items() for o in objs if float(o["score"]) > det_thr]
    frames_out = {fid: [] for fid in det["frames"]}
    c, d, bs = int(recognizer.c), int(recognizer.d), max(1, int(recognizer.batch_size))
    feats_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = feats_out.with_name(feats_out.name + ".tmp.npy")
    feats = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(kept), c, d))
    try:
        crops, pending, n_done = [], [], 0

        def flush():
            nonlocal n_done
            preds = recognizer.predict(crops)
            if len(preds) != len(crops):
                raise RuntimeError(f"recogniser returned {len(preds)} outputs for {len(crops)} crops")
            for (fid, o), p in zip(pending, preds):
                f = np.asarray(p["feat"])
                if f.shape != (c, d):
                    raise RuntimeError(f"feature shape {f.shape} != ({c}, {d})")
                feats[n_done] = f
                frames_out[fid].append({**o, **lsc_fields(p, n_done)})
                n_done += 1
            crops.clear()
            pending.clear()

        cur_fid, img = None, None
        for fid, o in kept:
            if fid != cur_fid:
                p = frame_dir / f"{int(fid):06d}.jpg"
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(f"cannot read frame {p}")
                cur_fid = fid
            crops.append(crop_quad(img, o["quad"], OUT_HW))
            pending.append((fid, o))
            if len(crops) >= bs:
                flush()
        if crops:
            flush()
        feats.flush()
        feats = None  # close the memmap before the rename (Windows cannot replace a mapped file)
        os.replace(tmp, feats_out)
    finally:
        feats = None
        if tmp.exists():
            tmp.unlink()
    meta = {k: v for k, v in det.items() if k != "frames"}
    return {**meta, "frames": frames_out}


def run(cfg: dict, ds: str, videos: list[str], recognizer, root: Path, log, det_exp: str,
        overwrite: bool = False) -> dict:
    """Run every video of one (dataset, split); returns timing/count stats."""
    det_dir = root / "runs" / det_exp / "det" / ds
    out_dir = root / "runs" / cfg["exp_id"] / STAGE / ds
    det_thr = float(cfg["lsc"]["det_thr"])
    n_boxes, n_pos, n_videos, t0 = 0, 0, 0, time.perf_counter()
    for video in videos:
        out, fo = out_dir / f"{video}.json", feats_path(out_dir, video)
        if out.exists() and fo.exists() and not overwrite:
            log.info("skip %s (exists)", video)
            continue
        det = load_json(det_dir / f"{video}.json")
        lsc = run_video(det, root / "data" / "frames" / ds / video, recognizer, det_thr, fo)
        save_json(lsc, out)  # after the feats file: an existing json implies existing feats
        objs = [o for v in lsc["frames"].values() for o in v]
        pos = sum(o["lang_hard"] for o in objs)
        n_boxes, n_pos, n_videos = n_boxes + len(objs), n_pos + pos, n_videos + 1
        log.info("%s: %d boxes > %.2f, lang_hard=1: %d", video, len(objs), det_thr, pos)
    dt = time.perf_counter() - t0
    return {"videos": n_videos, "boxes": n_boxes, "lang_hard_pos": n_pos, "det_exp": det_exp,
            "det_thr": det_thr, "seconds": round(dt, 2),
            "ms_per_box": round(1000.0 * dt / n_boxes, 3) if n_boxes else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--det_exp", default=None, help="exp id holding det/ outputs; default config exp_id")
    ap.add_argument("--datasets", nargs="+", default=None, choices=DATASETS, help="default: config dataset")
    ap.add_argument("--splits", nargs="+", default=None, choices=("train", "val", "test"),
                    help="default: config split")
    ap.add_argument("--final", action="store_true", help="allow the test split (rule 5)")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    import torch

    from logo.lsc.abinet_wrapper import ABINetRecognizer

    root = project_root()
    cfg = copy.deepcopy(load_yaml(args.config))
    datasets = args.datasets or [cfg["dataset"]]
    splits = [check_split(s, args.final) for s in (args.splits or [cfg["split"]])]
    det_exp = args.det_exp or cfg["exp_id"]
    ic = infer_settings(cfg)
    set_seed(cfg["seed"])
    run_dir = root / "runs" / cfg["exp_id"]
    log = get_logger("run_lsc", run_dir / "log.txt")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = pick_ckpt(cfg, root, args.ckpt)
    if ckpt is None or ckpt != run_dir / "checkpoints" / "best.pth":
        log.warning("not using a fine-tuned checkpoint of this exp: %s", ckpt or "ABINet repo default")
    rec = ABINetRecognizer(ckpt, device, fp16=ic["amp"])
    rec.batch_size = ic["batch_size"]
    log.info("exp=%s det_exp=%s datasets=%s splits=%s ckpt=%s device=%s bs=%d amp=%s", cfg["exp_id"],
             det_exp, datasets, splits, ckpt, device, ic["batch_size"], rec.fp16)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for ds in datasets:
        for split in splits:
            videos = read_videos(ds, split, root, final=args.final)
            log.info("=== %s/%s: %d videos", ds, split, len(videos))
            stats = run(cfg, ds, videos, rec, root, log, det_exp, args.overwrite)
            stats.update(ckpt=str(ckpt) if ckpt else None, device=device, batch_size=ic["batch_size"],
                         amp=rec.fp16, peak_gpu_mem_mb=(round(torch.cuda.max_memory_allocated() / 2**20, 1)
                                                        if device.startswith("cuda") else None))
            write_run_files(cfg, run_dir, stats, key=f"lsc_infer_{ds}_{split}")
            log.info("%s/%s: %s", ds, split, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
