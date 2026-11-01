"""Tracking stage: fused json -> tracks jsonl (one line per box, score = score_fused).

Reads runs/<fuse_exp>/fuse/<ds>/<video>.json, writes runs/<exp_id>/track/<ds>/<video>.jsonl and
runs/<exp_id>/{config.yaml,log.txt,metrics.json meta.track_<ds>_<split>}.
tracker.type embed additionally reads runs/<lsc_exp>/lsc/<ds>/<video>_feats.npy (memory-mapped) and the GT
json width/height, embeds every detection with the tracker.ckpt network (logo.tracking.train_embed;
default runs/<exp_id>/checkpoints/best.pth) and passes it to run_tracker as an in-memory "emb" key; the
jsonl output is unchanged.
tracker.type lst reads the same feats, runs VPMM + LSTAdapter.associate online frame by frame (tracker.ckpt
from logo.tracking.train_lst; architecture from the checkpoint, inference keys LST_INFER_KEYS from the
config). Detections with score_fused < tracker.thr (default DEFAULT_THR[ds]) are dropped before matching.
Box text = text of the highest text_conf detection among the track's last text_window matched detections.
Post-process (paper): trajectories with mean score < tracker.traj_thr (0.6) are removed and, if
eval.spotting, so are trajectories whose text (most frequent box text) is empty or a single character.

Usage: python -m logo.tracking.run_track --config configs/E3-lsc.yaml [--fuse_exp EXP] [--lsc_exp EXP]
           [--datasets icdar2015v] [--splits val] [--device cuda] [--final]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from logo.detector.infer_paddle import read_videos, write_run_files
from logo.tracking.lst_adapter import LSTAdapter
from logo.tracking.simple_tracker import embed_settings, run_tracker, tracker_settings
from logo.utils.io import get_logger, load_json, load_yaml, set_seed
from logo.utils.paths import DATASETS, check_split, project_root
from logo.vpmm.module import VPMM

STAGE = "track"
DEFAULT_THR = {"icdar2013v": 0.4, "icdar2015v": 0.4, "dstext": 0.3}
DEFAULT_TRAJ_THR = 0.6
LST_INFER_KEYS = ("test_len", "overlap_thr", "not_mult_thr", "with_iou", "decay_time", "max_center_dist")


def save_jsonl(rows: list[dict], p: Path) -> None:
    """One compact JSON object per line, written atomically; creates parent dirs."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()


def embed_ckpt(cfg: dict, root: Path) -> Path:
    """tracker.ckpt (relative to root) or runs/<exp_id>/checkpoints/best.pth."""
    p = Path(cfg["tracker"].get("ckpt") or Path("runs") / cfg["exp_id"] / "checkpoints" / "best.pth")
    return p if p.is_absolute() else root / p


def input_thr(cfg: dict, ds: str) -> float:
    """tracker.thr, else the per-dataset default (0.4 ICDAR, 0.3 DSText)."""
    t = cfg.get("tracker") or {}
    return float(t["thr"]) if t.get("thr") is not None else DEFAULT_THR[ds]


def traj_thr(cfg: dict) -> float:
    return float((cfg.get("tracker") or {}).get("traj_thr", DEFAULT_TRAJ_THR))


def load_lst(ckpt: Path, cfg: dict, device) -> tuple[VPMM, LSTAdapter]:
    """train_lst best.pth -> (VPMM, LSTAdapter) in eval mode; config overrides LST_INFER_KEYS only."""
    ck = torch.load(Path(ckpt), map_location="cpu")
    s = dict(ck["tracker"])
    t = cfg.get("tracker") or {}
    s.update({k: t[k] for k in LST_INFER_KEYS if k in t})
    vpmm = VPMM(ck["c"], ck["d"])
    vpmm.load_state_dict(ck["vpmm"])
    lst = LSTAdapter(2 * int(ck["d"]), {"tracker": s})
    lst.load_state_dict(ck["lst"])
    return vpmm.to(device).eval(), lst.to(device).eval()


@torch.no_grad()
def vpmm_features(fused: dict, feats, vpmm: VPMM, img_wh, device, use_amp: bool, batch: int = 256) -> np.ndarray:
    """float32 [N, 2d] VPMM vectors of every fused detection in file order; LSC row = feat_idx."""
    objs = [o for k in sorted(fused["frames"], key=int) for o in fused["frames"][k]]
    out = np.zeros((len(objs), 2 * vpmm.d), dtype=np.float32)
    rows = np.array([int(o["feat_idx"]) for o in objs], dtype=np.int64)
    if len(rows) and (rows.min() < 0 or rows.max() >= len(feats)):
        raise IndexError(f"feat_idx outside 0..{len(feats) - 1}")
    quads = np.array([o["quad"] for o in objs], dtype=np.float32).reshape(-1, 4, 2)
    for b in range(0, len(objs), batch):
        f = torch.from_numpy(np.asarray(feats[rows[b:b + batch]], dtype=np.float16)).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out[b:b + batch] = vpmm(torch.from_numpy(quads[b:b + batch]).to(device), f, img_wh).float().cpu().numpy()
    return out


@torch.no_grad()
def run_lst(fused: dict, emb: np.ndarray, lst: LSTAdapter, img_wh, thr: float, text_window: int) -> list[dict]:
    """Online LST association (score_fused >= thr only) -> track rows sorted by (frame, track_id)."""
    lst.reset()
    dev = next(lst.parameters()).device
    texts: dict[int, deque] = {}
    out, base = [], 0
    for frame in sorted(int(k) for k in fused["frames"]):
        dets = fused["frames"][str(frame)]
        e = emb[base:base + len(dets)]
        base += len(dets)
        scores = np.array([float(d["score_fused"]) for d in dets], dtype=np.float64)
        keep = np.nonzero(scores >= thr)[0]
        ids = lst.associate({"quads": np.array([dets[i]["quad"] for i in keep], dtype=np.float32).reshape(-1, 8),
                             "feats": torch.from_numpy(e[keep]).to(dev), "scores": scores[keep],
                             "img_wh": img_wh})
        for j in np.argsort(ids, kind="stable"):
            tid = int(ids[j])
            if tid < 0:
                continue
            d = dets[keep[j]]
            q = texts.setdefault(tid, deque(maxlen=text_window))
            q.append((float(d.get("text_conf", 0.0)), frame, str(d.get("text", ""))))
            out.append({"frame": frame, "track_id": tid, "quad": [float(v) for v in d["quad"]],
                        "score": float(scores[keep[j]]), "text": max(q, key=lambda t: (t[0], t[1]))[2]})
    return out


def filter_trajectories(rows: list[dict], min_score: float, spotting: bool) -> list[dict]:
    """Drops trajectories with mean score < min_score and, if spotting, with text (most frequent box text)
    of fewer than 2 characters."""
    by = defaultdict(list)
    for r in rows:
        by[r["track_id"]].append(r)
    drop = set()
    for tid, rs in by.items():
        if float(np.mean([r["score"] for r in rs])) < min_score:
            drop.add(tid)
        elif spotting and len(Counter(r["text"] for r in rs).most_common(1)[0][0].strip()) < 2:
            drop.add(tid)
    return [r for r in rows if r["track_id"] not in drop]


def video_inputs(root: Path, ds: str, video: str, lsc_exp: str):
    """(LSC feats memmap, (w, h) from the GT json)."""
    gt = load_json(root / "data" / "processed" / ds / f"{video}.json")
    feats = np.load(root / "runs" / lsc_exp / "lsc" / ds / f"{video}_feats.npy", mmap_mode="r")
    return feats, (int(gt["width"]), int(gt["height"]))


def run(cfg: dict, ds: str, videos: list[str], root: Path, log, fuse_exp: str, lsc_exp: str = None,
        device=None) -> dict:
    kind = cfg["tracker"].get("type", "simple")
    if kind not in ("simple", "embed", "lst"):
        raise NotImplementedError(f"tracker.type {kind!r} is not implemented by run_track")
    fuse_dir = root / "runs" / fuse_exp / "fuse" / ds
    out_dir = root / "runs" / cfg["exp_id"] / STAGE / ds
    extra = {}
    settings = tracker_settings(cfg)
    if kind in ("embed", "lst"):
        lsc_exp = lsc_exp or (cfg.get("pipeline") or {}).get("lsc_exp") or fuse_exp
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = embed_ckpt(cfg, root)
    if kind == "embed":
        from logo.tracking.train_embed import attach_embeddings, load_embed_net  # train_embed imports this module
        net = load_embed_net(ckpt, device)
        extra = {"lsc_exp": lsc_exp, "embed_ckpt": str(ckpt), "embed": embed_settings(cfg)}
    elif kind == "lst":
        vpmm, lst = load_lst(ckpt, cfg, device)
        thr, min_score = input_thr(cfg, ds), traj_thr(cfg)
        spotting = bool((cfg.get("eval") or {}).get("spotting", True))
        settings = {"type": "lst", "text_window": settings["text_window"]}
        extra = {"lsc_exp": lsc_exp, "lst_ckpt": str(ckpt), "lst": lst.s, "thr": thr, "traj_thr": min_score,
                 "text_filter": spotting}
    n_boxes = n_tracks = 0
    for video in videos:
        fused = load_json(fuse_dir / f"{video}.json")
        if kind == "embed":
            feats, wh = video_inputs(root, ds, video, lsc_exp)
            attach_embeddings(fused, feats, net, wh, device, device.type == "cuda")
            del feats
            rows = run_tracker(fused, cfg)
        elif kind == "lst":
            feats, wh = video_inputs(root, ds, video, lsc_exp)
            emb = vpmm_features(fused, feats, vpmm, wh, device, device.type == "cuda")
            del feats
            rows = filter_trajectories(run_lst(fused, emb, lst, wh, thr, settings["text_window"]), min_score,
                                       spotting)
        else:
            rows = run_tracker(fused, cfg)
        save_jsonl(rows, out_dir / f"{video}.jsonl")
        k = len({r["track_id"] for r in rows})
        n_boxes, n_tracks = n_boxes + len(rows), n_tracks + k
        log.info("%s: %d boxes, %d tracks", video, len(rows), k)
    return {"videos": len(videos), "boxes": n_boxes, "tracks": n_tracks, "fuse_exp": fuse_exp,
            "tracker": settings, **extra}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--fuse_exp", default=None, help="exp id holding fuse/ outputs; default config exp_id")
    ap.add_argument("--lsc_exp", default=None,
                    help="tracker.type embed|lst: exp id holding lsc/ feats; default config pipeline.lsc_exp, "
                         "else fuse_exp")
    ap.add_argument("--device", default=None, help="tracker.type embed|lst: default cuda if available")
    ap.add_argument("--datasets", nargs="+", default=None, choices=DATASETS, help="default: config dataset")
    ap.add_argument("--splits", nargs="+", default=None, choices=("train", "val", "test"),
                    help="default: config split")
    ap.add_argument("--final", action="store_true", help="allow the test split (rule 5)")
    args = ap.parse_args(argv)

    root = project_root()
    cfg = copy.deepcopy(load_yaml(args.config))
    datasets = args.datasets or [cfg["dataset"]]
    splits = [check_split(s, args.final) for s in (args.splits or [cfg["split"]])]
    fuse_exp = args.fuse_exp or cfg["exp_id"]
    set_seed(cfg["seed"])
    run_dir = root / "runs" / cfg["exp_id"]
    log = get_logger("run_track", run_dir / "log.txt")
    log.info("exp=%s fuse_exp=%s tracker=%s", cfg["exp_id"], fuse_exp, cfg["tracker"].get("type", "simple"))
    for ds in datasets:
        for split in splits:
            stats = run(cfg, ds, read_videos(ds, split, root, final=args.final), root, log, fuse_exp,
                        lsc_exp=args.lsc_exp, device=args.device)
            write_run_files(cfg, run_dir, stats, key=f"track_{ds}_{split}")
            log.info("%s/%s: %s", ds, split, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
