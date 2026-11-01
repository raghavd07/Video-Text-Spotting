"""Train VPMM + LSTAdapter (LST-Matcher association) on the LST cache (tools/build_lst_cache.py).

Per iteration: one LSTCacheDataset clip (clip_len frames of one video) -> VPMM(quads, LSC feats, img_wh)
[n,2d] -> LSTAdapter.train_step (long + short-term association loss); VPMM and the adapter are trained
jointly. AdamW(lr, weight_decay), linear warm-up from warmup_factor*lr, then cosine to min_lr; AMP
(GradScaler) on CUDA; full-model grad-norm clip = tracker grad_clip (GoMatching CLIP_VALUE). Clips without
any GT-matched instance give no loss and are skipped (not counted as iterations). On CUDA OOM clip_len
drops by 1 (down to min_clip_len) and a new epoch starts.

Validation every val_every iterations: LSTAdapter.associate over every frame of the val-split caches ->
runs/<exp_id>/val_track/<ds>/<video>.jsonl (score = score_fused, text = "") -> tools/evaluate.py
--mode track -> runs/<exp_id>/val_track/metrics.json. Best checkpoint = max (IDF1, MOTA).

Writes runs/<exp_id>/{config.yaml, log.txt, metrics.json meta.lst_train, checkpoints/{best,last}.pth}.
best.pth = {"vpmm", "lst", "c", "d", "tracker", "iter", "val"}; last.pth adds optimizer, scheduler,
scaler, RNG and data position for --resume. Settings: tracker.train (TRAIN_DEFAULTS).
--smoke: smoke_iters iterations, val every smoke_val_every on the first smoke_val_videos val videos,
exp_id <exp_id>_smoke.

Usage: python -m logo.tracking.train_lst --config configs/E5-lst.yaml [--smoke] [--resume] [--device cuda]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from logo.data.schema import write_split
from logo.detector.infer_paddle import read_videos
from logo.tracking.cache_dataset import CACHE_KEYS, LSTCacheDataset, cache_dir, collate_first, npz_memmap
from logo.tracking.lst_adapter import LSTAdapter, lst_settings
from logo.tracking.run_track import save_jsonl
from logo.utils.io import get_logger, load_json, load_yaml, save_json, set_seed
from logo.utils.paths import check_split, project_root
from logo.vpmm.module import VPMM

TRAIN_DEFAULTS = dict(train_split="train", val_split="val", iters=20000, lr=5.0e-5, weight_decay=1.0e-4,
                      warmup_iters=500, warmup_factor=0.001, min_lr=0.0, clip_len=6, min_clip_len=2,
                      stride=1, amp=True, num_workers=0, log_every=20, val_every=1000, smoke_iters=200,
                      smoke_val_every=100, smoke_val_videos=2)
_INT_KEYS = ("iters", "warmup_iters", "clip_len", "min_clip_len", "stride", "num_workers", "log_every",
             "val_every", "smoke_iters", "smoke_val_every", "smoke_val_videos")
_FLOAT_KEYS = ("lr", "weight_decay", "warmup_factor", "min_lr")
# tools/evaluate.py of this repository (LOGO_ROOT only moves data/runs, not code)
_EVALUATE_PY = Path(__file__).resolve().parents[3] / "tools" / "evaluate.py"


def train_settings(cfg: dict, smoke: bool = False) -> dict:
    """tracker.train merged over TRAIN_DEFAULTS; smoke replaces iters / val_every."""
    tc = dict(TRAIN_DEFAULTS, **(((cfg.get("tracker") or {}).get("train")) or {}))
    for k in _INT_KEYS:
        tc[k] = int(tc[k])
    for k in _FLOAT_KEYS:
        tc[k] = float(tc[k])
    tc["amp"] = bool(tc["amp"])
    if smoke:
        tc["iters"], tc["val_every"] = tc["smoke_iters"], tc["smoke_val_every"]
    if tc["iters"] < 1 or tc["val_every"] < 1 or tc["log_every"] < 1:
        raise ValueError("tracker.train: iters, val_every and log_every must be >= 1")
    if not 2 <= tc["min_clip_len"] <= tc["clip_len"]:
        raise ValueError("tracker.train: need 2 <= min_clip_len <= clip_len")
    tc["warmup_iters"] = min(tc["warmup_iters"], tc["iters"] // 10) if smoke else tc["warmup_iters"]
    return tc


def lr_factor(it: int, tc: dict) -> float:
    """Multiplier of lr after `it` optimizer steps: linear warm-up, then cosine down to min_lr."""
    warm, total = tc["warmup_iters"], tc["iters"]
    if it < warm:
        return tc["warmup_factor"] + (1.0 - tc["warmup_factor"]) * it / warm
    floor = tc["min_lr"] / tc["lr"]
    progress = min(1.0, (it - warm) / max(1, total - warm))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _evaluate_tool():
    name = "_logo_evaluate_tool"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _EVALUATE_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[name] = mod
    return sys.modules[name]


def cache_files(ds: str, split: str) -> list[Path]:
    """Cache npz of every video of the split (never the test split here: rule 5)."""
    root = project_root()
    files = [cache_dir(ds) / f"{v}.npz" for v in read_videos(ds, check_split(split), root)]
    missing = [p.stem for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"no LST cache for {ds}/{split} videos {missing[:5]}; run "
                                f"tools/build_lst_cache.py --splits {split}")
    return files


def feat_shape(path: Path) -> tuple[int, int]:
    c, d = npz_memmap(path, "feats").shape[1:]
    return int(c), int(d)


def embed_clip(vpmm: VPMM, clip: list[dict], device) -> list[dict]:
    """Clip frame dicts with raw LSC "feats" [n,c,d] replaced by VPMM output [n,2d] (one batched call)."""
    n = [len(f["scores"]) for f in clip]
    quads = torch.from_numpy(np.concatenate([f["quads"].reshape(-1, 8) for f in clip])).view(-1, 4, 2)
    feats = torch.cat([f["feats"] for f in clip]).to(device, non_blocking=True)
    emb = vpmm(quads.to(device), feats, clip[0]["img_wh"]).split(n, dim=0)
    return [dict(f, feats=e) for f, e in zip(clip, emb)]


@torch.no_grad()
def track_cache_video(vpmm: VPMM, lst: LSTAdapter, path: Path, device, use_amp: bool) -> list[dict]:
    """Runs LSTAdapter.associate over every frame of one cache npz -> track rows (CONTRACT tracks)."""
    with np.load(path) as z:
        m = {k: z[k] for k in CACHE_KEYS if k != "feats"}
    feats = npz_memmap(path, "feats")
    wh = (int(m["img_wh"][0]), int(m["img_wh"][1]))
    off = m["frame_offsets"]
    lst.reset()
    rows = []
    for k, frame in enumerate(m["frames"]):
        lo, hi = int(off[k]), int(off[k + 1])
        quads = m["quads"][lo:hi].astype(np.float32)
        f = torch.from_numpy(np.array(feats[lo:hi], dtype=np.float16)).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            emb = vpmm(torch.from_numpy(quads).view(-1, 4, 2).to(device), f, wh)
        ids = lst.associate({"quads": quads, "feats": emb.float(), "scores": m["score_fused"][lo:hi],
                             "img_wh": wh})
        for j in np.argsort(ids, kind="stable"):
            if ids[j] >= 0:
                rows.append({"frame": int(frame), "track_id": int(ids[j]),
                             "quad": [round(float(v), 2) for v in quads[j]],
                             "score": round(float(m["score_fused"][lo + j]), 5), "text": ""})
    return rows


def validate(vpmm, lst, files: list[Path], ds: str, out_dir: Path, device, use_amp: bool, iou: float) -> dict:
    """Tracks the val caches into out_dir/<ds>/ and scores them with tools/evaluate.py --mode track."""
    vpmm.eval()
    lst.eval()
    for p in files:
        save_jsonl(track_cache_video(vpmm, lst, p, device, use_amp), out_dir / ds / f"{p.stem}.jsonl")
    split_txt = out_dir / f"{ds}_val.txt"
    write_split(split_txt, [p.stem for p in files])
    out = out_dir / "metrics.json"
    _evaluate_tool().main(["--gt_dir", str(project_root() / "data" / "processed" / ds), "--pred_dir",
                           str(out_dir / ds), "--videos", str(split_txt), "--mode", "track", "--out", str(out),
                           "--iou", str(iou)])
    vpmm.train()
    lst.train()
    return dict(load_json(out)["track"])


def _rng_state() -> dict:
    s = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _set_rng_state(s: dict) -> None:
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def _write_metrics(run_dir: Path, result: dict) -> None:
    path = run_dir / "metrics.json"
    metrics = {"det": {}, "track": {}, "spot": {}, "per_video": {}, "meta": {}}
    if path.is_file():
        metrics.update(load_json(path))
    metrics.setdefault("meta", {})["lst_train"] = result
    save_json(metrics, path)


def _peak_mb(device) -> dict:
    if device.type != "cuda":
        return {"peak_alloc_mb": 0.0, "peak_reserved_mb": 0.0}
    return {"peak_alloc_mb": round(torch.cuda.max_memory_allocated(device) / 2 ** 20, 1),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 2 ** 20, 1)}


def train(cfg: dict, smoke: bool = False, resume: bool = False, device=None) -> dict:
    """Runs the training described by `cfg`; returns meta.lst_train."""
    root = project_root()
    tc = train_settings(cfg, smoke)
    seed, ds = int(cfg["seed"]), cfg["dataset"]
    exp_id = f"{cfg['exp_id']}_smoke" if smoke else str(cfg["exp_id"])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = tc["amp"] and device.type == "cuda"
    iou = float((cfg.get("eval") or {}).get("iou", 0.5))

    set_seed(seed)
    run_dir = root / "runs" / exp_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("train_lst", run_dir / "log.txt")
    dump = copy.deepcopy(cfg)
    dump["exp_id"] = exp_id
    dump.setdefault("tracker", {})["train"] = dict(tc, smoke=bool(smoke))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dump, sort_keys=False), encoding="utf-8")

    train_files = cache_files(ds, tc["train_split"])
    val_files = cache_files(ds, tc["val_split"])
    if smoke:
        val_files = val_files[:tc["smoke_val_videos"]]
    c, d = feat_shape(train_files[0])
    vpmm = VPMM(c, d).to(device)
    lst = LSTAdapter(2 * d, cfg).to(device)
    params = list(vpmm.parameters()) + list(lst.parameters())
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: lr_factor(it, tc))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_clip = lst.s["grad_clip"]

    st = {"iter": 0, "epoch": 0, "pos": 0, "clip_len": tc["clip_len"], "best": None, "history": [],
          "skipped": 0, "oom": []}
    last = ckpt_dir / "last.pth"
    if resume and last.is_file():
        ck = torch.load(last, map_location="cpu")
        vpmm.load_state_dict(ck["vpmm"])
        lst.load_state_dict(ck["lst"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        _set_rng_state(ck["rng"])
        st = ck["state"]
        log.info("resumed %s at iter %d (epoch %d, pos %d, clip_len %d)", last, st["iter"], st["epoch"],
                 st["pos"], st["clip_len"])
    elif resume:
        log.warning("--resume: %s not found, starting from scratch", last)

    def save(path: Path, full: bool, val=None) -> None:
        ck = {"vpmm": vpmm.state_dict(), "lst": lst.state_dict(), "c": c, "d": d, "tracker": lst.s,
              "iter": st["iter"], "val": val}
        if full:
            ck.update(opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                      rng=_rng_state(), state=st)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(ck, tmp)
        tmp.replace(path)

    ds_train = LSTCacheDataset(train_files, clip_len=st["clip_len"], stride=tc["stride"])
    if len(ds_train) == 0:
        raise ValueError(f"no {st['clip_len']}-frame clips in the {ds}/{tc['train_split']} cache")
    result = {"exp_id": exp_id, "smoke": bool(smoke), "device": str(device), "amp": use_amp, "c": c, "d": d,
              "settings": tc, "tracker": lst.s, "num_train_videos": len(train_files),
              "num_val_videos": len(val_files), "num_params": sum(p.numel() for p in params)}
    log.info("exp=%s smoke=%s device=%s amp=%s feats=[%d,%d] train=%d videos/%d clips val=%d videos "
             "params=%d iters=%d", exp_id, smoke, device, use_amp, c, d, len(train_files), len(ds_train),
             len(val_files), result["num_params"], tc["iters"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    vpmm.train()
    lst.train()
    t0, loss_win, n_win = time.time(), {}, 0
    while st["iter"] < tc["iters"]:
        order = np.random.default_rng(seed + st["epoch"]).permutation(len(ds_train)).tolist()
        loader = DataLoader(ds_train, batch_size=1, sampler=order[st["pos"]:], collate_fn=collate_first,
                            num_workers=tc["num_workers"])
        useful, oom, start_pos = 0, False, st["pos"]
        for clip in loader:
            st["pos"] += 1
            try:
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    out = lst.train_step(embed_clip(vpmm, clip, device))
                if not out["loss"].requires_grad:  # no GT-matched instance above asso_thr
                    st["skipped"] += 1
                    continue
                opt.zero_grad(set_to_none=True)
                scaler.scale(out["loss"]).backward()
                scaler.unscale_(opt)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                scaler.step(opt)
                scaler.update()
                sched.step()
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if st["clip_len"] <= tc["min_clip_len"]:
                    raise
                st["clip_len"] -= 1
                st["oom"].append({"iter": st["iter"], "clip_len": st["clip_len"]})
                log.warning("CUDA OOM at iter %d: clip_len -> %d, new epoch", st["iter"], st["clip_len"])
                ds_train = LSTCacheDataset(train_files, clip_len=st["clip_len"], stride=tc["stride"])
                oom = True
                break
            useful += 1
            st["iter"] += 1
            for k, v in out.items():
                loss_win[k] = loss_win.get(k, 0.0) + float(v)
            n_win += 1
            it = st["iter"]
            if it % tc["log_every"] == 0 or it == tc["iters"]:
                log.info("iter %d/%d %s lr %.2e skipped %d %.1fs peak %.0f MB", it, tc["iters"],
                         " ".join(f"{k} {v / n_win:.4f}" for k, v in loss_win.items()),
                         opt.param_groups[0]["lr"], st["skipped"], time.time() - t0,
                         _peak_mb(device)["peak_alloc_mb"])
                loss_win, n_win = {}, 0
            if it % tc["val_every"] == 0 or it == tc["iters"]:
                val = validate(vpmm, lst, val_files, ds, run_dir / "val_track", device, use_amp, iou)
                rec = {"iter": it, **val, **_peak_mb(device), "sec": round(time.time() - t0, 1)}
                st["history"].append(rec)
                key = (val["IDF1"], val["MOTA"])
                is_best = st["best"] is None or key > (st["best"]["IDF1"], st["best"]["MOTA"])
                if is_best:
                    st["best"] = rec
                    save(ckpt_dir / "best.pth", False, val)
                save(last, True, val)
                log.info("val iter %d: IDF1 %.2f MOTA %.2f MOTP %.2f MT %.1f ML %.1f%s", it, val["IDF1"],
                         val["MOTA"], val["MOTP"], val["MT"], val["ML"], " *best*" if is_best else "")
                _write_metrics(run_dir, dict(result, iters_done=it, clip_len=st["clip_len"],
                                             skipped_clips=st["skipped"], oom=st["oom"], best=st["best"],
                                             best_ckpt=str(ckpt_dir / "best.pth"), history=st["history"],
                                             **_peak_mb(device)))
            if it >= tc["iters"]:
                break
        if st["iter"] >= tc["iters"]:
            break
        if not oom and useful == 0 and start_pos == 0:
            raise ValueError("a full epoch had no clip with GT-matched instances above asso_thr")
        st["epoch"] += 1
        st["pos"] = 0
    peak = _peak_mb(device)
    log.info("done: %d iters, best iter %s IDF1 %s, peak VRAM %.0f MB alloc / %.0f MB reserved", st["iter"],
             st["best"]["iter"] if st["best"] else None, st["best"]["IDF1"] if st["best"] else None,
             peak["peak_alloc_mb"], peak["peak_reserved_mb"])
    final = dict(result, iters_done=st["iter"], clip_len=st["clip_len"], skipped_clips=st["skipped"],
                 oom=st["oom"], best=st["best"], best_ckpt=str(ckpt_dir / "best.pth"), history=st["history"],
                 **peak)
    _write_metrics(run_dir, final)
    return final


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/E5-lst.yaml"))
    ap.add_argument("--smoke", action="store_true", help="smoke_iters iterations, exp_id <exp_id>_smoke")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<exp_id>/checkpoints/last.pth")
    ap.add_argument("--device", default=None, help="default: cuda if available")
    args = ap.parse_args(argv)
    train(load_yaml(args.config), smoke=args.smoke, resume=args.resume, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
