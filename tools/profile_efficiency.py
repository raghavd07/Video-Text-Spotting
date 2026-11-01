"""Per-stage efficiency at batch 1 (one frame per call) on the first N frames of a config's split.

Stages (each fed in memory by the previous one; det json from runs/<det_exp>/det feeds lsc):
  det    PaddleDetector (logo-paddle env) or UltralyticsOBB (Plan B, logo-torch); alone via --stages det
  lsc    frame read + crop_quad + ABINet on boxes > lsc.det_thr
  fuse   run_fuse.fuse_video
  vpmm   tracker.type lst: VPMM per frame; embed: attach_embeddings per frame; simple: not used
  track  simple|embed: run_tracker; lst: LSTAdapter.associate (run_lst) + filter_trajectories
Per stage: params, peak_mem_mb (torch.cuda.max_memory_allocated after a peak reset that follows model load
and `warmup` untimed frames, so weights are included; det: paddle max_memory_allocated of the process),
ms_per_frame (wall clock, CUDA-synchronised). e2e: total_ms_per_frame = sum over stages (they run back to
back at batch 1), fps = 1000 / total; det is a separate env, so its entry is read from
runs/<det_exp>/profile/efficiency_<ds>_<split>.json (written by --stages det with the detector config) and
e2e.det_included says whether it was found.
Frames: frames 1.. of the split's videos in split order until N frames are taken (frame files on disk).
Writes runs/<exp_id>/profile/efficiency_<ds>_<split>.json (merged per stage) and
runs/<exp_id>/{config.yaml,log.txt,metrics.json meta.efficiency_<ds>_<split>}.

Usage: python tools/profile_efficiency.py --config configs/E5-lst.yaml [--frames 200] [--warmup 5]
           [--device cuda] [--final]
       python tools/profile_efficiency.py --config configs/det_ppyoloe_r_text_s.yaml --stages det   (logo-paddle)
"""
from __future__ import annotations

import argparse
import copy
import gc
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from logo.detector.infer_paddle import list_frames, read_videos, resolve, write_run_files
from logo.utils.io import get_logger, load_json, load_yaml, save_json, set_seed
from logo.utils.paths import check_dataset, check_split, project_root

STAGES = ("det", "lsc", "fuse", "vpmm", "track")
TORCH_STAGES = STAGES[1:]


def out_path(root: Path, exp: str, ds: str, split: str) -> Path:
    return root / "runs" / exp / "profile" / f"efficiency_{ds}_{split}.json"


def det_exp_of(cfg: dict) -> str:
    return (cfg.get("pipeline") or {}).get("det_exp") or cfg["exp_id"]


def count_params(model) -> int:
    """torch nn.Module or paddle nn.Layer; 0 for None."""
    if model is None:
        return 0
    return int(sum(int(np.prod(tuple(p.shape))) for p in model.parameters()))


def take_frames(root: Path, ds: str, videos: list[str], n: int) -> list[tuple[str, list[tuple[int, Path]]]]:
    """[(video, [(frame id, path)])] covering the first n frames of the videos, in split order."""
    out, left = [], int(n)
    for v in videos:
        if left <= 0:
            break
        fr = list_frames(root / "data" / "frames" / ds / v)[:left]
        out.append((v, fr))
        left -= len(fr)
    if left > 0:
        raise ValueError(f"split has only {n - left} frames, {n} requested")
    return out


def stage_result(params: int, peak_mb: Optional[float], seconds: float, frames: int, **extra) -> dict:
    return {"params": int(params), "peak_mem_mb": peak_mb, "frames": int(frames),
            "ms_per_frame": round(1000.0 * seconds / frames, 3) if frames else None, **extra}


def e2e(stages: dict) -> dict:
    ms = [s["ms_per_frame"] for s in stages.values() if s.get("ms_per_frame") is not None]
    total = float(sum(ms))
    peaks = [s["peak_mem_mb"] for s in stages.values() if s.get("peak_mem_mb") is not None]
    return {"total_ms_per_frame": round(total, 3), "fps": round(1000.0 / total, 2) if total > 0 else None,
            "params": int(sum(s.get("params", 0) for s in stages.values())),
            "max_peak_mem_mb": max(peaks) if peaks else None, "det_included": "det" in stages,
            "stages": [k for k in STAGES if k in stages]}


# ------------------------------------------------------------------------------------------------ det
def profile_det(cfg: dict, root: Path, segments, warmup: int, device: str = "gpu",
                predictor: Optional[Callable] = None) -> dict:
    """PaddleDetector (detector.config set) or Plan B UltralyticsOBB at batch 1; `predictor` (paths -> rows)
    replaces it in tests."""
    det = cfg["detector"]
    if predictor is None and det.get("config"):
        from logo.detector.infer_paddle import PaddleDetector
        predictor = PaddleDetector(resolve(det["config"], root), resolve(det["weights"], root),
                                   det.get("input_hw"), device, int(cfg["seed"]))
    elif predictor is None:  # as logo.detector.infer_obb.main
        from logo.detector.infer_obb import UltralyticsOBB
        predictor = UltralyticsOBB(resolve(det["weights"], root), det.get("input_hw"), device,
                                   float(det.get("save_thr", 0.1)), bool(det.get("half", True)),
                                   int(det.get("max_det", 300)), float(det.get("nms_iou", 0.7)))
    paths = [p for _, fr in segments for _, p in fr]
    for _ in range(warmup):
        predictor([paths[0]])
    t0 = time.perf_counter()
    for p in paths:
        predictor([p])
    dt = time.perf_counter() - t0
    peak = predictor.peak_memory_mb() if hasattr(predictor, "peak_memory_mb") else None
    return stage_result(count_params(getattr(predictor, "model", None)), peak, dt, len(paths),
                        input_hw=(cfg.get("detector") or {}).get("input_hw"))


# ---------------------------------------------------------------------------------------------- torch
class _Cuda:
    def __init__(self, device):
        import torch
        self.torch, self.device = torch, torch.device(device)
        self.on = self.device.type == "cuda"

    def sync(self):
        if self.on:
            self.torch.cuda.synchronize(self.device)

    def reset(self):
        self.sync()
        if self.on:
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def peak(self) -> Optional[float]:
        self.sync()
        return round(self.torch.cuda.max_memory_allocated(self.device) / 2 ** 20, 1) if self.on else None

    def free(self):
        gc.collect()
        if self.on:
            self.torch.cuda.empty_cache()


def make_recognizer(cfg: dict, root: Path, device: str):
    from logo.lsc import run_lsc
    from logo.lsc.abinet_wrapper import ABINetRecognizer
    ic = run_lsc.infer_settings(cfg)
    rec = ABINetRecognizer(run_lsc.pick_ckpt(cfg, root), device, fp16=ic["amp"])
    rec.batch_size = ic["batch_size"]
    return rec


def profile_lsc(cfg: dict, segments, dets: dict, recognizer, cu: _Cuda, warmup: int):
    """-> (stage result, {video: lsc dict of the selected frames}, {video: float16 [N,c,d]})."""
    import cv2

    from logo.lsc import run_lsc
    from logo.lsc.roi_rotate import crop_quad
    thr = float(cfg["lsc"]["det_thr"])

    def kept(v, fid):
        return [o for o in dets[v]["frames"].get(str(fid), []) if float(o["score"]) > thr]

    first = next(((v, p, kept(v, f)) for v, fr in segments for f, p in fr if kept(v, f)), None)
    if first is not None:
        img = cv2.imread(str(first[1]), cv2.IMREAD_COLOR)
        crops = [crop_quad(img, o["quad"], run_lsc.OUT_HW) for o in first[2]]
        for _ in range(warmup):
            recognizer.predict(crops)
    cu.reset()
    lsc, feats, n_boxes = {}, {}, 0
    t0 = time.perf_counter()
    for v, fr in segments:
        rows, frames = [], {}
        for fid, path in fr:
            objs, out = kept(v, fid), []
            if objs:
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(f"cannot read frame {path}")
                preds = recognizer.predict([crop_quad(img, o["quad"], run_lsc.OUT_HW) for o in objs])
                for o, p in zip(objs, preds):
                    out.append({**o, **run_lsc.lsc_fields(p, len(rows))})
                    rows.append(np.asarray(p["feat"], dtype=np.float16))
            frames[str(fid)] = out
        lsc[v] = {"video": v, "dataset": dets[v].get("dataset"), "frames": frames}
        c, d = int(recognizer.c), int(recognizer.d)
        feats[v] = np.stack(rows) if rows else np.zeros((0, c, d), np.float16)
        n_boxes += len(rows)
    cu.sync()
    dt = time.perf_counter() - t0
    res = stage_result(count_params(getattr(recognizer, "model", None)), cu.peak(), dt,
                       sum(len(fr) for _, fr in segments), boxes=n_boxes,
                       batch_size=int(getattr(recognizer, "batch_size", 1)))
    return res, lsc, feats


def profile_fuse(cfg: dict, lsc: dict, n_frames: int):
    from logo.scoring import run_fuse
    mode, w = cfg["fusion"]["mode"], float(cfg["fusion"]["w"])
    t0 = time.perf_counter()
    fused = {v: run_fuse.fuse_video(x, mode, w) for v, x in lsc.items()}
    return stage_result(0, None, time.perf_counter() - t0, n_frames), fused


def profile_vpmm_track(cfg: dict, root: Path, ds: str, fused: dict, feats: dict, cu: _Cuda,
                       warmup: int) -> dict:
    """{"vpmm": ..., "track": ...} (simple: track only); `fused` objects gain in-memory keys (embed)."""
    from logo.tracking import run_track
    from logo.tracking.simple_tracker import run_tracker, tracker_settings
    kind = (cfg.get("tracker") or {}).get("type", "simple")
    n_frames = sum(len(f["frames"]) for f in fused.values())
    if kind == "simple":
        cu.reset()
        t0 = time.perf_counter()
        for f in fused.values():
            run_tracker(f, cfg)
        return {"track": stage_result(0, None, time.perf_counter() - t0, n_frames, type=kind)}
    if kind not in ("lst", "embed"):
        raise NotImplementedError(f"tracker.type {kind!r}")
    wh = {v: (int(g["width"]), int(g["height"])) for v in fused
          for g in [load_json(root / "data" / "processed" / ds / f"{v}.json")]}
    ckpt = run_track.embed_ckpt(cfg, root)
    amp = cu.on
    if kind == "lst":
        vpmm, lst = run_track.load_lst(ckpt, cfg, cu.device)
        thr, min_score = run_track.input_thr(cfg, ds), run_track.traj_thr(cfg)
        spotting = bool((cfg.get("eval") or {}).get("spotting", True))
        text_window = tracker_settings(cfg)["text_window"]

        def embed_frame(v, fid, objs):
            return run_track.vpmm_features({"frames": {fid: objs}}, feats[v], vpmm, wh[v], cu.device, amp)
    else:
        from logo.tracking.train_embed import attach_embeddings, load_embed_net
        vpmm = load_embed_net(ckpt, cu.device)

        def embed_frame(v, fid, objs):
            attach_embeddings({"frames": {fid: objs}}, feats[v], vpmm, wh[v], cu.device, amp)

    first = next(((v, k, o) for v, f in fused.items() for k, o in f["frames"].items() if o), None)
    if first is not None:
        for _ in range(warmup):
            embed_frame(*first)
    cu.reset()
    emb = {}
    t0 = time.perf_counter()
    for v, f in fused.items():
        parts = [embed_frame(v, k, f["frames"][k]) for k in sorted(f["frames"], key=int)]
        if kind == "lst":
            emb[v] = (np.concatenate(parts) if parts else np.zeros((0, 2 * vpmm.d), np.float32))
    cu.sync()
    out = {"vpmm": stage_result(count_params(vpmm), cu.peak(), time.perf_counter() - t0, n_frames,
                                ckpt=str(ckpt))}
    if kind == "embed":
        cu.reset()
        t0 = time.perf_counter()
        for f in fused.values():
            run_tracker(f, cfg)
        out["track"] = stage_result(0, cu.peak(), time.perf_counter() - t0, n_frames, type=kind)
        return out
    if first is not None:
        v = first[0]
        head = {k: fused[v]["frames"][k] for k in sorted(fused[v]["frames"], key=int)[:max(warmup, 1)]}
        run_track.run_lst({"frames": head}, emb[v], lst, wh[v], thr, text_window)
    cu.reset()
    t0 = time.perf_counter()
    for v, f in fused.items():
        rows = run_track.run_lst(f, emb[v], lst, wh[v], thr, text_window)
        run_track.filter_trajectories(rows, min_score, spotting)
    cu.sync()
    out["track"] = stage_result(count_params(lst), cu.peak(), time.perf_counter() - t0, n_frames, type=kind,
                                thr=thr, traj_thr=min_score)
    return out


# ----------------------------------------------------------------------------------------------- main
def profile(cfg: dict, root: Path, frames: int = 200, warmup: int = 5, stages=TORCH_STAGES,
            device: Optional[str] = None, final: bool = False, log=None, recognizer=None,
            predictor=None) -> dict:
    """Profiles `stages` of cfg on its dataset/split; merges into and returns the efficiency json dict.

    stages: ("det",) (logo-paddle env) or any of TORCH_STAGES; lsc/fuse/vpmm/track are profiled together
    because each feeds the next. `recognizer` / `predictor` replace ABINet / PP-YOLOE-R in tests.
    """
    ds, split = check_dataset(cfg["dataset"]), check_split(cfg["split"], final)
    stages = [s for s in STAGES if s in set(stages)]
    if "det" in stages and len(stages) > 1:
        raise ValueError("det runs in the logo-paddle env: profile it alone (--stages det)")
    log = log or get_logger("profile_efficiency", root / "runs" / cfg["exp_id"] / "log.txt")
    set_seed(cfg["seed"])
    videos = read_videos(ds, split, root, final=final)
    segments = take_frames(root, ds, videos, frames)
    p_out = out_path(root, cfg["exp_id"], ds, split)
    res = load_json(p_out) if p_out.is_file() else {}
    if res.get("frames") != int(frames):
        res = {}
    res.update({"exp_id": cfg["exp_id"], "dataset": ds, "split": split, "frames": int(frames),
                "warmup": int(warmup), "videos": [v for v, _ in segments]})
    st = res.setdefault("stages", {})

    if stages == ["det"]:
        st["det"] = profile_det(cfg, root, segments, warmup, device or "gpu", predictor)
        log.info("det: %s", st["det"])
    else:
        import torch
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cu = _Cuda(dev)
        res["device"] = torch.cuda.get_device_name(cu.device) if cu.on else "cpu"
        st.pop("det", None)
        st.pop("vpmm", None)
        det_exp = det_exp_of(cfg)
        dets = {v: load_json(root / "runs" / det_exp / "det" / ds / f"{v}.json") for v, _ in segments}
        rec = recognizer if recognizer is not None else make_recognizer(cfg, root, dev)
        st["lsc"], lsc, feats = profile_lsc(cfg, segments, dets, rec, cu, warmup)
        del rec
        cu.free()
        st["fuse"], fused = profile_fuse(cfg, lsc, sum(len(fr) for _, fr in segments))
        st.update(profile_vpmm_track(cfg, root, ds, fused, feats, cu, warmup))
        cu.free()
        det_p = out_path(root, det_exp, ds, split)
        det_res = load_json(det_p) if det_p.is_file() else {}
        if det_res.get("frames") == int(frames) and "det" in det_res.get("stages", {}):
            st["det"] = {**det_res["stages"]["det"], "source": str(det_p)}
        else:
            log.warning("no det profile at %s for %d frames: e2e excludes det (run --stages det in logo-paddle)",
                        det_p, frames)
        for k in ("lsc", "fuse", "vpmm", "track"):
            if k in st:
                log.info("%s: %s", k, st[k])
    res["stages"] = {k: st[k] for k in STAGES if k in st}
    res["e2e"] = e2e(res["stages"])
    log.info("e2e: %s", res["e2e"])
    save_json(res, p_out)
    write_run_files(cfg, root / "runs" / cfg["exp_id"], res, key=f"efficiency_{ds}_{split}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--stages", default=",".join(TORCH_STAGES),
                    help="det (logo-paddle env) or lsc,fuse,vpmm,track (torch env)")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default=None, help="torch: cuda|cpu; det: gpu|cpu")
    ap.add_argument("--final", action="store_true", help="allow split=test (rule 5)")
    args = ap.parse_args(argv)
    cfg = copy.deepcopy(load_yaml(args.config))
    if cfg.get("split") == "test" and not args.final:
        ap.error("config split is test: pass --final")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if not stages or set(stages) - set(STAGES):
        ap.error(f"--stages must be a comma list of {STAGES}")
    root = project_root()
    res = profile(cfg, root, args.frames, args.warmup, stages, args.device, args.final)
    print(f"{cfg['exp_id']}: " + " | ".join(f"{k} {v['ms_per_frame']} ms" for k, v in res["stages"].items())
          + f" | total {res['e2e']['total_ms_per_frame']} ms/frame, {res['e2e']['fps']} FPS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
