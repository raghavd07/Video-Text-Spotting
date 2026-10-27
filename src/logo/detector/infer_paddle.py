"""PP-YOLOE-R inference on extracted frames -> runs/<exp_id>/det/<ds>/<video>.json (det schema).

Runs in the logo-paddle env (.venv-paddle); imports only paddle, ppdet, numpy and logo.utils.io.
The model is built the way PaddleDetection tools/infer.py does it (load_config -> create(architecture)
-> load_pretrain_weight) and frames go through the config's own TestReader transforms, with Resize.target_size
replaced by detector.input_hw. Trainer.predict has no AMP path, so inference is FP32.
Every class of the checkpoint is kept (class-agnostic); boxes with score > detector.save_thr are saved.
Peak GPU memory and ms/frame go to runs/<exp_id>/det/timing.json and metrics.json meta.det_infer.

Usage: python -m logo.detector.infer_paddle --config configs/det_ppyoloe_r_s.yaml
           [--split val] [--videos v1 v2] [--max_frames N] [--device gpu|cpu] [--overwrite] [--final]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from logo.utils.io import get_logger, load_json, load_yaml, save_json, set_seed

SPLITS = ("train", "val", "test")
STAGE = "det"

# predictor(list of frame paths) -> one float array [n, 10] = [cls, score, x1, y1, ..., x4, y4] per frame
Predictor = Callable[[Sequence[Path]], list]


def project_root() -> Path:
    env = os.environ.get("LOGO_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[3]


def resolve(p, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else root / p


def read_videos(ds: str, split: str, root: Path, final: bool = False) -> list[str]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if split == "test" and not final:
        raise PermissionError("test split requires --final")
    lines = (root / "splits" / f"{ds}_{split}.txt").read_text(encoding="utf-8").splitlines()
    return [v.strip() for v in lines if v.strip()]


def list_frames(frame_dir: Path) -> list[tuple[int, Path]]:
    """(1-based frame id, path) for every <id>.jpg in frame_dir, sorted by id."""
    out = [(int(p.stem), p) for p in frame_dir.glob("*.jpg") if p.stem.isdigit()]
    if not out:
        raise FileNotFoundError(f"no frames in {frame_dir}")
    return sorted(out)


def rows_to_objects(rows: np.ndarray, save_thr: float) -> list[dict]:
    """[n, 10] ppdet rotated rows -> det objects {"quad": [8 floats], "score": float}, score-sorted."""
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, 10)  # model dtype; thr compared in float32
    rows = rows[(rows[:, 0] >= 0) & (rows[:, 1] > np.float32(save_thr))]
    rows = rows[np.argsort(-rows[:, 1], kind="stable")]
    return [{"quad": [round(float(v), 2) for v in r[2:10]], "score": round(float(r[1]), 5)}
            for r in rows]


def detect_video(frames: list[tuple[int, Path]], predictor: Predictor, batch_size: int,
                 save_thr: float) -> dict:
    """Runs the predictor batch by batch; only one batch of images is in memory at a time."""
    out = {}
    for i in range(0, len(frames), batch_size):
        chunk = frames[i:i + batch_size]
        preds = predictor([p for _, p in chunk])
        if len(preds) != len(chunk):
            raise RuntimeError(f"predictor returned {len(preds)} results for {len(chunk)} frames")
        for (fid, _), rows in zip(chunk, preds):
            out[str(fid)] = rows_to_objects(rows, save_thr)
    return out


class PaddleDetector:
    """PP-YOLOE-R predictor; `model_seconds` accumulates synchronized forward+post-process time."""

    def __init__(self, ppdet_config: Path, weights: Optional[Path], input_hw=None, device="gpu",
                 seed: int = 0):
        import paddle
        from ppdet.core.workspace import create, load_config
        from ppdet.data import transform
        from ppdet.utils.checkpoint import load_pretrain_weight

        self.paddle = paddle
        self.use_gpu = device == "gpu"
        if self.use_gpu and not paddle.is_compiled_with_cuda():
            raise RuntimeError("device=gpu but this paddle build has no CUDA")
        paddle.set_device("gpu" if self.use_gpu else "cpu")
        paddle.seed(seed)

        cfg = load_config(str(ppdet_config))
        reader = cfg["TestReader"]
        self.sample_tf = [self._op(transform, item, input_hw) for item in reader["sample_transforms"]]
        self.batch_tf = [self._op(transform, item, input_hw)
                         for item in reader.get("batch_transforms") or []]
        self.model = create(cfg.architecture)
        if weights is not None:  # as Trainer.load_weights; the fixed angle_proj_conv is not in ckpts
            load_pretrain_weight(self.model, str(weights))
        self.model.eval()
        self.model_seconds = 0.0

    @staticmethod
    def _op(transform, item: dict, input_hw):
        (name, kwargs), = item.items()
        kwargs = dict(kwargs or {})
        if name == "Resize" and input_hw is not None:
            kwargs["target_size"] = [int(v) for v in input_hw]
        return getattr(transform, name)(**kwargs)

    def __call__(self, paths: Sequence[Path]) -> list:
        paddle = self.paddle
        samples = []
        for p in paths:
            s = {"im_file": str(p), "im_id": np.array([len(samples)])}
            for op in self.sample_tf:
                s = op(s)
            samples.append(s)
        for op in self.batch_tf:
            samples = op(samples)
        inputs = {k: paddle.to_tensor(np.stack([s[k] for s in samples]).astype(np.float32))
                  for k in ("image", "im_shape", "scale_factor")}

        t0 = time.perf_counter()
        with paddle.no_grad():
            outs = self.model(inputs)
        bbox, bbox_num = outs["bbox"].numpy(), outs["bbox_num"].numpy()
        if self.use_gpu:
            paddle.device.cuda.synchronize()
        self.model_seconds += time.perf_counter() - t0

        res, start = [], 0
        for n in bbox_num.tolist():
            res.append(bbox[start:start + n] if n > 0 else np.zeros((0, 10), np.float32))
            start += n
        return res

    def peak_memory_mb(self) -> Optional[float]:
        if not self.use_gpu:
            return None
        return round(self.paddle.device.cuda.max_memory_allocated() / 2 ** 20, 1)


def write_run_files(cfg: dict, run_dir: Path, timing: dict, key: str = "det_infer") -> None:
    """config.yaml (if absent; JSON is valid YAML) and metrics.json with timing under meta.<key>."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_out = run_dir / "config.yaml"
    if not cfg_out.exists():
        cfg_out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    m_out = run_dir / "metrics.json"
    metrics = {"det": {}, "track": {}, "spot": {}, "per_video": {}, "meta": {}}
    if m_out.is_file():
        metrics.update(load_json(m_out))
    metrics.setdefault("meta", {})[key] = timing
    save_json(metrics, m_out)


def run(cfg: dict, videos: list[str], predictor: Predictor, root: Path, log,
        max_frames: Optional[int] = None, overwrite: bool = False,
        peak_mem_fn: Optional[Callable[[], Optional[float]]] = None, amp: bool = False) -> dict:
    ds, det = cfg["dataset"], cfg["detector"]
    batch_size, save_thr = int(det.get("batch_size", 2)), float(det.get("save_thr", 0.1))
    run_dir = root / "runs" / cfg["exp_id"]
    out_dir = run_dir / STAGE / ds

    n_frames, n_done, t0 = 0, 0, time.perf_counter()
    model_s0 = getattr(predictor, "model_seconds", None)
    for video in videos:
        out = out_dir / f"{video}.json"
        if out.exists() and not overwrite:
            log.info("skip %s (exists)", video)
            continue
        frames = list_frames(root / "data" / "frames" / ds / video)
        if max_frames:
            frames = frames[:max_frames]
        tv = time.perf_counter()
        dets = detect_video(frames, predictor, batch_size, save_thr)
        save_json({"video": video, "dataset": ds, "frames": dets}, out)
        n_frames += len(frames)
        n_done += 1
        log.info("%s: %d frames, %d boxes, %.1f ms/frame", video, len(frames),
                 sum(len(v) for v in dets.values()), 1000 * (time.perf_counter() - tv) / len(frames))

    wall = time.perf_counter() - t0
    model_s = (predictor.model_seconds - model_s0) if model_s0 is not None else None
    timing = {
        "videos": n_done, "frames": n_frames, "batch_size": batch_size,
        "input_hw": det.get("input_hw"), "save_thr": save_thr,
        "ms_per_frame": round(1000 * wall / n_frames, 2) if n_frames else None,
        "ms_per_frame_model": round(1000 * model_s / n_frames, 2) if n_frames and model_s is not None
        else None,
        "peak_gpu_mem_mb": peak_mem_fn() if peak_mem_fn else None,
        "amp": bool(amp),
    }
    save_json(timing, run_dir / STAGE / "timing.json")
    write_run_files(cfg, run_dir, timing)
    log.info("done: %d videos, %d frames, %s ms/frame (model %s), peak GPU mem %s MB", n_done,
             n_frames, timing["ms_per_frame"], timing["ms_per_frame_model"], timing["peak_gpu_mem_mb"])
    print(f"peak GPU memory: {timing['peak_gpu_mem_mb']} MB | ms/frame: {timing['ms_per_frame']} "
          f"(model only: {timing['ms_per_frame_model']})")
    return timing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", default=None, help="default: config split")
    ap.add_argument("--videos", nargs="*", default=None, help="subset of the split's videos")
    ap.add_argument("--max_frames", type=int, default=None, help="per video (smoke runs)")
    ap.add_argument("--device", choices=("gpu", "cpu"), default=None,
                    help="default: config detector.device, else gpu")
    ap.add_argument("--overwrite", action="store_true", help="redo videos whose json exists")
    ap.add_argument("--final", action="store_true", help="allow the test split (rule 5)")
    args = ap.parse_args(argv)

    root = project_root()
    cfg = load_yaml(args.config)
    if args.split:
        cfg["split"] = args.split
    det = cfg["detector"]
    videos = read_videos(cfg["dataset"], cfg["split"], root, final=args.final)
    if args.videos:
        missing = sorted(set(args.videos) - set(videos))
        if missing:
            raise SystemExit(f"videos not in split {cfg['split']}: {missing}")
        videos = [v for v in videos if v in args.videos]

    set_seed(cfg["seed"])
    log = get_logger("infer_paddle", root / "runs" / cfg["exp_id"] / "log.txt")
    device = args.device or det.get("device", "gpu")
    log.info("exp=%s ds=%s split=%s videos=%d device=%s", cfg["exp_id"], cfg["dataset"],
             cfg["split"], len(videos), device)
    predictor = PaddleDetector(resolve(det["config"], root), resolve(det["weights"], root),
                               det.get("input_hw"), device, int(cfg["seed"]))
    run(cfg, videos, predictor, root, log, args.max_frames, args.overwrite, predictor.peak_memory_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
