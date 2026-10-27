"""Ultralytics YOLO-OBB inference on extracted frames -> runs/<exp_id>/det/<ds>/<video>.json (det schema).

Plan B detector, runs in the main torch env (.venv, ultralytics 8.3.40). Ultralytics calls used:
  YOLO(weights, task="obb")
  YOLO.predict(list_of_BGR_uint8_arrays, imgsz=[h, w], conf, iou, device, half, agnostic_nms, max_det,
               verbose=False) -> list[Results]   (a list of arrays is one batch: LoadPilAndNumpy)
  Results.obb.xyxyxyxy [N,4,2] (original-image pixels), Results.obb.conf [N], Results.obb.cls [N]
Corners are re-ordered with logo.utils.geometry.order_quad (same convention as GT). NMS is class-agnostic
and every class is kept; boxes with score > detector.save_thr are saved. FP16 on GPU when detector.half.
Frame loop, det json writing, timing.json and metrics meta.det_infer are shared with infer_paddle.run.

Usage: python -m logo.detector.infer_obb --config configs/det_yolo11s_obb.yaml
           [--split val] [--videos v1 v2] [--max_frames N] [--device gpu|cpu] [--overwrite] [--final]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from logo.detector.infer_paddle import project_root, read_videos, resolve, run
from logo.utils.geometry import order_quad
from logo.utils.io import get_logger, load_yaml, set_seed


def results_to_rows(xyxyxyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """OBB result arrays -> [n, 10] rows [cls, score, x1, y1, ..., x4, y4] with ordered corners."""
    pts = np.asarray(xyxyxyxy, dtype=np.float32).reshape(-1, 8)
    if len(pts) == 0:
        return np.zeros((0, 10), np.float32)
    quads = np.array([order_quad(q) for q in pts], np.float32)
    head = np.stack([np.asarray(cls, np.float32).reshape(-1), np.asarray(conf, np.float32).reshape(-1)], 1)
    return np.concatenate([head, quads], 1)


class UltralyticsOBB:
    """YOLO-OBB predictor; `model_seconds` accumulates synchronized predict() time (pre+infer+post)."""

    def __init__(self, weights: Path, input_hw=None, device="gpu", conf: float = 0.1,
                 half: bool = True, max_det: int = 300, iou: float = 0.7):
        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.use_gpu = device == "gpu"
        if self.use_gpu and not torch.cuda.is_available():
            raise RuntimeError("device=gpu but torch.cuda is not available")
        self.device = 0 if self.use_gpu else "cpu"
        self.half = bool(half) and self.use_gpu
        self.imgsz = [int(v) for v in input_hw] if input_hw is not None else 640
        self.conf, self.iou, self.max_det = float(conf), float(iou), int(max_det)
        self.model = YOLO(str(weights), task="obb")
        if self.use_gpu:
            torch.cuda.reset_peak_memory_stats()
        self.model_seconds = 0.0

    def _sync(self):
        if self.use_gpu:
            self.torch.cuda.synchronize()

    def __call__(self, paths: Sequence[Path]) -> list:
        import cv2

        imgs = []
        for p in paths:
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(f"cannot read {p}")
            imgs.append(im)
        self._sync()
        t0 = time.perf_counter()
        results = self.model.predict(imgs, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                     device=self.device, half=self.half, agnostic_nms=True,
                                     max_det=self.max_det, verbose=False)
        out = []
        for r in results:
            o = r.obb
            if o is None or len(o) == 0:
                out.append(np.zeros((0, 10), np.float32))
            else:
                out.append(results_to_rows(o.xyxyxyxy.cpu().numpy(), o.conf.cpu().numpy(),
                                           o.cls.cpu().numpy()))
        self._sync()
        self.model_seconds += time.perf_counter() - t0
        return out

    def peak_memory_mb(self) -> Optional[float]:
        if not self.use_gpu:
            return None
        return round(self.torch.cuda.max_memory_allocated() / 2 ** 20, 1)


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
    log = get_logger("infer_obb", root / "runs" / cfg["exp_id"] / "log.txt")
    device = args.device or det.get("device", "gpu")
    log.info("exp=%s ds=%s split=%s videos=%d device=%s", cfg["exp_id"], cfg["dataset"],
             cfg["split"], len(videos), device)
    predictor = UltralyticsOBB(resolve(det["weights"], root), det.get("input_hw"), device,
                               float(det.get("save_thr", 0.1)), bool(det.get("half", True)),
                               int(det.get("max_det", 300)), float(det.get("nms_iou", 0.7)))
    run(cfg, videos, predictor, root, log, args.max_frames, args.overwrite, predictor.peak_memory_mb,
        amp=predictor.half)
    return 0


if __name__ == "__main__":
    sys.exit(main())
