"""Demo: one mp4 -> det -> lsc -> fuse -> track -> vote -> rendered mp4 with quads, track ids and voted text.

Every stage is the pipeline's own run() function, run against a throwaway project tree in a temp dir:
  <tmp>/data/frames/<ds>/demo/%06d.jpg   logo.data.frames.extract_frames (stride 1)
  <tmp>/data/processed/<ds>/demo.json    stub GT (width/height only; read by tracker.type embed|lst)
  <tmp>/splits/<ds>_val.txt              "demo" (read by the paddle subprocess)
  <tmp>/runs/demo/{det,lsc,fuse,track,vote}/<ds>/demo.*
The temp dir is deleted at the end. Detector settings: --det_config, else config detector{} if it has weights,
else configs/<pipeline.det_exp>.yaml, else runs/<pipeline.det_exp>/config.yaml. detector.config (a ppdet yml)
means Plan A: logo.detector.infer_paddle runs in the logo-paddle interpreter (detector.paddle_python, else
$PADDLE_PY, else .venv-paddle) through subprocess with LOGO_ROOT=<tmp>; otherwise Plan B, Ultralytics OBB in
this process. Checkpoints resolve as in the pipeline (lsc: run_lsc.pick_ckpt, tracker: run_track.embed_ckpt).
Writes runs/demo/{config.yaml,log.txt,metrics.json meta.demo_<stem>} and prints a summary.

Usage: python scripts/demo.py --video in.mp4 --out out.mp4 [--config configs/FINAL.yaml] [--det_config Y]
           [--device cuda|cpu]
       python scripts/demo.py --gradio [--port 7860]       (needs: uv pip install gradio==4.44.1)
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import yaml

from logo.data.frames import extract_frames
from logo.detector import infer_paddle
from logo.detector.infer_paddle import list_frames, resolve
from logo.lsc import run_lsc
from logo.scoring import run_fuse
from logo.tracking import run_track
from logo.utils.io import get_logger, load_yaml, save_json, set_seed
from logo.utils.paths import check_dataset, project_root
from logo.voting import track_vote

_spec = importlib.util.spec_from_file_location(
    "run_pipeline", Path(__file__).resolve().parent / "run_pipeline.py")
run_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_pipeline)

DEMO_EXP = "demo"
VIDEO = "demo"
SPLIT = "val"
DEFAULT_CONFIG = Path("configs") / "FINAL.yaml"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def detector_config(cfg: dict, root: Path, det_config: Optional[Path] = None) -> dict:
    """The detector{} section to run (see module docstring for the lookup order)."""
    if det_config is not None:
        return dict(load_yaml(resolve(det_config, root))["detector"])
    if (cfg.get("detector") or {}).get("weights"):
        return dict(cfg["detector"])
    det_exp = (cfg.get("pipeline") or {}).get("det_exp")
    for p in ([root / "configs" / f"{det_exp}.yaml", root / "runs" / det_exp / "config.yaml"] if det_exp else []):
        if p.is_file():
            return dict(load_yaml(p)["detector"])
    raise FileNotFoundError(f"no detector settings: config detector{{}} has no weights and no config found for "
                            f"pipeline.det_exp={det_exp!r}; pass --det_config")


def detector_plan(det: dict) -> str:
    """"paddle" (Plan A, PP-YOLOE-R) if detector.config is set, else "obb" (Plan B, Ultralytics)."""
    return "paddle" if det.get("config") else "obb"


def paddle_python(det: dict, root: Path) -> Path:
    """detector.paddle_python > $PADDLE_PY > .venv-paddle (Windows, then Linux layout)."""
    if det.get("paddle_python"):
        return resolve(det["paddle_python"], root)
    if os.environ.get("PADDLE_PY"):
        return Path(os.environ["PADDLE_PY"])
    for p in (root / ".venv-paddle" / "Scripts" / "python.exe", root / ".venv-paddle" / "bin" / "python"):
        if p.is_file():
            return p
    raise FileNotFoundError("logo-paddle interpreter not found; set detector.paddle_python or $PADDLE_PY")


def det_run_config(cfg: dict, det: dict, root: Path, device: str) -> dict:
    """Config for infer_paddle.run on the temp tree: weights/config made absolute, device gpu|cpu."""
    d = dict(det)
    for k in ("weights", "config"):
        if d.get(k):
            d[k] = str(resolve(d[k], root))
    d["device"] = "gpu" if str(device).startswith("cuda") else "cpu"
    return {"exp_id": DEMO_EXP, "dataset": cfg["dataset"], "split": SPLIT, "seed": int(cfg["seed"]), "detector": d}


def paddle_command(dcfg: dict, det: dict, troot: Path, root: Path) -> tuple[list[str], dict]:
    """Writes <troot>/det_config.yaml; returns (argv, env) for the logo-paddle subprocess."""
    p = troot / "det_config.yaml"
    p.write_text(yaml.safe_dump(dcfg, sort_keys=False), encoding="utf-8")
    env = dict(os.environ, LOGO_ROOT=str(troot))
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    cmd = [str(paddle_python(det, root)), "-m", "logo.detector.infer_paddle", "--config", str(p),
           "--split", SPLIT, "--overwrite"]
    return cmd, env


def stage_config(cfg: dict, root: Path) -> dict:
    """cfg for the temp tree: exp_id/pipeline -> demo, checkpoints resolved against the real root."""
    s = copy.deepcopy(cfg)
    s["exp_id"], s["split"] = DEMO_EXP, SPLIT
    s["pipeline"] = {"det_exp": DEMO_EXP, "lsc_exp": DEMO_EXP}
    ck = run_lsc.pick_ckpt(cfg, root)
    s.setdefault("lsc", {})["ckpt"] = None if ck is None else str(ck)
    if (cfg.get("tracker") or {}).get("type", "simple") in ("embed", "lst"):
        s["tracker"]["ckpt"] = str(run_track.embed_ckpt(cfg, root))
    return s


def free_gpu() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_detector(cfg: dict, det: dict, troot: Path, root: Path, log, device: str,
                 predictor_factory: Optional[Callable] = None) -> None:
    dcfg = det_run_config(cfg, det, root, device)
    if predictor_factory is not None:
        infer_paddle.run(dcfg, [VIDEO], predictor_factory(dcfg), troot, log, overwrite=True)
    elif detector_plan(det) == "paddle":
        cmd, env = paddle_command(dcfg, det, troot, root)
        log.info("detector subprocess: %s", " ".join(cmd))
        subprocess.run(cmd, env=env, cwd=str(root), check=True)
    else:
        from logo.detector.infer_obb import UltralyticsOBB

        d = dcfg["detector"]
        pred = UltralyticsOBB(Path(d["weights"]), d.get("input_hw"), d["device"], float(d.get("save_thr", 0.1)),
                              bool(d.get("half", True)), int(d.get("max_det", 300)), float(d.get("nms_iou", 0.7)))
        infer_paddle.run(dcfg, [VIDEO], pred, troot, log, overwrite=True, peak_mem_fn=pred.peak_memory_mb,
                         amp=pred.half)
        del pred
        free_gpu()


def track_color(tid: int) -> tuple[int, int, int]:
    hsv = np.uint8([[[(int(tid) * 47) % 180, 210, 255]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw_tracks(img: np.ndarray, rows: list[dict]) -> np.ndarray:
    """Quads, "#id text" labels (ASCII; other characters become '?') drawn in place on a BGR frame."""
    h, w = img.shape[:2]
    for r in rows:
        pts = np.round(np.asarray(r["quad"], np.float64).reshape(4, 2)).astype(np.int32)
        col = track_color(r["track_id"])
        cv2.polylines(img, [pts], True, col, 2, cv2.LINE_AA)
        label = f"#{r['track_id']} {r.get('text', '')}".strip().encode("ascii", "replace").decode("ascii")
        (tw, th), base = cv2.getTextSize(label, FONT, 0.5, 1)
        x = int(np.clip(pts[:, 0].min(), 0, max(0, w - tw)))
        y = int(max(th + base, pts[:, 1].min() - 2))
        cv2.rectangle(img, (x, y - th - base), (x + tw, y), col, -1)
        cv2.putText(img, label, (x, y - base), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def render(frame_dir: Path, rows: list[dict], out: Path, fps: float) -> int:
    """Writes every frame of frame_dir with its track boxes to out (mp4v); returns frames written."""
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[int(r["frame"])].append(r)
    frames = list_frames(frame_dir)
    first = cv2.imread(str(frames[0][1]), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    tmp_out = frame_dir.parent / f"{VIDEO}_render.mp4"  # cv2 cannot write non-ASCII paths on Windows
    vw = cv2.VideoWriter(str(tmp_out), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not vw.isOpened():
        raise OSError(f"cannot open video writer for {tmp_out}")
    try:
        for fid, p in frames:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"cannot read frame {p}")
            vw.write(draw_tracks(img, by_frame.get(fid, [])))
    finally:
        vw.release()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    shutil.move(str(tmp_out), str(out))
    return len(frames)


def video_fps(video: Path, default: float = 25.0) -> float:
    cap = cv2.VideoCapture(str(video))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
    finally:
        cap.release()
    return fps if np.isfinite(fps) and fps > 0 else default


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["track_id"]].append(r)
    longest = None
    if by:
        tid = max(by, key=lambda t: (len(by[t]), -t))
        fr = [r["frame"] for r in by[tid]]
        longest = {"track_id": tid, "boxes": len(fr), "first_frame": min(fr), "last_frame": max(fr),
                   "text": by[tid][0].get("text", "")}
    return {"tracks": len(by), "boxes": len(rows), "longest_track": longest}


def format_summary(s: dict) -> str:
    lt = s["longest_track"]
    sec = s["seconds"]
    lines = [f"LOGO demo: {s['video']} -> {s['out']}",
             f"  frames {s['frames']} ({s['width']}x{s['height']} @ {s['video_fps']:.2f} fps), detector {s['detector']}",
             f"  tracks found: {s['tracks']} | track boxes: {s['boxes']}",
             "  longest track: " + (f"#{lt['track_id']}, {lt['boxes']} boxes (frames {lt['first_frame']}-"
                                    f"{lt['last_frame']}), text {lt['text']!r}" if lt else "none"),
             "  seconds: " + " ".join(f"{k}={v:.2f}" for k, v in sec.items()),
             f"  FPS: {s['fps_pipeline']:.2f} (det->vote), {s['fps_total']:.2f} end-to-end"]
    return "\n".join(lines)


def record_run(cfg: dict, det: dict, summary: dict, root: Path) -> None:
    """runs/demo/config.yaml (overwritten: the config of the latest demo) and metrics.json meta.demo_<stem>."""
    run_dir = root / "runs" / DEMO_EXP
    run_dir.mkdir(parents=True, exist_ok=True)
    saved = {**copy.deepcopy(cfg), "exp_id": DEMO_EXP, "detector": det}
    (run_dir / "config.yaml").write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
    infer_paddle.write_run_files(saved, run_dir, summary, key=f"demo_{Path(summary['video']).stem}")


def run_demo(video: Path, out: Path, cfg: dict, root: Path, log, device: Optional[str] = None,
             det: Optional[dict] = None, predictor_factory: Optional[Callable] = None,
             recognizer_factory: Optional[Callable] = None, tmp_parent: Optional[Path] = None) -> dict:
    """Full demo on one video; returns the summary dict (also recorded under runs/demo)."""
    import torch

    video, out = Path(video), Path(out)
    ds = check_dataset(cfg["dataset"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    det = det if det is not None else detector_config(cfg, root)
    set_seed(int(cfg["seed"]))
    scfg = stage_config(cfg, root)
    voting = bool((cfg.get("voting") or {}).get("enabled"))
    sec = {}
    t_all = time.perf_counter()
    log.info("demo video=%s out=%s exp=%s ds=%s detector=%s device=%s", video, out, cfg["exp_id"], ds,
             detector_plan(det), device)
    with tempfile.TemporaryDirectory(prefix="logo_demo_", dir=tmp_parent, ignore_cleanup_errors=True) as tmp:
        troot = Path(tmp)
        frame_dir = troot / "data" / "frames" / ds / VIDEO

        t0 = time.perf_counter()
        n = extract_frames(video, frame_dir)
        if n == 0:
            raise ValueError(f"no frames decoded from {video}")
        h, w = cv2.imread(str(list_frames(frame_dir)[0][1]), cv2.IMREAD_COLOR).shape[:2]
        save_json({"video": VIDEO, "dataset": ds, "width": w, "height": h, "num_frames": n, "frames": {}},
                  troot / "data" / "processed" / ds / f"{VIDEO}.json")
        (troot / "splits").mkdir(parents=True, exist_ok=True)
        (troot / "splits" / f"{ds}_{SPLIT}.txt").write_text(VIDEO + "\n", encoding="utf-8")
        sec["extract"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        run_detector(cfg, det, troot, root, log, device, predictor_factory)
        sec["det"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        rec = (recognizer_factory or run_pipeline.make_recognizer)(scfg, troot, log, device)
        run_lsc.run(scfg, ds, [VIDEO], rec, troot, log, DEMO_EXP, overwrite=True)
        del rec
        free_gpu()
        sec["lsc"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        run_fuse.run(scfg, ds, [VIDEO], troot, log, DEMO_EXP)
        sec["fuse"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        run_track.run(scfg, ds, [VIDEO], troot, log, DEMO_EXP, lsc_exp=DEMO_EXP, device=device)
        sec["track"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if voting:
            track_vote.run(scfg, ds, [VIDEO], troot, log)
        rows = track_vote.load_jsonl(troot / "runs" / DEMO_EXP / ("vote" if voting else "track") / ds
                                     / f"{VIDEO}.jsonl")
        sec["vote"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        fps = video_fps(video)
        render(frame_dir, rows, out, fps)
        sec["render"] = time.perf_counter() - t0

    total = time.perf_counter() - t_all
    pipe = sum(sec[k] for k in ("det", "lsc", "fuse", "track", "vote"))
    summary = {"video": str(video), "out": str(out), "source_exp": cfg["exp_id"], "detector": detector_plan(det),
               "device": device, "voting": voting, "frames": n, "width": w, "height": h, "video_fps": fps,
               **summarize(rows), "seconds": {k: round(v, 3) for k, v in sec.items()},
               "fps_pipeline": n / pipe if pipe > 0 else 0.0, "fps_total": n / total if total > 0 else 0.0}
    record_run(cfg, det, summary, root)
    log.info("demo done: %d tracks, %.2f fps (det->vote)", summary["tracks"], summary["fps_pipeline"])
    return summary


def launch_gradio(cfg: dict, root: Path, log, device: Optional[str], det: dict, port: int) -> None:
    try:
        import gradio as gr
    except ImportError:
        raise SystemExit("gradio is not installed: uv pip install gradio==4.44.1")
    out_dir = root / "runs" / DEMO_EXP / "gradio"

    def infer(path):
        if not path:
            raise gr.Error("upload a video first")
        out = out_dir / f"{Path(path).stem}_spotted.mp4"
        s = run_demo(Path(path), out, cfg, root, log, device=device, det=det)
        return str(out), format_summary(s)

    gr.Interface(fn=infer, inputs=gr.Video(label="input video"),
                 outputs=[gr.Video(label="spotted video"), gr.Textbox(label="summary", lines=7)],
                 title="LOGO video text spotting").launch(server_port=port)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="default: configs/FINAL.yaml")
    ap.add_argument("--det_config", type=Path, default=None, help="config whose detector{} to use")
    ap.add_argument("--device", default=None, help="cuda|cpu; default cuda if available")
    ap.add_argument("--gradio", action="store_true", help="launch an upload page instead of --video/--out")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)

    root = project_root()
    cfg_path = resolve(args.config, root) if not args.config.is_file() else args.config
    if not cfg_path.is_file():
        ap.error(f"config not found: {args.config}")
    if not args.gradio:
        if args.video is None or args.out is None:
            ap.error("--video and --out are required (or pass --gradio)")
        if not args.video.is_file():
            ap.error(f"video not found: {args.video}")
    cfg = copy.deepcopy(load_yaml(cfg_path))
    try:
        det = detector_config(cfg, root, args.det_config)
    except FileNotFoundError as e:
        ap.error(str(e))
    log = get_logger("demo", root / "runs" / DEMO_EXP / "log.txt")
    if args.gradio:
        launch_gradio(cfg, root, log, args.device, det, args.port)
        return 0
    print(format_summary(run_demo(args.video, args.out, cfg, root, log, device=args.device, det=det)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
