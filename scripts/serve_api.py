"""HTTP API for the web frontend: upload a video, run the demo pipeline as a background job, fetch results.

Wraps scripts/demo.py run_demo() without changing it. Jobs run one at a time (the pipeline holds the GPU) in a
worker thread and live under runs/api/jobs/<id>/:
  input<ext>        uploaded video
  output.mp4        rendered video from demo.render (mp4v)
  output_web.mp4    H.264 copy for browsers (only if ffmpeg is on PATH)
  result.json       per-frame track boxes + per-track summary (captured from the rows demo.render draws)
  log.txt           pipeline log for this job
  job.json          job record (status, stage, timings, summary, error)

The ML stack (torch, cv2, logo.*) is imported on first use, so /api/health, /api/configs and /api/benchmarks work
in an environment without it; /api/health reports why the pipeline is unavailable.

Endpoints (all JSON unless noted):
  GET    /api/health                    pipeline availability, device, ffmpeg
  GET    /api/configs                   pipeline configs in configs/ with their settings and missing checkpoints
  GET    /api/benchmarks                result tables parsed from docs/tables/*.md
  GET    /api/jobs                      all jobs, newest first
  POST   /api/jobs                      multipart: video (file); optional config (default: the served config,
                                        FINAL), device (auto|cuda|cpu). 503 if that config's weights are missing.
  GET    /api/jobs/{id}                 one job
  DELETE /api/jobs/{id}                 delete a finished job and its files
  GET    /api/jobs/{id}/result          boxes per frame + tracks (succeeded jobs)
  GET    /api/jobs/{id}/log             last lines of the job log (text/plain)
  GET    /api/jobs/{id}/video/{kind}    kind = source | output (video file)

Pipeline: the trained LOGO config given by --config / $LOGO_CONFIG (default configs/FINAL.yaml). Its artifacts
(detector weights, ABINet repo + LSC checkpoint, LST tracker checkpoint) are resolved against the backend root exactly
as scripts/demo.py resolves them; /api/health lists any that are missing and jobs are refused until they exist.
"pretrained" (logo.pretrained, EasyOCR) is only used when a client asks for it explicitly; the web UI never does.

Usage: uv pip install -r requirements-api.txt
       python scripts/serve_api.py [--host 127.0.0.1] [--port 8000] [--config FINAL] [--jobs-dir DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

JOBS_DIR = ROOT / "runs" / "api" / "jobs"
CONFIG_DIR = ROOT / "configs"
TABLES_DIR = ROOT / "docs" / "tables"
ABINET_ROOT = ROOT / "third_party" / "ABINet"  # = logo.lsc.abinet_wrapper.ABINET_ROOT
ABINET_DEFAULT_CKPT = ABINET_ROOT / "workdir" / "train-abinet" / "best-train-abinet.pth"
SERVED = {"config": os.environ.get("LOGO_CONFIG", "FINAL")}  # set by --config
STAGES = ("extract", "det", "lsc", "fuse", "track", "vote", "render")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
NON_PIPELINE = {"base", "ablation", "paper", "reported_numbers", "freeze_baseline"}
API_VERSION = "1.0.0"


# ---------------------------------------------------------------- configs

def load_cfg(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def detector_weights(cfg: dict) -> Optional[str]:
    """Same lookup order as demo.detector_config, without importing the ML stack."""
    det = cfg.get("detector") or {}
    if det.get("weights"):
        return det["weights"]
    det_exp = (cfg.get("pipeline") or {}).get("det_exp")
    for p in ([CONFIG_DIR / f"{det_exp}.yaml", ROOT / "runs" / str(det_exp) / "config.yaml"] if det_exp else []):
        if p.is_file():
            return (load_cfg(p).get("detector") or {}).get("weights")
    return None


def _rel(p: Path) -> str:
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def lsc_ckpt(cfg: dict) -> Path:
    """run_lsc.pick_ckpt: runs/<exp_id>/checkpoints/best.pth > lsc.ckpt > ABINet's own default checkpoint."""
    best = ROOT / "runs" / str(cfg.get("exp_id")) / "checkpoints" / "best.pth"
    if best.is_file():
        return best
    c = (cfg.get("lsc") or {}).get("ckpt")
    return ABINET_DEFAULT_CKPT if c is None else resolve(c)


def requirements(cfg: dict) -> list[dict]:
    """Every artifact the trained pipeline loads, resolved against the backend root (not the working directory),
    with the step that produces it. Each entry: component, path, ok, hint."""
    tr = cfg.get("tracker") or {}
    w = detector_weights(cfg)
    det_exp = (cfg.get("pipeline") or {}).get("det_exp")
    items = [
        ("detector", resolve(w) if w else None,
         f"YOLO11s-OBB weights: python tools/train_obb.py --config configs/{det_exp}.yaml"),
        ("abinet", ABINET_ROOT / "demo.py",
         "ABINet source: git clone https://github.com/FangShancheng/ABINet third_party/ABINet"),
        ("lsc", lsc_ckpt(cfg), "ABINet LSC checkpoint: python -m logo.lsc.train_lsc --config configs/E3-lsc.yaml"),
    ]
    if tr.get("type", "simple") in ("embed", "lst"):
        ck = Path(tr.get("ckpt") or Path("runs") / str(cfg.get("exp_id")) / "checkpoints" / "best.pth")
        items.append(("tracker", resolve(ck), "VPMM + LST-Matcher checkpoint: python -m logo.tracking.train_lst "
                                              "--config configs/E5-lst.yaml"))
    if tr.get("type") == "lst":
        items.append(("gomatching", ROOT / "third_party" / "GoMatching" / "gomatching" / "modeling" / "roi_heads"
                      / "transformer.py", "LST-Matcher source: git clone https://github.com/Hxyz-123/GoMatching "
                                          "third_party/GoMatching"))
    if importlib.util.find_spec("ultralytics") is None and w and str(w).endswith(".pt"):
        items.append(("ultralytics", None, "uv pip install ultralytics==8.3.40"))
    out = []
    for comp, path, hint in items:
        ok = path is not None and path.is_file() and path.stat().st_size > 0
        out.append({"component": comp, "path": _rel(path) if path is not None else None, "ok": ok, "hint": hint})
    return out


def config_summary(p: Path) -> dict:
    cfg = load_cfg(p)
    tr, fu, vo = cfg.get("tracker") or {}, cfg.get("fusion") or {}, cfg.get("voting") or {}
    missing = [r for r in requirements(cfg) if not r["ok"]]
    return {"name": p.stem, "file": f"configs/{p.name}", "exp_id": cfg.get("exp_id"),
            "dataset": cfg.get("dataset"), "fusion": fu.get("mode", "hard"),
            "tracker": tr.get("type", "simple"), "voting": bool(vo.get("enabled")),
            "det_exp": (cfg.get("pipeline") or {}).get("det_exp"), "missing": missing, "ready": not missing,
            "kind": "logo", "label": p.stem, "recommended": p.stem == SERVED["config"] and not missing,
            "description": "LOGO pipeline with this project's trained detector, recognizer and tracker."}


PRETRAINED = "pretrained"


def pretrained_summary() -> dict:
    from logo import pretrained

    ok, err = pretrained.available()
    return {"name": PRETRAINED, "file": None, "exp_id": pretrained.EXP_ID, "dataset": "any video",
            "fusion": "soft", "tracker": "simple", "voting": True, "det_exp": None,
            "missing": [] if ok else [{"component": "easyocr", "path": err}], "ready": ok,
            "kind": "pretrained", "label": "Pretrained (EasyOCR)", "recommended": ok,
            "description": "Works out of the box: pretrained CRAFT detector + CRNN recognizer (EasyOCR), "
                           "with this project's soft score fusion, IoU tracker and track-level text voting."}


def list_configs() -> list[dict]:
    out = [pretrained_summary()]
    for p in sorted(CONFIG_DIR.glob("*.yaml")):
        if p.stem in NON_PIPELINE or p.stem.startswith("det_"):
            continue
        try:
            cfg = load_cfg(p)
        except yaml.YAMLError:
            continue
        if isinstance(cfg, dict) and cfg.get("exp_id") and cfg.get("dataset") and "tracker" in cfg:
            out.append(config_summary(p))
    out.sort(key=lambda c: (c["name"] != SERVED["config"], not c["ready"], c["name"] == PRETRAINED, c["name"]))
    return out


# ---------------------------------------------------------------- benchmark tables

_BOLD = re.compile(r"^\*\*(.*)\*\*$")
_ITAL = re.compile(r"^\*(.*)\*$")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_table(p: Path) -> Optional[dict]:
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if not ln.startswith("<!--")]
    tbl = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("|")]
    if len(tbl) < 2:
        return None
    caption = " ".join(ln.strip() for ln in lines[:tbl[0]] if ln.strip())
    m = re.match(r"^\*\*(.+?)\*\*\s*(.*)$", caption)
    title, note = (m.group(1), m.group(2)) if m else (caption, "")
    if ". " in title:  # whole caption in bold: first sentence is the title, the rest the note
        title, rest = title.split(". ", 1)
        note = f"{rest} {note}".strip()
    headers = _cells(lines[tbl[0]])
    aligns = ["right" if c.endswith(":") else "left" for c in _cells(lines[tbl[1]])]
    rows = []
    for i in tbl[2:]:
        raw = _cells(lines[i])
        if raw[0] and all(not c for c in raw[1:]) and _ITAL.match(raw[0]):
            rows.append({"section": _ITAL.match(raw[0]).group(1)})
            continue
        cells = []
        for c in raw:
            b = _BOLD.match(c)
            cells.append({"value": (b.group(1) if b else c).replace("\\", ""), "best": bool(b)})
        rows.append({"cells": cells})
    tail = " ".join(ln.strip() for ln in lines[tbl[-1] + 1:] if ln.strip())
    sources = re.findall(r"`([^`]+)`", tail)
    return {"id": p.stem, "title": title.rstrip("."), "note": note, "headers": headers, "align": aligns,
            "rows": rows, "sources": sources}


def list_benchmarks() -> list[dict]:
    order = ["ablation", "comparison", "efficiency", "ci", "errors", "threshold", "voting"]
    tables = [t for t in (parse_table(p) for p in sorted(TABLES_DIR.glob("*.md"))) if t]
    tables.sort(key=lambda t: next((i for i, k in enumerate(order) if t["id"].startswith(k)), len(order)))
    return tables


# ---------------------------------------------------------------- jobs

class JobStore:
    def __init__(self, base: Path):
        self.base = base
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        base.mkdir(parents=True, exist_ok=True)
        for f in base.glob("*/job.json"):
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if j.get("status") in ("queued", "running"):  # server stopped mid-job
                j.update(status="failed", error="Server restarted before the job finished.", finished_at=time.time())
                f.write_text(json.dumps(j, indent=2), encoding="utf-8")
            self.jobs[j["id"]] = j

    def dir(self, jid: str) -> Path:
        return self.base / jid

    def save(self, j: dict) -> None:
        (self.dir(j["id"]) / "job.json").write_text(json.dumps(j, indent=2, default=str), encoding="utf-8")

    def update(self, jid: str, **kw) -> dict:
        with self.lock:
            j = self.jobs[jid]
            j.update(kw)
            self.save(j)
            return dict(j)

    def get(self, jid: str) -> dict:
        with self.lock:
            if jid not in self.jobs:
                raise HTTPException(404, f"job {jid} not found")
            return dict(self.jobs[jid])

    def all(self) -> list[dict]:
        with self.lock:
            return sorted((dict(j) for j in self.jobs.values()), key=lambda j: -j["created_at"])

    def add(self, j: dict) -> None:
        with self.lock:
            self.jobs[j["id"]] = j
            self.save(j)

    def remove(self, jid: str) -> None:
        with self.lock:
            self.jobs.pop(jid, None)
        shutil.rmtree(self.dir(jid), ignore_errors=True)


class Pipeline:
    """Lazily loads scripts/demo.py and wraps its stage calls so the current job sees stage changes and the
    final track rows. One job at a time, so a single 'current job' slot is enough."""

    def __init__(self, store: JobStore):
        self.store = store
        self.demo = None
        self.error: Optional[str] = None
        self.current: Optional[str] = None
        self.rows: list[dict] = []
        self._load_lock = threading.Lock()

    def load(self):
        with self._load_lock:
            if self.demo is not None:
                return self.demo
            try:
                spec = importlib.util.spec_from_file_location("logo_demo", ROOT / "scripts" / "demo.py")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            except Exception as e:  # noqa: BLE001 - report any import failure to the client
                self.error = f"{type(e).__name__}: {e}"
                raise
            self._instrument(mod)
            self.demo, self.error = mod, None
            return mod

    def available(self) -> tuple[bool, Optional[str]]:
        if self.demo is not None:
            return True, None
        try:
            self.load()
            return True, None
        except Exception:  # noqa: BLE001
            return False, self.error

    def mark(self, stage: str) -> None:
        if not self.current:
            return
        now = time.time()
        j = self.store.get(self.current)
        timeline = j.get("timeline") or {}
        if stage in timeline:
            return
        timeline[stage] = now
        self.store.update(self.current, stage=stage, timeline=timeline)

    def progress(self, stage: str, done: int, total: int) -> None:
        """Per-frame progress of the current stage, written at most every 0.5 s (and on the last frame)."""
        if not self.current:
            return
        now = time.time()
        if done < total and now - getattr(self, "_last_progress", 0.0) < 0.5:
            return
        self._last_progress = now
        self.store.update(self.current, progress={"stage": stage, "done": done, "total": total})

    def _instrument(self, demo) -> None:
        pipe = self

        def wrap(fn, stage, capture=False):
            def inner(*a, **k):
                pipe.mark(stage)
                if capture:  # render(frame_dir, rows, out, fps)
                    pipe.rows = list(a[1] if len(a) > 1 else k["rows"])
                return fn(*a, **k)
            return inner

        class StageModule:
            def __init__(self, mod, stages: dict[str, str]):
                self._mod, self._stages = mod, stages

            def __getattr__(self, name):
                attr = getattr(self._mod, name)
                return wrap(attr, self._stages[name]) if name in self._stages else attr

        demo.extract_frames = wrap(demo.extract_frames, "extract")
        demo.run_detector = wrap(demo.run_detector, "det")
        demo.render = wrap(demo.render, "render", capture=True)
        demo.run_lsc = StageModule(demo.run_lsc, {"run": "lsc"})
        demo.run_fuse = StageModule(demo.run_fuse, {"run": "fuse"})
        demo.run_track = StageModule(demo.run_track, {"run": "track"})
        demo.track_vote = StageModule(demo.track_vote, {"run": "vote", "load_jsonl": "vote"})


def build_result(rows: list[dict], summary: dict) -> dict:
    by_frame: dict[int, list] = defaultdict(list)
    by_track: dict[int, list] = defaultdict(list)
    for r in rows:
        f, t = int(r["frame"]), int(r["track_id"])
        by_frame[f].append({"t": t, "q": [round(float(v), 1) for v in r["quad"]],
                            "s": round(float(r.get("score", 0.0)), 3)})
        by_track[t].append(r)
    tracks = []
    for t, rs in by_track.items():
        fr = [int(r["frame"]) for r in rs]
        texts = [str(r.get("text", "")) for r in rs if r.get("text")]
        text = max(set(texts), key=texts.count) if texts else ""
        tracks.append({"id": t, "text": text, "first_frame": min(fr), "last_frame": max(fr), "boxes": len(rs),
                       "mean_score": round(sum(float(r.get("score", 0.0)) for r in rs) / len(rs), 3)})
    tracks.sort(key=lambda x: (x["first_frame"], x["id"]))
    texts_by_track = {t["id"]: t["text"] for t in tracks}
    return {"fps": summary.get("video_fps"), "width": summary.get("width"), "height": summary.get("height"),
            "num_frames": summary.get("frames"), "tracks": tracks, "texts": texts_by_track,
            "frames": {str(k): v for k, v in sorted(by_frame.items())}}


def ffmpeg_exe() -> Optional[str]:
    """ffmpeg on PATH, else the binary bundled with imageio-ffmpeg (requirements-api.txt)."""
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def transcode(src: Path, dst: Path, log, fps: Optional[float] = None, audio: Optional[Path] = None) -> bool:
    """H.264/yuv420p mp4 for browsers. With fps the output is constant-rate at that fps, so frame i of the
    pipeline (decoded sequentially by OpenCV) is shown at time (i - 1) / fps, matching the overlay. With audio,
    the first audio stream of that file (if it has one) is muxed in as AAC."""
    ff = ffmpeg_exe()
    if not ff:
        return False
    cmd = [ff, "-y", "-loglevel", "error", "-i", str(src)]
    if audio is not None and Path(audio) != Path(src):
        cmd += ["-i", str(audio)]
    a_in = None if audio is None else (0 if Path(audio) == Path(src) else 1)
    cmd += ["-map", "0:v:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart"]
    cmd += ["-an"] if a_in is None else ["-map", f"{a_in}:a:0?", "-c:a", "aac", "-b:a", "128k", "-shortest"]
    if fps:
        cmd += ["-r", f"{fps:.6f}", "-vsync", "cfr"]
    cmd.append(str(dst))
    try:
        subprocess.run(cmd, check=True, timeout=1800)
        return True
    except (subprocess.SubprocessError, OSError) as e:
        log.warning("ffmpeg transcode failed: %s", e)
        dst.unlink(missing_ok=True)
        return False


def job_logger(jid: str, path: Path) -> logging.Logger:
    log = logging.getLogger(f"api.job.{jid}")
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def run_job(store: JobStore, pipe: Pipeline, jid: str) -> None:
    j = store.get(jid)
    d = store.dir(jid)
    log = job_logger(jid, d / "log.txt")
    store.update(jid, status="running", started_at=time.time(), stage=None, timeline={})
    try:
        demo = pipe.load()
        device = None if j["device"] == "auto" else j["device"]
        pipe.current, pipe.rows = jid, []
        if j["config"] == PRETRAINED:
            from logo import pretrained

            work = d / "work"
            try:
                summary, rows = pretrained.run_pretrained(
                    d / j["input"], d / "output.mp4", work, log, demo.render, device=device,
                    mode=j.get("mode") or "fast", on_stage=pipe.mark, on_progress=pipe.progress)
            finally:
                shutil.rmtree(work, ignore_errors=True)
        else:
            cfg = load_cfg(CONFIG_DIR / f"{j['config']}.yaml")
            det = demo.detector_config(cfg, ROOT)
            summary = demo.run_demo(d / j["input"], d / "output.mp4", cfg, ROOT, log, device=device, det=det)
            rows = pipe.rows
        pipe.current = None
        (d / "result.json").write_text(json.dumps(build_result(rows, summary)), encoding="utf-8")
        store.update(jid, progress={"stage": "render", "done": 0, "total": 0})
        web = transcode(d / "output.mp4", d / "output_web.mp4", log, audio=d / j["input"])
        transcode(d / j["input"], d / "source_web.mp4", log, fps=summary.get("video_fps"), audio=d / j["input"])
        summary = {k: v for k, v in summary.items() if k not in ("video", "out")}
        store.update(jid, status="succeeded", stage=None, progress=None, finished_at=time.time(),
                     summary=summary, web_video=web)
    except Exception as e:  # noqa: BLE001 - any failure becomes a failed job with the message shown in the UI
        log.error("job failed:\n%s", traceback.format_exc())
        store.update(jid, status="failed", finished_at=time.time(), error=f"{type(e).__name__}: {e}")
    finally:
        pipe.current = None
        for h in list(log.handlers):
            h.close()
            log.removeHandler(h)


def worker(store: JobStore, pipe: Pipeline, q: "queue.Queue[str]") -> None:
    while True:
        jid = q.get()
        try:
            if jid in store.jobs:
                run_job(store, pipe, jid)
        except Exception:  # noqa: BLE001 - never let the worker thread die
            traceback.print_exc()
        finally:
            q.task_done()


# ---------------------------------------------------------------- app

def create_app(origins: Optional[list[str]] = None, jobs_dir: Optional[Path] = None) -> FastAPI:
    store = JobStore(jobs_dir or JOBS_DIR)
    pipe = Pipeline(store)
    q: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=worker, args=(store, pipe, q), daemon=True, name="logo-worker").start()

    app = FastAPI(title="LOGO Video Text Spotting API", version=API_VERSION)
    app.add_middleware(CORSMiddleware, allow_origins=origins or ["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])

    def public(j: dict) -> dict:
        j = dict(j)
        j["queue_position"] = None
        if j["status"] == "queued":
            queued = [x["id"] for x in sorted(store.all(), key=lambda x: x["created_at"]) if x["status"] == "queued"]
            j["queue_position"] = queued.index(j["id"]) + 1 if j["id"] in queued else None
        return j

    @app.get("/api/health")
    def health() -> dict:
        ok, err = pipe.available()
        dev: dict[str, Any] = {"cuda": False, "name": None}
        if ok:
            try:
                import torch

                if torch.cuda.is_available():
                    dev = {"cuda": True, "name": torch.cuda.get_device_name(0)}
            except Exception:  # noqa: BLE001
                pass
        from logo import pretrained

        p_ok, p_err = pretrained.available()
        name = SERVED["config"]
        cp = CONFIG_DIR / f"{name}.yaml"
        reqs = requirements(load_cfg(cp)) if cp.is_file() else []
        missing = [r for r in reqs if not r["ok"]]
        model = {"config": name, "file": _rel(cp), "found": cp.is_file(), "requirements": reqs,
                 "missing": missing, "ready": ok and cp.is_file() and not missing}
        return {"version": API_VERSION, "model": model, "pipeline": {"available": ok, "error": err},
                "pretrained": {"available": ok and p_ok, "error": err or p_err}, "device": dev,
                "ffmpeg": ffmpeg_exe() is not None,
                "busy": any(j["status"] == "running" for j in store.all())}

    @app.get("/api/configs")
    def configs() -> list[dict]:
        return list_configs()

    @app.get("/api/benchmarks")
    def benchmarks() -> list[dict]:
        return list_benchmarks()

    @app.get("/api/jobs")
    def jobs() -> list[dict]:
        return [public(j) for j in store.all()]

    @app.post("/api/jobs", status_code=201)
    async def create_job(video: UploadFile = File(...), config: Optional[str] = Form(None),
                         device: str = Form("auto"), mode: str = Form("fast")) -> dict:
        ext = Path(video.filename or "").suffix.lower()
        if ext not in VIDEO_EXTS:
            raise HTTPException(422, f"unsupported file type {ext or '(none)'}; expected one of "
                                     f"{', '.join(sorted(VIDEO_EXTS))}")
        if device not in ("auto", "cuda", "cpu"):
            raise HTTPException(422, "device must be auto, cuda or cpu")
        if mode not in ("fast", "accurate"):
            raise HTTPException(422, "mode must be fast or accurate")
        config = config or SERVED["config"]
        cfg_info = next((c for c in list_configs() if c["name"] == config), None)
        if cfg_info is None:
            raise HTTPException(422, f"unknown pipeline config {config!r}")
        if not cfg_info["ready"]:
            what = "; ".join(f"{m['component']}: {m['path'] or 'not configured'}" for m in cfg_info["missing"])
            raise HTTPException(503, f"The trained {config} model cannot run because these files are missing "
                                     f"under {ROOT}: {what}.")
        ok, err = pipe.available()
        if not ok:
            raise HTTPException(503, f"The pipeline cannot be imported on the server: {err}")
        jid = uuid.uuid4().hex[:12]
        d = store.dir(jid)
        d.mkdir(parents=True)
        size = 0
        with open(d / f"input{ext}", "wb") as f:
            while chunk := await video.read(1 << 20):
                f.write(chunk)
                size += len(chunk)
        if size == 0:
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(422, "uploaded file is empty")
        j = {"id": jid, "filename": video.filename, "input": f"input{ext}", "size_bytes": size, "config": config,
             "device": device, "mode": mode, "status": "queued", "progress": None, "stage": None, "timeline": {}, "created_at": time.time(),
             "started_at": None, "finished_at": None, "summary": None, "error": None, "web_video": False}
        store.add(j)
        q.put(jid)
        return public(j)

    @app.get("/api/jobs/{jid}")
    def job(jid: str) -> dict:
        return public(store.get(jid))

    @app.delete("/api/jobs/{jid}", status_code=204, response_class=Response)
    def delete_job(jid: str) -> Response:
        j = store.get(jid)
        if j["status"] in ("queued", "running"):
            raise HTTPException(409, "job is still queued or running")
        store.remove(jid)
        return Response(status_code=204)

    @app.get("/api/jobs/{jid}/result")
    def result(jid: str) -> FileResponse:
        p = store.dir(store.get(jid)["id"]) / "result.json"
        if not p.is_file():
            raise HTTPException(404, "no result for this job")
        return FileResponse(p, media_type="application/json")

    @app.get("/api/jobs/{jid}/log", response_class=PlainTextResponse)
    def job_log(jid: str, lines: int = 200) -> str:
        p = store.dir(store.get(jid)["id"]) / "log.txt"
        if not p.is_file():
            return ""
        return "\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, lines):])

    @app.get("/api/jobs/{jid}/video/{kind}")
    def job_video(jid: str, kind: str, download: bool = False) -> FileResponse:
        j = store.get(jid)
        d = store.dir(jid)
        if kind == "source":
            web = d / "source_web.mp4"
            p = web if web.is_file() and not download else d / j["input"]
        elif kind == "output":
            p = d / ("output_web.mp4" if (d / "output_web.mp4").is_file() else "output.mp4")
        else:
            raise HTTPException(404, "kind must be source or output")
        if not p.is_file():
            raise HTTPException(404, f"{kind} video not available")
        stem = Path(j["filename"] or "video").stem
        name = f"{stem}{p.suffix}" if kind == "source" else f"{stem}_spotted.mp4"
        return FileResponse(p, filename=name if download else None,
                            content_disposition_type="attachment" if download else "inline")

    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", default=SERVED["config"],
                    help="pipeline config served to the UI (configs/<name>.yaml); default FINAL or $LOGO_CONFIG")
    ap.add_argument("--origin", action="append", help="extra CORS origin (repeatable)")
    ap.add_argument("--jobs-dir", type=Path, default=None, help=f"default: {JOBS_DIR.relative_to(ROOT)}")
    args = ap.parse_args(argv)
    if not (CONFIG_DIR / f"{args.config}.yaml").is_file():
        ap.error(f"config not found: {CONFIG_DIR / (args.config + '.yaml')}")
    SERVED["config"] = args.config
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    origins = ["http://localhost:5173", "http://127.0.0.1:5173"] + (args.origin or [])
    uvicorn.run(create_app(origins, args.jobs_dir), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
