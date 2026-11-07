"""Pretrained video text spotting: runs without any project-trained weights.

Same stage layout as the LOGO pipeline (scripts/demo.py), with off-the-shelf models for detection and
recognition and this project's own tracking and voting:

  extract  logo.data.frames.extract_frames (every frame, 1-based)
  det      EasyOCR CRAFT text detector on every `step`-th frame
  lsc      EasyOCR CRNN recognizer on each detected box -> text, text_conf
  fuse     score_fused = (1 - w) * det + w * text_conf (C2 soft fusion; CRAFT gives no box score, det = 1)
  track    logo.tracking.simple_tracker.run_tracker (ByteTrack-style IoU tracker)
  vote     logo.voting.track_vote.vote_tracks (C3), then quads linearly interpolated over skipped frames
  render   caller-supplied render(frame_dir, rows, out, fps) (demo.render)

EasyOCR downloads its weights (~100 MB) to ~/.EasyOCR/model on first use.
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from logo.data.frames import extract_frames
from logo.eval.match import normalize_text
from logo.scoring.fusion import fuse
from logo.tracking.simple_tracker import run_tracker
from logo.voting.track_vote import vote_tracks

EXP_ID = "pretrained-easyocr"
MODES = {
    # analysis frames per second of video, CRAFT magnification
    "fast": {"analysis_fps": 10.0, "mag_ratio": 1.0},
    "accurate": {"analysis_fps": None, "mag_ratio": 1.5},  # None: every frame
}
FUSION_W = 0.5
MIN_TEXT_CONF = 0.15  # recognizer confidence below this: box dropped before tracking
VOTING = {"tau": 0.35, "min_len": 2}  # mean fused language score / min boxes per track
MIN_TEXT_CHARS = 2  # voted text must keep >= 2 characters after normalize_text (a-z0-9)

StageFn = Callable[[str], None]
ProgressFn = Callable[[str, int, int], None]


def available() -> tuple[bool, Optional[str]]:
    try:
        import easyocr  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    return True, None


_READERS: dict[tuple[str, bool], object] = {}


def get_reader(langs: tuple[str, ...], gpu: bool):
    """Cached easyocr.Reader (model load takes several seconds)."""
    import easyocr

    key = (",".join(langs), gpu)
    if key not in _READERS:
        _READERS[key] = easyocr.Reader(list(langs), gpu=gpu, verbose=False)
    return _READERS[key]


def _read_frame(p: Path):
    import cv2

    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read frame {p}")
    return img


def _quad(pts) -> list[float]:
    from logo.utils.geometry import order_quad

    return order_quad([float(v) for xy in pts for v in xy])


def interpolate(rows: list[dict], max_gap: int) -> list[dict]:
    """Adds linearly interpolated boxes for frames between two consecutive boxes of the same track
    (gap <= max_gap), so a track sampled every few frames is drawn on every frame."""
    by = defaultdict(list)
    for r in rows:
        by[r["track_id"]].append(r)
    out = list(rows)
    for rs in by.values():
        rs.sort(key=lambda r: r["frame"])
        for a, b in zip(rs, rs[1:]):
            gap = b["frame"] - a["frame"]
            if gap <= 1 or gap > max_gap:
                continue
            qa, qb = np.asarray(a["quad"]), np.asarray(b["quad"])
            for k in range(1, gap):
                t = k / gap
                out.append({"frame": a["frame"] + k, "track_id": a["track_id"],
                            "quad": [float(v) for v in (1 - t) * qa + t * qb],
                            "score": (1 - t) * a["score"] + t * b["score"], "text": a["text"],
                            "interpolated": True})
    return sorted(out, key=lambda r: (r["frame"], r["track_id"]))


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


def run_pretrained(video: Path, out: Path, work_dir: Path, log, render: Callable, device: Optional[str] = None,
                   mode: str = "fast", langs: tuple[str, ...] = ("en",), on_stage: StageFn = lambda s: None,
                   on_progress: ProgressFn = lambda s, d, t: None) -> tuple[dict, list[dict]]:
    """Full pipeline on one video; frames go to work_dir/frames. Returns (summary, rows drawn per frame)."""
    import cv2
    import torch

    if mode not in MODES:
        raise ValueError(f"mode must be one of {tuple(MODES)}, got {mode!r}")
    gpu = (device or ("cuda" if torch.cuda.is_available() else "cpu")).startswith("cuda") and torch.cuda.is_available()
    sec: dict[str, float] = {}
    t_all = time.perf_counter()

    on_stage("extract")
    t0 = time.perf_counter()
    frame_dir = work_dir / "frames"
    n = extract_frames(video, frame_dir, quality=92)
    if n == 0:
        raise ValueError(f"no frames decoded from {video}")
    frames = sorted((int(p.stem), p) for p in frame_dir.glob("[0-9]" * 6 + ".jpg"))
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
    cap.release()
    fps = fps if np.isfinite(fps) and 1.0 <= fps <= 240.0 else 25.0
    h, w = _read_frame(frames[0][1]).shape[:2]
    m = MODES[mode]
    step = 1 if m["analysis_fps"] is None else max(1, int(round(fps / m["analysis_fps"])))
    sampled = [(f, p) for f, p in frames if (f - 1) % step == 0]
    sec["extract"] = time.perf_counter() - t0
    log.info("pretrained: %s %dx%d %d frames @ %.2f fps, mode=%s step=%d (%d analysed), device=%s",
             video.name, w, h, n, fps, mode, step, len(sampled), "cuda" if gpu else "cpu")

    on_stage("det")
    t0 = time.perf_counter()
    reader = get_reader(langs, gpu)
    canvas = int(min(2560, max(h, w) * m["mag_ratio"]))
    boxes: dict[int, tuple[list, list]] = {}
    for i, (f, p) in enumerate(sampled):
        rgb = cv2.cvtColor(_read_frame(p), cv2.COLOR_BGR2RGB)
        hl, fl = reader.detect(rgb, canvas_size=canvas, mag_ratio=m["mag_ratio"], min_size=8, text_threshold=0.6,
                               low_text=0.35, link_threshold=0.4)
        boxes[f] = (hl[0], fl[0])
        on_progress("det", i + 1, len(sampled))
    sec["det"] = time.perf_counter() - t0

    on_stage("lsc")
    t0 = time.perf_counter()
    lsc: dict[int, list[dict]] = {}
    for i, (f, p) in enumerate(sampled):
        hl, fl = boxes[f]
        objs = []
        if hl or fl:
            grey = cv2.cvtColor(_read_frame(p), cv2.COLOR_BGR2GRAY)
            for pts, text, conf in reader.recognize(grey, horizontal_list=hl, free_list=fl, detail=1):
                text = str(text).strip()
                if text:
                    objs.append({"quad": _quad(pts), "text": text, "text_conf": float(conf)})
        lsc[f] = objs
        on_progress("lsc", i + 1, len(sampled))
    sec["lsc"] = time.perf_counter() - t0
    if gpu:
        torch.cuda.empty_cache()

    on_stage("fuse")
    t0 = time.perf_counter()
    fused = {"frames": {}}
    for f, objs in lsc.items():
        keep = [o for o in objs if o["text_conf"] >= MIN_TEXT_CONF]
        fused["frames"][str(f)] = [{**o, "det_score": 1.0,
                                    "score_fused": fuse(1.0, o["text_conf"], "soft", FUSION_W)} for o in keep]
    sec["fuse"] = time.perf_counter() - t0

    on_stage("track")
    t0 = time.perf_counter()
    tracker = {"tracker": {"type": "simple", "high_thr": 0.6, "low_thr": 0.5, "iou_gate": 0.2,
                           "max_age": max(15, 4 * step), "text_window": 10}}
    tracked = run_tracker(fused, tracker)
    sec["track"] = time.perf_counter() - t0

    on_stage("vote")
    t0 = time.perf_counter()
    conf = {(str(fr), tuple(o["quad"])): o["text_conf"] for fr, objs in fused["frames"].items() for o in objs}
    for r in tracked:
        c = conf.get((str(r["frame"]), tuple(r["quad"])), r["score"])
        r["text_conf"] = r["lang"] = float(c)
    voted = vote_tracks(tracked, {"voting": VOTING})
    # as run_track.filter_trajectories in spotting mode: drop tracks whose text has < 2 letters/digits
    voted = [r for r in voted if len(normalize_text(r["text"])) >= MIN_TEXT_CHARS]
    rows = interpolate(voted, max_gap=2 * step) if step > 1 else voted
    sec["vote"] = time.perf_counter() - t0

    on_stage("render")
    t0 = time.perf_counter()
    render(frame_dir, rows, out, fps)
    sec["render"] = time.perf_counter() - t0

    total = time.perf_counter() - t_all
    pipe = sum(sec[k] for k in ("det", "lsc", "fuse", "track", "vote"))
    summary = {"source_exp": EXP_ID, "detector": "easyocr-craft", "device": "cuda" if gpu else "cpu",
               "voting": True, "mode": mode, "step": step, "frames": n, "width": w, "height": h, "video_fps": fps,
               **summarize(rows), "seconds": {k: round(v, 3) for k, v in sec.items()},
               "fps_pipeline": n / pipe if pipe > 0 else 0.0, "fps_total": n / total if total > 0 else 0.0}
    log.info("pretrained done: %d tracks, %d boxes, %.1f s", summary["tracks"], summary["boxes"], total)
    return summary, rows
