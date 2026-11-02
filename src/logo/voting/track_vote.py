"""C3 track-level text voting: one text per track, weighted by per-box text_conf * lang; weak tracks removed.

vote_tracks(tracks, cfg): tracks are track-jsonl rows, optionally carrying the extra keys "text_conf" and "lang"
(each defaults to the row's score). Per track_id every box text is normalized (logo.eval.match.normalize_text: lowercase, a-z0-9 only) and
weighted by text_conf * lang; the normalized string with the highest summed weight wins (tie -> highest mean
text_conf, then the lexicographically smallest string). Empty normalized strings win only when no box has
any other text. The output text is the winning group's original spelling with the highest summed weight
(tie -> first seen) and is written to every box of the track. Tracks with mean lang < voting.tau or fewer than
voting.min_len boxes are dropped. Returned rows lose the extra keys (track-jsonl contract).

Stage (scripts/run_pipeline.py "vote"): runs/<exp_id>/track/<ds>/<video>.jsonl + runs/<exp_id>/fuse/<ds>/
<video>.json -> runs/<exp_id>/vote/<ds>/<video>.jsonl. text_conf and lang (logo.scoring.fusion.lang_score with
fusion.mode / fusion.soft_source) are recovered from the fused detection with the same frame and quad.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from logo.eval.match import normalize_text
from logo.scoring.fusion import lang_score
from logo.scoring.run_fuse import soft_source
from logo.tracking.run_track import save_jsonl
from logo.utils.io import load_json

STAGE = "vote"
EXTRA_KEYS = ("text_conf", "lang")


def voting_settings(cfg: dict) -> dict:
    v = cfg.get("voting") or {}
    s = {"tau": float(v.get("tau", 0.5)), "min_len": int(v.get("min_len", 3))}
    if s["min_len"] < 0:
        raise ValueError(f"voting.min_len must be >= 0, got {s['min_len']}")
    return s


def _vote_text(rows: list[dict]) -> str:
    """Winning original text of one track (see module docstring)."""
    groups: dict[str, dict] = {}
    for r in rows:
        conf = float(r.get("text_conf", r["score"]))
        wt = conf * float(r.get("lang", r["score"]))
        text = str(r.get("text", ""))
        g = groups.setdefault(normalize_text(text), {"w": 0.0, "conf": [], "orig": {}})
        g["w"] += wt
        g["conf"].append(conf)
        g["orig"][text] = g["orig"].get(text, 0.0) + wt  # dict keeps first-seen order for ties
    cands = [k for k in groups if k] or list(groups)
    best = min(cands, key=lambda k: (-round(groups[k]["w"], 9),
                                     -round(sum(groups[k]["conf"]) / len(groups[k]["conf"]), 9), k))
    orig = groups[best]["orig"]
    return max(orig, key=lambda t: round(orig[t], 9))


def vote_tracks(tracks, cfg) -> list[dict]:
    """Track rows -> voted rows (same order, filtered tracks removed, extra keys dropped)."""
    s = voting_settings(cfg)
    by = defaultdict(list)
    for r in tracks:
        by[r["track_id"]].append(r)
    texts = {}
    for tid, rs in by.items():
        mean_lang = sum(float(r.get("lang", r["score"])) for r in rs) / len(rs)
        if len(rs) >= s["min_len"] and mean_lang >= s["tau"]:
            texts[tid] = _vote_text(rs)
    return [{**{k: v for k, v in r.items() if k not in EXTRA_KEYS}, "text": texts[r["track_id"]]}
            for r in tracks if r["track_id"] in texts]


def _quad_key(frame, quad) -> tuple:
    return (int(frame), *(round(float(v), 4) for v in quad))


def attach_scores(rows: list[dict], fused: dict, mode: str, source: str = "p_eos") -> list[dict]:
    """Copies of rows with "text_conf" and "lang" from the fused detection of the same frame and quad
    (first one if several); rows without a match are returned unchanged (vote_tracks then uses score)."""
    index = {}
    for fid, objs in fused["frames"].items():
        for o in objs:
            index.setdefault(_quad_key(fid, o["quad"]), o)
    out = []
    for r in rows:
        o = index.get(_quad_key(r["frame"], r["quad"]))
        if o is None:
            out.append(dict(r))
        else:
            out.append({**r, "text_conf": float(o.get("text_conf", r["score"])),
                        "lang": float(lang_score(o, mode, source))})
    return out


def load_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_video(root: Path, exp: str, ds: str, video: str, cfg: dict) -> list[dict]:
    """runs/<exp>/track/<ds>/<video>.jsonl rows with text_conf / lang attached from runs/<exp>/fuse/."""
    rows = load_jsonl(root / "runs" / exp / "track" / ds / f"{video}.jsonl")
    fused = load_json(root / "runs" / exp / "fuse" / ds / f"{video}.json")
    return attach_scores(rows, fused, cfg["fusion"]["mode"], soft_source(cfg))


def run(cfg: dict, ds: str, videos: list[str], root: Path, log) -> dict:
    """Votes every video of runs/<exp_id>/track/<ds> into runs/<exp_id>/vote/<ds>."""
    exp, s = cfg["exp_id"], voting_settings(cfg)
    out_dir = root / "runs" / exp / STAGE / ds
    n_in = n_out = t_in = t_out = 0
    for video in videos:
        rows = load_video(root, exp, ds, video, cfg)
        voted = vote_tracks(rows, cfg)
        save_jsonl(voted, out_dir / f"{video}.jsonl")
        ti, to = len({r["track_id"] for r in rows}), len({r["track_id"] for r in voted})
        n_in, n_out, t_in, t_out = n_in + len(rows), n_out + len(voted), t_in + ti, t_out + to
        log.info("%s: %d/%d tracks kept, %d/%d boxes", video, to, ti, len(voted), len(rows))
    return {"videos": len(videos), "mode": "vote", **s, "boxes_in": n_in, "boxes_out": n_out,
            "tracks_in": t_in, "tracks_out": t_out}
