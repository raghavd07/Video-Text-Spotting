import logging
import random

import pytest

from logo.tracking.run_track import save_jsonl
from logo.utils.io import save_json
from logo.voting.track_vote import attach_scores, load_jsonl, run, vote_tracks

CFG = {"voting": {"enabled": True, "tau": 0.5, "min_len": 3}}


def _row(f, tid, text, conf=None, lang=None, score=0.8, x=10.0):
    r = {"frame": f, "track_id": tid, "quad": [x, 10.0, x + 40, 10.0, x + 40, 22.0, x, 22.0], "score": score,
         "text": text}
    if conf is not None:
        r["text_conf"] = conf
    if lang is not None:
        r["lang"] = lang
    return r


def _noisy(truth: str, rng: random.Random) -> str:
    s = list(truth)
    i = rng.randrange(len(s))
    s[i] = rng.choice("abcdefghijklmnopqrstuvwxyz0123456789")
    return "".join(s)


def test_noisy_strings_converge_to_truth():
    rng = random.Random(0)
    rows = []
    for f in range(1, 31):  # 12 exact (case / punctuation variants), 18 single-character corruptions
        if f % 5 in (0, 2):
            text, conf = rng.choice(["Hello", "HELLO!", "hello"]), rng.uniform(0.6, 0.9)
        else:
            text, conf = _noisy("hello", rng), rng.uniform(0.6, 0.9)
        rows.append(_row(f, 7, text, conf, rng.uniform(0.7, 1.0)))
    out = vote_tracks(rows, CFG)
    assert len(out) == 30 and {r["text"].lower().strip("!") for r in out} == {"hello"}
    assert all(set(r) == {"frame", "track_id", "quad", "score", "text"} for r in out)
    assert [r["frame"] for r in out] == list(range(1, 31))


def test_weights_beat_counts_and_tie_breaks():
    # "world" appears twice at low confidence, "w0rld" once at high: summed weight decides (0.18 < 0.81)
    rows = [_row(1, 1, "world", 0.3, 0.3), _row(2, 1, "world", 0.3, 0.3), _row(3, 1, "w0rld", 0.9, 0.9)]
    assert {r["text"] for r in vote_tracks(rows, {"voting": {"tau": 0.0, "min_len": 1}})} == {"w0rld"}
    # equal summed weight 0.5: "abc" mean conf 1.0 (lang 0.5) beats "abd" mean conf 0.5 (lang 1.0)
    rows = [_row(1, 1, "abd", 0.5, 1.0), _row(2, 1, "ABC", 1.0, 0.5)]
    assert {r["text"] for r in vote_tracks(rows, {"voting": {"tau": 0.0, "min_len": 1}})} == {"ABC"}
    # empty normalized text never wins against real text
    rows = [_row(1, 1, "", 1.0, 1.0), _row(2, 1, "!!", 1.0, 1.0), _row(3, 1, "ok", 0.1, 0.1)]
    assert {r["text"] for r in vote_tracks(rows, {"voting": {"tau": 0.0, "min_len": 1}})} == {"ok"}


def test_short_and_low_confidence_tracks_removed():
    rows = ([_row(f, 1, "good", 0.9, 0.9) for f in range(1, 11)]
            + [_row(f, 2, "zz", 0.2, 0.1, x=200.0) for f in range(1, 11)]      # long but mean lang 0.1 < tau
            + [_row(f, 3, "tiny", 0.9, 0.9, x=400.0) for f in range(1, 3)]     # high lang but 2 boxes < min_len
            + [_row(f, 4, "x", 0.3, 0.2, x=600.0) for f in range(1, 3)])       # short and weak
    out = vote_tracks(rows, CFG)
    assert {r["track_id"] for r in out} == {1}
    assert {r["track_id"] for r in vote_tracks(rows, {"voting": {"tau": 0.0, "min_len": 1}})} == {1, 2, 3, 4}


def test_missing_extras_default_to_score():
    rows = [_row(f, 1, "abc", score=0.4) for f in range(1, 5)]
    assert vote_tracks(rows, CFG) == []                            # mean lang = score 0.4 < tau 0.5
    rows = [_row(f, 1, "abc", score=0.6) for f in range(1, 5)]
    assert len(vote_tracks(rows, CFG)) == 4
    with pytest.raises(ValueError):
        vote_tracks(rows, {"voting": {"tau": 0.5, "min_len": -1}})


def test_attach_scores_matches_frame_and_quad():
    det = _row(1, 0, "abc")
    fused = {"frames": {"1": [{"quad": det["quad"], "score": 0.8, "text": "abc", "text_conf": 0.7, "p_eos": 0.2,
                               "lang_hard": 1, "score_fused": 0.9}], "2": []}}
    rows = [_row(1, 3, "abc"), _row(2, 3, "abc"), _row(1, 4, "abc", x=99.0)]
    soft = attach_scores(rows, fused, "soft", "p_eos")
    assert soft[0]["text_conf"] == pytest.approx(0.7) and soft[0]["lang"] == pytest.approx(0.8)
    assert "lang" not in soft[1] and "lang" not in soft[2] and "lang" not in rows[0]
    assert attach_scores(rows, fused, "hard")[0]["lang"] == 1.0


def test_run_writes_vote_dir(tmp_path):
    ds, cfg = "icdar2015v", {"exp_id": "E", "fusion": {"mode": "soft", "w": 0.5, "soft_source": "text_conf"},
                             "voting": {"enabled": True, "tau": 0.5, "min_len": 2}}
    rows = [_row(1, 1, "cat"), _row(2, 1, "cot"), _row(3, 1, "cat"), _row(1, 2, "dog", x=300.0)]
    fused = {"frames": {str(f): [{"quad": r["quad"], "score": 0.8, "text": r["text"], "text_conf": 0.9,
                                  "p_eos": 0.1, "lang_hard": 1, "score_fused": 0.85}
                                 for r in rows if r["frame"] == f] for f in (1, 2, 3)}}
    save_jsonl(rows, tmp_path / "runs/E/track" / ds / "v.jsonl")
    save_json(fused, tmp_path / "runs/E/fuse" / ds / "v.json")
    stats = run(cfg, ds, ["v"], tmp_path, logging.getLogger("t"))
    out = load_jsonl(tmp_path / "runs/E/vote" / ds / "v.jsonl")
    assert [r["text"] for r in out] == ["cat"] * 3 and stats["tracks_in"] == 2 and stats["tracks_out"] == 1
