"""End-to-end pipeline driver: det -> lsc -> fuse -> track -> vote -> eval for the config's dataset/split.

Stages run in that order (whatever order --stage lists them in). Each one reads the previous stage's files:
  det   runs/<det_exp>/det/<ds>/<video>.json                      check only (made in the logo-paddle env)
  lsc   runs/<lsc_exp>/lsc/<ds>/<video>.{json,_feats.npy}         logo.lsc.run_lsc; check only if lsc_exp != exp_id
  fuse  runs/<exp_id>/fuse/<ds>/<video>.json                      logo.scoring.run_fuse
  track runs/<exp_id>/track/<ds>/<video>.jsonl                    logo.tracking.run_track
  vote  runs/<exp_id>/vote/<ds>/<video>.jsonl                     pass-through copy of track until P26
  eval  runs/<exp_id>/metrics.json                                det on fuse (score_fused > eval.det_score_thr),
                                                                  track + spot (if eval.spotting) on vote
A stage is skipped when all its outputs exist, unless --force or an earlier stage ran in this call.
Stages between the first and last requested ones also run (implicit): a stage never reads files older than
an upstream stage re-run in the same call. Implicit stages ignore --force and run only when an earlier stage
ran or their outputs are missing (e.g. --stage fuse,track,eval also refreshes vote).
det_exp / lsc_exp: CLI > config pipeline.{det_exp,lsc_exp} > exp_id. Per-stage status and seconds go to
log.txt and metrics.json meta.pipeline_<ds>_<split>. split=test needs --final.

Usage: python scripts/run_pipeline.py --config configs/E4-lsc.yaml --stage det,lsc,fuse,track,vote,eval
           [--force] [--final] [--det_exp EXP] [--lsc_exp EXP] [--device cuda]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from logo.detector.infer_paddle import read_videos, write_run_files
from logo.lsc import run_lsc
from logo.scoring import run_fuse
from logo.tracking import run_track
from logo.utils.io import get_logger, load_json, load_yaml, save_json, set_seed
from logo.utils.paths import check_dataset, check_split, project_root

_spec = importlib.util.spec_from_file_location(
    "evaluate_tool", Path(__file__).resolve().parents[1] / "tools" / "evaluate.py")
evaluate_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_tool)

STAGES = ("det", "lsc", "fuse", "track", "vote", "eval")
DET_SCORE_THR = 0.5  # default eval.det_score_thr


def parse_stages(s: str) -> list[str]:
    names = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in names if x not in STAGES]
    if bad or not names:
        raise ValueError(f"unknown stage(s) {bad or s!r}; expected a comma list of {STAGES}")
    return [x for x in STAGES if x in names]


def source_exps(cfg: dict, det_exp: Optional[str] = None, lsc_exp: Optional[str] = None) -> tuple[str, str]:
    pc = cfg.get("pipeline") or {}
    return (det_exp or pc.get("det_exp") or cfg["exp_id"], lsc_exp or pc.get("lsc_exp") or cfg["exp_id"])


def plan_stages(stages: list[str]) -> list[str]:
    """Every stage from the first to the last requested one, in pipeline order."""
    idx = [STAGES.index(s) for s in stages]
    return list(STAGES[min(idx):max(idx) + 1])


def stage_dir(root: Path, exp: str, stage: str, ds: str) -> Path:
    return root / "runs" / exp / stage / ds


def outputs(stage: str, root: Path, cfg: dict, ds: str, split: str, videos: list[str],
            det_exp: str, lsc_exp: str) -> list[Path]:
    """Files whose existence marks the stage as done (eval: metrics.json, checked separately)."""
    exp = cfg["exp_id"]
    if stage == "det":
        return [stage_dir(root, det_exp, "det", ds) / f"{v}.json" for v in videos]
    if stage == "lsc":
        d = stage_dir(root, lsc_exp, "lsc", ds)
        return [p for v in videos for p in (d / f"{v}.json", run_lsc.feats_path(d, v))]
    if stage == "fuse":
        return [stage_dir(root, exp, "fuse", ds) / f"{v}.json" for v in videos]
    if stage in ("track", "vote"):
        return [stage_dir(root, exp, stage, ds) / f"{v}.jsonl" for v in videos]
    return [root / "runs" / exp / "metrics.json"]


def eval_done(root: Path, cfg: dict, ds: str, split: str) -> bool:
    m = root / "runs" / cfg["exp_id"] / "metrics.json"
    return m.is_file() and f"eval_{ds}_{split}" in (load_json(m).get("meta") or {})


def require(paths: list[Path], what: str) -> None:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{what}: {len(missing)} missing file(s), e.g. {missing[0]}")


def make_recognizer(cfg: dict, root: Path, log, device: Optional[str] = None):
    """ABINetRecognizer as in logo.lsc.run_lsc.main (ckpt via run_lsc.pick_ckpt, lsc.infer settings)."""
    import torch

    from logo.lsc.abinet_wrapper import ABINetRecognizer

    ic = run_lsc.infer_settings(cfg)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = run_lsc.pick_ckpt(cfg, root)
    rec = ABINetRecognizer(ckpt, device, fp16=ic["amp"])
    rec.batch_size = ic["batch_size"]
    log.info("lsc recognizer: ckpt=%s device=%s bs=%d amp=%s", ckpt, device, rec.batch_size, rec.fp16)
    return rec


def copy_votes(root: Path, cfg: dict, ds: str, videos: list[str]) -> dict:
    src, dst = stage_dir(root, cfg["exp_id"], "track", ds), stage_dir(root, cfg["exp_id"], "vote", ds)
    dst.mkdir(parents=True, exist_ok=True)
    for v in videos:
        shutil.copyfile(src / f"{v}.jsonl", dst / f"{v}.jsonl")
    return {"videos": len(videos), "mode": "pass-through"}


def evaluate(root: Path, cfg: dict, ds: str, split: str, videos: list[str], log) -> dict:
    """Writes det/track/spot/per_video and meta.{det,track,spot,eval_<ds>_<split>} into metrics.json."""
    ec = cfg.get("eval") or {}
    iou, spotting = float(ec.get("iou", 0.5)), bool(ec.get("spotting", True))
    thr = float(ec.get("det_score_thr", DET_SCORE_THR))
    gt_dir = root / "data" / "processed" / ds
    fuse_dir = stage_dir(root, cfg["exp_id"], "fuse", ds)
    vote_dir = stage_dir(root, cfg["exp_id"], "vote", ds)
    m_out = root / "runs" / cfg["exp_id"] / "metrics.json"
    metrics = evaluate_tool.empty_metrics()
    if m_out.is_file():
        metrics.update(load_json(m_out))
    meta = metrics.setdefault("meta", {})

    with tempfile.TemporaryDirectory() as tmp:
        pred = evaluate_tool.filter_preds(fuse_dir, videos, Path(tmp), "score_fused", thr)
        metrics["det"], meta["det"] = evaluate_tool.run(gt_dir, pred, videos, "det", iou)
    meta["det"].update(pred_dir=str(fuse_dir), score_key="score_fused", score_thr=thr)
    metrics["track"], meta["track"] = evaluate_tool.run(gt_dir, vote_dir, videos, "track", iou)
    metrics["per_video"] = meta["track"].pop("per_video")
    meta["track"]["pred_dir"] = str(vote_dir)
    if spotting:
        metrics["spot"], meta["spot"] = evaluate_tool.run(gt_dir, vote_dir, videos, "spot", iou)
        meta["spot"]["pred_dir"] = str(vote_dir)
    else:
        metrics["spot"] = {}
        meta.pop("spot", None)
    meta[f"eval_{ds}_{split}"] = {"iou": iou, "spotting": spotting, "det_score_thr": thr,
                                  "num_videos": len(videos), "gt_dir": str(gt_dir)}
    save_json(metrics, m_out)
    for k in ("det", "track", "spot"):
        if metrics[k]:
            log.info("eval %s: %s", k, " ".join(f"{n}={v:.2f}" for n, v in metrics[k].items()))
    return {k: metrics[k] for k in ("det", "track", "spot")}


def run_pipeline(cfg: dict, stages: list[str], root: Path, log, force: bool = False, final: bool = False,
                 det_exp: Optional[str] = None, lsc_exp: Optional[str] = None,
                 recognizer_factory: Optional[Callable] = None, device: Optional[str] = None) -> dict:
    """Runs the requested stages for cfg dataset/split; returns {stage: {"status", "seconds"}}."""
    ds, split = check_dataset(cfg["dataset"]), check_split(cfg["split"], final)
    exp = cfg["exp_id"]
    det_exp, lsc_exp = source_exps(cfg, det_exp, lsc_exp)
    videos = read_videos(ds, split, root, final=final)
    run_dir = root / "runs" / exp
    set_seed(cfg["seed"])
    log.info("pipeline exp=%s ds=%s split=%s videos=%d stages=%s det_exp=%s lsc_exp=%s force=%s", exp, ds,
             split, len(videos), ",".join(stages), det_exp, lsc_exp, force)
    report, dirty = {}, False
    for stage in plan_stages(stages):
        implicit = stage not in stages
        t0 = time.perf_counter()
        outs = outputs(stage, root, cfg, ds, split, videos, det_exp, lsc_exp)
        check_only = stage == "det" or (stage == "lsc" and lsc_exp != exp)
        done = eval_done(root, cfg, ds, split) if stage == "eval" else all(p.is_file() for p in outs)
        if check_only:
            require(outs, f"{stage} outputs of exp {det_exp if stage == 'det' else lsc_exp}")
            status = "checked"
        elif done and not dirty and (implicit or not force):
            status = "skipped"
        else:
            status, dirty = "ran", True
            if stage == "lsc":
                require(outputs("det", root, cfg, ds, split, videos, det_exp, lsc_exp), "lsc input")
                rec = (recognizer_factory or make_recognizer)(cfg, root, log, device)
                stats = run_lsc.run(cfg, ds, videos, rec, root, log, det_exp, overwrite=True)
                write_run_files(cfg, run_dir, stats, key=f"lsc_infer_{ds}_{split}")
            elif stage == "fuse":
                require(outputs("lsc", root, cfg, ds, split, videos, det_exp, lsc_exp), "fuse input")
                stats = run_fuse.run(cfg, ds, videos, root, log, lsc_exp)
                write_run_files(cfg, run_dir, stats, key=f"fuse_{ds}_{split}")
            elif stage == "track":
                require(outputs("fuse", root, cfg, ds, split, videos, det_exp, lsc_exp), "track input")
                stats = run_track.run(cfg, ds, videos, root, log, exp, lsc_exp=lsc_exp)
                write_run_files(cfg, run_dir, stats, key=f"track_{ds}_{split}")
            elif stage == "vote":
                require(outputs("track", root, cfg, ds, split, videos, det_exp, lsc_exp), "vote input")
                if (cfg.get("voting") or {}).get("enabled"):
                    log.warning("voting.enabled is set but track voting arrives in P26; copying tracks")
                copy_votes(root, cfg, ds, videos)
            elif stage == "eval":
                require(outputs("fuse", root, cfg, ds, split, videos, det_exp, lsc_exp)
                        + outputs("vote", root, cfg, ds, split, videos, det_exp, lsc_exp), "eval input")
                evaluate(root, cfg, ds, split, videos, log)
        dt = round(time.perf_counter() - t0, 3)
        report[stage] = {"status": status, "seconds": dt, **({"implicit": True} if implicit else {})}
        log.info("stage %-5s %-7s %.2f s%s", stage, status, dt, " (implicit)" if implicit else "")
    write_run_files(cfg, run_dir, {"stages": report, "det_exp": det_exp, "lsc_exp": lsc_exp,
                                   "force": force, "final": final}, key=f"pipeline_{ds}_{split}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--stage", default=",".join(STAGES), help=f"comma list of {','.join(STAGES)}")
    ap.add_argument("--force", action="store_true", help="re-run stages whose outputs exist")
    ap.add_argument("--final", action="store_true", help="allow split=test (rule 5)")
    ap.add_argument("--det_exp", default=None)
    ap.add_argument("--lsc_exp", default=None)
    ap.add_argument("--device", default=None, help="lsc device; default cuda if available else cpu")
    args = ap.parse_args(argv)

    try:
        stages = parse_stages(args.stage)
    except ValueError as e:
        ap.error(str(e))
    cfg = copy.deepcopy(load_yaml(args.config))
    if cfg.get("split") == "test" and not args.final:
        ap.error("config split is test: pass --final")
    root = project_root()
    log = get_logger("run_pipeline", root / "runs" / cfg["exp_id"] / "log.txt")
    t0 = time.perf_counter()
    run_pipeline(cfg, stages, root, log, force=args.force, final=args.final, det_exp=args.det_exp,
                 lsc_exp=args.lsc_exp, device=args.device)
    log.info("pipeline done in %.2f s", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
