"""P28: final evaluation of one frozen config on each dataset's held-out split (refuses to run without --final).

Held-out split per dataset: the last docs/decisions.md table row whose decision cell reads
"held-out <ds>: test" or "held-out <ds>: val" (val when test labels are unavailable). icdar2015v and
icdar2013v are required; dstext runs only when it has a record and det outputs for every held-out video.
Every dataset needs GT (data/processed), frames and det outputs (runs/<det_exp>/det/<ds>, made beforehand by
the detector env with --final) for all held-out videos; otherwise nothing runs.
Per dataset the config is copied and only exp_id = FINAL-<ds>, dataset, split and pipeline.lsc_exp = FINAL-<ds>
(LSC re-runs on the held-out videos) change. Any other difference, or an LSC / tracker checkpoint that would
resolve to another file under the new exp_id, is refused: hyperparameters come only from --config.
All run_pipeline stages run into runs/FINAL-<ds>/. An existing runs/FINAL-<ds>/config.yaml that differs from
the new config is refused unless --force (then every stage re-runs).
metrics.json meta.final: git commit + dirty flag, config path + sha256, seed, GPU name, split + its source.
Unless --skip_profile, tools/profile_efficiency.profile runs per dataset and runs/FINAL-efficiency.json is
written (params, peak VRAM, ms/frame per stage; det entry only if profile_efficiency --stages det --final was
run for that det_exp, dataset and split).

Usage: python scripts/final_eval.py --config runs/A-all_yolo/config.yaml --final [--datasets icdar2015v ...]
           [--force] [--skip_profile] [--frames 200] [--warmup 5] [--device cuda]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from logo.detector.infer_paddle import read_videos, resolve, write_run_files
from logo.lsc import run_lsc
from logo.tracking import run_track
from logo.utils.io import get_logger, load_json, set_seed
from logo.utils.paths import DATASETS, check_dataset, project_root

_REPO = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ra = _load(_REPO / "tools" / "run_ablation.py", "run_ablation_tool")
fb, rp = ra.fb, ra.rp
pe = fb.pe

REQUIRED = ("icdar2015v", "icdar2013v")
OPTIONAL = ("dstext",)
ALLOWED_CHANGES = ("exp_id", "dataset", "split", "pipeline.lsc_exp")
_HELD_OUT = re.compile(r"held-out\s+(" + "|".join(DATASETS) + r")\s*[:=]\s*(test|val)\b", re.I)


def exp_of(ds: str) -> str:
    return f"FINAL-{ds}"


def held_out_splits(path: Path) -> dict:
    """{ds: split} from docs/decisions.md table rows (decision cell "held-out <ds>: test|val"; last row wins)."""
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [re.sub(r"\\([_\[\]|*`])", r"\1", c).strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
        if len(cells) < 3:
            continue
        m = _HELD_OUT.search(cells[2])
        if m:
            out[m.group(1).lower()] = m.group(2).lower()
    return out


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def changed_keys(a: dict, b: dict) -> list[str]:
    fa, fb_ = flatten(a), flatten(b)
    return sorted(k for k in set(fa) | set(fb_) if fa.get(k, KeyError) != fb_.get(k, KeyError))


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(ra._canon(cfg).encode("utf-8")).hexdigest()


def frozen_ckpts(cfg: dict, root: Path, lsc_exp: str) -> dict:
    """Checkpoint files the config resolves to (LSC as run_lsc.pick_ckpt for lsc_exp, tracker as run_track)."""
    out = {"lsc": run_lsc.pick_ckpt({**cfg, "exp_id": lsc_exp}, root)}
    if (cfg.get("tracker") or {}).get("type", "simple") in ("embed", "lst"):
        out["tracker"] = run_track.embed_ckpt(cfg, root)
    return {k: None if v is None else Path(v).resolve() for k, v in out.items()}


def final_config(best: dict, ds: str, split: str, root: Path) -> dict:
    """Copy of `best` for FINAL-<ds>; raises ValueError if anything beyond ALLOWED_CHANGES would differ."""
    cfg = copy.deepcopy(best)
    cfg.update(exp_id=exp_of(ds), dataset=ds, split=split)
    cfg["pipeline"] = {**(cfg.get("pipeline") or {}), "lsc_exp": exp_of(ds)}
    bad = [k for k in changed_keys(best, cfg) if k not in ALLOWED_CHANGES]
    if bad:
        raise ValueError(f"final config would change {bad}")
    src = frozen_ckpts(best, root, rp.source_exps(best)[1])
    new = frozen_ckpts(cfg, root, cfg["exp_id"])
    moved = {k: (str(src[k]), str(new[k])) for k in src if src[k] != new[k]}
    if moved:
        raise ValueError(f"checkpoint(s) would resolve differently under {cfg['exp_id']}: {moved}; "
                         "set lsc.ckpt / tracker.ckpt explicitly in the config")
    for k, p in new.items():
        if p is not None and not p.is_file():
            raise FileNotFoundError(f"{k} checkpoint {p} not found")
    return cfg


def preflight(ds: str, split: Optional[str], det_exp: str, root: Path) -> tuple[list[str], list[str]]:
    """(held-out videos, problems) for ds on split."""
    if split is None:
        return [], [f"{ds}: no 'held-out {ds}: test|val' row in docs/decisions.md"]
    try:
        videos = read_videos(ds, split, root, final=True)
    except FileNotFoundError:
        return [], [f"{ds}: splits/{ds}_{split}.txt not found"]
    if not videos:
        return [], [f"{ds}: splits/{ds}_{split}.txt is empty"]
    probs = []
    for what, paths in (
            ("GT" + (" (test labels unavailable? record 'held-out " + ds + ": val')" if split == "test" else ""),
             [root / "data" / "processed" / ds / f"{v}.json" for v in videos]),
            (f"det outputs of {det_exp}", [rp.stage_dir(root, det_exp, "det", ds) / f"{v}.json" for v in videos])):
        miss = [p for p in paths if not p.is_file()]
        if miss:
            probs.append(f"{ds}/{split}: {len(miss)}/{len(videos)} missing {what}, e.g. {miss[0]}")
    miss = [v for v in videos if not (root / "data" / "frames" / ds / v).is_dir()]
    if miss:
        probs.append(f"{ds}/{split}: {len(miss)}/{len(videos)} frame dirs missing, e.g. {miss[0]}")
    return videos, probs


def git_info(root: Path) -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, timeout=10)
    try:
        head = git("rev-parse", "HEAD").stdout.strip() or "unknown"
        dirty = git("diff", "--quiet", "HEAD").returncode != 0 if head != "unknown" else None
    except (OSError, subprocess.SubprocessError):
        head, dirty = "unknown", None
    return {"git_commit": head, "git_dirty": dirty}


def gpu_name(device: Optional[str]) -> str:
    import torch
    if device == "cpu" or not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(torch.device(device or "cuda"))


def _close_files(log) -> None:
    for h in [h for h in log.handlers if hasattr(h, "baseFilename")]:
        h.close()
        log.removeHandler(h)


def final_eval(best: dict, config_path: str, root: Path, final: bool = False, datasets=REQUIRED + OPTIONAL,
               force: bool = False, skip_profile: bool = False, frames: int = 200, warmup: int = 5,
               device: Optional[str] = None, recognizer_factory: Optional[Callable] = None,
               profile_kw: Optional[dict] = None) -> dict:
    """Runs every available dataset; returns {"results": {ds: metrics}, "skipped": {ds: reason},
    "efficiency": FINAL-efficiency.json dict or None}."""
    if not final:
        raise PermissionError("final_eval needs --final")
    datasets = [check_dataset(d) for d in datasets]
    set_seed(best["seed"])
    splits = held_out_splits(root / "docs" / "decisions.md")
    det_exp = rp.source_exps(best)[0]
    plan, skipped, errors = {}, {}, []
    for ds in datasets:
        videos, probs = preflight(ds, splits.get(ds), det_exp, root)
        if probs and ds not in REQUIRED:
            skipped[ds] = "; ".join(probs)
            continue
        errors += probs
        if not probs:
            plan[ds] = (splits[ds], videos)
    if errors:
        raise FileNotFoundError("final_eval preflight failed:\n  " + "\n  ".join(errors))
    cfgs = {ds: final_config(best, ds, split, root) for ds, (split, _) in plan.items()}
    for ds, cfg in cfgs.items():
        saved = root / "runs" / cfg["exp_id"] / "config.yaml"
        if saved.is_file() and ra._canon(ra.saved_config(saved)) != ra._canon(cfg) and not force:
            raise ValueError(f"{saved} differs from the new final config: pass --force to re-run {cfg['exp_id']}")

    meta_common = {**git_info(root), "config": str(config_path), "config_hash": config_hash(best),
                   "seed": int(best["seed"]), "gpu": gpu_name(device), "date": date.today().isoformat()}
    results, eff = {}, None
    for ds, cfg in cfgs.items():
        split, videos = plan[ds]
        run_dir = root / "runs" / cfg["exp_id"]
        log = get_logger("final_eval", run_dir / "log.txt")
        log.info("=== %s: held-out split %s (%d videos), config %s sha256 %s", ds, split, len(videos),
                 config_path, meta_common["config_hash"][:12])
        if force and (run_dir / "config.yaml").is_file():
            (run_dir / "config.yaml").unlink()  # write_run_files keeps an existing config.yaml
        fb.run_row({"cfg": cfg}, root, force=force, final=True, device=device,
                   recognizer_factory=recognizer_factory)
        meta = {**meta_common, "dataset": ds, "split": split, "split_source": "docs/decisions.md",
                "num_videos": len(videos), "final_config_hash": config_hash(cfg),
                "tuned_on": f"{best.get('dataset')}/{best.get('split')}", "det_exp": det_exp}
        write_run_files(cfg, run_dir, meta, key="final")
        results[ds] = load_json(run_dir / "metrics.json")
        for k in ("det", "track", "spot"):
            if results[ds].get(k):
                log.info("FINAL %s %s: %s", ds, k, " ".join(f"{n}={v:.2f}" for n, v in results[ds][k].items()))
        if not skip_profile:
            res = pe.profile(cfg, root, frames, warmup, device=device, final=True, log=log, **(profile_kw or {}))
            eff = pe.update_final_efficiency(root, res, {k: meta_common[k] for k in
                                                          ("git_commit", "git_dirty", "config_hash", "seed", "gpu")})
        _close_files(log)
    for ds, why in skipped.items():
        print(f"skipped {ds}: {why}")
    return {"results": results, "skipped": skipped, "efficiency": eff}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True, help="best config (YAML or a runs/<exp>/config.yaml)")
    ap.add_argument("--final", action="store_true", help="required: this script reads held-out labels")
    ap.add_argument("--datasets", nargs="+", default=list(REQUIRED + OPTIONAL), choices=DATASETS)
    ap.add_argument("--force", action="store_true", help="re-run every stage of the FINAL runs")
    ap.add_argument("--skip_profile", action="store_true")
    ap.add_argument("--frames", type=int, default=200, help="profile frames per dataset")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default=None, help="cuda|cpu; default cuda if available")
    args = ap.parse_args(argv)
    if not args.final:
        ap.error("refusing to run without --final")
    root = project_root()
    path = resolve(args.config, root)
    if not path.is_file():
        ap.error(f"config {path} not found; pass a config YAML or a finished run's runs/<exp_id>/config.yaml "
                 "(e.g. runs/A-all_yolo/config.yaml)")
    best = copy.deepcopy(ra.saved_config(path))
    res = final_eval(best, str(args.config), root, True, args.datasets, args.force, args.skip_profile,
                     args.frames, args.warmup, args.device)
    for ds, m in res["results"].items():
        print(f"{exp_of(ds)}: " + " | ".join(f"{k} " + " ".join(f"{n}={v:.2f}" for n, v in m[k].items())
                                             for k in ("det", "track", "spot") if m.get(k)))
    if res["efficiency"]:
        print(f"efficiency: {pe.final_efficiency_path(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
