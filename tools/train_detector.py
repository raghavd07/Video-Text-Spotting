"""Fine-tunes PP-YOLOE-R on data/det_dota by running PaddleDetection tools/train.py in-process (logo-paddle).

Adds what train.py lacks: seed from the logo config, AMP on GPU (--amp), auto-resume from the newest epoch
checkpoint (<save_dir>/<k>.pdparams + <k>.pdopt; train.py writes one per epoch, the last as model_final),
and runs/<exp_id>/{config.yaml,log.txt,metrics.json}. All stdout/stderr (ppdet logs too) go to log.txt.
--smoke uses <ppdet config stem>_smoke.yml (1 epoch, 200 images) and runs/<exp_id>_smoke/.

Usage: python tools/train_detector.py --config configs/det_ppyoloe_r_text_s.yaml [--smoke]
           [--resume auto|none|<ckpt>] [--eval] [--device gpu|cpu] [--force]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import runpy
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

from logo.utils.io import get_logger, load_yaml, set_seed
from logo.utils.paths import project_root

TRAIN_PY = Path("third_party/PaddleDetection/tools/train.py")
EXT_OP_HELP = ("ext_op (rotated IoU op, needed by RotatedTaskAlignedAssigner) is not installed in this env; "
               "build third_party/PaddleDetection/ppdet/ext_op (docs/env_setup.md section 2.4)")


def smoke_config(ppdet_cfg: Path) -> Path:
    return ppdet_cfg.with_name(f"{ppdet_cfg.stem}_smoke{ppdet_cfg.suffix}")


def is_finished(save_dir: Path) -> bool:
    return (save_dir / "model_final.pdparams").is_file()


def latest_checkpoint(save_dir: Path) -> Optional[Path]:
    """Newest '<epoch>' checkpoint (extensionless path, as train.py -r expects) with params and optimizer."""
    if not save_dir.is_dir():
        return None
    epochs = [int(p.stem) for p in save_dir.glob("*.pdparams")
              if p.stem.isdigit() and p.with_suffix(".pdopt").is_file()]
    return save_dir / str(max(epochs)) if epochs else None


def resolve_resume(resume: str, save_dir: Path) -> Optional[Path]:
    if resume == "none":
        return None
    if resume == "auto":
        return latest_checkpoint(save_dir)
    p = Path(resume)
    p = p.with_suffix("") if p.suffix == ".pdparams" else p
    if not p.with_suffix(".pdparams").is_file():
        raise SystemExit(f"resume checkpoint not found: {p}.pdparams")
    return p


def build_argv(train_py: Path, ppdet_cfg: Path, save_dir: Path, resume: Optional[Path], device: str,
               eval_: bool) -> list[str]:
    argv = [str(train_py), "-c", ppdet_cfg.as_posix()]
    if device == "gpu":
        argv.append("--amp")  # paddle AMP O1; cfg.amp in the yml would be overwritten by merge_args
    if resume is not None:
        argv += ["-r", str(resume)]
    if eval_:
        argv.append("--eval")
    argv += ["-o", f"save_dir={save_dir.as_posix()}", f"use_gpu={'true' if device == 'gpu' else 'false'}"]
    return argv


def check_ext_op() -> None:
    if importlib.util.find_spec("ext_op") is None:
        raise SystemExit(EXT_OP_HELP)


def seed_all(seed: int) -> None:
    set_seed(seed)
    try:
        import paddle
    except ImportError:
        return
    paddle.seed(int(seed))


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()

    def isatty(self):
        return False


def rel(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def write_run_files(cfg: dict, run_dir: Path, info: dict) -> None:
    from logo.detector.infer_paddle import write_run_files as _write  # numpy + logo only
    _write(cfg, run_dir, info, key="det_train")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="200 images, 1 epoch -> runs/<exp_id>_smoke")
    ap.add_argument("--resume", default="auto", help="auto (newest epoch ckpt) | none | <ckpt path>")
    ap.add_argument("--eval", action="store_true", help="evaluate on det_dota/val.json after each epoch")
    ap.add_argument("--device", choices=("gpu", "cpu"), default=None)
    ap.add_argument("--force", action="store_true", help="train even if model_final exists")
    ap.add_argument("--root", type=Path, default=None, help="project root (default: auto)")
    args = ap.parse_args(argv)

    root = (args.root or project_root()).resolve()
    os.chdir(root)  # ppdet yml paths are relative to the project root
    cfg = copy.deepcopy(load_yaml(args.config if args.config.is_absolute() else root / args.config))
    det = cfg["detector"]
    ppdet_cfg = Path(det["config"])
    if args.smoke:
        cfg["exp_id"] = f"{cfg['exp_id']}_smoke"
        ppdet_cfg = smoke_config(ppdet_cfg)
    device = args.device or det.get("device", "gpu")
    run_dir = root / "runs" / cfg["exp_id"]
    save_dir = run_dir / "ckpt"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_f = open(run_dir / "log.txt", "a", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(old_out, log_f), Tee(old_err, log_f)
    old_argv = sys.argv
    log = get_logger(f"train_detector.{cfg['exp_id']}")  # console handler -> the Tee -> log.txt
    try:
        if is_finished(save_dir) and not args.force:
            log.info("%s exists; nothing to do (use --force to retrain)", save_dir / "model_final.pdparams")
            return 0
        check_ext_op()
        resume = resolve_resume(args.resume, save_dir)
        seed_all(cfg["seed"])
        shutil.copyfile(ppdet_cfg, run_dir / "ppdet_config.yml")
        train_argv = build_argv(root / TRAIN_PY, ppdet_cfg, save_dir, resume, device, args.eval)
        log.info("exp=%s seed=%s device=%s resume=%s", cfg["exp_id"], cfg["seed"], device, resume)
        log.info("train.py argv: %s", " ".join(train_argv[1:]))
        t0 = time.perf_counter()
        sys.argv = train_argv
        runpy.run_path(str(root / TRAIN_PY), run_name="__main__")
        info = {
            "ppdet_config": ppdet_cfg.as_posix(), "save_dir": rel(save_dir, root),
            "resumed_from": rel(resume, root) if resume else None,
            "final_weights": rel(save_dir / "model_final.pdparams", root) if is_finished(save_dir)
            else None,
            "seconds_this_session": round(time.perf_counter() - t0, 1),
            "seed": cfg["seed"], "amp": device == "gpu", "device": device, "smoke": args.smoke,
        }
        write_run_files(cfg, run_dir, info)
        log.info("done: %s", info)
    finally:
        for h in list(log.handlers):  # bound to this call's Tee; a later call adds fresh ones
            log.removeHandler(h)
        sys.argv = old_argv
        sys.stdout, sys.stderr = old_out, old_err
        log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
