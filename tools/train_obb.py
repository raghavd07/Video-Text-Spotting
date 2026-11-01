"""Plan B: fine-tunes Ultralytics YOLO-OBB on the prepared OBB dataset (logo-torch env, AMP on GPU).

Data: <detector.obb_data.root>/<ds>/<ds>.yaml from logo.detector.prepare_obb_data (run that first).
Init weights: detector.train.init. Ultralytics writes runs/<exp_id>/train/weights/{last,best}.pt; set
detector.weights to the best.pt for logo.detector.infer_obb. An unfinished run (last.pt without best.pt
being final) is resumed with YOLO(last.pt).train(resume=True) unless --resume none.
Writes runs/<exp_id>/{config.yaml,log.txt,metrics.json meta.det_train}. --smoke: 1 epoch on
detector.train.smoke_fraction of the train images -> runs/<exp_id>_smoke/.

Usage: python tools/train_obb.py --config configs/det_yolo11s_obb_text.yaml [--smoke] [--resume auto|none]
           [--device gpu|cpu] [--force]
"""
from __future__ import annotations

import argparse
import copy
import shutil
import sys
import time
from pathlib import Path

from logo.detector.infer_paddle import resolve, write_run_files
from logo.utils.io import get_logger, load_yaml, set_seed
from logo.utils.paths import check_dataset, project_root

TRAIN_DEFAULTS = dict(init="models/yolo_obb/yolo11s-obb.pt", epochs=50, batch=4, imgsz=1024, workers=2,
                      patience=15, close_mosaic=5, fliplr=0.5, degrees=0.0, smoke_fraction=0.1,
                      val_conf=0.05)  # validator conf (ultralytics default 0.001 OOMs rotated NMS on 6 GB)


def train_settings(cfg: dict) -> dict:
    return dict(TRAIN_DEFAULTS, **((cfg.get("detector") or {}).get("train") or {}))


def data_yaml(cfg: dict, root: Path) -> Path:
    ds = check_dataset(cfg["dataset"])
    od = (cfg.get("detector") or {}).get("obb_data") or {}
    p = resolve(od.get("root", "data/obb"), root) / ds / f"{ds}.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing: run python -m logo.detector.prepare_obb_data --config ...")
    return p


def train_kwargs(cfg: dict, root: Path, run_dir: Path, device: str, smoke: bool) -> dict:
    """Arguments for ultralytics YOLO.train (names as in ultralytics/cfg/default.yaml)."""
    t = train_settings(cfg)
    gpu = device == "gpu"
    return dict(data=data_yaml(cfg, root).as_posix(), epochs=1 if smoke else int(t["epochs"]),
                batch=int(t["batch"]), imgsz=int(t["imgsz"]), workers=int(t["workers"]),
                patience=int(t["patience"]), close_mosaic=0 if smoke else int(t["close_mosaic"]),
                fliplr=float(t["fliplr"]), degrees=float(t["degrees"]), conf=float(t["val_conf"]),
                fraction=float(t["smoke_fraction"]) if smoke else 1.0, device=0 if gpu else "cpu", amp=gpu,
                seed=int(cfg["seed"]), deterministic=True, project=run_dir.as_posix(), name="train",
                exist_ok=True, plots=False, verbose=True)


def main(argv=None, yolo_cls=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", choices=("auto", "none"), default="auto")
    ap.add_argument("--device", choices=("gpu", "cpu"), default=None)
    ap.add_argument("--force", action="store_true", help="retrain from init even if weights exist")
    args = ap.parse_args(argv)

    root = project_root()
    cfg = copy.deepcopy(load_yaml(resolve(args.config, root)))
    if args.smoke:
        cfg["exp_id"] = f"{cfg['exp_id']}_smoke"
    device = args.device or (cfg.get("detector") or {}).get("device", "gpu")
    run_dir = root / "runs" / cfg["exp_id"]
    wdir = run_dir / "train" / "weights"
    log = get_logger("train_obb", run_dir / "log.txt")
    if args.force and (run_dir / "train").exists():
        shutil.rmtree(run_dir / "train")
    if (run_dir / "metrics.json").is_file() and (wdir / "best.pt").is_file() and not args.force:
        from logo.utils.io import load_json
        if (load_json(run_dir / "metrics.json").get("meta") or {}).get("det_train", {}).get("finished"):
            log.info("%s finished; nothing to do (use --force to retrain)", wdir / "best.pt")
            return 0
    if yolo_cls is None:
        from ultralytics import YOLO as yolo_cls

    set_seed(cfg["seed"])
    kw = train_kwargs(cfg, root, run_dir, device, args.smoke)
    last = wdir / "last.pt"
    t0 = time.perf_counter()
    if last.is_file() and args.resume == "auto":
        log.info("resuming %s", last)
        yolo_cls(str(last), task="obb").train(resume=True)
    else:
        init = resolve(train_settings(cfg)["init"], root)
        log.info("exp=%s init=%s train kwargs=%s", cfg["exp_id"], init, kw)
        yolo_cls(str(init), task="obb").train(**kw)
    info = {"finished": (wdir / "best.pt").is_file(), "best": (wdir / "best.pt").as_posix(),
            "seconds": round(time.perf_counter() - t0, 1), "smoke": args.smoke, **kw}
    write_run_files(cfg, run_dir, info, key="det_train")
    log.info("done: %s", info)
    return 0 if info["finished"] else 1


if __name__ == "__main__":
    sys.exit(main())
