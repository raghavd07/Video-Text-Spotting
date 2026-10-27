"""Detection P/R/F before vs after LSC re-scoring, via tools/evaluate.py; writes docs/baseline_report.md.

Rows per dataset: det score > 0.3 (runs/<det_exp>/det), score_fused > 0.3 and > 0.5
(runs/<exp_id>/fuse). Each row is one tools/evaluate.py call writing runs/<exp_id>/metrics_det_<tag>_<ds>.json
(+ log.txt / config.yaml there); the table also goes to runs/<exp_id>/metrics.json meta.baseline_det_<split>.

Usage: python tools/baseline_report.py --config configs/E3-lsc.yaml --det_exp DET_EXP
           [--datasets icdar2015v] [--split val] [--final] [--out docs/baseline_report.md]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from datetime import date
from pathlib import Path

from logo.detector.infer_paddle import write_run_files
from logo.utils.io import get_logger, load_json, load_yaml
from logo.utils.paths import DATASETS, check_split, project_root

_spec = importlib.util.spec_from_file_location("evaluate_tool",
                                               Path(__file__).resolve().parent / "evaluate.py")
evaluate_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_tool)

# (tag, stage, score key, threshold, label)
ROWS = (("det_0.3", "det", "score", 0.3, "before: det score > 0.3"),
        ("fused_0.3", "fuse", "score_fused", 0.3, "after: fused score > 0.3"),
        ("fused_0.5", "fuse", "score_fused", 0.5, "after: fused score > 0.5"))


def evaluate_rows(cfg: dict, cfg_path: Path, ds: str, split: str, det_exp: str, root: Path,
                  final: bool) -> list[dict]:
    run_dir = root / "runs" / cfg["exp_id"]
    iou = float(cfg.get("eval", {}).get("iou", 0.5))
    out = []
    for tag, stage, key, thr, label in ROWS:
        exp = det_exp if stage == "det" else cfg["exp_id"]
        m_out = run_dir / f"metrics_det_{tag}_{ds}.json"
        argv = ["--gt_dir", str(root / "data" / "processed" / ds),
                "--pred_dir", str(root / "runs" / exp / stage / ds),
                "--videos", str(root / "splits" / f"{ds}_{split}.txt"), "--mode", "det",
                "--out", str(m_out), "--iou", str(iou), "--score_key", key, "--score_thr", str(thr),
                "--config", str(cfg_path)]
        if final:
            argv.append("--final")
        evaluate_tool.main(argv)
        m = load_json(m_out)
        out.append({"dataset": ds, "row": label, "tag": tag, **m["det"], **m["meta"]["det"]["counts"]})
    return out


def to_markdown(rows: list[dict], cfg: dict, det_exp: str, split: str) -> str:
    lines = ["# Baseline: detection before vs after LSC re-scoring", "",
             f"{date.today().isoformat()} · exp `{cfg['exp_id']}` · det_exp `{det_exp}` · split `{split}` · "
             f"fusion `{cfg['fusion']['mode']}` w={cfg['fusion']['w']} · lsc.det_thr {cfg['lsc']['det_thr']} · "
             f"IoU {cfg.get('eval', {}).get('iou', 0.5)}", "",
             "| Dataset | Boxes kept | P | R | F | TP | FP | FN |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['row']} | {r['precision']:.2f} | {r['recall']:.2f} | "
                     f"{r['f']:.2f} | {r['tp']} | {r['fp']} | {r['fn']} |")
    w = float(cfg["fusion"]["w"])
    if cfg["fusion"]["mode"] == "hard":
        lines += ["", f"Hard fusion: fused = {1 - w:g}*det + {w:g}*lang_hard on boxes with det > "
                      f"{cfg['lsc']['det_thr']}, so a threshold t keeps lang_hard=1 boxes with det > "
                      f"(t-{w:g})/{1 - w:g} and lang_hard=0 boxes with det > t/{1 - w:g}."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--det_exp", default=None, help="exp id holding det/ outputs; default config exp_id")
    ap.add_argument("--datasets", nargs="+", default=None, choices=DATASETS, help="default: config dataset")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--final", action="store_true", help="allow the test split (rule 5)")
    ap.add_argument("--out", type=Path, default=None, help="default: docs/baseline_report.md")
    args = ap.parse_args(argv)

    root = project_root()
    split = check_split(args.split, args.final)
    cfg = copy.deepcopy(load_yaml(args.config))
    det_exp = args.det_exp or cfg["exp_id"]
    run_dir = root / "runs" / cfg["exp_id"]
    log = get_logger("baseline_report", run_dir / "log.txt")
    rows = []
    for ds in args.datasets or [cfg["dataset"]]:
        rows += evaluate_rows(cfg, args.config, ds, split, det_exp, root, args.final)
    md = to_markdown(rows, cfg, det_exp, split)
    out = args.out or root / "docs" / "baseline_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    write_run_files(cfg, run_dir, {"det_exp": det_exp, "rows": rows}, key=f"baseline_det_{split}")
    log.info("wrote %s\n%s", out, md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
