"""P27: run every ablation row (configs/ablation.yaml) through scripts/run_pipeline and tabulate metrics.json.

Plan file: exp_id, dataset, split, seed, optional det_exp / exp_suffix (as tools/freeze_baseline.load_rows),
rows [{label, config, optional lsc_exp, exp_id, set {dotted.key: value}}], metrics (<section>.<key> paths).
A row is finished when runs/<exp_id>/metrics.json has meta.eval_<ds>_<split> and runs/<exp_id>/config.yaml
equals the row config; finished rows are skipped unless --force. A finished row whose saved config differs
is re-run with force (config.yaml is rewritten first). Rows providing another row's lsc_exp run first.
Prints a Markdown table; writes runs/<plan exp_id>/{config.yaml,log.txt,metrics.json meta.ablation_<ds>_<split>}
and, with --out, the table to that file. split=test needs --final.

Usage: python tools/run_ablation.py [--plan configs/ablation.yaml] [--only LABEL ...] [--force] [--final]
           [--device cuda] [--out docs/ablation.md]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from logo.detector.infer_paddle import write_run_files
from logo.utils.io import get_logger, load_json, load_yaml, set_seed
from logo.utils.paths import check_dataset, check_split, project_root

_REPO = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fb = _load(_REPO / "tools" / "freeze_baseline.py", "freeze_baseline_tool")
rp = fb.rp


def set_key(cfg: dict, dotted: str, value) -> None:
    *parents, last = dotted.split(".")
    d = cfg
    for k in parents:
        d[k] = dict(d.get(k) or {})
        d = d[k]
    d[last] = copy.deepcopy(value)


def load_rows(plan: dict, root: Path) -> list[dict]:
    """freeze_baseline.load_rows plus per-row exp_id (+ plan exp_suffix) and `set` overrides; ids unique."""
    suffix = plan.get("exp_suffix") or ""
    rows = fb.load_rows(plan, root)
    for spec, row in zip(plan["rows"], rows):
        if spec.get("exp_id"):
            row["cfg"]["exp_id"] = spec["exp_id"] + suffix
        for k, v in (spec.get("set") or {}).items():
            if k.split(".")[0] in ("exp_id", "dataset", "split"):
                raise ValueError(f"row {spec['label']!r}: set may not change {k!r}")
            set_key(row["cfg"], k, v)
    ids = [r["cfg"]["exp_id"] for r in rows]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"duplicate row exp_id(s) {dup}: give rows sharing a config their own exp_id")
    return rows


def _canon(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, ensure_ascii=False)


def saved_config(p: Path) -> dict:
    """config.yaml as written by write_run_files (JSON; YAML 1.1 would read 5e-05 as a string), else YAML."""
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return load_yaml(p)


def row_state(row: dict, root: Path) -> str:
    """'finished' | 'stale' (evaluated with another config) | 'todo'."""
    cfg = row["cfg"]
    if not rp.eval_done(root, cfg, cfg["dataset"], cfg["split"]):
        return "todo"
    saved = root / "runs" / cfg["exp_id"] / "config.yaml"
    return "finished" if saved.is_file() and _canon(saved_config(saved)) == _canon(cfg) else "stale"


def run_all(plan: dict, rows: list[dict], root: Path, log, force: bool = False, final: bool = False,
            device: Optional[str] = None, recognizer_factory: Optional[Callable] = None) -> list[dict]:
    """Runs unfinished rows (all with force); returns [{label, exp_id, status, seconds, metrics}] in plan order."""
    out = {}
    for i in fb.run_order(rows):
        row, t0 = rows[i], time.perf_counter()
        exp = row["cfg"]["exp_id"]
        state = row_state(row, root)
        if state == "finished" and not force:
            status = "skipped"
        else:
            if state == "stale":
                log.warning("%s: metrics.json was made with another config; re-running with force", exp)
            cfg_p = root / "runs" / exp / "config.yaml"
            if cfg_p.is_file():
                cfg_p.unlink()  # write_run_files keeps an existing config.yaml
            status = "forced" if (force or state == "stale") else "ran"
            fb.run_row(row, root, force=force or state == "stale", final=final, device=device,
                       recognizer_factory=recognizer_factory)
        dt = round(time.perf_counter() - t0, 3)
        out[i] = {"label": row["label"], "exp_id": exp, "status": status, "seconds": dt,
                  "metrics": load_json(root / "runs" / exp / "metrics.json")}
        log.info("row %-10s %-14s %-8s %.1f s", row["label"], exp, status, dt)
    return [out[i] for i in range(len(rows))]


def to_markdown(plan: dict, results: list[dict]) -> str:
    ms = plan["metrics"]
    lines = [f"Ablation on `{plan['dataset']}` split `{plan['split']}` (values in percent)", "",
             "| Row | exp_id | " + " | ".join(ms) + " |", "|---|---|" + "---:|" * len(ms)]
    for r in results:
        lines.append(f"| {r['label']} | `{r['exp_id']}` | "
                     + " | ".join(fb.fmt(fb.lookup(r["metrics"], k)) for k in ms) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", type=Path, default=Path("configs/ablation.yaml"))
    ap.add_argument("--only", nargs="+", default=None, help="row labels to run (others are not touched)")
    ap.add_argument("--force", action="store_true", help="re-run finished rows")
    ap.add_argument("--final", action="store_true", help="allow split=test (rule 5)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=None, help="also write the Markdown table here")
    args = ap.parse_args(argv)

    root = project_root()
    plan = copy.deepcopy(load_yaml(fb.resolve(args.plan, root)))
    ds = check_dataset(plan["dataset"])
    try:
        split = check_split(plan["split"], args.final)
    except PermissionError as e:
        ap.error(str(e))
    rows = load_rows(plan, root)
    if args.only:
        unknown = set(args.only) - {r["label"] for r in rows}
        if unknown:
            ap.error(f"unknown row label(s) {sorted(unknown)}")
        rows = [r for r in rows if r["label"] in args.only]
    run_dir = root / "runs" / plan["exp_id"]
    set_seed(plan["seed"])
    log = get_logger("run_ablation", run_dir / "log.txt")
    log.info("ablation ds=%s split=%s rows=%s force=%s", ds, split, [r["label"] for r in rows], args.force)
    results = run_all(plan, rows, root, log, force=args.force, final=args.final, device=args.device)
    md = to_markdown(plan, results)
    print(md, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
    write_run_files(plan, run_dir, {"rows": [{k: r[k] for k in ("label", "exp_id", "status", "seconds")}
                                             | {m: fb.lookup(r["metrics"], m) for m in plan["metrics"]}
                                             for r in results]},
                    key=f"ablation_{ds}_{split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
