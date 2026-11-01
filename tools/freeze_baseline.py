"""P23: freeze the val baseline (ablation rows, efficiency, reproducibility) into docs/baseline_report.md.

Plan file (configs/freeze_baseline.yaml): exp_id, dataset, split, seed, optional det_exp / exp_suffix,
rows [{label, config, optional lsc_exp}] (see load_rows), metrics
(<section>.<key> paths into metrics.json), paper_table_v {label: {metric: value|null}}, repro {row, tol},
profile {frames, warmup}.
  1) every row through scripts/run_pipeline.run_pipeline (all stages, plan dataset/split; outputs reused
     unless --force). Rows whose exp_id is another row's lsc_exp run first.
  2) tools/profile_efficiency.profile per row (det entry: run that tool with --stages det in logo-paddle).
  3) repro.row re-run with force=True; every det/track/spot value must match the step-1 metrics.json
     within repro.tol (percent points), else exit code 1 (the report is still written).
  4) docs/baseline_report.md: the block between FREEZE_BEGIN/FREEZE_END is replaced (appended if absent),
     the rest of the file is kept. Deviations = docs/decisions.md rows with a date and impact != none.
  5) prints the git commands that tag v1.0-base (nothing is run).
Writes runs/<plan exp_id>/{config.yaml,log.txt,metrics.json meta.freeze_<ds>_<split>}.

Usage: python tools/freeze_baseline.py [--plan configs/freeze_baseline.yaml] [--force] [--skip_profile]
           [--device cuda] [--out docs/baseline_report.md] [--final]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from logo.detector.infer_paddle import resolve, write_run_files
from logo.utils.io import get_logger, load_json, load_yaml, set_seed
from logo.utils.paths import check_dataset, check_split, project_root

_REPO = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rp = _load(_REPO / "scripts" / "run_pipeline.py", "run_pipeline_tool")
pe = _load(_REPO / "tools" / "profile_efficiency.py", "profile_efficiency_tool")

FREEZE_BEGIN, FREEZE_END = "<!-- freeze_baseline:begin -->", "<!-- freeze_baseline:end -->"
SECTIONS = ("det", "track", "spot")
TAG = "v1.0-base"


def lookup(metrics: dict, path: str):
    v = metrics
    for k in path.split("."):
        if not isinstance(v, dict) or k not in v:
            return None
        v = v[k]
    return v


def fmt(v, nd: int = 2) -> str:
    return "-" if v is None else f"{v:.{nd}f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def load_rows(plan: dict, root: Path) -> list[dict]:
    """[{label, config path, cfg}] with dataset/split forced to the plan's.

    Optional plan.det_exp replaces every row's pipeline.det_exp; plan.exp_suffix is appended to every row
    exp_id and to lsc_exp values naming a row (earlier runs stay untouched); row.lsc_exp overrides the
    config's pipeline.lsc_exp (a row exp_id before the suffix). With det_exp set, an lsc_exp outside the
    rows is refused: its lsc outputs were made from other detections.
    """
    suffix, det = plan.get("exp_suffix") or "", plan.get("det_exp")
    cfgs = [copy.deepcopy(load_yaml(resolve(r["config"], root))) for r in plan["rows"]]
    ids = {c["exp_id"] for c in cfgs}
    out = []
    for r, cfg in zip(plan["rows"], cfgs):
        cfg.update(dataset=plan["dataset"], split=plan["split"])
        pc = cfg["pipeline"] = dict(cfg.get("pipeline") or {})
        if det:
            pc["det_exp"] = det
        lsc = r.get("lsc_exp") or pc.get("lsc_exp")
        if lsc in ids:
            pc["lsc_exp"] = lsc + suffix
        elif lsc and det:
            raise ValueError(f"row {r['label']!r}: lsc_exp {lsc!r} is not a plan row but det_exp is overridden; "
                             "set the row's lsc_exp to a row exp_id")
        cfg["exp_id"] += suffix
        out.append({"label": r["label"], "config": str(r["config"]), "cfg": cfg})
    return out


def run_order(rows: list[dict]) -> list[int]:
    """Row indices; rows providing another row's lsc_exp first, otherwise plan order."""
    providers = {rp.source_exps(r["cfg"])[1] for r in rows if rp.source_exps(r["cfg"])[1] != r["cfg"]["exp_id"]}
    return sorted(range(len(rows)), key=lambda i: rows[i]["cfg"]["exp_id"] not in providers)


def run_row(row: dict, root: Path, force: bool, final: bool, device: Optional[str],
            recognizer_factory: Optional[Callable] = None) -> dict:
    """run_pipeline on all stages; returns the row's metrics.json."""
    cfg = row["cfg"]
    log = get_logger("run_pipeline", root / "runs" / cfg["exp_id"] / "log.txt")
    rp.run_pipeline(cfg, list(rp.STAGES), root, log, force=force, final=final, device=device,
                    recognizer_factory=recognizer_factory)
    for h in [h for h in log.handlers if hasattr(h, "baseFilename")]:  # next row logs to its own file
        h.close()
        log.removeHandler(h)
    return load_json(root / "runs" / cfg["exp_id"] / "metrics.json")


def metric_diffs(old: dict, new: dict) -> dict:
    """{section.key: |old - new|} over det/track/spot keys present in either."""
    out = {}
    for s in SECTIONS:
        for k in sorted(set(old.get(s) or {}) | set(new.get(s) or {})):
            a, b = (old.get(s) or {}).get(k), (new.get(s) or {}).get(k)
            out[f"{s}.{k}"] = abs(float(a) - float(b)) if a is not None and b is not None else float("inf")
    return out


def read_deviations(path: Path) -> list[str]:
    """docs/decisions.md table rows with a date and an impact other than none -> '<date> <task>: ...'."""
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [re.sub(r"\\([_\[\]|*`])", r"\1", c).strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
        if len(cells) < 5 or not cells[0] or cells[0].lower() == "date" or set("".join(cells)) <= set("-: "):
            continue
        date_, task, decision, why, impact = cells[:5]
        if impact.lower() in ("", "none"):
            continue
        out.append(f"{date_} {task}: {decision} (why: {why}; impact: {impact})")
    return out


def git_head(root: Path) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                           timeout=10)
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def paper_cell(plan: dict, label: str) -> str:
    vals = (plan.get("paper_table_v") or {}).get(label) or {}
    cells = [fmt(vals.get(m)) for m in plan["metrics"]]
    return "n/r" if all(c == "-" for c in cells) else " / ".join(cells)


def to_markdown(plan: dict, rows: list[dict], results: dict, effs: dict, repro: dict, deviations: list[str],
                commit: str) -> str:
    ms = plan["metrics"]
    det_exp = {rp.source_exps(r["cfg"])[0] for r in rows}
    lines = [FREEZE_BEGIN, f"## P23: frozen baseline ({TAG})", "",
             f"{date.today().isoformat()} · commit `{commit}` · `{plan['dataset']}` split `{plan['split']}` · "
             f"det_exp {', '.join(f'`{d}`' for d in sorted(det_exp))} · IoU "
             f"{(rows[0]['cfg'].get('eval') or {}).get('iou', 0.5)} · all values in percent", "",
             "### Reproduction", "",
             "| Row | exp_id | " + " | ".join(ms) + " | paper reported (Table V) |",
             "|---|---|" + "---:|" * len(ms) + "---|"]
    for r in rows:
        m = results[r["cfg"]["exp_id"]]
        lines.append(f"| {r['label']} | `{r['cfg']['exp_id']}` | " + " | ".join(fmt(lookup(m, k)) for k in ms)
                     + f" | {paper_cell(plan, r['label'])} |")
    lines += ["", f"Paper column: {' / '.join(ms)} from arXiv 2405.19194 Table V (n/r: not reported / not "
                  "filled in configs/freeze_baseline.yaml).", ""]
    if effs:
        any_e = next(iter(effs.values()))
        lines += ["### Efficiency", "",
                  f"Batch 1, {any_e['frames']} frames ({', '.join(any_e['videos'])}), {any_e.get('device', '?')}; "
                  "ms/frame per stage, e2e FPS = 1000 / sum of stage ms.", "",
                  "| Row | " + " | ".join(f"{s} ms" for s in pe.STAGES)
                  + " | total ms/frame | FPS | params (M) | peak GPU MB |",
                  "|---|" + "---:|" * (len(pe.STAGES) + 4)]
        for r in rows:
            e = effs.get(r["cfg"]["exp_id"])
            if e is None:
                continue
            st, ee = e["stages"], e["e2e"]
            lines.append(f"| {r['label']} | " + " | ".join(fmt((st.get(s) or {}).get("ms_per_frame"))
                                                         for s in pe.STAGES)
                         + f" | {fmt(ee['total_ms_per_frame'])} | {fmt(ee['fps'])}{'' if ee['det_included'] else '*'}"
                           f" | {ee['params'] / 1e6:.2f} | {fmt(ee['max_peak_mem_mb'], 1)} |")
        if not all(e["e2e"]["det_included"] for e in effs.values()):
            lines.append("")
            lines.append("\\* det not profiled (logo-paddle env): FPS excludes the detector.")
        lines += ["", "| Row | stage | params | peak GPU MB | ms/frame |", "|---|---|---:|---:|---:|"]
        for r in rows:
            e = effs.get(r["cfg"]["exp_id"])
            for s, v in (e or {}).get("stages", {}).items():
                lines.append(f"| {r['label']} | {s} | {v['params']:,} | {fmt(v.get('peak_mem_mb'), 1)} | "
                             f"{fmt(v.get('ms_per_frame'), 3)} |")
        lines.append("")
    if repro:
        worst = max(repro["diffs"].items(), key=lambda kv: kv[1]) if repro["diffs"] else ("-", 0.0)
        lines += ["### Reproducibility", "",
                  f"`{repro['exp_id']}` ({repro['label']}) re-run with --force: max |Δ| = {fmt(worst[1], 3)} "
                  f"({worst[0]}), tolerance {repro['tol']} → **{'PASS' if repro['ok'] else 'FAIL'}**.", ""]
    lines += ["### Deviations from the paper (docs/decisions.md)", ""]
    lines += [f"- {d}" for d in deviations] or ["- none recorded"]
    lines.append(FREEZE_END)
    return "\n".join(lines) + "\n"


def write_report(path: Path, block: str) -> None:
    """Replaces the freeze block in `path` (or appends it); other content is kept."""
    old = path.read_text(encoding="utf-8") if path.is_file() else ""
    i, j = old.find(FREEZE_BEGIN), old.find(FREEZE_END)
    if i >= 0 and j > i:
        new = old[:i] + block.rstrip("\n") + old[j + len(FREEZE_END):]
    else:
        new = old.rstrip("\n") + ("\n\n" if old.strip() else "") + block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new if new.endswith("\n") else new + "\n", encoding="utf-8")


def git_commands(report: Path, root: Path, summary: str) -> list[str]:
    rel = report.relative_to(root).as_posix() if report.is_relative_to(root) else str(report)
    return ["git status --short",
            f"git add configs docs scripts src tests tools splits requirements.txt {rel}",
            f'git commit -m "P23: freeze baseline {TAG}"',
            f'git tag -a {TAG} -m "Frozen val baseline: {summary}"',
            "git push origin main",
            f"git push origin {TAG}"]


def freeze(plan: dict, root: Path, log, force: bool = False, final: bool = False, skip_profile: bool = False,
           device: Optional[str] = None, out: Optional[Path] = None, recognizer_factory: Optional[Callable] = None,
           profile_kw: Optional[dict] = None) -> dict:
    ds, split = check_dataset(plan["dataset"]), check_split(plan["split"], final)
    set_seed(plan["seed"])
    rows = load_rows(plan, root)
    labels = [r["label"] for r in rows]
    rep_cfg = dict({"row": labels[min(1, len(labels) - 1)], "tol": 0.3}, **(plan.get("repro") or {}))
    if rep_cfg["row"] not in labels:
        raise ValueError(f"repro.row {rep_cfg['row']!r} is not one of {labels}")

    results = {}
    for i in run_order(rows):
        r = rows[i]
        log.info("=== row %s (%s, %s)", r["label"], r["cfg"]["exp_id"], r["config"])
        results[r["cfg"]["exp_id"]] = run_row(r, root, force, final, device, recognizer_factory)

    effs = {}
    if not skip_profile:
        pc = dict({"frames": 200, "warmup": 5}, **(plan.get("profile") or {}))
        for r in rows:
            log.info("=== profile %s", r["cfg"]["exp_id"])
            effs[r["cfg"]["exp_id"]] = pe.profile(r["cfg"], root, int(pc["frames"]), int(pc["warmup"]),
                                                  device=device, final=final, log=log, **(profile_kw or {}))

    row = rows[labels.index(rep_cfg["row"])]
    exp = row["cfg"]["exp_id"]
    stored = copy.deepcopy(results[exp])
    log.info("=== reproducibility: re-running %s with force", exp)
    rerun = run_row(row, root, True, final, device, recognizer_factory)
    diffs = metric_diffs(stored, rerun)
    tol = float(rep_cfg["tol"])
    repro = {"label": row["label"], "exp_id": exp, "tol": tol, "diffs": diffs,
             "ok": bool(diffs) and all(d <= tol for d in diffs.values())}
    results[exp] = rerun
    for k, d in diffs.items():
        (log.info if d <= tol else log.error)("repro %s: |delta| = %.4f", k, d)

    report = out or root / "docs" / "baseline_report.md"
    block = to_markdown(plan, rows, results, effs, repro, read_deviations(root / "docs" / "decisions.md"),
                        git_head(root))
    write_report(report, block)
    log.info("report: %s", report)
    summary = ", ".join(f"{r['label']} " + "/".join(fmt(lookup(results[r['cfg']['exp_id']], k)) for k in
                                                     plan["metrics"][:2]) for r in rows)
    meta = {"rows": [{"label": r["label"], "exp_id": r["cfg"]["exp_id"], "config": r["config"],
                      **{k: lookup(results[r["cfg"]["exp_id"]], k) for k in plan["metrics"]}} for r in rows],
            "efficiency": {k: v["e2e"] for k, v in effs.items()}, "repro": repro, "report": str(report)}
    write_run_files(plan, root / "runs" / plan["exp_id"], meta, key=f"freeze_{ds}_{split}")
    return {"repro": repro, "report": report, "git": git_commands(report, root, summary)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", type=Path, default=Path("configs/freeze_baseline.yaml"))
    ap.add_argument("--force", action="store_true", help="re-run every row's stages in step 1")
    ap.add_argument("--skip_profile", action="store_true")
    ap.add_argument("--device", default=None, help="lsc / tracker device; default cuda if available")
    ap.add_argument("--out", type=Path, default=None, help="default: docs/baseline_report.md")
    ap.add_argument("--final", action="store_true", help="allow split=test (rule 5)")
    args = ap.parse_args(argv)
    root = project_root()
    plan = copy.deepcopy(load_yaml(resolve(args.plan, root)))
    if plan.get("split") == "test" and not args.final:
        ap.error("plan split is test: pass --final")
    log = get_logger("freeze_baseline", root / "runs" / plan["exp_id"] / "log.txt")
    res = freeze(plan, root, log, args.force, args.final, args.skip_profile, args.device,
                 resolve(args.out, root) if args.out else None)
    print(f"\nreport: {res['report']}\nreproducibility: {'PASS' if res['repro']['ok'] else 'FAIL'}\n"
          f"\n# tag {TAG}:")
    print("\n".join(res["git"]))
    return 0 if res["repro"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
