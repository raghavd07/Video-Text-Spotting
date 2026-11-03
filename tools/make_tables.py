"""P29: paper tables from runs/*/metrics.json -> paper/tables/<name>.tex (booktabs) + docs/tables/<name>.md.

Every number is read from a metrics.json (or, for the 'Reported' block only, configs/reported_numbers.yaml);
nothing is typed by hand. Tables (one file per discovered meta entry; <ds>_<split> from the meta key):
  ablation_<ds>_<split>            S1  meta.ablation_<ds>_<split>.rows (tools/run_ablation.py)
  comparison_<ds>                  --  top-level track/spot of the runs in <config>.comparison.ours (dataset and
                                       split from runs/<exp>/config.yaml) + 'Reported' block per dataset
  threshold_<exp>_<ds>_<split>     S2  meta.sweep_fusion_* (tools/sweep_fusion.py), hard vs soft at
                                       <config>.threshold.w (nearest swept w)
  voting_<exp>_<ds>_<split>        S3  meta.sweep_voting_* (tools/sweep_voting.py)
  efficiency_<ds>_<split>          S4  meta.efficiency_* (tools/profile_efficiency.py); detector-only entries
                                       are skipped; rows without the detector are marked with a dagger
  errors_<ds>_<split>              S5  meta.errors_* totals per run (tools/error_analysis.py)
  errors_bins_<exp>_<ds>_<split>   S5  category x text-height bins of one run
  ci                               --  meta.bootstrap_* (tools/bootstrap_ci.py)
Formatting: percent 2 decimals, ms 1, FPS 2, params (M) 1, VRAM (MB) 1, counts integer. Per column the best
value (after rounding) is bold; columns where all rows tie are not bolded. CI: the better of A/B and a
significant difference are bold. Entries of the test split are skipped unless --final (rule 5).
Writes runs/<exp_id>/{config.yaml,log.txt,metrics.json meta.tables}.

Usage: python tools/make_tables.py [--config configs/paper.yaml] [--final] [--tex_dir paper/tables]
           [--md_dir docs/tables]
"""
from __future__ import annotations

import argparse
import copy
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from logo.detector.infer_paddle import resolve, write_run_files
from logo.utils.io import get_logger, load_json, load_yaml, set_seed
from logo.utils.paths import DATASETS, SPLITS, project_root

DEC = {"pct": 2, "pct1": 1, "ms": 1, "fps": 2, "mega": 1, "mb": 1, "int": 0}
TRACK_KEYS = ("MOTA", "MOTP", "IDF1", "MT", "ML")
SPOT_KEYS = ("MOTA", "MOTP", "IDF1")
BETTER = {"MOTA": "max", "MOTP": "max", "IDF1": "max", "MT": "max", "ML": "min", "precision": "max",
          "recall": "max", "f": "max"}
STAGES = ("det", "lsc", "fuse", "vpmm", "track")
ERR_CATS = ("missed_gt", "false_positive", "wrong_text", "id_switch", "fragmentation")
ERR_LABELS = {"missed_gt": "Missed GT", "false_positive": "False pos.", "wrong_text": "Wrong text",
              "id_switch": "ID switch", "fragmentation": "Fragment."}
SIZE_BINS = ("<20", "20-50", ">50")
DAGGER = "†"


# ----------------------------------------------------------------------------------------------- table model
@dataclass
class Col:
    key: str
    head: str
    group: str = ""
    fmt: str = "pct"               # DEC key or "text"
    best: Optional[str] = None     # "max" | "min" | None
    head_tex: Optional[str] = None


@dataclass
class Table:
    name: str
    caption: str
    cols: list[Col]
    rows: list[dict]               # values by col key + "_label", optional "_block", "_nobest", "_bold", "_label_tex"
    label_head: str = "Method"
    sources: list[str] = field(default_factory=list)


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def fmt_value(v, kind: str) -> Optional[str]:
    """Display string, or None for a missing value."""
    if kind == "text":
        return None if v is None or v == "" else str(v)
    if not is_num(v):
        return None
    if kind == "mega":
        v = v / 1e6
    return f"{int(round(v))}" if kind == "int" else f"{v:.{DEC[kind]}f}"


def bold_cells(t: Table) -> set[tuple[int, str]]:
    """(row index, col key) pairs to print bold: per-column best after rounding, plus rows' forced _bold."""
    out = {(i, k) for i, r in enumerate(t.rows) for k in r.get("_bold", ())}
    for c in t.cols:
        if c.best is None:
            continue
        vals = {}
        for i, r in enumerate(t.rows):
            s = fmt_value(r.get(c.key), c.fmt)
            if s is not None and c.key not in r.get("_nobest", ()):
                vals[i] = float(s)
        if len(set(vals.values())) < 2:
            continue
        best = max(vals.values()) if c.best == "max" else min(vals.values())
        out |= {(i, c.key) for i, v in vals.items() if v == best}
    return out


_TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
            "<": r"\textless{}", ">": r"\textgreater{}", DAGGER: r"$^\dagger$"}
_NEG = re.compile(r"(?<![\w.])-(?=\d)")


def tex_escape(s: str) -> str:
    return "".join(_TEX_ESC.get(ch, ch) for ch in str(s))


def tex_cell(s: Optional[str], bold: bool) -> str:
    if s is None:
        return "--"
    s = _NEG.sub("$-$", tex_escape(s))
    return rf"\textbf{{{s}}}" if bold else s


def _groups(cols: list[Col]) -> list[tuple[str, int, int]]:
    """(group, first col index, last col index) runs of equal group names (0-based over data columns)."""
    out = []
    for i, c in enumerate(cols):
        if out and out[-1][0] == c.group:
            out[-1] = (c.group, out[-1][1], i)
        else:
            out.append((c.group, i, i))
    return out


def to_latex(t: Table) -> str:
    bold = bold_cells(t)
    n = len(t.cols) + 1
    spec = "l" + "".join("l" if c.fmt == "text" else "r" for c in t.cols)
    lines = ["% Generated by tools/make_tables.py from runs/*/metrics.json -- do not edit by hand.",
             r"% Requires \usepackage{booktabs}.",
             r"\begin{table}[t]", r"\centering", rf"\caption{{{tex_escape(t.caption)}}}",
             rf"\label{{tab:{t.name}}}", r"\small", rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    groups = _groups(t.cols)
    if any(g for g, _, _ in groups):
        cells, rules = [""], []
        for g, a, b in groups:
            span = b - a + 1
            cells.append(rf"\multicolumn{{{span}}}{{c}}{{{tex_escape(g)}}}" if span > 1 else tex_escape(g))
            if g:
                rules.append(rf"\cmidrule(lr){{{a + 2}-{b + 2}}}")
        lines.append(" & ".join(cells) + r" \\")
        lines.append(" ".join(rules))
    lines.append(" & ".join([tex_escape(t.label_head)] + [c.head_tex or tex_escape(c.head) for c in t.cols])
                 + r" \\")
    lines.append(r"\midrule")
    block = None
    for i, r in enumerate(t.rows):
        if r.get("_block") is not None and r.get("_block") != block:
            if block is not None:
                lines.append(r"\midrule")
            block = r["_block"]
            lines.append(rf"\multicolumn{{{n}}}{{l}}{{\textit{{{tex_escape(block)}}}}} \\")
        label = r.get("_label_tex") or tex_escape(r["_label"])
        cells = [label] + [tex_cell(fmt_value(r.get(c.key), c.fmt), (i, c.key) in bold) for c in t.cols]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def md_escape(s: str) -> str:
    return str(s).replace("|", r"\|")


def to_markdown(t: Table) -> str:
    bold = bold_cells(t)
    heads = [t.label_head] + [f"{c.group} {c.head}".strip() for c in t.cols]
    lines = ["<!-- Generated by tools/make_tables.py from runs/*/metrics.json -- do not edit by hand. -->",
             f"**{md_escape(t.caption)}**", "",
             "| " + " | ".join(md_escape(h) for h in heads) + " |",
             "|---|" + "".join("---|" if c.fmt == "text" else "---:|" for c in t.cols)]
    block = None
    for i, r in enumerate(t.rows):
        if r.get("_block") is not None and r.get("_block") != block:
            block = r["_block"]
            lines.append(f"| *{md_escape(block)}* |" + " |" * len(t.cols))
        cells = [md_escape(r["_label"])]
        for c in t.cols:
            s = fmt_value(r.get(c.key), c.fmt)
            s = "-" if s is None else md_escape(s)
            cells.append(f"**{s}**" if (i, c.key) in bold and s != "-" else s)
        lines.append("| " + " | ".join(cells) + " |")
    if t.sources:
        lines += ["", "Sources: " + ", ".join(f"`{s}`" for s in t.sources)]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------- discovery
def parse_key(key: str, prefix: str) -> Optional[tuple[str, str]]:
    """(ds, split) of a meta key '<prefix>_<ds>_<split>', else None."""
    if not key.startswith(prefix + "_"):
        return None
    rest = key[len(prefix) + 1:]
    if "_" not in rest:
        return None
    ds, split = rest.rsplit("_", 1)
    return (ds, split) if ds in DATASETS and split in SPLITS else None


def load_runs(runs_dir: Path) -> dict[str, dict]:
    """{exp_id: metrics} for every runs/<exp_id>/metrics.json, sorted by exp id."""
    out = {}
    for p in sorted(Path(runs_dir).glob("*/metrics.json")):
        m = load_json(p)
        if isinstance(m, dict):
            out[p.parent.name] = m
    return out


def iter_meta(runs: dict, prefix: str, final: bool, log=None) -> Iterator[tuple[str, str, str, dict]]:
    """(exp, ds, split, entry) for every meta.<prefix>_<ds>_<split>; test split only with final."""
    for exp, m in runs.items():
        for key, entry in (m.get("meta") or {}).items():
            ds_split = parse_key(key, prefix)
            if ds_split is None or not isinstance(entry, dict):
                continue
            if ds_split[1] == "test" and not final:
                if log:
                    log.info("skip %s meta.%s (test split needs --final)", exp, key)
                continue
            yield exp, ds_split[0], ds_split[1], entry


def pick_w(table: list[dict], w: float) -> float:
    ws = sorted({float(r["w"]) for r in table if is_num(r.get("w"))})
    if not ws:
        raise ValueError("sweep_fusion table has no w values")
    return min(ws, key=lambda x: (abs(x - w), x))


def run_config(runs_dir: Path, exp: str) -> dict:
    p = Path(runs_dir) / exp / "config.yaml"
    return load_yaml(p) if p.is_file() else {}


def _pct_cols(group: str, keys, prefix: str) -> list[Col]:
    return [Col(f"{prefix}{k}", k, group, "pct", BETTER[k]) for k in keys]


# ----------------------------------------------------------------------------------------------- builders
def ablation_tables(runs: dict, final: bool, log=None) -> list[Table]:
    cols = ([Col("det.f", "F", "Det", "pct", "max")] + _pct_cols("Track", SPOT_KEYS, "track.")
            + _pct_cols("Spot", SPOT_KEYS, "spot."))
    out = []
    for exp, ds, split, e in iter_meta(runs, "ablation", final, log):
        rows = [dict({c.key: r.get(c.key) for c in cols}, _label=r.get("label", r.get("exp_id", "?")))
                for r in e.get("rows", [])]
        if rows:
            out.append(Table(f"ablation_{ds}_{split}", f"Ablation (S1) on {ds} {split}.", cols, rows,
                             label_head="Configuration", sources=[f"runs/{exp}/metrics.json"]))
    return out


def _reported_rows(entries: list) -> list[dict]:
    rows = []
    for e in entries or []:
        r = {"_block": "Reported", "_label": str(e.get("method", "?"))}
        if e.get("source"):
            r["_label"] += f" ({e['source']})"
        if e.get("cite"):
            r["_label_tex"] = tex_escape(e.get("method", "?")) + rf"~\cite{{{e['cite']}}}"
        for sec, keys in (("track", TRACK_KEYS), ("spot", SPOT_KEYS)):
            for k in keys:
                r[f"{sec}.{k}"] = (e.get(sec) or {}).get(k)
        rows.append(r)
    return rows


def comparison_tables(cfg: dict, runs: dict, runs_dir: Path, root: Path, final: bool, log=None) -> list[Table]:
    comp = cfg.get("comparison") or {}
    cols = _pct_cols("Tracking", TRACK_KEYS, "track.") + _pct_cols("Spotting", SPOT_KEYS, "spot.")
    ours: dict[str, list[dict]] = {}
    splits: dict[str, set] = {}
    srcs: dict[str, list[str]] = {}
    for item in comp.get("ours") or []:
        exp = item["exp_id"]
        if exp not in runs:
            if log:
                log.warning("comparison: runs/%s/metrics.json not found; row skipped", exp)
            continue
        rc = run_config(runs_dir, exp)
        ds, split = rc.get("dataset"), rc.get("split")
        if ds not in DATASETS or split not in SPLITS:
            if log:
                log.warning("comparison: runs/%s/config.yaml lacks dataset/split; row skipped", exp)
            continue
        if split == "test" and not final:
            if log:
                log.info("comparison: skip %s (test split needs --final)", exp)
            continue
        m = runs[exp]
        r = {"_block": "Reproduced (this work)", "_label": f"{item.get('label', exp)} [{split}]"}
        for sec, keys in (("track", TRACK_KEYS), ("spot", SPOT_KEYS)):
            for k in keys:
                r[f"{sec}.{k}"] = (m.get(sec) or {}).get(k)
        ours.setdefault(ds, []).append(r)
        splits.setdefault(ds, set()).add(split)
        srcs.setdefault(ds, []).append(f"runs/{exp}/metrics.json")
    rep_path = resolve(comp.get("reported", "configs/reported_numbers.yaml"), root)
    reported = load_yaml(rep_path) if rep_path.is_file() else {}
    if not rep_path.is_file() and log:
        log.warning("comparison: %s not found; no 'Reported' block", rep_path)
    out = []
    for ds in DATASETS:
        rows = ours.get(ds, []) + _reported_rows((reported or {}).get(ds))
        if not rows:
            continue
        sp = "/".join(sorted(splits.get(ds, ()))) or "-"
        out.append(Table(f"comparison_{ds}", f"Comparison on {ds}. Reproduced rows: {sp} split; reported rows "
                         f"as published (configs/reported_numbers.yaml).", cols, rows,
                         sources=srcs.get(ds, []) + [str(comp.get("reported", "configs/reported_numbers.yaml"))]))
    return out


def threshold_tables(cfg: dict, runs: dict, final: bool, log=None) -> list[Table]:
    w_want = float((cfg.get("threshold") or {}).get("w", 0.5))
    out = []
    for exp, ds, split, e in iter_meta(runs, "sweep_fusion", final, log):
        table = e.get("table") or []
        if not table:
            continue
        w = pick_w(table, w_want)
        modes = [m for m in ("hard", "soft") if any(r.get("mode") == m for r in table)]
        cols = []
        for mode in modes:
            g = f"{mode}, w={w:.2f}"
            cols += [Col(f"{mode}.track_MOTA", "Trk MOTA", g, "pct", "max"),
                     Col(f"{mode}.track_IDF1", "Trk IDF1", g, "pct", "max"),
                     Col(f"{mode}.spot_MOTA", "Spot MOTA", g, "pct", "max"),
                     Col(f"{mode}.spot_IDF1", "Spot IDF1", g, "pct", "max")]
        by_thr: dict[float, dict] = {}
        for r in table:
            if not is_num(r.get("w")) or abs(float(r["w"]) - w) > 1e-9:
                continue
            row = by_thr.setdefault(float(r["thr"]), {"_label": f"{float(r['thr']):.2f}"})
            for k in ("track_MOTA", "track_IDF1", "spot_MOTA", "spot_IDF1"):
                row[f"{r['mode']}.{k}"] = r.get(k)
        rows = [by_thr[t] for t in sorted(by_thr)]
        out.append(Table(f"threshold_{exp}_{ds}_{split}",
                         f"Score-threshold sensitivity (S2), hard vs soft fusion, {exp} on {ds} {split} "
                         f"(soft source {e.get('soft_source', '-')}, tracker {e.get('tracker', '-')}).",
                         cols, rows, label_head="thr", sources=[f"runs/{exp}/metrics.json"]))
    return out


def voting_tables(runs: dict, final: bool, log=None) -> list[Table]:
    cols = [Col("tau", "tau", "", "text", head_tex=r"$\tau$"), Col("min_len", "min_len", "", "text",
                                                                     head_tex=r"$n_{\min}$"),
            Col("tracks", "tracks", "", "int"),
            Col("track_MOTA", "MOTA", "Track", "pct", "max"), Col("track_IDF1", "IDF1", "Track", "pct", "max"),
            Col("spot_MOTA", "MOTA", "Spot", "pct", "max"), Col("spot_MOTP", "MOTP", "Spot", "pct", "max"),
            Col("spot_IDF1", "IDF1", "Spot", "pct", "max")]
    out = []
    for exp, ds, split, e in iter_meta(runs, "sweep_voting", final, log):
        rows = []
        for r in e.get("table") or []:
            voted = r.get("voted") in (True, "True", "true", 1)
            row = {c.key: r.get(c.key) for c in cols}
            row["tau"] = f"{float(r['tau']):.2f}" if voted and is_num(r.get("tau")) else None
            row["min_len"] = str(int(r["min_len"])) if voted and is_num(r.get("min_len")) else None
            row["_label"] = "on" if voted else "off"
            rows.append(row)
        if rows:
            out.append(Table(f"voting_{exp}_{ds}_{split}", f"Track-level text voting (S3), {exp} on {ds} {split}.",
                             cols, rows, label_head="Voting", sources=[f"runs/{exp}/metrics.json"]))
    return out


def efficiency_tables(runs: dict, final: bool, log=None) -> list[Table]:
    cols = ([Col(f"ms.{s}", s, "ms / frame", "ms", "min") for s in STAGES]
            + [Col("ms.total", "total", "ms / frame", "ms", "min"), Col("fps", "FPS", "", "fps", "max"),
               Col("params", "Params (M)", "", "mega", "min"), Col("mem", "VRAM (MB)", "", "mb", "min")])
    groups: dict[tuple[str, str], list[dict]] = {}
    srcs: dict[tuple[str, str], list[str]] = {}
    for exp, ds, split, e in iter_meta(runs, "efficiency", final, log):
        stages = e.get("stages") or {}
        if not set(stages) - {"det"}:
            continue  # detector-only profile
        e2e = e.get("e2e") or {}
        row = {f"ms.{s}": (stages.get(s) or {}).get("ms_per_frame") for s in STAGES}
        row.update({"ms.total": e2e.get("total_ms_per_frame"), "fps": e2e.get("fps"),
                    "params": e2e.get("params"), "mem": e2e.get("max_peak_mem_mb"), "_label": exp})
        if not e2e.get("det_included", False):
            row["_label"] += DAGGER
            row["_nobest"] = {"ms.total", "fps", "params"}
        groups.setdefault((ds, split), []).append(row)
        srcs.setdefault((ds, split), []).append(f"runs/{exp}/metrics.json")
    return [Table(f"efficiency_{ds}_{split}",
                  f"Efficiency (S4) at batch 1 on {ds} {split}, one GPU. {DAGGER}: detector not included "
                  f"(total, FPS and params exclude it and are not ranked).", cols, rows, label_head="Run",
                  sources=srcs[(ds, split)])
            for (ds, split), rows in sorted(groups.items())]


def error_tables(runs: dict, final: bool, log=None) -> list[Table]:
    out = []
    summary: dict[tuple[str, str], list[dict]] = {}
    srcs: dict[tuple[str, str], list[str]] = {}
    for exp, ds, split, e in iter_meta(runs, "errors", final, log):
        counts, totals = e.get("counts") or {}, e.get("totals") or {}
        all_err = sum(int(totals.get(c, 0)) for c in ERR_CATS)
        row = {c: totals.get(c) for c in ERR_CATS}
        row.update({"all": all_err, "_label": f"{exp} ({e.get('stage', '-')})"})
        summary.setdefault((ds, split), []).append(row)
        srcs.setdefault((ds, split), []).append(f"runs/{exp}/metrics.json")
        rows = []
        for c in ERR_CATS:
            r = {b: (counts.get(c) or {}).get(b) for b in SIZE_BINS}
            r.update({"all": totals.get(c), "_label": ERR_LABELS[c],
                      "share": 100.0 * float(totals.get(c, 0)) / all_err if all_err else None})
            rows.append(r)
        cols = ([Col(b, f"{b} px", "text height", "int") for b in SIZE_BINS]
                + [Col("all", "all", "", "int"), Col("share", "share (%)", "", "pct1")])
        out.append(Table(f"errors_bins_{exp}_{ds}_{split}",
                         f"Errors (S5) by category and text height, {exp} on {ds} {split}, IoU "
                         f"{e.get('iou', '-')}.", cols, rows, label_head="Category",
                         sources=[f"runs/{exp}/metrics.json"]))
    cols = [Col(c, ERR_LABELS[c], "", "int", "min") for c in ERR_CATS] + [Col("all", "Total", "", "int", "min")]
    for (ds, split), rows in sorted(summary.items()):
        out.insert(0, Table(f"errors_{ds}_{split}", f"Error counts (S5) on {ds} {split}.", cols, rows,
                            label_head="Run (stage)", sources=srcs[(ds, split)]))
    return out


def _exp_name(p) -> str:
    p = Path(str(p).replace("\\", "/"))
    return p.parent.name if p.name == "metrics.json" else p.stem


def ci_tables(runs: dict, log=None) -> list[Table]:
    cols = [Col("metric", "Metric", "", "text"), Col("a", "A", "", "pct"), Col("a_ci", "A CI", "", "text"),
            Col("b", "B", "", "pct"), Col("b_ci", "B CI", "", "text"),
            Col("d", "B-A", "", "pct", head_tex=r"$\Delta$ (B$-$A)"), Col("d_ci", "diff CI", "", "text",
                                                                          head_tex=r"$\Delta$ CI"),
            Col("sig", "sig.", "", "text")]
    rows, seen, levels, srcs = [], set(), set(), []
    for exp, m in runs.items():
        for key, e in (m.get("meta") or {}).items():
            if not key.startswith("bootstrap_") or key in seen or not isinstance(e, dict):
                continue
            seen.add(key)
            srcs.append(f"runs/{exp}/metrics.json")
            levels.add(round(100 * (1 - float(e.get("alpha", 0.05))), 1))
            a, b = _exp_name(e.get("a", "A")), _exp_name(e.get("b", "B"))
            for metric, r in (e.get("results") or {}).items():
                ra, rb, rd = r["a"], r["b"], r["diff"]
                row = {"_label": f"{a} vs {b}", "metric": metric, "a": ra["value"], "b": rb["value"],
                       "d": rd["value"], "a_ci": f"[{ra['lo']:.2f}, {ra['hi']:.2f}]",
                       "b_ci": f"[{rb['lo']:.2f}, {rb['hi']:.2f}]", "d_ci": f"[{rd['lo']:.2f}, {rd['hi']:.2f}]",
                       "sig": "yes" if rd.get("significant") else "no", "_bold": set()}
                if fmt_value(ra["value"], "pct") != fmt_value(rb["value"], "pct"):
                    row["_bold"].add("a" if ra["value"] > rb["value"] else "b")
                if rd.get("significant"):
                    row["_bold"] |= {"d", "d_ci"}
                rows.append(row)
    if not rows:
        return []
    lvl = "/".join(f"{x:g}" for x in sorted(levels))
    return [Table("ci", f"Paired video-level bootstrap {lvl}% intervals (B - A); bold difference = interval "
                  f"excludes zero.", cols, rows, label_head="A vs B", sources=sorted(set(srcs)))]


# ----------------------------------------------------------------------------------------------- main
def build_all(cfg: dict, root: Path, runs_dir: Path, final: bool = False, log=None) -> list[Table]:
    runs = load_runs(runs_dir)
    tables = (ablation_tables(runs, final, log) + comparison_tables(cfg, runs, runs_dir, root, final, log)
              + threshold_tables(cfg, runs, final, log) + voting_tables(runs, final, log)
              + efficiency_tables(runs, final, log) + error_tables(runs, final, log) + ci_tables(runs, log))
    names = set()
    for t in tables:
        if t.name in names:
            raise ValueError(f"duplicate table name {t.name}")
        names.add(t.name)
    return tables


def write_tables(tables: list[Table], tex_dir: Path, md_dir: Path) -> list[dict]:
    tex_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for t in tables:
        tp, mp = tex_dir / f"{t.name}.tex", md_dir / f"{t.name}.md"
        tp.write_text(to_latex(t), encoding="utf-8")
        mp.write_text(to_markdown(t), encoding="utf-8")
        out.append({"name": t.name, "tex": str(tp), "md": str(mp), "rows": len(t.rows), "sources": t.sources})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    ap.add_argument("--final", action="store_true", help="include test-split entries (rule 5)")
    ap.add_argument("--runs_dir", type=Path, default=None, help="default: <project root>/runs")
    ap.add_argument("--tex_dir", type=Path, default=None, help="default: <project root>/paper/tables")
    ap.add_argument("--md_dir", type=Path, default=None, help="default: <project root>/docs/tables")
    args = ap.parse_args(argv)
    root = project_root()
    cfg_path = resolve(args.config, root)
    if not cfg_path.is_file():
        ap.error(f"config not found: {cfg_path}")
    cfg = copy.deepcopy(load_yaml(cfg_path))
    set_seed(int(cfg["seed"]))
    run_dir = root / "runs" / cfg["exp_id"]
    log = get_logger("make_tables", run_dir / "log.txt")
    runs_dir = args.runs_dir or root / "runs"
    tables = build_all(cfg, root, runs_dir, args.final, log)
    written = write_tables(tables, args.tex_dir or root / "paper" / "tables", args.md_dir or root / "docs" / "tables")
    for w in written:
        log.info("table %s: %d rows -> %s", w["name"], w["rows"], w["tex"])
    write_run_files(cfg, run_dir, {"final": bool(args.final), "tables": written}, key="tables")
    log.info("%d tables written", len(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
