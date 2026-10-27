# Decisions and deviations from the paper

One entry per decision: date, task id, what, why, impact on comparability with the paper.

| Date | Task | Decision | Why | Impact |
|------|------|----------|-----|--------|
| 2026-10-26 | P01 | `save_json` writes compact UTF-8 JSON atomically (tmp + replace) and converts NumPy scalars/arrays | interrupted runs never leave half-written stage files | none |
| 2026-10-26 | P01 | `get_logger` appends to the log file and logs to stdout; repeated calls do not duplicate handlers | stages are re-run into the same `runs/<exp_id>/log.txt` | none |
| | | (example) detector: Plan A PP-YOLOE-R small, input 1024x576 | 6 GB GPU | lower det recall than paper |
