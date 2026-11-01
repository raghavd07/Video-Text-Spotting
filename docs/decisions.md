# Decisions and deviations from the paper

One entry per decision: date, task id, what, why, impact on comparability with the paper.

|Date|Task|Decision|Why|Impact|
|-|-|-|-|-|
|2026-10-26|P01|`save\_json` writes compact UTF-8 JSON atomically (tmp + replace) and converts NumPy scalars/arrays|interrupted runs never leave half-written stage files|none|
|2026-10-26|P01|`get\_logger` appends to the log file and logs to stdout; repeated calls do not duplicate handlers|stages are re-run into the same `runs/<exp\_id>/log.txt`|none|
|||(example) detector: Plan A PP-YOLOE-R small, input 1024x576|6 GB GPU|lower det recall than paper|
|2026-11-01|P19|Wrap GoMatching’s association Transformer and heads in `src/logo/tracking/lst\_adapter.py`; use VPMM features `\[N,2d]` instead of DeepSolo query features; leave `third\_party` files unchanged|Keep upstream code untouched while adapting the matcher to the project’s VPMM output|Deviates from the original DeepSolo query-feature input; real-data integration remains unverified|



