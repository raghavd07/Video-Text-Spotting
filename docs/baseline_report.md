### P13: Detection Evaluation and Hard Fusion

**Evaluation setup.** The YOLO-OBB detector and hard-fusion pipeline were evaluated on the five-video ICDAR2015 video validation split. The evaluation used an IoU threshold of 0.50. The baseline detector was evaluated at a score threshold of 0.30; fused detections were evaluated at thresholds of 0.30 and 0.50.

**Results**

| Method            | Score threshold | Precision | Recall | F-score |
| ----------------- | --------------: | --------: | -----: | ------: |
| YOLO-OBB baseline |            0.30 |    0.1338 | 0.0554 |  0.0784 |
| Hard fusion       |            0.30 |    0.1338 | 0.0554 |  0.0784 |
| Hard fusion       |            0.50 |    0.1338 | 0.0554 |  0.0784 |

The baseline evaluation recorded 4 true positives, 2,986 false positives, and 7,210 false negatives. The same counts were obtained for both fused evaluation thresholds.

**Interpretation.** Hard fusion did not improve the measured detection metrics. During LSC inference, all 2,995 retained detections received `lang_hard = 1`. Consequently, the language signal did not distinguish detections that matched the ground truth from false positives. Applying the fusion formula increased detection scores but did not change the evaluated detection counts at the tested thresholds.

The experiment demonstrates that the inference, recognition, fusion, visualization, and evaluation stages execute successfully. However, the measured results do not demonstrate a detection-quality improvement. The ABINet checkpoint was also reported as not fine-tuned for this experiment, which should be considered when interpreting the language-rescoring results.

**Conclusion.** P13 pipeline execution succeeded, but the precision-improvement acceptance criterion was not met. Further work is needed to improve detector quality and obtain a more discriminative language-recognition signal before claiming that hard fusion improves detection precision.

<!-- freeze_baseline:begin -->
## P23: frozen baseline (v1.0-base)

2026-11-02 · commit `8844a06` · `icdar2015v` split `val` · det_exp `det_yolo11s_obb_text` · IoU 0.5 · all values in percent

### Reproduction

| Row | exp_id | det.f | track.MOTA | track.MOTP | track.IDF1 | spot.MOTA | spot.MOTP | spot.IDF1 | paper reported (Table V) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | `E4-base_yolo` | 71.80 | 46.01 | 74.74 | 57.80 | -6.20 | 77.50 | 33.90 | n/r |
| +LSC hard | `E4-lsc_yolo` | 73.51 | 43.83 | 74.58 | 56.44 | -9.84 | 77.33 | 33.38 | n/r |
| +GS | `E4-glyph_yolo` | 73.30 | 43.51 | 74.56 | 56.32 | -4.24 | 76.80 | 36.08 | n/r |
| +VPMM/LST | `E5-lst_yolo` | 73.51 | 45.52 | 74.69 | 51.95 | -10.47 | 77.43 | 30.31 | n/r |

Paper column: det.f / track.MOTA / track.MOTP / track.IDF1 / spot.MOTA / spot.MOTP / spot.IDF1 from arXiv 2405.19194 Table V (n/r: not reported / not filled in configs/freeze_baseline.yaml).

### Efficiency

Batch 1, 200 frames (Video_2_1_2), NVIDIA GeForce RTX 3050 6GB Laptop GPU; ms/frame per stage, e2e FPS = 1000 / sum of stage ms.

| Row | det ms | lsc ms | fuse ms | vpmm ms | track ms | total ms/frame | FPS | params (M) | peak GPU MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 25.66 | 36.46 | 0.00 | - | 0.56 | 62.68 | 15.95 | 46.44 | 303.5 |
| +LSC hard | 25.66 | 55.58 | 0.00 | - | 0.13 | 81.37 | 12.29 | 46.44 | 304.5 |
| +GS | 25.66 | 26.10 | 0.00 | - | 0.15 | 51.91 | 19.26 | 46.44 | 304.5 |
| +VPMM/LST | 25.66 | 26.35 | 0.00 | 2.56 | 3.58 | 58.15 | 17.20 | 88.67 | 304.5 |

| Row | stage | params | peak GPU MB | ms/frame |
|---|---|---:|---:|---:|
| Baseline | det | 9,699,174 | 209.0 | 25.657 |
| Baseline | lsc | 36,738,703 | 303.5 | 36.457 |
| Baseline | fuse | 0 | - | 0.001 |
| Baseline | track | 0 | - | 0.564 |
| +LSC hard | det | 9,699,174 | 209.0 | 25.657 |
| +LSC hard | lsc | 36,738,703 | 304.5 | 55.581 |
| +LSC hard | fuse | 0 | - | 0.001 |
| +LSC hard | track | 0 | - | 0.133 |
| +GS | det | 9,699,174 | 209.0 | 25.657 |
| +GS | lsc | 36,738,703 | 304.5 | 26.101 |
| +GS | fuse | 0 | - | 0.001 |
| +GS | track | 0 | - | 0.155 |
| +VPMM/LST | det | 9,699,174 | 209.0 | 25.657 |
| +VPMM/LST | lsc | 36,738,703 | 304.5 | 26.346 |
| +VPMM/LST | fuse | 0 | - | 0.002 |
| +VPMM/LST | vpmm | 14,946,304 | 199.9 | 2.565 |
| +VPMM/LST | track | 27,289,600 | 170.6 | 3.582 |

### Reproducibility

`E4-lsc_yolo` (+LSC hard) re-run with --force: max |Δ| = 0.000 (det.f), tolerance 0.3 → **PASS**.

### Deviations from the paper (docs/decisions.md)

- 2026-11-01 P19: Wrap GoMatching’s association Transformer and heads in `src/logo/tracking/lst_adapter.py`; use VPMM features `[N,2d]` instead of DeepSolo query features; leave `third_party` files unchanged (why: Keep upstream code untouched while adapting the matcher to the project’s VPMM output; impact: Deviates from the original DeepSolo query-feature input; real-data integration remains unverified)
<!-- freeze_baseline:end -->
