Ablation on `icdar2015v` split `val` (values in percent)

| Row | exp_id | det.f | track.MOTA | track.MOTP | track.IDF1 | spot.MOTA | spot.MOTP | spot.IDF1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | `E4-base_yolo` | 71.80 | 46.01 | 74.74 | 57.80 | -6.20 | 77.50 | 33.90 |
| +LSC hard | `E4-lsc_yolo` | 73.51 | 43.83 | 74.58 | 56.44 | -9.84 | 77.33 | 33.38 |
| +GS | `E4-glyph_yolo` | 73.30 | 43.51 | 74.56 | 56.32 | -4.24 | 76.80 | 36.08 |
| +VPMM/LST | `E5-lst_yolo` | 73.51 | 45.52 | 74.69 | 51.95 | -10.47 | 77.43 | 30.31 |
| +soft | `A-soft_yolo` | 73.35 | 45.13 | 74.64 | 52.76 | -10.77 | 77.40 | 30.97 |
| +voting | `A-vote_yolo` | 73.51 | 46.23 | 74.62 | 52.26 | -13.89 | 76.83 | 30.81 |
| all | `A-all_yolo` | 73.35 | 45.80 | 74.58 | 53.06 | -13.13 | 76.94 | 32.15 |
