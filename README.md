# LOGO: Video Text Spotting

Find, read and follow text in videos: shop signs, road signs, captions, jerseys and so on.

Given a video, the system draws a box around every piece of text, reads what it says, and keeps
the same ID on each piece of text from frame to frame. The output is a video with labelled boxes
such as `#12 STOP`.

This project re-implements **LOGO** ([arXiv 2405.19194](https://arxiv.org/abs/2405.19194)) and adds
two extensions of its own:

- **C2, soft language re-scoring:** the recognizer's confidence adjusts each detection's score
  gradually, instead of a hard keep-or-drop decision.
- **C3, track-level text voting:** a whole track gets one transcription, chosen by voting across
  all the frames it appears in, so the label stops flickering from frame to frame.

---

## How it works

```
video ─► frames ─► detect ─► read (LSC) ─► fuse scores ─► track ─► vote ─► evaluate / render
```

| Stage | What it does | Code |
|---|---|---|
| **Frames** | Decodes the video into JPEG frames | `logo.data.frames` |
| **Detect** | Finds rotated text boxes in each frame (YOLO11-OBB or PP-YOLOE-R) | `logo.detector` |
| **Read (LSC)** | An ABINet-based recognizer reads each box and judges whether it is real text | `logo.lsc` |
| **Glyph branch** | Optional extra supervision from character-shape masks | `logo.glyph` |
| **Fuse** | Combines the detector and recognizer scores (hard, or C2 soft) | `logo.scoring` |
| **Track** | Links boxes across frames using VPMM features and an LST-Matcher tracker | `logo.vpmm`, `logo.tracking` |
| **Vote** | Gives each track a single text label (C3) | `logo.voting` |
| **Evaluate** | Detection P/R/F, tracking and spotting MOTA / MOTP / IDF1 | `logo.eval` |

Each stage reads the previous stage's output files from `runs/<exp_id>/<stage>/<dataset>/`. That
means you can re-run any single stage, inspect what it produced, or swap one component without
touching the others.

---

## Quick start

### 1. Install

Python 3.11 and an NVIDIA GPU are recommended. Everything also runs on CPU, just more slowly.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate              # Windows: .venv\Scripts\Activate.ps1
uv pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
uv pip install -e ".[dev]"
python tools/check_env.py              # add --allow-cpu on a machine without CUDA
pytest -q                              # about 250 tests on synthetic data, CPU only
```

The optional PP-YOLOE-R detector needs a separate PaddlePaddle environment. See
[docs/env_setup.md](docs/env_setup.md). The default YOLO11-OBB detector doesn't need it.

### 2. Run the demo on your own video

```bash
python scripts/demo.py --video my_clip.mp4 --out my_clip_spotted.mp4
python scripts/demo.py --video my_clip.mp4 --out out.mp4 --config configs/E4-base.yaml --device cpu
```

The demo prints the number of tracks found, the longest track and the processing speed (FPS).

To use a browser upload page instead:

```bash
uv pip install gradio==4.44.1
python scripts/demo.py --gradio        # then open http://127.0.0.1:7860
```

> **Note:** trained weights are not included in this repository. Train them yourself (see below)
> and place them where the configs expect, for example `runs/E3-lsc/checkpoints/best.pth`.

---

## Datasets

Three public benchmarks are supported. Download them from their official sources and place the raw
files under `data/raw/<dataset>/`:

| Name in code | Dataset |
|---|---|
| `icdar2013v` | ICDAR 2013 Video Text |
| `icdar2015v` | ICDAR 2015 Text in Videos |
| `dstext` | DSText (dense, small video text) |

Datasets are **not** included in this repository. The train/val/test video lists are in
[splits/](splits/).

Then prepare the data:

```bash
python scripts/prepare_gt.py     --ds icdar2015v                # XML annotations -> JSON
python scripts/extract_frames.py --ds icdar2015v --split train  # video -> frames
python scripts/extract_frames.py --ds icdar2015v --split val
python scripts/make_splits.py    --ds icdar2015v                # only if the split files are missing
```

The test split is locked: any script that touches it must be run with `--final`. This keeps
tuning honest.

---

## Training

```bash
python tools/train_obb.py --config configs/det_yolo11s_obb_text.yaml     # text detector
python -m logo.lsc.train_lsc --config configs/E3-lsc.yaml                # recognizer / classifier
python -m logo.tracking.train_lst --config configs/E5-lst.yaml           # tracker
```

Add `--smoke` to any of these for a short sanity-check run.

## Running the pipeline and evaluating

```bash
# full pipeline for one experiment
python scripts/run_pipeline.py --config configs/FINAL.yaml --stage det,lsc,fuse,track,vote,eval

# ablation study (every row in configs/ablation.yaml)
python tools/run_ablation.py

# tables and figures for the report
python tools/make_tables.py
python tools/make_figures.py
```

Each run writes `runs/<exp_id>/config.yaml`, `log.txt` and `metrics.json`. All metrics are
percentages.

---

## Results

All results below are on the **ICDAR 2015 video validation split**, because the official test
labels are not public. Higher is better.

| Configuration | Det F | Track MOTA | Track IDF1 | Spot IDF1 |
|---|---:|---:|---:|---:|
| Baseline (detector + simple tracker) | 71.80 | 46.01 | **57.80** | 33.90 |
| + LSC (hard filtering) | **73.51** | 43.83 | 56.44 | 33.38 |
| + Glyph supervision | 73.30 | 43.51 | 56.32 | **36.08** |
| + VPMM / LST-Matcher (LOGO reproduced) | **73.51** | 45.52 | 51.95 | 30.31 |
| + C2 soft re-scoring | 73.35 | 45.13 | 52.76 | 30.97 |
| + C3 track voting | **73.51** | **46.23** | 52.26 | 30.81 |
| **Final (LOGO + C2 + C3)** | 73.35 | 45.80 | 53.06 | 32.15 |

On **ICDAR 2013 video** (validation) the final system reaches Track MOTA 66.18, Track IDF1 61.54
and Spot IDF1 49.57.

On top of the reproduced LOGO model, C2 and C3 together raise tracking IDF1 by +1.1 and spotting
IDF1 by +1.8. More tables (confidence intervals, threshold sweeps, error breakdowns, speed) are in
[docs/tables/](docs/tables/). Known deviations from the original paper are listed in
[docs/decisions.md](docs/decisions.md).

---

## Repository layout

```
src/logo/          the Python package
  data/            annotation conversion, frame extraction, splits
  detector/        text detectors (YOLO11-OBB, PP-YOLOE-R)
  lsc/             language synergy classifier (ABINet-based recognizer)
  glyph/           glyph-shape supervision branch
  vpmm/            visual-positional features for tracking
  tracking/        simple IoU tracker, embedding tracker, LST-Matcher
  scoring/         score fusion (hard and soft / C2)
  voting/          track-level text voting (C3)
  eval/            detection, tracking and spotting metrics
  utils/           I/O, geometry, paths
configs/           experiment configs (base.yaml documents every key)
scripts/           main entry points: data prep, pipeline, final eval, demo
tools/             training, sweeps, ablations, analysis, tables and figures
splits/            train / val / test video lists per dataset
tests/             pytest suite (CPU-only, synthetic data)
docs/              setup guide, data report, decisions, result tables and figures
paper/             generated LaTeX tables and PDF figures
notebooks/         dataset exploration
```

## Acknowledgements

- LOGO: *Video Text Spotting with Language Collaboration and Glyph Perception Model*
  ([arXiv 2405.19194](https://arxiv.org/abs/2405.19194))
- [ABINet](https://github.com/FangShancheng/ABINet) for the recognizer,
  [GoMatching](https://github.com/Hxyz-123/GoMatching) for the LST-Matcher,
  [Ultralytics](https://github.com/ultralytics/ultralytics) and
  [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) for the detectors.
  Third-party code is not redistributed here. Clone ABINet, GoMatching and PaddleDetection into
  `third_party/<name>/` yourself (see [docs/env_setup.md](docs/env_setup.md)).
- Datasets: ICDAR 2013 / 2015 Robust Reading video tasks, and DSText.

Built as a B.Tech major project.
