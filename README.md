# LOGO: Video Text Spotting

B.Tech major project. Re-implementation of [LOGO (arXiv 2405.19194)](https://arxiv.org/abs/2405.19194):
detector -> language synergy classifier (LSC) -> score fusion -> VPMM features -> LST-Matcher tracker,
plus two extensions: **C2** soft language re-scoring and **C3** track-level text voting.

## Setup
```
uv venv --python 3.11 .venv
.venv\Scripts\Activate.ps1          # Linux/WSL: source .venv/bin/activate
uv pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
uv pip install -e ".[dev]"
pytest -q
```
The detector may run in a separate PaddlePaddle env (`logo-paddle`).

## Pipeline
frames -> det -> lsc -> fuse -> track -> vote -> eval. Each stage reads the previous stage's files from
`runs/<exp_id>/<stage>/<ds>/`. Datasets: `icdar2013v`, `icdar2015v`, `dstext`.

## Layout
- `src/logo/{utils,data,detector,lsc,glyph,vpmm,tracking,scoring,voting,eval}`: package `logo`
- `configs/`: experiment configs (`base.yaml` has every key)
- `splits/`: `<ds>_{train,val,test}.txt`
- `scripts/`, `tools/`: entry points and dataset utilities
- `docs/decisions.md`: deviations from the paper
- `PROJECT_STATE.md`: progress log

## Results
TBD.
