# Environment setup (Windows 11, RTX 3050 6 GB Laptop, driver 592.27)

Two environments. Only the detector stage (Plan A) runs in `logo-paddle`; every other stage runs in `logo-torch`.
Stages talk only through files under `runs/<exp_id>/<stage>/<ds>/`, so the two envs never import each other.
All commands are PowerShell, run from the repo root `E:\Logo_Video_Text_Spotting`.

## 1. logo-torch (`.venv`, Python 3.11, torch 2.1.2+cu121)

Already created with `python -m venv .venv` and torch installed via
`pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121`.
Add the remaining packages. `numpy==1.26.4` and the torch pins are repeated on the same command line so pip
can never upgrade them as a side effect:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install opencv-python==4.10.0.84 shapely pyclipper scikit-learn pyyaml tqdm pytest motmetrics matplotlib `
    numpy==1.26.4 torch==2.1.2 torchvision==0.16.2 --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install -e . --no-deps
python -m pip freeze > docs\pip_freeze_logo-torch.txt
python tools\check_env.py          # must print OK and exit 0
python -m pytest -q
```

`tools/check_env.py` exit codes: 0 ok, 1 missing required package or Python < 3.11, 2 CUDA unavailable
(`--allow-cpu` turns 2 into a warning, e.g. on a CPU-only CI machine).

## 2. logo-paddle (`.venv-paddle`, Python 3.10, PaddlePaddle-GPU + PaddleDetection)

### 2.1 Python 3.10 and the venv
```powershell
winget install --id Python.Python.3.10 -e      # or the python.org installer; adds `py -3.10`
py -3.10 -m venv .venv-paddle
.\.venv-paddle\Scripts\Activate.ps1
python --version                                # must print 3.10.x
python -m pip install --upgrade pip
```

### 2.2 PaddlePaddle-GPU
Open the official selector: https://www.paddlepaddle.org.cn/en/install/quick
Select: version = latest stable 2.x/3.x that PaddleDetection's README lists as supported, OS = Windows,
package = pip, compute platform = a CUDA version **not higher than** the "CUDA Version" shown by `nvidia-smi`
(driver 592.27). Copy the exact `python -m pip install paddlepaddle-gpu==... -i ...` line it prints and run it
inside `.venv-paddle`. Choose the version from the official selector; do not guess. Then:

```powershell
python -c "import paddle; paddle.utils.run_check()"   # must end with 'PaddlePaddle is installed successfully!'
python -c "import paddle; print(paddle.__version__, paddle.version.cuda(), paddle.version.cudnn())"
```
If `run_check` fails with a missing cuDNN/CUDA DLL, follow the selector page's Windows notes for that version.

### 2.3 PaddleDetection, pinned
```powershell
git clone https://github.com/PaddlePaddle/PaddleDetection.git third_party\PaddleDetection
cd third_party\PaddleDetection
git branch -r                                     # pick the release/2.x branch whose README matches your Paddle version
git checkout <that-branch>
git rev-parse HEAD                                # copy this SHA into docs/decisions.md (P02 row)
git checkout <SHA>                                # detached HEAD = pinned; re-run this on any new machine
python -m pip install -r requirements.txt
python -m pip install -e .
python -c "import ppdet; print(ppdet.__file__)"   # must point into third_party\PaddleDetection
cd ..\..
```

### 2.4 Rotated-IoU custom op (required by PP-YOLOE-R training)
Needs a C++/CUDA toolchain matching the Paddle build:
- Visual Studio 2019 or 2022 Build Tools with "Desktop development with C++";
- CUDA Toolkit with the **same major.minor** as `paddle.version.cuda()` (nvcc on PATH).

Open **"x64 Native Tools Command Prompt for VS"**, then:
```bat
cd /d E:\Logo_Video_Text_Spotting
.venv-paddle\Scripts\activate.bat
cd third_party\PaddleDetection\ppdet\ext_op
python setup.py install
for %f in (unittest\test_*.py) do python %f
```
All unittests must pass. If the build fails on Windows (common: nvcc/MSVC version mismatch), repeat sections
2.1–2.4 inside WSL2 Ubuntu using the Linux selector options; data and `runs/` are shared via `/mnt/e/...`.
If that also fails, use Plan B.

### 2.5 Final check
```powershell
.\.venv-paddle\Scripts\Activate.ps1
python -c "import paddle, ppdet; paddle.utils.run_check(); print(ppdet.__file__)"
python -m pip freeze > docs\pip_freeze_logo-paddle.txt
```

The `logo` package requires Python >= 3.11 and is **not** installed in `logo-paddle`. Detector scripts run
there must only depend on the stdlib, numpy, OpenCV and PaddleDetection, and write the det json of the CONTRACT.

## 3. Plan B: PyTorch rotated detector (Ultralytics YOLO OBB) in logo-torch

Use when the Paddle env or the rotated-IoU op cannot be built, or to remove the second env.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install ultralytics numpy==1.26.4 torch==2.1.2 torchvision==0.16.2 opencv-python==4.10.0.84 `
    --extra-index-url https://download.pytorch.org/whl/cu121
python -c "import ultralytics, torch; ultralytics.checks(); print(torch.__version__)"   # torch must still be 2.1.2+cu121
python tools\check_env.py                                                              # 'ultralytics' now has a version
```
Record the installed ultralytics version in docs/decisions.md.

What changes:
- **Detector stage only.** Training data: GT json -> YOLO-OBB labels (`0 x1 y1 x2 y2 x3 y3 x4 y4`, coordinates
  normalised by width/height, one txt per frame; `dont_care` boxes dropped). Inference reads
  `result.obb.xyxyxyxy` (N,4,2 pixels) and `result.obb.conf`, flattens each quad to 8 floats, orders it with
  `logo.utils.geometry.order_quad`, and writes the same det json:
  `{"video","dataset","frames":{"1":[{"quad":[8 floats],"score":float}]}}`.
- Runs in `logo-torch` (one env, AMP via `half=True`, small batch for 6 GB).
- Config: only keys inside `detector{}` differ; no CONTRACT key, file name or function signature changes.
- **Downstream: nothing changes.** lsc, fuse, track, vote and eval read the det json exactly as in Plan A.
- Deviation from the paper (detector differs) is logged in docs/decisions.md; it affects absolute det/spot
  numbers but not the relative C2/C3 ablations, which share the same detections.
