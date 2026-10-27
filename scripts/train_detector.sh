#!/usr/bin/env bash
# Fine-tune PP-YOLOE-R s on text in the logo-paddle env. Resumable: rerun the same command after a stop;
# tools/train_detector.py picks the newest runs/<exp_id>/ckpt/<epoch>.pdparams+.pdopt.
#   scripts/train_detector.sh                 full run (configs/det_ppyoloe_r_text_s.yaml)
#   scripts/train_detector.sh --smoke         200 images, 1 epoch -> runs/<exp_id>_smoke/
#   extra args pass through: --resume none|<ckpt>  --eval  --device cpu  --force  --config <yaml>
# Prereqs: python tools/gt_to_dota.py --splits train val ; ext_op built (docs/env_setup.md 2.4).
# Interpreter: $PADDLE_PY, else .venv-paddle (Windows or Linux layout).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PADDLE_PY:-}" ]]; then
  PY="$PADDLE_PY"
elif [[ -x "$ROOT/.venv-paddle/Scripts/python.exe" ]]; then
  PY="$ROOT/.venv-paddle/Scripts/python.exe"
elif [[ -x "$ROOT/.venv-paddle/bin/python" ]]; then
  PY="$ROOT/.venv-paddle/bin/python"
else
  echo "logo-paddle interpreter not found; set PADDLE_PY" >&2
  exit 1
fi
cd "$ROOT"
exec "$PY" tools/train_detector.py --config configs/det_ppyoloe_r_text_s.yaml "$@"
