#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONDONTWRITEBYTECODE=1

if [[ "${1:-}" == "--check" ]]; then
  [[ -f .env ]] || { echo 'Run bash setup.sh first.' >&2; exit 1; }
  source .env
  [[ -x "$LOWN_PYTHON" ]] || { echo 'The local environment is missing; rerun setup.sh.' >&2; exit 1; }
  "$LOWN_PYTHON" - <<'PY'
import sys
import torch
import transformers
from llava.model.language_model.llava_llada import LlavaLLaDAModelLM
from llava.mm_utils import process_images
assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"
assert transformers.__version__ == "4.39.3", "Use the supplied requirements.txt"
assert torch.__version__.startswith("2.6.0"), "PyTorch 2.6.0 is required"
print("Environment ready (Python 3.11, PyTorch 2.6.0, Transformers 4.39.3).")
PY
  exit 0
fi
[[ $# == 0 ]] || { echo 'Usage: bash setup.sh [--check]' >&2; exit 2; }

bootstrap="${PYTHON_BIN:-python3}"
"$bootstrap" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11 (set PYTHON_BIN if needed)"'
[[ -x .venv/bin/python ]] || "$bootstrap" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install -r requirements.txt
if [[ ! -f .env ]]; then
  printf '%s\n' 'LOWN_PYTHON=.venv/bin/python' > .env
fi
bash setup.sh --check
