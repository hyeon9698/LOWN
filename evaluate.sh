#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
[[ -f .env ]] || { echo 'Run bash setup.sh first.' >&2; exit 1; }
source .env
bash setup.sh --check
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

arguments=()
while (( $# )); do
  if [[ "$1" == "--gpu" ]]; then
    [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { echo '--gpu requires one GPU index.' >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES="$2"
    shift 2
  else
    arguments+=("$1")
    shift
  fi
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "$LOWN_PYTHON" demo.py "${arguments[@]}"
