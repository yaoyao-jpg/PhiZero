#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${PHIZERO_REASONER_PYTHON:?Set PHIZERO_REASONER_PYTHON.}"
: "${PHIZERO_DECODER_PYTHON:?Set PHIZERO_DECODER_PYTHON.}"
: "${PHIZERO_REASONER_CHECKPOINT:?Set PHIZERO_REASONER_CHECKPOINT.}"
: "${PHIZERO_DECODER_CHECKPOINT:?Set PHIZERO_DECODER_CHECKPOINT.}"
: "${PHYSICS_IQ_BASE:?Set PHYSICS_IQ_BASE.}"

command=(
  "${PHIZERO_REASONER_PYTHON}"
  "${script_dir}/run.py"
  --mode "${PHIZERO_MODE:-all}"
  --benchmark-base "${PHYSICS_IQ_BASE}"
  --reasoner-python "${PHIZERO_REASONER_PYTHON}"
  --decoder-python "${PHIZERO_DECODER_PYTHON}"
  --score-python "${PHIZERO_SCORE_PYTHON:-${PHIZERO_REASONER_PYTHON}}"
  --reasoner-checkpoint "${PHIZERO_REASONER_CHECKPOINT}"
  --decoder-checkpoint "${PHIZERO_DECODER_CHECKPOINT}"
  --output-root "${PHIZERO_OUTPUT_ROOT:-${script_dir}/outputs}"
  --gpus "${PHIZERO_GPUS:-0,1,2,3,4,5,6,7}"
  --score-processes "${PHIZERO_SCORE_PROCESSES:-0}"
)

if [[ -n "${PHIZERO_DIT_PATHS:-}" ]]; then
  read -r -a dit_paths <<< "${PHIZERO_DIT_PATHS}"
  command+=(--dit-path "${dit_paths[@]}")
fi
if [[ -n "${PHIZERO_VAE_PATH:-}" ]]; then
  command+=(--vae-path "${PHIZERO_VAE_PATH}")
fi

exec "${command[@]}" "$@"
