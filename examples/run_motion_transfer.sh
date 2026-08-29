#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

: "${PHIZERO_DECODER_PYTHON:?Set PHIZERO_DECODER_PYTHON.}"
: "${PHIZERO_DECODER_CHECKPOINT:?Set PHIZERO_DECODER_CHECKPOINT.}"

export DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

command=(
  "${PHIZERO_DECODER_PYTHON}"
  "${root}/DiffSynth-Studio/examples/phizero/motion_transfer.py"
  --checkpoint "${PHIZERO_DECODER_CHECKPOINT}"
)

if [[ -n "${PHIZERO_DIT_PATHS:-}" ]]; then
  read -r -a dit_paths <<< "${PHIZERO_DIT_PATHS}"
  command+=(--dit-path "${dit_paths[@]}")
fi
if [[ -n "${PHIZERO_VAE_PATH:-}" ]]; then
  command+=(--vae-path "${PHIZERO_VAE_PATH}")
fi

"${command[@]}" "$@"
