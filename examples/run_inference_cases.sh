#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

: "${PHIZERO_REASONER_PYTHON:?Set PHIZERO_REASONER_PYTHON.}"
: "${PHIZERO_DECODER_PYTHON:?Set PHIZERO_DECODER_PYTHON.}"
: "${PHIZERO_REASONER_CHECKPOINT:?Set PHIZERO_REASONER_CHECKPOINT.}"
: "${PHIZERO_DECODER_CHECKPOINT:?Set PHIZERO_DECODER_CHECKPOINT.}"

export DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMAGE_MAX_TOKEN_NUM=512
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12

"${PHIZERO_REASONER_PYTHON}" \
  "${root}/ms-swift/examples/infer/phizero_reason_cases.py" \
  --model "${PHIZERO_REASONER_CHECKPOINT}" \
  "$@"

decoder=(
  "${PHIZERO_DECODER_PYTHON}"
  "${root}/DiffSynth-Studio/examples/phizero/render_cases.py"
  --checkpoint "${PHIZERO_DECODER_CHECKPOINT}"
)

if [[ -n "${PHIZERO_DIT_PATHS:-}" ]]; then
  read -r -a dit_paths <<< "${PHIZERO_DIT_PATHS}"
  decoder+=(--dit-path "${dit_paths[@]}")
fi
if [[ -n "${PHIZERO_VAE_PATH:-}" ]]; then
  decoder+=(--vae-path "${PHIZERO_VAE_PATH}")
fi

"${decoder[@]}" "$@"
