#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="${PHIZERO_TRAIN_PYTHON:-${PHIZERO_DECODER_PYTHON:-}}"
checkpoint="${PHIZERO_INIT_CHECKPOINT:-${PHIZERO_DECODER_CHECKPOINT:-}}"
csv="${PHIZERO_TRAIN_CSV:-${root}/data/training/toy_videos.csv}"
output="${PHIZERO_TRAIN_OUTPUT_DIR:-${root}/outputs/training_toy}"
processes="${PHIZERO_NUM_PROCESSES:-1}"

: "${python:?Set PHIZERO_TRAIN_PYTHON or PHIZERO_DECODER_PYTHON.}"
: "${checkpoint:?Set PHIZERO_INIT_CHECKPOINT or PHIZERO_DECODER_CHECKPOINT.}"

export DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

command=(
  "${python}" -m accelerate.commands.launch
  --num_processes "${processes}"
  --num_machines 1
  --mixed_precision bf16
  --dynamo_backend no
)
if (( processes > 1 )); then
  command+=(--multi_gpu)
fi
command+=(
  "${root}/DiffSynth-Studio/examples/phizero/train.py"
  --csv "${csv}"
  --init-checkpoint "${checkpoint}"
  --output-dir "${output}"
  --max-steps "${PHIZERO_MAX_STEPS:-100}"
)

if [[ -n "${PHIZERO_DIT_PATHS:-}" ]]; then
  read -r -a dit_paths <<< "${PHIZERO_DIT_PATHS}"
  command+=(--dit-path "${dit_paths[@]}")
fi
if [[ -n "${PHIZERO_VAE_PATH:-}" ]]; then
  command+=(--vae-path "${PHIZERO_VAE_PATH}")
fi

"${command[@]}" "$@"
