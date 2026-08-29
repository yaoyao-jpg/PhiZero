#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output="${PHIZERO_ADAPT_OUTPUT_DIR:-${root}/outputs/motion_transfer_adaptation}"

export PHIZERO_TRAIN_CSV="${PHIZERO_ADAPT_CSV:-${root}/data/training/toy_videos.csv}"
export PHIZERO_TRAIN_OUTPUT_DIR="${output}"
export PHIZERO_MAX_STEPS="${PHIZERO_ADAPT_STEPS:-1000}"

exec bash "${root}/examples/run_training.sh" \
  --train-tokenizer \
  --gradient-accumulation-steps 1 \
  --warmup-steps 10 \
  --min-learning-rate 1e-5 \
  --motion-dropout 0 \
  --fsq-entropy-weight 0.05 \
  --save-every "${PHIZERO_ADAPT_SAVE_EVERY:-${PHIZERO_MAX_STEPS}}" \
  --gradient-checkpointing \
  "$@"
