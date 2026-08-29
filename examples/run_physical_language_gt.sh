#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="${PHIZERO_TOKENIZER_PYTHON:-${PHIZERO_DECODER_PYTHON:-}}"
checkpoint="${PHIZERO_TOKENIZER_CHECKPOINT:-${PHIZERO_DECODER_CHECKPOINT:-}}"
csv="${PHIZERO_GT_CSV:-}"
output="${PHIZERO_GT_OUTPUT_DIR:-${root}/outputs/physical_language_gt}"

: "${python:?Set PHIZERO_TOKENIZER_PYTHON or PHIZERO_DECODER_PYTHON.}"
: "${checkpoint:?Set PHIZERO_TOKENIZER_CHECKPOINT or PHIZERO_DECODER_CHECKPOINT.}"
: "${csv:?Set PHIZERO_GT_CSV to a CSV with video,prompt columns.}"

exec "${python}" "${root}/DiffSynth-Studio/examples/phizero/extract_physical_language_gt.py" \
  --csv "${csv}" \
  --checkpoint "${checkpoint}" \
  --output-dir "${output}" \
  "$@"
