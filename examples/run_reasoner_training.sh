#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="${PHIZERO_REASONER_PYTHON:-}"
model="${PHIZERO_REASONER_INIT_MODEL:-${root}/ckpt/reasoner}"
train_jsonl="${PHIZERO_REASONER_TRAIN_JSONL:-}"
val_jsonl="${PHIZERO_REASONER_VAL_JSONL:-}"
output="${PHIZERO_REASONER_OUTPUT_DIR:-${root}/outputs/reasoner_training}"
processes="${PHIZERO_REASONER_NUM_PROCESSES:-1}"

: "${python:?Set PHIZERO_REASONER_PYTHON.}"
: "${train_jsonl:?Set PHIZERO_REASONER_TRAIN_JSONL.}"

if [[ ! -x "${python}" ]]; then
  echo "PHIZERO_REASONER_PYTHON is not executable: ${python}" >&2
  exit 2
fi
if [[ ! -d "${model}" ]]; then
  echo "Reasoner initialization model is not a directory: ${model}" >&2
  exit 2
fi
if [[ ! -f "${train_jsonl}" ]]; then
  echo "Reasoner training JSONL does not exist: ${train_jsonl}" >&2
  exit 2
fi
if [[ -n "${val_jsonl}" ]] && [[ ! -f "${val_jsonl}" ]]; then
  echo "Reasoner validation JSONL does not exist: ${val_jsonl}" >&2
  exit 2
fi

if [[ ! "${processes}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PHIZERO_REASONER_NUM_PROCESSES must be a positive integer." >&2
  exit 2
fi

python="$(realpath -- "${python}")"
model="$(realpath -- "${model}")"
train_jsonl="$(realpath -- "${train_jsonl}")"
if [[ -n "${val_jsonl}" ]]; then
  val_jsonl="$(realpath -- "${val_jsonl}")"
fi
mkdir -p "${output}"
output="$(realpath -- "${output}")"

tokens="${output}/physical_language_tokens.txt"
prepare=(
  "${python}"
  "${root}/ms-swift/examples/train/prepare_phizero_reasoner_data.py"
  --dataset "${train_jsonl}"
  --tokens-output "${tokens}"
  --path-root "${root}"
)
if [[ -n "${val_jsonl}" ]]; then
  prepare+=(--dataset "${val_jsonl}")
fi
if [[ "${PHIZERO_REASONER_SKIP_IMAGE_CHECK:-0}" == "1" ]]; then
  prepare+=(--skip-image-check)
fi
"${prepare[@]}"

export PYTHONPATH="${root}/ms-swift${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMAGE_MAX_TOKEN_NUM=512
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12

if (( processes > 1 )); then
  export NPROC_PER_NODE="${processes}"
  export NNODES=1
  unset NODE_RANK MASTER_ADDR MASTER_PORT
else
  unset NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT
fi

command=(
  "${python}" -m swift.cli.main sft
  --model "${model}"
  --model_type qwen3_vl
  --template qwen3_vl
  --dataset "${train_jsonl}"
  --new_special_tokens "${tokens}"
  --tuner_type full
  --freeze_llm false
  --freeze_vit false
  --freeze_aligner false
  --loss_scale default
  --torch_dtype bfloat16
  --attn_impl flash_attn
  --num_train_epochs "${PHIZERO_REASONER_NUM_EPOCHS:-1}"
  --per_device_train_batch_size 1
  --per_device_eval_batch_size 1
  --learning_rate 1e-5
  --lr_scheduler_type cosine
  --weight_decay 0.1
  --adam_beta2 0.95
  --gradient_accumulation_steps 1
  --max_grad_norm 1.0
  --gradient_checkpointing true
  --vit_gradient_checkpointing true
  --eval_steps 100
  --save_steps 100
  --save_total_limit 2
  --save_only_model true
  --logging_steps 10
  --logging_nan_inf_filter false
  --max_length 1024
  --output_dir "${output}"
  --warmup_ratio 0.05
  --dataset_num_proc 1
  --dataloader_num_workers 4
  --load_from_cache_file true
  --split_dataset_ratio 0
  --add_non_thinking_prefix false
  --report_to none
)
if [[ -n "${val_jsonl}" ]]; then
  command+=(--val_dataset "${val_jsonl}")
fi

deepspeed="${PHIZERO_REASONER_DEEPSPEED:-}"
if [[ -z "${deepspeed}" ]] && (( processes > 1 )); then
  deepspeed=zero2
fi
if [[ -n "${deepspeed}" ]] && [[ "${deepspeed}" != "none" ]]; then
  command+=(--deepspeed "${deepspeed}")
fi

cd "${root}"
"${command[@]}" "$@"
