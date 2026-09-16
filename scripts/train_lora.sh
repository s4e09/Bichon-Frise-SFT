#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/style_lora_2024plus.env}"

if [[ -f "$CONFIG_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_PATH"
else
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

: "${MODEL_ID:=Qwen/Qwen3-4B-Instruct}"
: "${DATA_DIR:=$HOME/storage/SunJan10/qq_style_2024plus}"
: "${TRAIN_FILE:=style_train_alpaca.jsonl}"
: "${VAL_FILE:=style_val_alpaca.jsonl}"
: "${OUTPUT_DIR:=output/style_lora_2024plus}"

: "${LORA_RANK:=8}"
: "${LORA_ALPHA:=16}"
: "${LORA_TARGET_MODULES:=q_proj,k_proj,v_proj,o_proj}"
: "${LEARNING_RATE:=2e-4}"
: "${NUM_TRAIN_EPOCHS:=3}"
: "${MAX_LENGTH:=1024}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=4}"
: "${GRADIENT_ACCUMULATION_STEPS:=4}"
: "${TORCH_DTYPE:=bfloat16}"
: "${LOGGING_STEPS:=10}"
: "${SAVE_STEPS:=100}"
: "${EVAL_STEPS:=100}"
: "${SAVE_TOTAL_LIMIT:=3}"

TRAIN_PATH="$DATA_DIR/$TRAIN_FILE"
VAL_PATH="$DATA_DIR/$VAL_FILE"

if [[ ! -f "$TRAIN_PATH" ]]; then
  echo "Training dataset not found: $TRAIN_PATH" >&2
  exit 1
fi

if [[ ! -f "$VAL_PATH" ]]; then
  echo "Validation dataset not found: $VAL_PATH" >&2
  exit 1
fi

if ! command -v swift >/dev/null 2>&1; then
  echo "ms-swift is not available. Install it on the training machine before running this script." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Starting QQ style LoRA training:"
echo "  model:      $MODEL_ID"
echo "  train:      $TRAIN_PATH"
echo "  val:        $VAL_PATH"
echo "  output:     $OUTPUT_DIR"
echo "  lora_rank:  $LORA_RANK"
echo "  epochs:     $NUM_TRAIN_EPOCHS"

swift sft \
  --model "$MODEL_ID" \
  --dataset "$TRAIN_PATH" \
  --val_dataset "$VAL_PATH" \
  --train_type lora \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_target_modules "$LORA_TARGET_MODULES" \
  --learning_rate "$LEARNING_RATE" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --max_length "$MAX_LENGTH" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --torch_dtype "$TORCH_DTYPE" \
  --logging_steps "$LOGGING_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --eval_steps "$EVAL_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --output_dir "$OUTPUT_DIR"
