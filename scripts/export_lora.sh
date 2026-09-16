#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/style_lora_2024plus.env}"
ADAPTER_DIR="${2:-}"

if [[ -f "$CONFIG_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_PATH"
fi

: "${OUTPUT_DIR:=output/style_lora_2024plus}"
: "${MERGED_OUTPUT_DIR:=output/style_lora_2024plus_merged}"

if [[ -z "$ADAPTER_DIR" ]]; then
  ADAPTER_DIR="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
fi

if [[ -z "$ADAPTER_DIR" || ! -d "$ADAPTER_DIR" ]]; then
  echo "No adapter checkpoint found. Pass one explicitly: scripts/export_lora.sh <config> <checkpoint-dir>" >&2
  exit 1
fi

if ! command -v swift >/dev/null 2>&1; then
  echo "ms-swift is not available. Install it on the training machine before exporting." >&2
  exit 1
fi

mkdir -p "$MERGED_OUTPUT_DIR"

echo "Exporting LoRA adapter:"
echo "  adapter: $ADAPTER_DIR"
echo "  output:  $MERGED_OUTPUT_DIR"

swift export \
  --adapters "$ADAPTER_DIR" \
  --merge_lora true \
  --output_dir "$MERGED_OUTPUT_DIR"
