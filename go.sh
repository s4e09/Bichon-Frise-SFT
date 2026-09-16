#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/style_lora_2024plus.env}"
exec bash scripts/train_lora.sh "$CONFIG_PATH"
