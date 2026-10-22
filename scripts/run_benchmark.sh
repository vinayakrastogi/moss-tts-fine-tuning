#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ORIGINAL_MODEL="${1:-$ROOT/models/MOSS-TTS-Nano}"
ADAPTED_MODEL="${2:-$ROOT/training/indian_payment_tts/output/continued/checkpoint-last}"
AUDIO_TOKENIZER="${3:-$ROOT/models/MOSS-Audio-Tokenizer-Nano}"
LIMIT="${4:-24}"
DEVICE="${DEVICE:-auto}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/latest}"

"$PYTHON_BIN" "$ROOT/training/indian_payment_tts/evaluation/generate_original_model.py" \
  --device "$DEVICE" --limit "$LIMIT" --model-path "$ORIGINAL_MODEL" \
  --tokenizer-path "$AUDIO_TOKENIZER" --output-dir "$RUN_DIR/original"
"$PYTHON_BIN" "$ROOT/training/indian_payment_tts/evaluation/generate_adapted_model.py" \
  --device "$DEVICE" --limit "$LIMIT" --model-path "$ADAPTED_MODEL" \
  --tokenizer-path "$AUDIO_TOKENIZER" --output-dir "$RUN_DIR/adapted"
"$PYTHON_BIN" "$ROOT/scripts/summarize_metrics.py" \
  --original "$RUN_DIR/original/generated_manifest.csv" \
  --adapted "$RUN_DIR/adapted/generated_manifest.csv" \
  --output "$RUN_DIR/summary.json"
