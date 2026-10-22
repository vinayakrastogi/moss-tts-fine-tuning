#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORK="$ROOT/training/indian_payment_tts"
DATA="$WORK/external/aura_indian_english"

if [[ ! -f "$DATA/train-00000-of-00001.parquet" ]]; then
  echo "Download the source dataset parquet into $DATA first (see README.md)." >&2
  exit 1
fi
"$PYTHON_BIN" "$WORK/import_aura_dataset.py"
"$PYTHON_BIN" "$WORK/encode_moss_audio.py"

"$PYTHON_BIN" "$WORK/finetuning/sft.py" \
  --model-path "$ROOT/models/MOSS-TTS-Nano" \
  --codec-path "$ROOT/models/MOSS-Audio-Tokenizer-Nano" \
  --train-jsonl "$DATA/train_with_codes.jsonl" \
  --output-dir "$WORK/output/initial" \
  --num-epochs 1 --learning-rate 1e-6 --max-length 512 \
  --mixed-precision no --attn-implementation eager --seed 42

"$PYTHON_BIN" "$WORK/finetuning/sft.py" \
  --model-path "$WORK/output/initial/checkpoint-last" \
  --codec-path "$ROOT/models/MOSS-Audio-Tokenizer-Nano" \
  --train-jsonl "$DATA/train_with_codes.jsonl" \
  --output-dir "$WORK/output/continued" \
  --num-epochs 2 --learning-rate 5e-7 --max-length 512 \
  --mixed-precision no --attn-implementation eager --seed 42

echo "Final adapted checkpoint: $WORK/output/continued/checkpoint-last"
