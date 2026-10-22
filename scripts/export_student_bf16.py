#!/usr/bin/env python3
"""Save a trained student at the teacher checkpoint's bfloat16 precision."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SUPPORT_FILES = (
    "__init__.py", "configuration_moss_tts_nano.py", "gpt2_decoder.py",
    "modeling_moss_tts_nano.py", "prompting.py", "tokenization_moss_tts_nano.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, default=ROOT / "training/indian_payment_tts/output/student")
    parser.add_argument("--teacher", type=Path, default=ROOT / "models/MOSS-TTS-Nano")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "training/indian_payment_tts/output/student_bf16")
    args = parser.parse_args()
    teacher_weights = args.teacher / "pytorch_model.bin"
    if not teacher_weights.is_file():
        parser.error(f"Missing teacher weights: {teacher_weights}")
    teacher_state = torch.load(teacher_weights, map_location="cpu", weights_only=True)
    teacher_dtypes = {value.dtype for value in teacher_state.values()
                      if torch.is_tensor(value) and value.is_floating_point()}
    del teacher_state
    if teacher_dtypes != {torch.bfloat16}:
        parser.error(f"Expected a bfloat16 teacher checkpoint, found {teacher_dtypes}")
    model = AutoModelForCausalLM.from_pretrained(args.student, trust_remote_code=True,
                                                 torch_dtype=torch.float32)
    model.to(dtype=torch.bfloat16)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=False)
    AutoTokenizer.from_pretrained(args.student, trust_remote_code=True).save_pretrained(args.output_dir)
    for filename in SUPPORT_FILES:
        source = args.student / filename
        if source.exists():
            shutil.copy2(source, args.output_dir / filename)
    student_weights = args.output_dir / "pytorch_model.bin"
    result = {
        "teacher_dtype": "bfloat16", "student_dtype": "bfloat16",
        "teacher_weight_bytes": teacher_weights.stat().st_size,
        "student_weight_bytes": student_weights.stat().st_size,
        "weight_file_reduction_percent": 100 * (1 - student_weights.stat().st_size / teacher_weights.stat().st_size),
        "scope": "checkpoint weight files only; runtime, codec, and tokenizer assets excluded",
    }
    (args.output_dir / "checkpoint_size.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
