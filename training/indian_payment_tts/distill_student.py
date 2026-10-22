#!/usr/bin/env python3
"""Experimental 12-to-6-layer MOSS teacher/student feature distillation.

This is a real student training path, separate from the supervised fine-tuning
checkpoint used in the report's original/adapted benchmark. It does not by
itself establish speech quality or an edge memory reduction.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetuning.dataset import MossTTSNanoSFTDataset


ROOT = Path(__file__).resolve().parents[2]
SUPPORT_FILES = (
    "__init__.py", "configuration_moss_tts_nano.py", "gpt2_decoder.py",
    "modeling_moss_tts_nano.py", "prompting.py", "tokenization_moss_tts_nano.py",
)


def records(path: Path, limit: int) -> list[dict]:
    data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if limit:
        data = data[:limit]
    if not data or any("audio_codes" not in item for item in data):
        raise ValueError(f"No pre-encoded speech records in {path}")
    return data


def device_for(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def initialize_student(teacher, student) -> None:
    source = teacher.state_dict()
    target = student.state_dict()
    compatible = {key: value for key, value in source.items()
                  if key in target and target[key].shape == value.shape}
    student.load_state_dict(compatible, strict=False)
    source_blocks = teacher.transformer.h
    target_blocks = student.transformer.h
    for i, block in enumerate(target_blocks):
        teacher_index = round(i * (len(source_blocks) - 1) / max(1, len(target_blocks) - 1))
        block.load_state_dict(source_blocks[teacher_index].state_dict())


def hidden_loss(teacher, student, batch, device: torch.device) -> tuple[torch.Tensor, float]:
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    with torch.no_grad():
        target = teacher(input_ids=ids, attention_mask=mask,
                         use_cache=False, return_dict=True).global_hidden_states
    prediction = student(input_ids=ids, attention_mask=mask,
                         use_cache=False, return_dict=True).global_hidden_states
    if target is None or prediction is None:
        raise RuntimeError("MOSS forward did not return global hidden states")
    valid = mask.bool()
    target = target[valid].float()
    prediction = prediction[valid].float()
    l1 = F.l1_loss(prediction, target)
    cosine = F.cosine_similarity(prediction, target, dim=-1).mean()
    return l1 + (1 - cosine), float(cosine.detach())


def evaluate(teacher, student, loader, device: torch.device) -> dict:
    student.eval()
    losses, cosines = [], []
    with torch.no_grad():
        for batch in loader:
            loss, cosine = hidden_loss(teacher, student, batch, device)
            losses.append(float(loss))
            cosines.append(cosine)
    student.train()
    return {"n": len(losses), "feature_loss": sum(losses) / len(losses),
            "cosine": sum(cosines) / len(cosines)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=ROOT / "models/MOSS-TTS-Nano")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "training/indian_payment_tts/output/student")
    parser.add_argument("--student-layers", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.student_layers < 1 or args.max_length < 8 or args.epochs < 1:
        parser.error("student layers and epochs must be positive; max length must be >=8")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_for(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, trust_remote_code=True,
                                                   torch_dtype=torch.float32)
    teacher._set_attention_implementation("eager")
    teacher.eval().to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher_layers = int(teacher.config.gpt2_config.n_layer)
    if args.student_layers >= teacher_layers:
        parser.error(f"student must have fewer than {teacher_layers} teacher layers")
    student_config = copy.deepcopy(teacher.config)
    student_config.gpt2_config.n_layer = args.student_layers
    student = AutoModelForCausalLM.from_config(student_config, trust_remote_code=True)
    student._set_attention_implementation("eager")
    initialize_student(teacher, student)
    student.to(device).train()
    train = MossTTSNanoSFTDataset(records(args.train_jsonl, args.max_train_samples),
                                  tokenizer=tokenizer, model_config=teacher.config,
                                  max_length=args.max_length)
    validation = MossTTSNanoSFTDataset(records(args.validation_jsonl, args.max_validation_samples),
                                       tokenizer=tokenizer, model_config=teacher.config,
                                       max_length=args.max_length)
    train_loader = DataLoader(train, batch_size=1, shuffle=True, collate_fn=train.collate_fn)
    val_loader = DataLoader(validation, batch_size=1, shuffle=False, collate_fn=validation.collate_fn)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
    before = evaluate(teacher, student, val_loader, device)
    print("Validation before:", before, flush=True)
    steps = 0
    for epoch in range(args.epochs):
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = hidden_loss(teacher, student, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            steps += 1
            if steps % 10 == 0 or steps == 1:
                print(f"epoch={epoch + 1} step={steps} loss={float(loss.detach()):.4f}", flush=True)
            if args.max_steps and steps >= args.max_steps:
                break
        if args.max_steps and steps >= args.max_steps:
            break
    after = evaluate(teacher, student, val_loader, device)
    print("Validation after:", after, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(args.output_dir, safe_serialization=False)
    tokenizer.save_pretrained(args.output_dir)
    for filename in SUPPORT_FILES:
        source = args.teacher / filename
        if source.exists():
            shutil.copy2(source, args.output_dir / filename)
    params_teacher = sum(p.numel() for p in teacher.parameters())
    params_student = sum(p.numel() for p in student.parameters())
    summary = {
        "type": "experimental feature distillation", "teacher_layers": teacher_layers,
        "student_layers": args.student_layers, "teacher_parameters": params_teacher,
        "student_parameters": params_student,
        "parameter_reduction_percent": 100 * (1 - params_student / params_teacher),
        "training_steps": steps, "training_records": len(train),
        "validation_records": len(validation), "validation_before": before,
        "validation_after": after, "seed": args.seed,
        "claim_limit": "Feature matching is not a perceptual speech-quality or device-memory benchmark.",
    }
    (args.output_dir / "distillation_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
