#!/usr/bin/env python3
"""Encode extracted audio with the project's local MOSS ONNX audio codec."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio.functional as AF

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "external" / "aura_indian_english"
ENCODER = Path(__file__).resolve().parents[2] / "models" / "nano" / "tokenizer" / "moss_audio_tokenizer_encode.onnx"
TARGET_RATE = 48_000


def encode_one(session: ort.InferenceSession, path: Path) -> list[list[int]]:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    # soundfile returns [samples, channels]; MOSS codec expects [batch, channels, samples].
    audio = torch.from_numpy(waveform.T.copy())
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_RATE:
        audio = AF.resample(audio, sample_rate, TARGET_RATE)
    audio = audio.repeat(2, 1).contiguous()
    values = audio.unsqueeze(0).numpy().astype(np.float32, copy=False)
    lengths = np.asarray([values.shape[-1]], dtype=np.int32)
    codes, code_lengths = session.run(["audio_codes", "audio_code_lengths"], {
        "waveform": values,
        "input_lengths": lengths,
    })
    return np.asarray(codes[0, :int(code_lengths[0]), :], dtype=np.int64).tolist()


def main() -> None:
    if not ENCODER.is_file():
        raise SystemExit(f"Missing local MOSS codec encoder: {ENCODER}")
    session = ort.InferenceSession(str(ENCODER), providers=["CPUExecutionProvider"])
    for split in ("train", "validation", "test"):
        source = DATASET_DIR / f"{split}.jsonl"
        if not source.is_file():
            continue
        destination = DATASET_DIR / f"{split}_with_codes.jsonl"
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
        with destination.open("w", encoding="utf-8") as output:
            for index, record in enumerate(records, 1):
                record["audio_codes"] = encode_one(session, Path(record["audio"]))
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"{split}: encoded {index}/{len(records)}", flush=True)
        print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
