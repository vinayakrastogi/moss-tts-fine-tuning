"""Shared inference and rating-sheet utilities for the paired TTS evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import threading
import time
import wave
from pathlib import Path

import numpy as np
import psutil


EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_DIR.parents[2]
PAYMENT_MANIFEST = PROJECT_ROOT / "training" / "indian_payment_tts" / "manifest.csv"
AUDIO_TOKENIZER = PROJECT_ROOT / "models" / "MOSS-Audio-Tokenizer-Nano"
OUTPUT_ROOT = EVALUATION_DIR / "generated"
MODEL_PATHS = {
    "original": PROJECT_ROOT / "models" / "MOSS-TTS-Nano",
    "adapted": PROJECT_ROOT / "training" / "indian_payment_tts" / "output" / "continued" / "checkpoint-last",
    "student": PROJECT_ROOT / "training" / "indian_payment_tts" / "output" / "student",
}


def parse_args(model_key: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Generate the held-out payment utterances with the {model_key} MOSS-TTS-Nano model."
    )
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of test utterances; 0 uses all.")
    parser.add_argument("--seed", type=int, default=42, help="Seed reset for each utterance in both model runs.")
    parser.add_argument("--max-new-frames", type=int, default=160)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATHS[model_key])
    parser.add_argument("--tokenizer-path", type=Path, default=AUDIO_TOKENIZER)
    parser.add_argument("--manifest", type=Path, default=PAYMENT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / f"{model_key}_model")
    return parser.parse_args()


def _resolve_device(torch, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _save_wav(path: str, waveform, sample_rate: int, *_args, **_kwargs) -> None:
    """Use the standard library WAV writer and avoid TorchCodec/FFmpeg coupling."""
    audio = waveform.detach().float().cpu().numpy()
    if audio.ndim == 1:
        audio = audio[:, None]
    elif audio.ndim == 2:
        audio = audio.T
    else:
        raise ValueError(f"Unexpected waveform shape: {audio.shape}")
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as output:
        output.setnchannels(int(pcm16.shape[1]))
        output.setsampwidth(2)
        output.setframerate(int(sample_rate))
        output.writeframes(pcm16.tobytes())


def _load_test_rows(limit: int, manifest: Path) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row.get("split") == "test"]
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No test prompts found in {manifest}")
    return rows


def _write_listener_sheet(rows: list[dict[str, str]]) -> None:
    path = OUTPUT_ROOT / "listener_rating_sheet.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "prompt_text", "expected_amount_inr",
        "A_amount_correct_1_to_5", "A_intelligibility_1_to_5", "A_naturalness_1_to_5", "A_indian_english_pronunciation_1_to_5",
        "B_amount_correct_1_to_5", "B_intelligibility_1_to_5", "B_naturalness_1_to_5", "B_indian_english_pronunciation_1_to_5",
        "comments",
    ]
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sample_id": row["id"],
                "prompt_text": row["record_exactly"],
                "expected_amount_inr": row["amount_inr"],
            })


def run_for_model(model_key: str) -> None:
    args = parse_args(model_key)
    checkpoint = args.model_path.expanduser().resolve()
    audio_tokenizer = args.tokenizer_path.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
    if not audio_tokenizer.is_dir():
        raise FileNotFoundError(f"Local MOSS audio tokenizer not found: {audio_tokenizer}")

    try:
        import torch
        import torchaudio
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "MOSS evaluation dependencies are missing. Use this project's Python environment, "
            "or install compatible Torch, TorchAudio, Transformers, and SentencePiece packages."
        ) from exc

    device = _resolve_device(torch, args.device)
    print(f"Loading {model_key} checkpoint from {checkpoint} on {device}...")
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), trust_remote_code=True)
    model.to(device=device, dtype=torch.float32)
    model._set_attention_implementation("sdpa")
    model.eval()
    print(f"Loaded in {time.perf_counter() - load_started:.1f}s")
    process = psutil.Process()
    rss_after_load_mb = process.memory_info().rss / 1_048_576

    # The model calls torchaudio.save internally. Replace only that writer so
    # WAV export does not depend on platform-specific TorchCodec libraries.
    torchaudio.save = _save_wav

    rows = _load_test_rows(args.limit, args.manifest)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_listener_sheet(_load_test_rows(0, args.manifest))

    # The first inference loads the audio codec. Measure it separately so later
    # per-prompt latency is not confused with one-time initialization.
    warmup_path = output_dir / "_warmup.wav"
    model.inference(
        text=rows[0]["record_exactly"].strip(),
        output_audio_path=warmup_path,
        mode="continuation",
        audio_tokenizer_type="moss-audio-tokenizer-nano",
        audio_tokenizer_pretrained_name_or_path=str(audio_tokenizer),
        device=device,
        max_new_frames=args.max_new_frames,
        do_sample=True,
        use_kv_cache=True,
    )
    warmup_path.unlink(missing_ok=True)

    run_info = {
        "model": model_key, "checkpoint": str(checkpoint),
        "audio_tokenizer": str(audio_tokenizer), "device": str(device),
        "python": platform.python_version(), "torch": torch.__version__,
        "machine": platform.machine(), "platform": platform.platform(),
        "seed": args.seed, "max_new_frames": args.max_new_frames,
        "warmups": 1, "rss_after_model_load_mb": rss_after_load_mb,
        "manifest": str(args.manifest.resolve()), "test_prompts": len(rows),
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")

    manifest_path = output_dir / "generated_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        fields = ["sample_id", "amount_inr", "prompt_text", "spoken_text", "audio_path", "duration_seconds", "sample_rate", "inference_seconds", "rtf", "rss_after_inference_mb", "peak_process_rss_mb", "status"]
        writer = csv.DictWriter(manifest_file, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            text = row["record_exactly"].strip()
            output_path = output_dir / f"{row['id']}.wav"
            torch.manual_seed(args.seed + index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + index)
            print(f"[{index + 1}/{len(rows)}] {row['id']}: {text}", flush=True)
            stop_sampling = threading.Event()
            peak_mb = [process.memory_info().rss / 1_048_576]
            def sample_rss() -> None:
                while not stop_sampling.wait(0.02):
                    peak_mb[0] = max(peak_mb[0], process.memory_info().rss / 1_048_576)
            sampler = threading.Thread(target=sample_rss, daemon=True)
            sampler.start()
            started = time.perf_counter()
            result = model.inference(
                text=text,
                output_audio_path=output_path,
                mode="continuation",
                audio_tokenizer_type="moss-audio-tokenizer-nano",
                audio_tokenizer_pretrained_name_or_path=str(audio_tokenizer),
                device=device,
                max_new_frames=args.max_new_frames,
                do_sample=True,
                use_kv_cache=True,
            )
            elapsed = time.perf_counter() - started
            stop_sampling.set()
            sampler.join()
            rss_after_inference_mb = process.memory_info().rss / 1_048_576
            peak_mb[0] = max(peak_mb[0], rss_after_inference_mb)
            waveform = result["waveform"]
            duration = waveform.shape[-1] / float(result["sample_rate"]) if waveform.numel() else 0.0
            writer.writerow({
                "sample_id": row["id"],
                "amount_inr": row["amount_inr"],
                "prompt_text": row["text"],
                "spoken_text": text,
                "audio_path": str(output_path),
                "duration_seconds": f"{duration:.3f}",
                "sample_rate": result["sample_rate"],
                "inference_seconds": f"{elapsed:.6f}",
                "rtf": f"{elapsed / duration:.6f}" if duration else "",
                "rss_after_inference_mb": f"{rss_after_inference_mb:.3f}",
                "peak_process_rss_mb": f"{peak_mb[0]:.3f}",
                "status": "generated",
            })
            manifest_file.flush()
    print(f"Generated {len(rows)} samples in {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Listener sheet: {OUTPUT_ROOT / 'listener_rating_sheet.csv'}")
