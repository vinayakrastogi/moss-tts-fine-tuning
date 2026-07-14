#!/usr/bin/env python3
"""
MOSS-TTS-Nano interactive CLI.
Uses local ONNX model files — no pip install of the model package required.

PyTorch is not needed when using builtin voice presets (audio codes are
pre-embedded in the manifest).  We inject a MagicMock stub so that the
top-level `import torch` / `import torchaudio` in onnx_tts_runtime.py
succeeds on Python versions (e.g. 3.14) where torch wheels are not yet
available.  The stubbed code paths (_load_reference_audio) are never
reached unless you pass --prompt-audio-path.

Usage:
  python tts_cli.py              # normal interactive mode
  python tts_cli.py --profile    # same + prints memory/latency stats after each synthesis
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
import time
import tracemalloc
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parent
REF_REPO = WORKSPACE_ROOT / "ref-repo"
MODEL_DIR = WORKSPACE_ROOT / "models" / "nano"

# ---------------------------------------------------------------------------
# Stub torch / torchaudio when not available (Python 3.14 has no wheel yet)
# ---------------------------------------------------------------------------
def _inject_torch_stub() -> None:
    """Inject a MagicMock for torch so onnx_tts_runtime can be imported."""
    torch_stub = MagicMock(name="torch")
    torch_stub.float32 = "float32_stub"
    torch_stub.float16 = "float16_stub"
    torch_stub.bfloat16 = "bfloat16_stub"
    torch_stub.device = MagicMock(return_value=MagicMock())
    sys.modules.setdefault("torch", torch_stub)

    torchaudio_stub = MagicMock(name="torchaudio")
    sys.modules.setdefault("torchaudio", torchaudio_stub)
    sys.modules.setdefault("torchaudio.functional", MagicMock(name="torchaudio.functional"))

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    _inject_torch_stub()

# ---------------------------------------------------------------------------
# Add ref-repo to sys.path so runtime modules are imported directly from source
# ---------------------------------------------------------------------------
if str(REF_REPO) not in sys.path:
    sys.path.insert(0, str(REF_REPO))

from onnx_tts_runtime import OnnxTtsRuntime  # noqa: E402

# ---------------------------------------------------------------------------
# Language detection & voice auto-selection
# ---------------------------------------------------------------------------

# Voice groups as declared in the manifest
_VOICE_GROUPS: dict[str, list[str]] = {
    "chinese": ["Junhao", "Zhiming", "Weiguo", "Xiaoyu", "Yuewen", "Lingyu"],
    "english": ["Ava", "Bella", "Adam", "Nathan", "Trump"],
    "japanese": ["Soyo", "Saki", "Mortis", "Umiri", "Mei", "Anon", "Arisa"],
}

def _has_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
                0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF):
            return True
    return False

def _has_japanese_kana(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF:
            return True
    return False

def _auto_voice(text: str, available: list[str]) -> str:
    """Pick the best default voice based on detected script."""
    available_set = set(available)
    if _has_japanese_kana(text):
        group = _VOICE_GROUPS["japanese"]
    elif _has_cjk(text):
        group = _VOICE_GROUPS["chinese"]
    else:
        group = _VOICE_GROUPS["english"]
    for name in group:
        if name in available_set:
            return name
    return available[0]  # ultimate fallback


# ---------------------------------------------------------------------------
# Resource profiling helpers
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    """Current process RSS in MB (works on macOS and Linux)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        # Fallback: ru_maxrss via resource module
        # On macOS ru_maxrss is in bytes; on Linux it is in kilobytes.
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return raw / 1024 / 1024 if sys.platform == "darwin" else raw / 1024


def _model_sizes(model_dir: Path) -> dict[str, float]:
    """Return sizes in MB for each .onnx and .data file under model_dir."""
    sizes: dict[str, float] = {}
    for ext in ("*.onnx", "*.data"):
        for f in sorted(model_dir.rglob(ext)):
            sizes[f.name] = f.stat().st_size / 1024 / 1024
    return sizes


def _print_model_sizes(model_dir: Path) -> None:
    sizes = _model_sizes(model_dir)
    total = sum(sizes.values())
    print("\n  Model file sizes:")
    for name, mb in sizes.items():
        print(f"    {name:<50s} {mb:>8.1f} MB")
    print(f"    {'TOTAL':<50s} {total:>8.1f} MB")


def _print_profile(
    *,
    load_time_s: float,
    rss_after_load_mb: float,
    infer_time_s: float,
    rss_after_infer_mb: float,
    audio_duration_s: float,
    peak_heap_mb: float,
) -> None:
    rtf = infer_time_s / audio_duration_s if audio_duration_s > 0 else float("inf")
    print("\n  ── Resource profile ─────────────────────────────")
    print(f"  Model load time        : {load_time_s:.2f} s")
    print(f"  RSS after load         : {rss_after_load_mb:.1f} MB")
    print(f"  Inference time         : {infer_time_s:.2f} s")
    print(f"  RSS after inference    : {rss_after_infer_mb:.1f} MB")
    print(f"  Audio duration         : {audio_duration_s:.2f} s")
    print(f"  Real-time factor (RTF) : {rtf:.3f}  {'✓ faster than real-time' if rtf < 1 else '✗ slower than real-time'}")
    print(f"  Peak Python heap       : {peak_heap_mb:.1f} MB  (tracemalloc)")
    print("  ─────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Audio playback helper
# ---------------------------------------------------------------------------

def _play_wav(wav_path: str) -> None:
    wav_path = str(wav_path)
    try:
        import sounddevice as sd
        import soundfile as sf
        data, sr = sf.read(wav_path, dtype="float32")
        sd.play(data, sr)
        sd.wait()
        return
    except Exception:
        pass
    if os.system(f'afplay "{wav_path}"') == 0:
        return
    print(f"[audio saved → {wav_path}]  Open it manually to listen.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MOSS-TTS-Nano interactive CLI")
    p.add_argument(
        "--profile", action="store_true",
        help="Print memory and latency stats after each synthesis (useful for edge deployment analysis).",
    )
    p.add_argument(
        "--threads", type=int, default=max(1, os.cpu_count() or 4),
        help="ONNX Runtime intra-op thread count (default: all logical cores).",
    )
    p.add_argument(
        "--voice", default=None,
        help="Built-in voice name to use. Auto-selected from text language if omitted.",
    )
    p.add_argument(
        "--no-streaming", action="store_true",
        help="Disable realtime streaming codec decode (streaming is on by default for better quality).",
    )
    p.add_argument(
        "--list-voices", action="store_true",
        help="Print available voices and exit.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.profile:
        _print_model_sizes(MODEL_DIR)
        tracemalloc.start()

    print(f"\nLoading MOSS-TTS-Nano (ONNX, threads={args.threads}) …")
    t_load_start = time.perf_counter()
    runtime = OnnxTtsRuntime(model_dir=str(MODEL_DIR), thread_count=args.threads)
    load_time_s = time.perf_counter() - t_load_start
    rss_after_load = _rss_mb()

    all_voices = [v["voice"] for v in runtime.list_builtin_voices()]
    groups: dict[str, list[str]] = {}
    for v in runtime.list_builtin_voices():
        g = str(v.get("group") or "Other")
        groups.setdefault(g, []).append(str(v["voice"]))

    if args.list_voices:
        print("\nAvailable voices:")
        for group, names in groups.items():
            print(f"  {group}: {', '.join(names)}")
        return

    # Warmup — prevents the first real synthesis from being sluggish
    print(f"Model loaded in {load_time_s:.1f}s  |  RSS: {rss_after_load:.0f} MB  |  Warming up …", end="", flush=True)
    runtime.synthesize(
        text="Hi.",
        voice="Ava",
        streaming=True,
        enable_wetext=False,
        enable_normalize_tts_text=True,
    )
    print(" ready.")

    streaming = not args.no_streaming
    sample_rate = int(runtime.codec_meta["codec_config"]["sample_rate"])

    print(f"Streaming: {'on' if streaming else 'off'}")
    print("Voices auto-selected by detected language (CN/EN/JP). Override with --voice NAME.")
    print("'--list-voices' to see all.  'q' or Ctrl-C to quit.\n")

    while True:
        try:
            text = input("Text> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not text:
            continue
        if text.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            break

        # Resolve voice: explicit arg > auto-detect from text script
        voice = args.voice if (args.voice and args.voice in all_voices) else _auto_voice(text, all_voices)

        if args.profile:
            tracemalloc.clear_traces()

        print(f"Synthesising  [voice={voice}] …")
        t_infer_start = time.perf_counter()
        try:
            result = runtime.synthesize(
                text=text,
                voice=voice,
                streaming=streaming,
                enable_wetext=False,          # WeTextProcessing needs pynini (optional)
                enable_normalize_tts_text=True,
            )
        except Exception as exc:
            print(f"[error] {exc}")
            continue

        infer_time_s = time.perf_counter() - t_infer_start
        rss_after_infer = _rss_mb()

        audio_path = result["audio_path"]
        waveform = result["waveform"]
        audio_duration_s = waveform.shape[0] / sample_rate
        rtf = infer_time_s / audio_duration_s if audio_duration_s > 0 else 0

        print(f"Saved → {audio_path}  [{audio_duration_s:.1f}s audio, RTF={rtf:.2f}]")

        if args.profile:
            _current, peak = tracemalloc.get_traced_memory()
            _print_profile(
                load_time_s=load_time_s,
                rss_after_load_mb=rss_after_load,
                infer_time_s=infer_time_s,
                rss_after_infer_mb=rss_after_infer,
                audio_duration_s=audio_duration_s,
                peak_heap_mb=peak / 1024 / 1024,
            )

        _play_wav(audio_path)


if __name__ == "__main__":
    main()
