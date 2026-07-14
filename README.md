# MOSS-TTS-Nano Local CLI

Interactive text-to-speech CLI backed by [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) ONNX weights. Runs entirely from local model files — no pip-installed model package required, making it easy to swap in quantised or distilled variants for edge deployment.

---

## Requirements

- **Python 3.9+** (tested on 3.14.5)
- macOS / Linux
- ~1.3 GB free RAM at runtime
- ~730 MB disk for model weights

> PyTorch is **not** required. A lightweight stub handles the `import torch` in the upstream runtime, since builtin voices use pre-encoded audio codes and never call the torch audio-loading path.

---

## Installation

### 1. Clone this repo and the reference runtime

```bash
git clone <this-repo> MossEnv
cd MossEnv

# Clone the upstream runtime source (used directly, not pip-installed)
git clone https://github.com/OpenMOSS/MOSS-TTS-Nano ref-repo
```

### 2. Download model weights

```bash
# Install huggingface-cli if needed
pip install huggingface_hub

# TTS model (~640 MB)
huggingface-cli download OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX \
    --local-dir models/nano/tts

# Audio tokenizer / codec (~85 MB)
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX \
    --local-dir models/nano/tokenizer
```

### 3. Create the patched manifest

The original manifest uses a broken relative path to the codec. Run this once to fix it:

```bash
python3 -c "
import json
from pathlib import Path
src = Path('models/nano/tts/browser_poc_manifest.json')
m = json.loads(src.read_text())
m['model_files']['tts_meta']        = 'tts/tts_browser_onnx_meta.json'
m['model_files']['codec_meta']      = 'tokenizer/codec_browser_onnx_meta.json'
m['model_files']['tokenizer_model'] = 'tts/tokenizer.model'
Path('models/nano/browser_poc_manifest.json').write_text(json.dumps(m, indent=2))
print('Manifest written.')
"
```

### 4. Create a virtual environment and install dependencies

```bash
python3 -m venv moss-env

moss-env/bin/pip install onnxruntime sentencepiece sounddevice soundfile torchaudio
```

> `torchaudio` is installed for completeness but `torch` is intentionally omitted — the CLI stubs it automatically.

---

## Usage

```bash
moss-env/bin/python3 tts_cli.py
```

Type any text at the `Text>` prompt and press Enter. The CLI synthesises speech, prints the path to the saved WAV, and plays it back immediately. Enter `q` or press `Ctrl-C` to quit.

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--voice NAME` | auto | Force a specific voice (see `--list-voices`) |
| `--list-voices` | — | Print all 18 voices grouped by language and exit |
| `--threads N` | all cores | ONNX Runtime thread count — set low to simulate edge hardware |
| `--no-streaming` | off | Disable streaming codec decode (streaming on by default) |
| `--profile` | off | Print RAM, latency, and real-time factor after each synthesis |

### Examples

```bash
# Normal interactive mode
moss-env/bin/python3 tts_cli.py

# See all available voices
moss-env/bin/python3 tts_cli.py --list-voices

# Force a specific voice
moss-env/bin/python3 tts_cli.py --voice Adam

# Simulate a 2-core edge device with profiling
moss-env/bin/python3 tts_cli.py --threads 2 --profile
```

### Voice Auto-Selection

The CLI detects the script of your input text and picks a matching voice automatically:

| Input script | Voice group |
|---|---|
| Chinese / CJK | Chinese Male/Female (Junhao, Xiaoyu, …) |
| Japanese kana | Japanese Female (Soyo, Saki, …) |
| Everything else | English Male/Female (Ava, Adam, …) |

Override at any time with `--voice NAME`.

---

## Output

Generated audio is saved to `generated_audio/infer_onnx_output.wav` (48 kHz stereo WAV) and played back automatically via `sounddevice`. On macOS, `afplay` is used as a fallback if `sounddevice` is unavailable.

---

## Resource Usage (MacBook Air M4 baseline)

| Metric | Value |
|---|---|
| Disk (model weights) | ~723 MB |
| RAM after load | ~1265 MB |
| Model load time | ~1.6 s |
| Real-time factor | < 1.0 (faster than real-time) |

Run `--profile` to measure on your own hardware. Use `--threads N` to simulate reduced-core devices.

---

## Project Structure

```
MossEnv/
├── tts_cli.py          # CLI entry point
├── README.md           # this file
├── models/nano/        # model weights (gitignored — download separately)
│   ├── browser_poc_manifest.json   # patched manifest
│   ├── tts/            # TTS ONNX graphs + weights
│   └── tokenizer/      # codec ONNX graphs + weights
├── ref-repo/           # upstream OpenMOSS/MOSS-TTS-Nano source
└── moss-env/           # Python virtual environment (gitignored)
```

---

## Troubleshooting

**Audio cuts off after a few words**  
Do not pass `max_new_frames` to a warmup call — `OnnxTtsRuntime.synthesize()` permanently mutates the manifest default, capping all subsequent synthesis. The CLI handles this correctly.

**`ModuleNotFoundError: No module named 'torch'`**  
Expected on Python 3.14 — no torch wheel exists yet. The CLI injects a stub automatically. If you see this error directly, ensure you are running via `moss-env/bin/python3 tts_cli.py`, not a system Python that hasn't loaded the stub.

**No audio playback**  
The WAV is always saved to `generated_audio/infer_onnx_output.wav`. Open it manually if both `sounddevice` and `afplay` fail. Install `sounddevice` with `moss-env/bin/pip install sounddevice`.

**`FileNotFoundError: browser_poc_manifest.json not found`**  
Run the manifest patch command in step 3 of Installation.
