# Edge-Deployable Neural Text-to-Speech: Domain Specialization and Distillation

This is the small, reproducible project extracted from the larger development workspace. It contains two connected experiments: Indian-English supervised adaptation of MOSS-TTS-Nano, and a separately trained six-layer student using teacher–student feature distillation. The full student training run reduced parameter count and matched-precision weight-file size by **36.3%**, and improved held-out feature cosine from **0.222 to 0.937**. The project's scripts prepare licensed speech, train the models, synthesize 24 fixed payment prompts, and calculate latency, real-time factor (RTF), and process memory. Table 7.2 of the existing report compares the original and adapted checkpoints, not the distilled student.

The training speech is an NPTEL lecture subset, **not recorded payment speech**. The generated payment prompts test transfer to a payment-announcement setting; they do not establish successful payment-domain specialization or better accent quality without human ratings. The supported distillation claim concerns **parameters, model-weight storage, and feature matching**. The measured original/adapted checkpoints also had mean RTF below 1 on the tested Apple Silicon machine. Run the benchmark locally before quoting a device-specific number. Do not describe fine-tuning alone as knowledge distillation or claim compression from Table 7.2.

## Project contents

| Path | Purpose |
|---|---|
| `scripts/download_assets.py` | Fetch upstream MOSS models, audio codec, and source dataset. |
| `training/indian_payment_tts/import_aura_dataset.py` | Verify audio hashes and split the English lecture clips by source/seed. |
| `training/indian_payment_tts/encode_moss_audio.py` | Encode recordings to MOSS audio tokens. |
| `training/indian_payment_tts/finetuning/sft.py` | Supervised model training, 1 initial + 2 continued epochs. |
| `training/indian_payment_tts/distill_student.py` | Separate experimental teacher/student feature-distillation path. |
| `scripts/export_student_bf16.py` | Save the student at the teacher checkpoint's bfloat16 precision for a fair weight-file comparison. |
| `training/indian_payment_tts/manifest.csv` | Fixed payment prompts; 24 rows have `split=test`. |
| `training/indian_payment_tts/evaluation/` | Paired original/adapted WAV generation and per-sample measurements. |
| `scripts/summarize_metrics.py` | Calculate paired descriptive metrics from generated manifests. |
| `results/report_table_7_2/` | Earlier raw 18-run measurements and summary transcribed into the report. |

The large datasets, checkpoints, generated WAVs, and local run folders are ignored by Git. This directory can be used as the root of a separate GitHub repository. The old `Version 2`, `Version 3`, `Deploy`, and other prototypes in the parent workspace are intentionally outside this release.

## Setup

Tested locally on Apple Silicon with Python 3.14.7, PyTorch 2.13.0, TorchAudio 2.11.0, and the package versions in `requirements.txt`. A compatible PyTorch/TorchAudio installation for your system is required. From this directory:

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchaudio
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_assets.py
```

The downloads are large. The dataset source is [auraCodes/indian-english-hindi-tts-60min](https://huggingface.co/datasets/auraCodes/indian-english-hindi-tts-60min), whose card states CC-BY 4.0. Keep its attribution and per-clip provenance. Model source: [OpenMOSS-Team/MOSS-TTS-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano). Review the source model and dataset licenses before redistributing any weights or audio. No downloaded assets are included here.

## Reproduce adaptation training

```bash
PYTHON_BIN=.venv/bin/python bash scripts/train_pipeline.sh
```

This runs the same two-stage schedule as the local experiment: 144 English training clips, 1 epoch at `1e-6`, then 2 epochs at `5e-7`, fixed 512-row sequences, seed 42, and a held-out lecture for testing. It writes the final checkpoint to `training/indian_payment_tts/output/continued/checkpoint-last/`. The 19 validation and 62 lecture-test clips are excluded from weight updates. GPU/MPS is strongly recommended; full training is much slower than the checks below.

The exact configuration of the earlier local checkpoint is recorded in `results/training_config_initial.json` and `results/training_config_continued.json`. Checkpoint outputs are intentionally not uploaded. Small hardware/software differences can shift timing and random audio generation; use the included fixed prompts, seed, model identities, and run metadata when comparing results.

## Generate test audio and measure it

```bash
PYTHON_BIN=.venv/bin/python DEVICE=auto bash scripts/run_benchmark.sh
```

The runner executes original and adapted models in separate Python processes. It warms each model once, then generates 24 WAVs per model with the same prompt IDs, order, seed, and `max_new_frames=160`. It writes per-prompt inference seconds, generated-audio duration, RTF (`inference_seconds / audio_duration_seconds`), and sampled process RSS to `runs/latest/{original,adapted}/generated_manifest.csv`; `runs/latest/summary.json` contains the paired summary. For a quick check, pass paths and a limit:

```bash
PYTHON_BIN=.venv/bin/python DEVICE=mps bash scripts/run_benchmark.sh \
  models/MOSS-TTS-Nano \
  training/indian_payment_tts/output/continued/checkpoint-last \
  models/MOSS-Audio-Tokenizer-Nano 3
```

The `prompt_text` field includes numeric INR text; the model receives the `spoken_text` field with the number verbalized in Indian grouping. This keeps text normalization outside the TTS comparison. Ratings of amount correctness, intelligibility, naturalness, and perceived pronunciation require listeners; the generated rating sheet is blank by design.

**Metric interpretation:** RTF < 1 means audio is produced faster than its playback duration. Inference time is wall-clock time after a warm-up. RSS is resident memory for the whole Python process, sampled every 20 ms during inference; it includes framework and codec overhead and is sensitive to allocator/caching state. The original and adapted models retain the same architecture, so a lower RSS sample is an observation, not evidence of compression. Avoid making a cross-device memory claim from this experiment.

## Experimental knowledge distillation

```bash
.venv/bin/python training/indian_payment_tts/distill_student.py \
  --train-jsonl training/indian_payment_tts/external/aura_indian_english/train_with_codes.jsonl \
  --validation-jsonl training/indian_payment_tts/external/aura_indian_english/validation_with_codes.jsonl \
  --student-layers 6 --epochs 3
```

This trains a reduced-layer student against the frozen teacher's input-dependent global hidden states. It reports student/teacher parameter counts and feature loss/cosine on held-out validation clips, then saves a student checkpoint. The report's original-versus-adapted benchmark is **not** this student. Before claiming that distillation improved speech quality, run the student through the same audio-generation and listener-evaluation protocol. Before claiming an edge-memory improvement, benchmark the student in a separate process with all required codec assets included.

To save a matched-precision student checkpoint after training:

```bash
.venv/bin/python scripts/export_student_bf16.py
```

Generate a test sample from the student checkpoint with:

```bash
.venv/bin/python training/indian_payment_tts/evaluation/generate_student_model.py --limit 1
```

## Results and report alignment

`results/report_table_7_2/` preserves the previous nine-prompt, two-round timing data (`n=18` per checkpoint). It supports the report's RTF finding: mean RTF 0.788 (original) and 0.793 (adapted) on its recorded MPS run. It also shows the adapted checkpoint was **8.4% slower** in mean wall-clock inference time. The saved process-RSS values are descriptive and should not be presented as proof that supervised fine-tuning compressed the model.

The fresh local run is preserved in `results/local_validation_2026-10-22/` (24 paired prompts, one warm-up per checkpoint, Apple Silicon MPS). Mean RTF was **0.876 original** and **0.976 adapted**, so both means were below real time; some individual prompts were above 1. Mean inference time was 4.24 s versus 4.88 s, making the adapted model about 15.1% slower in this run. The newer process-RSS readings differed from the earlier report table and do **not** support a memory-reduction claim. The exact input checkpoint hashes and raw measurements are included. Timing is device- and run-dependent; repeat the run on the target machine before quoting a device-specific number.

The full three-epoch distillation run is in `results/distillation_validation_2026-10-22/`: 144 training clips and 19 held-out validation clips, 432 updates. The six-layer student has **74.8M parameters versus 117.3M** for the teacher (36.3% fewer). Validation feature cosine rose from **0.222 to 0.937** and feature loss fell from **1.722 to 0.283**. At the teacher's bfloat16 precision, the student weight file was **149.6 MB versus 234.7 MB** (36.3% smaller). It generated 24 nonempty payment-prompt WAVs. Its mean inference time (7.32 s) and peak sampled process RSS (1278.9 MB) were worse than the original teacher's in these local runs, so the result supports **parameter and checkpoint compression**, not runtime acceleration, lower runtime memory, or speech-quality improvement. See `RESULTS_FOR_REPORT.md` for ready-to-use report wording and limits.

The `runs/latest/` result from this repository's benchmark is the independently rerunnable result to quote for new work. If its values differ, use the new raw CSV and its `run_info.json` files, and update the report accordingly. No listening preference or intelligibility improvement is asserted without completed listener ratings.
