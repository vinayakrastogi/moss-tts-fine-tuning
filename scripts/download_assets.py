#!/usr/bin/env python3
"""Download upstream model files and the licensed source dataset."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    if args.models_only and args.data_only:
        parser.error("choose at most one of --models-only and --data-only")
    if not args.data_only:
        for repo_id, target in (
            ("OpenMOSS-Team/MOSS-TTS-Nano", ROOT / "models/MOSS-TTS-Nano"),
            ("OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano", ROOT / "models/MOSS-Audio-Tokenizer-Nano"),
            ("OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX", ROOT / "models/nano/tokenizer"),
        ):
            print(f"Downloading {repo_id} to {target}", flush=True)
            snapshot_download(repo_id=repo_id, local_dir=target)
    if not args.models_only:
        repo_id = "auraCodes/indian-english-hindi-tts-60min"
        print(f"Downloading {repo_id} dataset", flush=True)
        snapshot = Path(snapshot_download(repo_id=repo_id, repo_type="dataset",
                                         allow_patterns=["*.parquet", "README.md"]))
        candidates = sorted(snapshot.rglob("train-*.parquet"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one training parquet file; found {candidates}")
        destination = ROOT / "training/indian_payment_tts/external/aura_indian_english/train-00000-of-00001.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], destination)
        print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
