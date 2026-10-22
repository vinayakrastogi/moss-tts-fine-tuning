#!/usr/bin/env python3
"""Extract the English portion of auraCodes' CC-BY Indian-English corpus."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "external" / "aura_indian_english"
PARQUET = DATASET_DIR / "train-00000-of-00001.parquet"
AUDIO_DIR = DATASET_DIR / "audio"


def main() -> None:
    if not PARQUET.exists():
        raise SystemExit(f"Dataset file not found: {PARQUET}")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch in pq.ParquetFile(PARQUET).iter_batches(batch_size=32):
        for record in batch.to_pylist():
            if record.get("language") != "en-IN":
                continue
            audio = record.get("audio") or {}
            audio_bytes = audio.get("bytes")
            text = " ".join((record.get("text") or "").split())
            if not audio_bytes or len(text) < 3:
                continue
            expected_hash = record.get("sha256")
            actual_hash = hashlib.sha256(audio_bytes).hexdigest()
            if expected_hash and actual_hash != expected_hash:
                raise ValueError(f"Audio hash mismatch for {record.get('clip_id')}")
            filename = f"{record['clip_id']}.wav"
            audio_path = AUDIO_DIR / filename
            if not audio_path.exists():
                audio_path.write_bytes(audio_bytes)
            rows.append({
                "clip_id": record["clip_id"],
                "audio": str(audio_path.resolve()),
                "text": text,
                "language": "en",
                "speaker": record.get("speaker"),
                "source_title": record.get("source_title"),
                "source_url": record.get("source_url"),
                "duration_s": record.get("duration_s"),
                "snr_db": record.get("snr_db"),
                "license": record.get("license"),
            })

    rng = random.Random(20261020)
    train, validation, test = [], [], []
    for row in rows:
        title = row.get("source_title") or ""
        # Reserve an entire lecture as the test source. Randomly hold out 10% of
        # the other lecture's utterances for validation.
        if "Part 5" in title:
            row["split"] = "test"
            test.append(row)
        elif rng.random() < 0.10:
            row["split"] = "validation"
            validation.append(row)
        else:
            row["split"] = "train"
            train.append(row)

    for name, selected in (("train", train), ("validation", validation), ("test", test)):
        manifest = DATASET_DIR / f"{name}.jsonl"
        manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    stats = {
        "dataset": "auraCodes/indian-english-hindi-tts-60min",
        "source_card": "https://huggingface.co/datasets/auraCodes/indian-english-hindi-tts-60min",
        "license_claimed_by_dataset_card": "CC-BY-4.0",
        "citation_and_provenance": "See README_upstream.md; English clips are from one NPTEL psychrometry lecturer and include per-clip Creative Commons Attribution provenance.",
        "counts": dict(Counter(row["split"] for row in rows)),
        "duration_hours": round(sum(float(row.get("duration_s") or 0) for row in rows) / 3600, 3),
        "speaker_count": len({row.get("speaker") for row in rows}),
        "split_policy": "Psychrometry Part 5 held out for test; 10% of Parts 1 and 2 held out for validation.",
        "warning": "General educational speech, not payment-domain speech; verify transcript/audio alignment and license provenance before publication or redistribution.",
    }
    (DATASET_DIR / "dataset_provenance.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
