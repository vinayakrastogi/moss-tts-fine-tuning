#!/usr/bin/env python3
"""Create/validate a small Indian-English payment TTS recording corpus."""
from __future__ import annotations

import argparse
import csv
import json
import random
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORDS_0_19 = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def under_thousand(n: int) -> str:
    parts = []
    if n >= 100:
        parts.extend((WORDS_0_19[n // 100], "hundred"))
        n %= 100
    if n >= 20:
        parts.append(TENS[n // 10])
        n %= 10
    if n:
        parts.append(WORDS_0_19[n])
    return " ".join(parts)


def indian_number_words(n: int) -> str:
    if n == 0:
        return "zero"
    groups = [(10_000_000, "crore"), (100_000, "lakh"), (1_000, "thousand"), (100, "hundred")]
    parts = []
    for divisor, label in groups:
        count, n = divmod(n, divisor)
        if count:
            parts.extend((under_thousand(count), label))
    if n:
        parts.append(under_thousand(n))
    return " ".join(parts)


def make_rows(count: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    # Include boundaries and Indian grouping cases, then sample unique values.
    amounts = {0, 1, 9, 10, 19, 20, 99, 100, 101, 999, 1_000, 1_001,
               9_999, 10_000, 99_999, 1_00_000, 1_00_001, 9_99_999,
               10_00_000, 99_99_999, 1_00_00_000, 12_34_56_789}
    while len(amounts) < count:
        bucket = rng.random()
        if bucket < 0.65:
            value = rng.randint(10, 99_999)
        elif bucket < 0.95:
            value = rng.randint(1_00_000, 99_99_999)
        else:
            value = rng.randint(1_00_00_000, 99_99_99_999)
        amounts.add(value)
    amounts = list(amounts)
    rng.shuffle(amounts)
    templates = [
        ("Payment received: INR {formatted}.", "Payment received, {spoken} rupees."),
        ("You have received INR {formatted}.", "You have received {spoken} rupees."),
        ("INR {formatted} has been credited to your account.", "{spoken} rupees has been credited to your account."),
        ("Your payment of INR {formatted} was successful.", "Your payment of {spoken} rupees was successful."),
        ("Refund initiated for INR {formatted}.", "Refund initiated for {spoken} rupees."),
        ("Your account balance is INR {formatted}.", "Your account balance is {spoken} rupees."),
    ]
    rows = []
    for i, amount in enumerate(amounts[:count], 1):
        source, spoken = templates[(i - 1) % len(templates)]
        indian_grouping = f"{amount:,}"
        # Reformat grouping for the Indian convention (last 3 digits, then pairs).
        digits = str(amount)
        if len(digits) > 3:
            indian_grouping = digits[-3:]
            rest = digits[:-3]
            chunks = []
            while rest:
                chunks.append(rest[-2:])
                rest = rest[:-2]
            indian_grouping = ",".join(reversed(chunks)) + "," + indian_grouping
        source_text = source.format(formatted=indian_grouping)
        spoken_text = spoken.format(spoken=indian_number_words(amount))
        rows.append({"id": f"utt_{i:04d}", "split": "", "amount_inr": str(amount),
                     "text": source_text, "record_exactly": spoken_text,
                     "audio": f"audio/utt_{i:04d}.wav"})
    # Hold out entire amount bands for validation and test, reducing leakage.
    for row in rows:
        amount = int(row["amount_inr"])
        if amount % 7 == 0:
            row["split"] = "test"
        elif amount % 5 == 0:
            row["split"] = "validation"
        else:
            row["split"] = "train"
    return rows


def write_manifest(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_and_export(manifest: Path, output: Path) -> None:
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    ready = []
    missing = []
    for row in rows:
        if row["split"] == "train":
            path = (manifest.parent / row["audio"]).resolve()
            if not path.is_file():
                missing.append(path)
                continue
            try:
                with wave.open(str(path), "rb") as wav:
                    if wav.getnframes() < wav.getframerate() // 2:
                        raise ValueError("audio shorter than 0.5 seconds")
                    if wav.getnchannels() not in (1, 2):
                        raise ValueError("expected mono or stereo audio")
            except (wave.Error, EOFError) as exc:
                raise ValueError(f"Invalid WAV: {path}: {exc}") from exc
            ready.append({"audio": str(path), "text": row["text"], "language": "en"})
    if missing:
        print(f"Need recordings for {len(missing)} training prompts. Example: {missing[0]}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ready), encoding="utf-8")
    print(f"Wrote {len(ready)} training examples to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true", help="Validate recorded WAVs and export raw MOSS JSONL")
    args = parser.parse_args()
    manifest = ROOT / "manifest.csv"
    if args.validate:
        validate_and_export(manifest, ROOT / "data" / "train_raw.jsonl")
    else:
        rows = make_rows(args.count, args.seed)
        write_manifest(rows, manifest)
        counts = {key: sum(row["split"] == key for row in rows) for key in ("train", "validation", "test")}
        print(f"Created {manifest} with {len(rows)} prompts: {counts}")


if __name__ == "__main__":
    main()
