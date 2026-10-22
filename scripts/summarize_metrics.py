#!/usr/bin/env python3
"""Summarize two paired MOSS generation manifests without inventing quality scores."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "duration_seconds", "inference_seconds", "rtf", "peak_process_rss_mb"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Missing instrumented measurements in {path}")
    result = {row["sample_id"]: row for row in rows if row["status"] == "generated"}
    if len(result) != len(rows):
        raise ValueError(f"Incomplete or duplicate samples in {path}")
    return result


def summarize(rows: dict[str, dict[str, str]]) -> dict:
    def stats(key: str) -> dict:
        values = [float(row[key]) for row in rows.values()]
        return {
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values),
        }
    return {
        "n": len(rows),
        "inference_seconds": stats("inference_seconds"),
        "audio_duration_seconds": stats("duration_seconds"),
        "rtf": stats("rtf"),
        "all_rtf_below_one": all(float(row["rtf"]) < 1 for row in rows.values()),
        "maximum_observed_process_rss_mb": max(float(row["peak_process_rss_mb"]) for row in rows.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = read_rows(args.original)
    adapted = read_rows(args.adapted)
    if original.keys() != adapted.keys():
        raise ValueError("The two manifests must contain the same sample IDs")
    result = {
        "metric_definitions": {
            "inference_seconds": "wall-clock seconds for model.inference after one warm-up",
            "rtf": "inference seconds divided by generated audio duration; <1 is faster than playback",
            "maximum_observed_process_rss_mb": "maximum sampled resident memory of the Python process during measured generations; includes runtime/codec overhead",
        },
        "sample_ids": sorted(original),
        "original": summarize(original),
        "adapted": summarize(adapted),
    }
    a = result["adapted"]["inference_seconds"]["mean"]
    b = result["original"]["inference_seconds"]["mean"]
    result["adapted_inference_time_change_percent"] = 100 * (a / b - 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
