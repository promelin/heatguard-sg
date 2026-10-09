"""Measure candidate diversity, density fidelity and hard-constraint compliance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    conditions = np.load(args.data / manifest["files"]["conditions"], mmap_mode="r")
    condition = conditions[args.sample_index].astype(np.float32) / 255.0
    candidates = np.load(args.candidates, mmap_mode="r")
    site = condition[0] > 0.5
    hard = condition[5] > 0.5
    editable = condition[1] > 0.5
    target_coverage = float(condition[13][site].mean() * 0.35)
    channel_std = (candidates.astype(np.float32) / 255.0).std(axis=0)
    mean_std = [float(channel[site].mean()) for channel in channel_std]
    building = candidates[:, 0] >= 128
    coverages = building[:, site].mean(axis=1)
    coverage_errors = np.abs(coverages - target_coverage)
    violation = building[:, hard].mean(axis=1) if hard.any() else np.zeros(len(building))
    hashes = {
        hashlib.sha1(np.packbits(item[:, editable]).tobytes()).hexdigest()
        for item in (candidates >= 128)
    }
    result = {
        "candidate_count": int(len(candidates)),
        "unique_binarized_candidates": len(hashes),
        "unique_fraction": len(hashes) / len(candidates),
        "mean_pixel_std_by_channel": dict(zip(manifest["target_names"], mean_std)),
        "mean_pixel_std_all_channels": float(np.mean(mean_std)),
        "building_coverage": {
            "target": target_coverage,
            "mean": float(coverages.mean()),
            "std": float(coverages.std()),
            "min": float(coverages.min()),
            "max": float(coverages.max()),
            "mean_absolute_error": float(coverage_errors.mean()),
            "median_absolute_error": float(np.median(coverage_errors)),
            "within_0_02": int((coverage_errors <= 0.02).sum()),
            "within_0_05": int((coverage_errors <= 0.05).sum()),
            "within_0_10": int((coverage_errors <= 0.10).sum()),
        },
        "hard_exclusion_building_overlap_mean": float(violation.mean()),
        "hard_exclusion_building_overlap_max": float(violation.max()),
    }
    output = args.output or args.candidates.with_suffix(".evaluation.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
