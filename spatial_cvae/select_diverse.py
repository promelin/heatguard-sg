"""Select a spatial CVAE using validation quality and sampled-plan diversity."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for run_dir in sorted(path for path in args.results.iterdir() if path.is_dir()):
        metrics_path = run_dir / "metrics.jsonl"
        evaluation_path = run_dir / f"candidates_100_sample{args.sample_index}.evaluation.json"
        checkpoint = run_dir / "best.pt"
        if not metrics_path.exists() or not evaluation_path.exists() or not checkpoint.exists():
            continue
        records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        best = min(records, key=lambda row: row["val"]["total"])
        evaluation = json.loads(evaluation_path.read_text())
        coverage_error = abs(
            evaluation["building_coverage"]["mean"] - evaluation["building_coverage"]["target"]
        )
        unique_fraction = evaluation["unique_fraction"]
        pixel_std = evaluation["mean_pixel_std_all_channels"]
        hard_overlap = evaluation["hard_exclusion_building_overlap_max"]
        feasible = unique_fraction >= 0.90 and pixel_std >= 0.002 and hard_overlap <= 1e-6
        score = (
            best["val"]["total"]
            + 0.50 * coverage_error
            + 0.20 * (1.0 - unique_fraction)
            + 20.0 * max(0.0, 0.005 - pixel_std)
            + 10.0 * hard_overlap
        )
        rows.append(
            {
                "run": run_dir.name,
                "feasible": feasible,
                "selection_score": score,
                "best_epoch": best["epoch"],
                "best_val_loss": best["val"]["total"],
                "mean_binary_dice": best["val"]["mean_binary_dice"],
                "coverage_error": coverage_error,
                "candidate_evaluation": evaluation,
                "checkpoint": str(checkpoint),
            }
        )
    if not rows:
        raise SystemExit("No evaluated diversity sweep runs found")
    rows.sort(key=lambda row: (not row["feasible"], row["selection_score"]))
    payload = {
        "selection_rule": "feasibility gate, then validation/diversity/coverage composite",
        "thresholds": {"unique_fraction": 0.90, "mean_pixel_std": 0.002, "hard_overlap": 1e-6},
        "best": rows[0],
        "runs": rows,
    }
    summary = args.results / "diversity_selection.json"
    summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    selected = args.results / "selected_diverse_best.pt"
    shutil.copy2(rows[0]["checkpoint"], selected)
    print(json.dumps(payload, indent=2))
    print(f"SELECTED {selected}")


if __name__ == "__main__":
    main()
