"""Summarise CVAE sweep runs and select the best validation checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for metrics_path in sorted(args.results.glob("*/metrics.jsonl")):
        records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        if not records:
            continue
        best = min(records, key=lambda row: row["val"]["total"])
        peak_dice = max(records, key=lambda row: row["val"]["mean_binary_dice"])
        rows.append(
            {
                "run": metrics_path.parent.name,
                "epochs": len(records),
                "best_epoch": best["epoch"],
                "best_val_loss": best["val"]["total"],
                "dice_at_best_loss": best["val"]["mean_binary_dice"],
                "peak_dice_epoch": peak_dice["epoch"],
                "peak_mean_binary_dice": peak_dice["val"]["mean_binary_dice"],
                "best_metrics": best["val"],
                "checkpoint": str(metrics_path.parent / "best.pt"),
            }
        )
    if not rows:
        raise SystemExit("No completed sweep runs found")
    rows.sort(key=lambda row: (row["best_val_loss"], -row["dice_at_best_loss"]))
    output = args.output or args.results / "sweep_summary.json"
    payload = {"selection_rule": "minimum spatial-validation total loss", "best": rows[0], "runs": rows}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    selected = args.results / "selected_best.pt"
    shutil.copy2(rows[0]["checkpoint"], selected)
    print(json.dumps(payload, indent=2))
    print(f"SELECTED {selected}")


if __name__ == "__main__":
    main()
