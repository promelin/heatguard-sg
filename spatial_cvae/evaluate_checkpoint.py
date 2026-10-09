"""Evaluate a trained checkpoint on the held-out planning-area test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .model import model_from_checkpoint
from .train import SpatialLayoutDataset, evaluate, split_by_planning_area


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    dataset = SpatialLayoutDataset(args.data.resolve())
    train_args = checkpoint.get("args", {})
    seed = int(train_args.get("seed", 20261009))
    beta = float(train_args.get("beta", 0.003))
    split = split_by_planning_area(dataset.metadata, seed)
    test_indexes = split["test"]
    loader = DataLoader(
        Subset(dataset, test_indexes),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = model_from_checkpoint(checkpoint, device=device).eval()
    metrics = evaluate(model, loader, device, beta)
    test_areas = sorted({dataset.metadata[index]["planning_area"] for index in test_indexes})
    payload = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "split_method": "planning-area holdout",
        "test_samples": len(test_indexes),
        "test_planning_areas": test_areas,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
