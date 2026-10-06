#!/usr/bin/env python3
"""Strictly load the production TFT class and run a shape-level inference check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--module-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    sys.path.insert(0, str(args.module_dir))
    from backend.tft_model import FullTemporalFusionTransformer

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    configuration = checkpoint.get("arguments", {})
    model = FullTemporalFusionTransformer(
        history_variables=len(metadata["dynamic_names"]),
        future_real_variables=len(metadata["future_real_names"]),
        future_cat_cardinalities=[],
        stations=len(metadata["stations"]),
        hidden_size=int(configuration.get("hidden_size", 64)),
        heads=int(configuration.get("heads", 4)),
        dropout=float(configuration.get("dropout", 0.1)),
        cooling_gate_enabled=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    batch = 2
    with torch.inference_mode():
        output = model(
            torch.zeros(batch, 168, 23),
            torch.zeros(batch, 24, 5),
            torch.empty(batch, 24, 0, dtype=torch.long),
            torch.arange(batch, dtype=torch.long),
            torch.tensor(metadata["station_static"][:batch], dtype=torch.float32),
        )
    if tuple(output.shape) != (batch, 24, 3):
        raise RuntimeError(f"Unexpected output shape: {tuple(output.shape)}")
    print(json.dumps({
        "strict_load": True,
        "output_shape": list(output.shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "finite": bool(torch.isfinite(output).all()),
    }))


if __name__ == "__main__":
    main()
