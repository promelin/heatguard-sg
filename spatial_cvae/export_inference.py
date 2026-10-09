"""Export a compact, inference-only spatial CVAE checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {
        name: value.half() if torch.is_floating_point(value) else value
        for name, value in source["model_state"].items()
    }
    payload = {
        "format": "heatguard-spatial-cvae-inference-v1",
        "model_config": source["model_config"],
        "model_state": state,
        "dataset_manifest": source.get("dataset_manifest", {}),
        "training": {
            "epoch": source.get("epoch"),
            "best_val_loss": source.get("best_val_loss"),
            "args": source.get("args", {}),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    metadata = {
        "format": payload["format"],
        "file": args.output.name,
        "bytes": args.output.stat().st_size,
        "sha256": digest,
        "model_config": payload["model_config"],
        "training": payload["training"],
    }
    metadata_path = args.metadata or args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
