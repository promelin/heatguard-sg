"""Generate 100–1,000 candidate neighbourhood layouts from a trained CVAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--count", type=int, choices=[100, 1000], default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261009)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = model_from_checkpoint(checkpoint, device=device).eval()
    manifest = json.loads((args.data / "manifest.json").read_text())
    source = np.load(args.data / manifest["files"]["conditions"], mmap_mode="r")
    if not 0 <= args.sample_index < len(source):
        raise IndexError(args.sample_index)
    condition_numpy = np.array(source[args.sample_index], copy=True)
    condition = torch.from_numpy(condition_numpy).float().div_(255.0).unsqueeze(0).to(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.uint8,
        shape=(args.count, len(manifest["target_names"]), manifest["tile_size"], manifest["tile_size"]),
    )
    hard_exclusion = condition_numpy[5] > 0
    written = 0
    with torch.no_grad():
        while written < args.count:
            batch = min(args.batch_size, args.count - written)
            prediction = model.sample(condition, count=batch).cpu().numpy()
            prediction[:, 0, hard_exclusion] = 0.0
            candidates[written : written + batch] = np.rint(prediction * 255).astype(np.uint8)
            written += batch
            candidates.flush()
            print(f"GENERATED {written}/{args.count}", flush=True)
    del candidates
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "condition_data": str(args.data),
                "sample_index": args.sample_index,
                "count": args.count,
                "seed": args.seed,
                "target_names": manifest["target_names"],
                "shape": [args.count, len(manifest["target_names"]), manifest["tile_size"], manifest["tile_size"]],
                "dtype": "uint8",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
