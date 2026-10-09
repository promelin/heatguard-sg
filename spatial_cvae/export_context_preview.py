"""Render the inference context raster as a browser-ready training-data basemap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def blend(canvas: np.ndarray, colour: tuple[int, int, int], mask: np.ndarray) -> None:
    alpha = np.clip(mask, 0.0, 1.0)[..., None]
    canvas[:] = canvas * (1.0 - alpha) + np.asarray(colour, dtype=np.float32) * alpha


def main() -> None:
    args = parse_args()
    with rasterio.open(args.context) as source:
        data = source.read().astype(np.float32) / 255.0
        bounds = source.bounds
        to_wgs84 = Transformer.from_crs(source.crs, "EPSG:4326", always_xy=True)
    conserved, water, nature, _hard, building, road, walk_cycle, green, context_water, land_use = data
    visible_land = np.maximum.reduce([land_use, building, road, walk_cycle, green, nature])
    canvas = np.empty((data.shape[1], data.shape[2], 3), dtype=np.float32)
    canvas[:] = (246, 241, 236)
    blend(canvas, (232, 226, 219), np.clip(land_use * 0.42, 0, 0.42))
    blend(canvas, (142, 174, 136), green * 0.72)
    blend(canvas, (86, 139, 101), nature * 0.92)
    blend(canvas, (113, 165, 181), np.maximum(water, context_water) * 0.95)
    blend(canvas, (193, 170, 156), road * 0.74)
    blend(canvas, (43, 126, 127), walk_cycle * 0.82)
    blend(canvas, (100, 82, 82), building * 0.82)
    blend(canvas, (122, 37, 52), conserved * 0.96)
    alpha = np.where(visible_land > 0, 244, 0).astype(np.uint8)
    rgba = np.concatenate([
        np.rint(np.clip(canvas, 0, 255)).astype(np.uint8),
        alpha[..., None],
    ], axis=-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(args.output, optimize=True)
    left, bottom = to_wgs84.transform(bounds.left, bounds.bottom)
    right, top = to_wgs84.transform(bounds.right, bounds.top)
    payload = {
        "source": "spatial_context.tif",
        "bounds": {"minLon": left, "minLat": bottom, "maxLon": right, "maxLat": top},
        "width": int(rgba.shape[1]),
        "height": int(rgba.shape[0]),
        "layers": ["buildings", "roads", "walking and cycling", "vegetation", "water", "conserved buildings"],
    }
    args.metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE {args.output} ({args.output.stat().st_size} bytes)")
    print(f"WROTE {args.metadata}")


if __name__ == "__main__":
    main()
