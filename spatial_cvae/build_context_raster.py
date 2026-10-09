"""Build a compact Singapore-wide context raster for arbitrary-site inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from shapely.geometry import box

from .make_dataset import HARD_EXCLUSION_LAND_USE, burn, combine, read_layer


BAND_NAMES = [
    "conserved_building",
    "water",
    "nature",
    "hard_exclusion",
    "building",
    "road",
    "walk_cycle",
    "green_context",
    "water_context",
    "environment_land_use",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.processed_root.resolve()
    source_path = root / "rasters/worldcover_singapore_10m_epsg3414.tif"
    layers = {
        "land_use": read_layer(root, "context/land_use_mp2025"),
        "buildings": read_layer(root, "context/buildings_mp2025"),
        "conserved": read_layer(root, "preserve/conserved_buildings"),
        "nature": read_layer(root, "preserve/nature_boundaries"),
        "tree_conservation": read_layer(root, "preserve/tree_conservation_areas"),
        "heritage_trees": read_layer(root, "preserve/heritage_trees"),
        "water": read_layer(root, "blue_green/waterbodies"),
        "nature_reserves": read_layer(root, "blue_green/parks_nature_reserves"),
        "roads": read_layer(root, "networks/roads"),
        "footpaths": read_layer(root, "networks/footpaths"),
        "covered_links": read_layer(root, "networks/covered_linkways"),
        "cycling": read_layer(root, "networks/cycling_paths"),
        "park_connectors": read_layer(root, "networks/park_connectors"),
        "pedestrian_links": read_layer(root, "networks/pedestrian_links"),
    }
    land_use_classes = sorted(layers["land_use"]["LU_DESC"].dropna().unique().tolist())
    land_use_code = {name: index + 1 for index, name in enumerate(land_use_classes)}
    layers["land_use"]["environment_code"] = (
        layers["land_use"]["LU_DESC"].map(land_use_code).fillna(0).astype(float)
        / len(land_use_code)
    )
    exclusion_land = layers["land_use"][
        layers["land_use"]["LU_DESC"].isin(HARD_EXCLUSION_LAND_USE)
    ].copy()

    with rasterio.open(source_path) as source:
        shape = (source.height, source.width)
        transform = source.transform
        bounds = source.bounds
        extent = box(*source.bounds)
        wc = source.read(1)

        def layer(name: str, *, value_column: str | None = None, point_buffer_m: float = 0.0):
            print(f"RASTERISE {name}", flush=True)
            return burn(
                layers[name], extent, transform, shape,
                value_column=value_column, point_buffer_m=point_buffer_m,
            )

        conserved = layer("conserved")
        water = layer("water")
        nature = combine(
            layer("nature"),
            layer("nature_reserves"),
            layer("tree_conservation"),
            layer("heritage_trees", point_buffer_m=20),
        )
        land_exclusion = burn(exclusion_land, extent, transform, shape)
        hard_exclusion = combine(conserved, water, nature, land_exclusion)
        building = layer("buildings")
        road = layer("roads")
        walk_cycle = combine(
            layer("footpaths"),
            layer("covered_links"),
            layer("pedestrian_links"),
            layer("cycling"),
            layer("park_connectors"),
        )
        green_context = np.isin(wc, [10, 20, 30, 40, 90, 95]).astype(np.float32)
        water_context = (wc == 80).astype(np.float32)
        land_use = burn(
            layers["land_use"], extent, transform, shape,
            value_column="environment_code",
        )
        data = np.stack([
            conserved,
            water,
            nature,
            hard_exclusion,
            building,
            road,
            walk_cycle,
            green_context,
            water_context,
            land_use,
        ])
        data = np.rint(np.clip(data, 0, 1) * 255).astype(np.uint8)
        profile = source.profile.copy()
        profile.update(
            count=len(BAND_NAMES),
            dtype="uint8",
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            nodata=0,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(args.output, "w", **profile) as target:
            target.write(data)
            for index, name in enumerate(BAND_NAMES, start=1):
                target.set_band_description(index, name)

    payload = {
        "version": "1.0",
        "crs": "EPSG:3414",
        "pixel_size_m": 10,
        "width": int(shape[1]),
        "height": int(shape[0]),
        "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
        "band_names": BAND_NAMES,
        "land_use_codes": land_use_code,
        "source": "Singapore public planning, built-environment and WorldCover layers",
    }
    args.metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE {args.output} ({args.output.stat().st_size} bytes)")
    print(f"WROTE {args.metadata}")


if __name__ == "__main__":
    main()
