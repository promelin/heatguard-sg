#!/usr/bin/env python3
"""Export lazy-loaded planning-area building footprints for the web planner."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
from shapely.geometry import mapping


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def rounded_coordinates(value, digits: int = 7):
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (int, float)):
            return [round(float(item), digits) for item in value]
        return [rounded_coordinates(item, digits) for item in value]
    return value


def compact_geometry(geometry) -> dict:
    value = mapping(geometry)
    value["coordinates"] = rounded_coordinates(value["coordinates"])
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    buildings = gpd.read_parquet(
        args.data_root / "vectors" / "context" / "buildings_mp2025.parquet"
    )
    areas = gpd.read_parquet(
        args.data_root / "vectors" / "site" / "planning_areas.parquet"
    ).to_crs(buildings.crs)
    buildings = buildings[["BLDG_TYPE", "geometry"]].copy()
    buildings.geometry = buildings.geometry.simplify(0.2, preserve_topology=True)
    spatial_index = buildings.sindex
    output_dir = args.output / "planner-buildings"
    index: dict[str, dict] = {}

    for area in areas.itertuples():
        candidate_ids = list(spatial_index.query(area.geometry, predicate="intersects"))
        selected = buildings.iloc[candidate_ids].to_crs("EPSG:4326")
        filename = f"planner-buildings/{slug(area.display_name)}.geojson"
        features = []
        for row in selected.itertuples():
            features.append({
                "type": "Feature",
                "properties": {"type": str(row.BLDG_TYPE or "Building")},
                "geometry": compact_geometry(row.geometry),
            })
        write_json(args.output / filename, {"type": "FeatureCollection", "features": features})
        index[str(area.id)] = {
            "name": str(area.display_name),
            "file": filename,
            "buildingCount": len(features),
        }

    write_json(args.output / "planner-buildings-index.json", index)
    print(json.dumps({
        "planningAreas": len(index),
        "uniqueBuildings": len(buildings),
        "assignedBuildingFeatures": sum(row["buildingCount"] for row in index.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
