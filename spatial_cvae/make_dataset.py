"""Rasterise Singapore planning layers into conditional-layout training tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import Point, box


CONDITION_NAMES = [
    "site_mask",
    "editable_mask",
    "must_retain_building",
    "must_retain_water",
    "must_retain_nature",
    "hard_exclusion",
    "context_building",
    "context_road",
    "context_walk_cycle",
    "context_green",
    "context_water",
    "environment_land_use",
    "target_dwelling_density",
    "target_building_coverage",
]

TARGET_NAMES = [
    "hdb_footprint",
    "hdb_height_normalized",
    "park_green",
    "forest_nature",
    "community_green",
    "water_blue_green",
    "road",
    "walking_network",
    "cycling_network",
    "school_facility",
    "eldercare_facility",
    "community_facility",
    "public_open_space",
]

HARD_EXCLUSION_LAND_USE = {
    "WATERBODY",
    "ROAD",
    "MASS RAPID TRANSIT",
    "LIGHT RAPID TRANSIT",
    "TRANSPORT FACILITIES",
    "PARK",
    "OPEN SPACE",
    "BEACH AREA",
    "CEMETERY",
    "PORT / AIRPORT",
}


def read_layer(root: Path, relative: str) -> gpd.GeoDataFrame:
    return gpd.read_parquet(root / "vectors" / f"{relative}.parquet")


def subset(gdf: gpd.GeoDataFrame, tile_geometry) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    indexes = gdf.sindex.query(tile_geometry, predicate="intersects")
    return gdf.iloc[indexes]


def burn(
    gdf: gpd.GeoDataFrame,
    tile_geometry,
    transform,
    shape: tuple[int, int],
    value: float = 1.0,
    value_column: str | None = None,
    point_buffer_m: float = 0.0,
) -> np.ndarray:
    selected = subset(gdf, tile_geometry)
    if selected.empty:
        return np.zeros(shape, dtype=np.float32)
    geometries = selected.geometry
    if point_buffer_m:
        geometries = geometries.buffer(point_buffer_m)
    if value_column:
        values = selected[value_column].fillna(0).astype(float).to_numpy()
        shapes = ((geometry, float(item)) for geometry, item in zip(geometries, values) if not geometry.is_empty)
    else:
        shapes = ((geometry, value) for geometry in geometries if not geometry.is_empty)
    return rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0.0,
        all_touched=True,
        dtype="float32",
    )


def combine(*arrays: np.ndarray) -> np.ndarray:
    return np.maximum.reduce(arrays) if len(arrays) > 1 else arrays[0]


def mask_outside(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return array * mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--min-hdb-blocks", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.processed_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tile_size = args.tile_size
    shape = (tile_size, tile_size)

    layers = {
        "subzones": read_layer(root, "site/subzones"),
        "land_use": read_layer(root, "context/land_use_mp2025"),
        "buildings": read_layer(root, "context/buildings_mp2025"),
        "conserved": read_layer(root, "preserve/conserved_buildings"),
        "nature": read_layer(root, "preserve/nature_boundaries"),
        "tree_conservation": read_layer(root, "preserve/tree_conservation_areas"),
        "heritage_trees": read_layer(root, "preserve/heritage_trees"),
        "water": read_layer(root, "blue_green/waterbodies"),
        "parks": read_layer(root, "blue_green/parks_mp2025"),
        "nature_reserves": read_layer(root, "blue_green/parks_nature_reserves"),
        "community_gardens": read_layer(root, "blue_green/community_gardens"),
        "abc_completed": read_layer(root, "blue_green/abc_waters_completed"),
        "roads": read_layer(root, "networks/roads"),
        "footpaths": read_layer(root, "networks/footpaths"),
        "covered_links": read_layer(root, "networks/covered_linkways"),
        "cycling": read_layer(root, "networks/cycling_paths"),
        "park_connectors": read_layer(root, "networks/park_connectors"),
        "pedestrian_links": read_layer(root, "networks/pedestrian_links"),
        "schools": read_layer(root, "facilities/schools"),
        "eldercare": read_layer(root, "facilities/eldercare"),
        "community_clubs": read_layer(root, "facilities/community_clubs"),
        "public_spaces": read_layer(root, "outputs/public_spaces"),
        "hdb": read_layer(root, "outputs/hdb_blocks_enriched"),
    }
    layers["hdb"] = layers["hdb"][layers["hdb"]["residential"].eq("Y")].copy()
    layers["hdb"]["height_normalized"] = (
        layers["hdb"]["max_floor_lvl"].fillna(0).astype(float) / 60.0
    ).clip(0, 1)

    land_use_classes = sorted(layers["land_use"]["LU_DESC"].dropna().unique().tolist())
    land_use_code = {name: index + 1 for index, name in enumerate(land_use_classes)}
    layers["land_use"]["environment_code"] = (
        layers["land_use"]["LU_DESC"].map(land_use_code).fillna(0).astype(float) / len(land_use_code)
    )
    exclusion_land = layers["land_use"][
        layers["land_use"]["LU_DESC"].isin(HARD_EXCLUSION_LAND_USE)
    ].copy()
    open_space_land = layers["land_use"][layers["land_use"]["LU_DESC"].eq("OPEN SPACE")].copy()

    subzone_lookup = {
        str(row["name"]).upper(): row.geometry for _, row in layers["subzones"].iterrows()
    }
    seed_rows = layers["hdb"].reset_index(drop=True)

    conditions_path = output / "conditions.npy"
    targets_path = output / "targets.npy"
    conditions = np.lib.format.open_memmap(
        conditions_path,
        mode="w+",
        dtype=np.uint8,
        shape=(args.num_samples, len(CONDITION_NAMES), tile_size, tile_size),
    )
    targets = np.lib.format.open_memmap(
        targets_path,
        mode="w+",
        dtype=np.uint8,
        shape=(args.num_samples, len(TARGET_NAMES), tile_size, tile_size),
    )
    metadata = []
    attempts = 0
    worldcover_path = root / "rasters/worldcover_singapore_10m_epsg3414.tif"

    with rasterio.open(worldcover_path) as worldcover:
        while len(metadata) < args.num_samples:
            attempts += 1
            if attempts > args.num_samples * 30:
                raise RuntimeError("Unable to create enough valid HDB neighbourhood samples")
            seed = seed_rows.iloc[int(rng.integers(0, len(seed_rows)))]
            centroid = seed.geometry.centroid
            center_x = centroid.x + float(rng.uniform(-180, 180))
            center_y = centroid.y + float(rng.uniform(-180, 180))
            col = int(round((center_x - worldcover.transform.c) / worldcover.transform.a))
            row = int(round((worldcover.transform.f - center_y) / abs(worldcover.transform.e)))
            half = tile_size // 2
            window = Window(col - half, row - half, tile_size, tile_size)
            if (
                window.col_off < 0
                or window.row_off < 0
                or window.col_off + tile_size > worldcover.width
                or window.row_off + tile_size > worldcover.height
            ):
                continue
            minx, miny, maxx, maxy = window_bounds(window, worldcover.transform)
            tile_geometry = box(minx, miny, maxx, maxy)
            transform = rasterio.windows.transform(window, worldcover.transform)
            wc = worldcover.read(1, window=window)

            subzone_name = str(seed.get("subzone", "")).upper()
            subzone_geometry = subzone_lookup.get(subzone_name)
            if subzone_geometry is None:
                matches = subset(layers["subzones"], Point(center_x, center_y))
                if matches.empty:
                    continue
                subzone_geometry = matches.iloc[0].geometry
                subzone_name = str(matches.iloc[0]["name"])
            radius = float(rng.uniform(520, 980))
            site_geometry = subzone_geometry.intersection(Point(center_x, center_y).buffer(radius))
            if site_geometry.is_empty or site_geometry.area < 350_000:
                continue

            site = rasterize(
                [(site_geometry, 1.0)],
                out_shape=shape,
                transform=transform,
                fill=0.0,
                all_touched=True,
                dtype="float32",
            )
            if site.sum() < 3500:
                continue

            hdb_in_site = subset(layers["hdb"], site_geometry)
            hdb_in_site = hdb_in_site[hdb_in_site.geometry.intersects(site_geometry)]
            if len(hdb_in_site) < args.min_hdb_blocks:
                continue

            hdb_footprint = burn(layers["hdb"], tile_geometry, transform, shape)
            hdb_height = burn(
                layers["hdb"], tile_geometry, transform, shape, value_column="height_normalized"
            )
            building = burn(layers["buildings"], tile_geometry, transform, shape)
            conserved = burn(layers["conserved"], tile_geometry, transform, shape)
            water = burn(layers["water"], tile_geometry, transform, shape)
            abc_water = burn(layers["abc_completed"], tile_geometry, transform, shape)
            parks = burn(layers["parks"], tile_geometry, transform, shape)
            nature = combine(
                burn(layers["nature"], tile_geometry, transform, shape),
                burn(layers["nature_reserves"], tile_geometry, transform, shape),
                burn(layers["tree_conservation"], tile_geometry, transform, shape),
                burn(layers["heritage_trees"], tile_geometry, transform, shape, point_buffer_m=20),
            )
            road = burn(layers["roads"], tile_geometry, transform, shape)
            walking = combine(
                burn(layers["footpaths"], tile_geometry, transform, shape),
                burn(layers["covered_links"], tile_geometry, transform, shape),
                burn(layers["pedestrian_links"], tile_geometry, transform, shape),
            )
            cycling = combine(
                burn(layers["cycling"], tile_geometry, transform, shape),
                burn(layers["park_connectors"], tile_geometry, transform, shape),
            )
            community_green = burn(
                layers["community_gardens"], tile_geometry, transform, shape, point_buffer_m=35
            )
            school = burn(layers["schools"], tile_geometry, transform, shape, point_buffer_m=40)
            eldercare = burn(layers["eldercare"], tile_geometry, transform, shape, point_buffer_m=40)
            community = burn(
                layers["community_clubs"], tile_geometry, transform, shape, point_buffer_m=40
            )
            public_open = combine(
                burn(layers["public_spaces"], tile_geometry, transform, shape, point_buffer_m=45),
                burn(open_space_land, tile_geometry, transform, shape),
            )
            exclusion = burn(exclusion_land, tile_geometry, transform, shape)
            hard_exclusion = combine(conserved, water, nature, exclusion)
            editable = site * (1.0 - np.clip(hard_exclusion, 0, 1))
            outside_site = 1.0 - site

            land_use_environment = burn(
                layers["land_use"],
                tile_geometry,
                transform,
                shape,
                value_column="environment_code",
            )
            green_context = np.isin(wc, [10, 20, 30, 40, 90, 95]).astype(np.float32)
            tree_cover = np.isin(wc, [10, 95]).astype(np.float32)
            water_context = (wc == 80).astype(np.float32)

            site_area_ha = site_geometry.area / 10_000.0
            dwelling_units = float(hdb_in_site["total_dwelling_units"].fillna(0).sum())
            dwelling_density = dwelling_units / max(site_area_ha, 1e-6)
            footprint_coverage = float((hdb_footprint * site).sum() / max(site.sum(), 1.0))
            density_condition = np.clip(dwelling_density / 250.0, 0, 1)
            coverage_condition = np.clip(footprint_coverage / 0.35, 0, 1)

            condition = np.stack(
                [
                    site,
                    editable,
                    conserved * site,
                    water * site,
                    nature * site,
                    hard_exclusion * site,
                    building * outside_site,
                    road * outside_site,
                    combine(walking, cycling) * outside_site,
                    green_context * outside_site,
                    combine(water, water_context) * outside_site,
                    land_use_environment,
                    site * density_condition,
                    site * coverage_condition,
                ]
            )
            target = np.stack(
                [
                    mask_outside(hdb_footprint, site),
                    mask_outside(hdb_height, site),
                    mask_outside(parks, site),
                    mask_outside(tree_cover, site),
                    mask_outside(community_green, site),
                    mask_outside(combine(water, abc_water), site),
                    mask_outside(road, site),
                    mask_outside(walking, site),
                    mask_outside(cycling, site),
                    mask_outside(school, site),
                    mask_outside(eldercare, site),
                    mask_outside(community, site),
                    mask_outside(public_open, site),
                ]
            )
            index = len(metadata)
            conditions[index] = np.rint(np.clip(condition, 0, 1) * 255).astype(np.uint8)
            targets[index] = np.rint(np.clip(target, 0, 1) * 255).astype(np.uint8)
            metadata.append(
                {
                    "sample_id": index,
                    "planning_area": str(seed.get("planning_area", "")),
                    "subzone": subzone_name,
                    "center_epsg3414": [round(center_x, 3), round(center_y, 3)],
                    "site_area_ha": round(site_area_ha, 3),
                    "hdb_blocks": int(len(hdb_in_site)),
                    "dwelling_units": round(dwelling_units, 1),
                    "dwelling_density_per_ha": round(dwelling_density, 3),
                    "hdb_footprint_coverage": round(footprint_coverage, 5),
                }
            )
            if len(metadata) % 25 == 0 or len(metadata) == args.num_samples:
                conditions.flush()
                targets.flush()
                print(
                    f"SAMPLES {len(metadata)}/{args.num_samples}; attempts={attempts}",
                    flush=True,
                )

    del conditions
    del targets
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest = {
        "version": "1.0",
        "seed": args.seed,
        "num_samples": args.num_samples,
        "tile_size": tile_size,
        "pixel_size_m": 10,
        "crs": "EPSG:3414",
        "condition_names": CONDITION_NAMES,
        "target_names": TARGET_NAMES,
        "land_use_codes": land_use_code,
        "normalization": {
            "height_floors_max": 60,
            "dwelling_density_per_ha_max": 250,
            "hdb_footprint_coverage_max": 0.35,
        },
        "files": {"conditions": "conditions.npy", "targets": "targets.npy", "metadata": "metadata.json"},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
