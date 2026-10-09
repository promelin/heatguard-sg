"""Local micro-environment heat-index estimation.

The model estimates a location's heat index under a supplied island-wide
background heat condition.  Static features are derived from nearby parks,
HDB building footprints, coastline exposure and geographic position.  It is a
spatial downscaling model, not a weather forecast and not an official alert.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np


EARTH_RADIUS_M = 6_371_000.0
CBD = (103.851959, 1.290270)
FEATURE_NAMES = [
    "background_heat_index_c",
    "park_cover_proxy_pct",
    "building_coverage_pct",
    "building_floor_area_ratio",
    "building_density_per_km2",
    "mean_building_height_floors",
    "distance_to_coast_m",
    "coastal_proximity",
    "distance_to_cbd_km",
    "latitude",
    "longitude",
]


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    value = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(value)))


def _rings(geometry: dict[str, Any]) -> Iterable[list[list[float]]]:
    coordinates = geometry.get("coordinates") or []
    if geometry.get("type") == "Polygon":
        yield from coordinates
    elif geometry.get("type") == "MultiPolygon":
        for polygon in coordinates:
            yield from polygon


def _geometry_centre(geometry: dict[str, Any]) -> tuple[float, float]:
    points = [point for ring in _rings(geometry) for point in ring]
    if not points:
        return 0.0, 0.0
    return (
        (min(point[0] for point in points) + max(point[0] for point in points)) / 2.0,
        (min(point[1] for point in points) + max(point[1] for point in points)) / 2.0,
    )


def _point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    previous = ring[-1]
    for current in ring:
        x1, y1 = previous
        x2, y2 = current
        crosses = (y1 > lat) != (y2 > lat)
        if crosses and lon < (x2 - x1) * (lat - y1) / ((y2 - y1) or 1e-12) + x1:
            inside = not inside
        previous = current
    return inside


def _point_in_geometry(lon: float, lat: float, geometry: dict[str, Any]) -> bool:
    coordinates = geometry.get("coordinates") or []
    polygons = [coordinates] if geometry.get("type") == "Polygon" else coordinates
    for polygon in polygons:
        if not polygon or not _point_in_ring(lon, lat, polygon[0]):
            continue
        if not any(_point_in_ring(lon, lat, hole) for hole in polygon[1:]):
            return True
    return False


def _distance_to_segment_m(
    lon: float, lat: float, start: tuple[float, float], end: tuple[float, float]
) -> float:
    metres_per_lon = 111_320.0 * math.cos(math.radians(lat))
    metres_per_lat = 110_540.0
    ax = (start[0] - lon) * metres_per_lon
    ay = (start[1] - lat) * metres_per_lat
    bx = (end[0] - lon) * metres_per_lon
    by = (end[1] - lat) * metres_per_lat
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(ax, ay)
    t = max(0.0, min(1.0, -(ax * dx + ay * dy) / length_sq))
    return math.hypot(ax + t * dx, ay + t * dy)


@dataclass(frozen=True)
class ParkRecord:
    lon: float
    lat: float
    area_m2: float


@dataclass(frozen=True)
class BuildingRecord:
    lon: float
    lat: float
    footprint_m2: float
    floors: float


class LocalFeatureExtractor:
    """Derive transparent spatial features within a circular search radius."""

    def __init__(self, spatial_dir: Path) -> None:
        self.spatial_dir = spatial_dir
        self.areas = json.loads((spatial_dir / "areas.geojson").read_text(encoding="utf-8"))["features"]
        park_features = json.loads((spatial_dir / "parks.geojson").read_text(encoding="utf-8"))["features"]
        self.parks = [
            ParkRecord(*_geometry_centre(item["geometry"]), float(item["properties"].get("areaKm2") or 0.0) * 1_000_000.0)
            for item in park_features
        ]
        buildings: list[BuildingRecord] = []
        for path in sorted((spatial_dir / "blocks").glob("*.geojson")):
            for feature in json.loads(path.read_text(encoding="utf-8"))["features"]:
                properties = feature["properties"]
                buildings.append(BuildingRecord(
                    float(properties.get("lon") or 0.0),
                    float(properties.get("lat") or 0.0),
                    max(0.0, float(properties.get("footprint_area_m2") or 0.0)),
                    max(1.0, float(properties.get("max_floor_lvl") or 1.0)),
                ))
        self.buildings = buildings
        self.coast_segments = self._coast_segments()

    def _coast_segments(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        counts: dict[tuple[tuple[float, float], tuple[float, float]], int] = {}
        originals: dict[tuple[tuple[float, float], tuple[float, float]], tuple[tuple[float, float], tuple[float, float]]] = {}
        for feature in self.areas:
            for ring in _rings(feature["geometry"]):
                for first, second in zip(ring, ring[1:]):
                    a = (round(float(first[0]), 6), round(float(first[1]), 6))
                    b = (round(float(second[0]), 6), round(float(second[1]), 6))
                    key = tuple(sorted((a, b)))
                    counts[key] = counts.get(key, 0) + 1
                    originals[key] = ((float(first[0]), float(first[1])), (float(second[0]), float(second[1])))
        return [originals[key] for key, count in counts.items() if count == 1]

    def planning_area(self, longitude: float, latitude: float) -> str | None:
        for feature in self.areas:
            if _point_in_geometry(longitude, latitude, feature["geometry"]):
                return str(feature["properties"].get("display_name") or feature["properties"].get("name"))
        return None

    @lru_cache(maxsize=2048)
    def extract(self, longitude: float, latitude: float, radius_m: int = 500) -> dict[str, float | str | None]:
        radius_m = int(max(200, min(2_000, radius_m)))
        circle_area_m2 = math.pi * radius_m * radius_m

        effective_park_area = 0.0
        park_count = 0
        for park in self.parks:
            distance = _haversine_m(longitude, latitude, park.lon, park.lat)
            if distance <= radius_m:
                park_count += 1
                effective_park_area += park.area_m2 * (1.0 - distance / radius_m)

        nearby_buildings: list[BuildingRecord] = []
        for building in self.buildings:
            if _haversine_m(longitude, latitude, building.lon, building.lat) <= radius_m:
                nearby_buildings.append(building)
        footprint = sum(item.footprint_m2 for item in nearby_buildings)
        floor_area = sum(item.footprint_m2 * item.floors for item in nearby_buildings)
        mean_floors = (
            sum(item.floors * item.footprint_m2 for item in nearby_buildings) / footprint
            if footprint else 0.0
        )
        coast_distance = min(
            (_distance_to_segment_m(longitude, latitude, start, end) for start, end in self.coast_segments),
            default=25_000.0,
        )
        coastal_proximity = max(0.0, 1.0 - coast_distance / radius_m)
        return {
            "longitude": round(float(longitude), 6),
            "latitude": round(float(latitude), 6),
            "radius_m": radius_m,
            "planning_area": self.planning_area(longitude, latitude),
            "park_count": park_count,
            "park_cover_proxy_pct": round(min(100.0, effective_park_area / circle_area_m2 * 100.0), 4),
            "building_count": len(nearby_buildings),
            "building_coverage_pct": round(min(100.0, footprint / circle_area_m2 * 100.0), 4),
            "building_floor_area_ratio": round(floor_area / circle_area_m2, 4),
            "building_density_per_km2": round(len(nearby_buildings) / (circle_area_m2 / 1_000_000.0), 4),
            "mean_building_height_floors": round(mean_floors, 4),
            "distance_to_coast_m": round(coast_distance, 2),
            "coastal_proximity": round(coastal_proximity, 6),
            "distance_to_cbd_km": round(_haversine_m(longitude, latitude, CBD[0], CBD[1]) / 1_000.0, 4),
        }


class LocalHeatIndexModel:
    """Load the private model artifact and return calibrated local estimates."""

    def __init__(self, model_path: Path, spatial_dir: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Local heat model is missing: {model_path}. Run scripts/train_local_heat_model.py first."
            )
        bundle = joblib.load(model_path)
        if bundle.get("schema_version") != 1:
            raise ValueError("Unsupported local heat model schema")
        self.model = bundle["model"]
        self.feature_names = list(bundle["feature_names"])
        self.validation = dict(bundle["validation"])
        self.training = dict(bundle["training"])
        self.spatial_support = dict(bundle.get("spatial_support") or {})
        self.extractor = LocalFeatureExtractor(spatial_dir)

    def predict(
        self,
        longitude: float,
        latitude: float,
        radius_m: int,
        background_heat_index_c: float,
    ) -> dict[str, Any]:
        spatial = self.extractor.extract(float(longitude), float(latitude), int(radius_m))
        values = {**spatial, "background_heat_index_c": float(background_heat_index_c)}
        vector = np.asarray([[float(values[name]) for name in self.feature_names]], dtype=float)
        raw_delta = float(self.model.predict(vector)[0])
        delta_bounds = self.training.get("delta_bounds_c", [-4.0, 4.0])
        local_delta = max(float(delta_bounds[0]), min(float(delta_bounds[1]), raw_delta))
        support_weight = 1.0
        support_distance = 0.0
        if self.spatial_support:
            support_names = self.spatial_support["feature_names"]
            profile = np.asarray([float(values[name]) for name in support_names], dtype=float)
            mean = np.asarray(self.spatial_support["mean"], dtype=float)
            std = np.asarray(self.spatial_support["std"], dtype=float)
            stations = np.asarray(self.spatial_support["station_profiles"], dtype=float)
            support_distance = float(np.sqrt(np.sum(((stations - profile) / std) ** 2, axis=1)).min())
            support_weight = math.exp(-support_distance / float(self.spatial_support.get("decay_scale", 3.0)))
            local_delta *= support_weight
        estimate = float(background_heat_index_c) + local_delta
        lower_offset = float(self.validation.get("residual_p10_c", -2.0))
        upper_offset = float(self.validation.get("residual_p90_c", 2.0))
        estimate = max(20.0, min(55.0, estimate))
        if estimate >= 40.0:
            risk = "Very high"
        elif estimate >= 37.0:
            risk = "High"
        elif estimate >= 34.0:
            risk = "Elevated"
        else:
            risk = "Watch"
        return {
            "model": "local-environment-v1",
            "estimate_c": round(estimate, 2),
            "interval_c": [round(max(20.0, estimate + lower_offset), 2), round(min(55.0, estimate + upper_offset), 2)],
            "risk": risk,
            "background_heat_index_c": round(float(background_heat_index_c), 2),
            "local_adjustment_c": round(local_delta, 2),
            "spatial_support": {
                "weight": round(support_weight, 4),
                "distance": round(support_distance, 4),
                "interpretation": "Closer to 1 means the local feature profile is better represented by training stations.",
            },
            "spatial": spatial,
            "validation": {
                "grouped_mae_c": self.validation.get("grouped_mae_c"),
                "grouped_rmse_c": self.validation.get("grouped_rmse_c"),
                "held_out_by": "weather station",
            },
            "notice": "Research estimate from local environment and a supplied background heat condition; not an observation, weather forecast or official alert.",
        }
