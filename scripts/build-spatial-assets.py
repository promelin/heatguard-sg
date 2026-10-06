#!/usr/bin/env python3
"""Build compact browser assets for HeatGuard SG spatial-detail modules."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EARTH_KM_PER_DEGREE = 111.32


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def iter_points(geometry):
    coords = geometry["coordinates"]
    if geometry["type"] == "Point":
        yield coords
    elif geometry["type"] in {"LineString", "MultiPoint"}:
        yield from coords
    elif geometry["type"] in {"Polygon", "MultiLineString"}:
        for part in coords:
            yield from part
    elif geometry["type"] == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                yield from ring


def bbox(geometry):
    points = list(iter_points(geometry))
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def centroid(geometry):
    min_x, min_y, max_x, max_y = bbox(geometry)
    return [round((min_x + max_x) / 2, 6), round((min_y + max_y) / 2, 6)]


def geometry_area_km2(geometry):
    def ring_area(ring):
        if len(ring) < 3:
            return 0.0
        mean_lat = math.radians(sum(point[1] for point in ring) / len(ring))
        scale_x = EARTH_KM_PER_DEGREE * math.cos(mean_lat)
        scale_y = EARTH_KM_PER_DEGREE
        return abs(sum(
            (ring[index][0] * scale_x) * (ring[(index + 1) % len(ring)][1] * scale_y)
            - (ring[(index + 1) % len(ring)][0] * scale_x) * (ring[index][1] * scale_y)
            for index in range(len(ring))
        )) / 2

    polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    return sum(max(0.0, ring_area(polygon[0]) - sum(ring_area(hole) for hole in polygon[1:])) for polygon in polygons)


def point_in_ring(point, ring):
    x, y = point
    inside = False
    previous = ring[-1]
    for current in ring:
        x1, y1 = previous
        x2, y2 = current
        if ((y1 > y) != (y2 > y)) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
            inside = not inside
        previous = current
    return inside


def point_in_polygon(point, polygon):
    return bool(polygon and point_in_ring(point, polygon[0]) and not any(point_in_ring(point, hole) for hole in polygon[1:]))


def point_in_geometry(point, geometry):
    if geometry["type"] == "Polygon":
        return point_in_polygon(point, geometry["coordinates"])
    if geometry["type"] == "MultiPolygon":
        return any(point_in_polygon(point, polygon) for polygon in geometry["coordinates"])
    return False


def distance_km(a, b):
    mean_lat = math.radians((a[1] + b[1]) / 2)
    dx = (b[0] - a[0]) * EARTH_KM_PER_DEGREE * math.cos(mean_lat)
    dy = (b[1] - a[1]) * EARTH_KM_PER_DEGREE
    return math.hypot(dx, dy)


def iter_lines(geometry):
    if geometry["type"] == "LineString":
        yield geometry["coordinates"]
    elif geometry["type"] == "MultiLineString":
        yield from geometry["coordinates"]


def percentile_ranks(values):
    ordered = sorted((value, index) for index, value in enumerate(values))
    output = [0.0] * len(values)
    denominator = max(1, len(values) - 1)
    for rank, (_, index) in enumerate(ordered):
        output[index] = rank / denominator * 100
    return output


def locate_area(point, area_features):
    for feature in area_features:
        bounds = feature["properties"]["bbox"]
        if bounds[0] <= point[0] <= bounds[2] and bounds[1] <= point[1] <= bounds[3]:
            if point_in_geometry(point, feature["geometry"]):
                return feature
    return None


def representative_point(geometry):
    centre = centroid(geometry)
    if geometry["type"] in {"Polygon", "MultiPolygon"} and point_in_geometry(centre, geometry):
        return centre
    return next(iter_points(geometry))


def build(args):
    detail_root = Path(args.detail_root)
    green_root = Path(args.green_root)
    output_root = Path(args.output)
    existing = read_json(Path(args.existing_geo))
    existing_by_name = {feature["properties"]["name"].upper(): feature["properties"] for feature in existing["features"]}

    areas = read_json(detail_root / "web" / "areas.min.geojson")
    for feature in areas["features"]:
        props = feature["properties"]
        old = existing_by_name.get(props["display_name"].upper(), {})
        props["centroid"] = old.get("centroid") or centroid(feature["geometry"])
        props["bbox"] = bbox(feature["geometry"])
        props["area_km2"] = old.get("areaKm2", 0)
        props["cooling_centres"] = old.get("coolingCentres", 0)

    subzones = read_json(detail_root / "web" / "subzones.min.geojson")
    for feature in subzones["features"]:
        props = feature["properties"]
        props["centroid"] = centroid(feature["geometry"])
        props["bbox"] = bbox(feature["geometry"])
        props["area_km2"] = round(geometry_area_km2(feature["geometry"]), 4)
        props["cooling_centres"] = 0

    for centre in existing.get("coolingCentres", []):
        point = [centre["lon"], centre["lat"]]
        for feature in subzones["features"]:
            if point_in_geometry(point, feature["geometry"]):
                feature["properties"]["cooling_centres"] += 1
                break

    write_json(output_root / "areas.geojson", areas)
    write_json(output_root / "subzones.geojson", subzones)

    blocks_index = read_json(detail_root / "web" / "blocks_index.json")
    write_json(output_root / "blocks_index.json", blocks_index)
    for area_id, item in blocks_index.items():
        source = detail_root / "web" / item["file"]
        block_data = read_json(source)
        for feature in block_data["features"]:
            props = feature["properties"]
            props["display_name"] = f"Block {props.get('blk_no') or props.get('block_id')}"
        write_json(output_root / item["file"], block_data)

    area_metrics = {
        feature["properties"]["id"]: {
            "id": feature["properties"]["id"],
            "name": feature["properties"]["display_name"],
            "parkCount": 0,
            "parkAreaKm2": 0.0,
            "cyclingKm": 0.0,
            "parkConnectorKm": 0.0,
        }
        for feature in areas["features"]
    }

    parks = read_json(green_root / "NParksParksandNatureReserves.geojson")
    compact_parks = {"type": "FeatureCollection", "features": []}
    for feature in parks["features"]:
        props = feature["properties"]
        area = locate_area(representative_point(feature["geometry"]), areas["features"])
        area_id = area["properties"]["id"] if area else None
        park_area = float(props.get("SHAPE_1.AREA") or 0) / 1_000_000
        if area_id:
            area_metrics[area_id]["parkCount"] += 1
            area_metrics[area_id]["parkAreaKm2"] += park_area
        compact_parks["features"].append({
            "type": "Feature",
            "geometry": feature["geometry"],
            "properties": {
                "id": props.get("L_CODE"),
                "name": props.get("NAME") or "Park / nature reserve",
                "natureReserve": bool(props.get("N_RESERVE")),
                "areaKm2": round(park_area, 5),
                "planningAreaId": area_id,
            },
        })

    def compact_network(filename, title_key, output_name, metric_key):
        source = read_json(green_root / filename)
        compact = {"type": "FeatureCollection", "features": []}
        for feature in source["features"]:
            allocations = {}
            for line in iter_lines(feature["geometry"]):
                for start, end in zip(line, line[1:]):
                    length = distance_km(start, end)
                    midpoint = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2]
                    area = locate_area(midpoint, areas["features"])
                    if area:
                        area_id = area["properties"]["id"]
                        allocations[area_id] = allocations.get(area_id, 0.0) + length
                        area_metrics[area_id][metric_key] += length
            props = feature["properties"]
            compact["features"].append({
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": {
                    "name": props.get(title_key) or output_name,
                    "planningAreaId": max(allocations, key=allocations.get) if allocations else None,
                    "lengthKm": round(sum(allocations.values()), 4),
                },
            })
        write_json(output_root / f"{output_name}.geojson", compact)

    write_json(output_root / "parks.geojson", compact_parks)
    compact_network("CyclingPathNetworkGEOJSON.geojson", "CYL_PATH", "cycling", "cyclingKm")
    compact_network("ParkConnectorLoop.geojson", "PCN_LOOP", "park-connectors", "parkConnectorKm")

    metric_rows = list(area_metrics.values())
    area_by_id = {feature["properties"]["id"]: feature for feature in areas["features"]}
    for row in metric_rows:
        area_km2 = max(0.01, area_by_id[row["id"]]["properties"].get("area_km2") or 0.01)
        row["parkCoveragePct"] = row["parkAreaKm2"] / area_km2 * 100
        row["cyclingDensity"] = row["cyclingKm"] / area_km2
        row["parkConnectorDensity"] = row["parkConnectorKm"] / area_km2

    coverage_rank = percentile_ranks([row["parkCoveragePct"] for row in metric_rows])
    cycling_rank = percentile_ranks([row["cyclingDensity"] for row in metric_rows])
    connector_rank = percentile_ranks([row["parkConnectorDensity"] for row in metric_rows])
    for index, row in enumerate(metric_rows):
        row["greenBaselineScore"] = round(0.5 * coverage_rank[index] + 0.3 * cycling_rank[index] + 0.2 * connector_rank[index], 1)
        for key in ("parkAreaKm2", "cyclingKm", "parkConnectorKm", "parkCoveragePct", "cyclingDensity", "parkConnectorDensity"):
            row[key] = round(row[key], 3)

    write_json(output_root / "green-metrics.json", {"areas": metric_rows, "method": {
        "greenBaselineScore": "50% park-cover percentile + 30% cycling-network density percentile + 20% park-connector density percentile",
        "parkAssignment": "Park area is assigned by representative point; cross-boundary parks are not clipped.",
        "lineAssignment": "Network segments are assigned by segment midpoint.",
    }})
    write_json(output_root / "metadata.json", {
        "planningAreas": len(areas["features"]),
        "subzones": len(subzones["features"]),
        "blocks": sum(item["block_count"] for item in blocks_index.values()),
        "parks": len(compact_parks["features"]),
        "cyclingSegments": len(read_json(output_root / "cycling.geojson")["features"]),
        "parkConnectorSegments": len(read_json(output_root / "park-connectors.geojson")["features"]),
        "demographics": "SingStat June 2026 at planning-area and subzone level; block values are estimates.",
        "geometry": "URA Master Plan 2019 boundaries.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail-root", required=True)
    parser.add_argument("--green-root", required=True)
    parser.add_argument("--existing-geo", required=True)
    parser.add_argument("--output", required=True)
    build(parser.parse_args())
