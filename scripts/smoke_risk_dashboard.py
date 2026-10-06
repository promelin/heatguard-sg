#!/usr/bin/env python3
"""Smoke-check the packaged replay through the deployed planning-area model."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


STATIONS = {
    "S104": (103.78538, 1.44387), "S108": (103.8703, 1.2799),
    "S109": (103.8492, 1.3764), "S111": (103.8365, 1.31055),
    "S115": (103.61843, 1.29377), "S117": (103.679, 1.256),
    "S121": (103.72244, 1.37288), "S24": (103.9826, 1.3678),
    "S43": (103.8878, 1.3399), "S44": (103.68166, 1.34583),
    "S50": (103.7768, 1.3337), "S60": (103.8279, 1.25),
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    replay = json.loads((root / "dist/heatguard-data.json").read_text(encoding="utf-8"))
    areas = json.loads((root / "dist/data/spatial/areas.geojson").read_text(encoding="utf-8"))["features"]
    package = joblib.load(root / "backend/model/risk_model.joblib")
    target_time = replay["peakTime"]
    observations = []
    for station in replay["stations"]:
        point = next((item for item in station["series"] if item["time"] == target_time), None)
        if point and station["id"] in STATIONS:
            observations.append((*STATIONS[station["id"]], float(point["predicted"])))

    rows = []
    for feature in areas:
        p = feature["properties"]
        lon, lat = p["centroid"]
        weights = []
        values = []
        for station_lon, station_lat, heat in observations:
            dx = (lon - station_lon) * math.cos(lat * math.pi / 180)
            dy = lat - station_lat
            weights.append(1 / (dx * dx + dy * dy + 0.000025))
            values.append(heat)
        local_heat = float(np.average(values, weights=weights))
        residents = float(p.get("population_2026") or 0)
        seniors = float(p.get("elderly_65plus_2026") or 0)
        elderly75 = float(p.get("elderly_75plus_2026") or 0)
        area_km2 = max(float(p.get("area_km2") or 0), 0.01)
        access = 10_000 * float(p.get("cooling_centres") or 0) / seniors if seniors else 0
        values = pd.DataFrame([[
            local_heat,
            seniors / area_km2,
            100 * seniors / residents if residents else 0,
            100 * elderly75 / residents if residents else 0,
            access,
        ]], columns=package["features"])
        probability = package["model"].predict_proba(values)[0]
        score = float(50 * probability[1] + 100 * probability[2])
        low_medium = package["thresholds"]["low_medium"]
        medium_high = package["thresholds"]["medium_high"]
        tier = "High" if score >= medium_high else "Medium" if score >= low_medium else "Low"
        rows.append((p["display_name"], local_heat, score, tier))
    rows.sort(key=lambda row: row[2], reverse=True)
    tier_counts = Counter(row[3] for row in rows)
    if len(tier_counts) < 2:
        raise SystemExit(f"Risk map collapsed to one tier at the packaged peak: {dict(tier_counts)}")
    print(json.dumps({
        "time": target_time,
        "tier_counts": dict(tier_counts),
        "score_range": [round(min(row[2] for row in rows), 2), round(max(row[2] for row in rows), 2)],
        "top_five": [{"area": row[0], "heat": round(row[1], 2), "score": round(row[2], 2), "tier": row[3]} for row in rows[:5]],
    }, indent=2))


if __name__ == "__main__":
    main()
