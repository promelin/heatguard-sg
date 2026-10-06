#!/usr/bin/env python3
"""Offline smoke test for hourly aggregation, preprocessing and V4 inference."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--module-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    sys.path.insert(0, str(args.module_dir))
    from backend.predictor import V4Predictor

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    end = datetime.now(timezone.utc).replace(minute=30, second=0, microsecond=0)
    rows = []
    for hour_index in range(170):
        timestamp = end - timedelta(hours=169 - hour_index)
        for station_index, station in enumerate(metadata["stations"]):
            phase = 2 * math.pi * timestamp.hour / 24.0
            temperature = 29.0 + 3.5 * math.sin(phase - 1.2) + station_index * 0.025
            humidity = 72.0 - 12.0 * math.sin(phase - 1.2)
            rows.extend([
                {"metric": "temperature", "timestamp": timestamp.isoformat(), "station_id": station,
                 "value": temperature, "name": station, "latitude": None, "longitude": None},
                {"metric": "humidity", "timestamp": timestamp.isoformat(), "station_id": station,
                 "value": humidity, "name": station, "latitude": None, "longitude": None},
            ])
        rows.append({
            "metric": "rainfall", "timestamp": timestamp.isoformat(), "station_id": "R001",
            "value": 0.0, "name": "Synthetic rain gauge", "latitude": 1.35, "longitude": 103.82,
        })
    payload = V4Predictor(args.checkpoint, args.metadata).predict(rows)
    assert len(payload["stations"]) == 12
    assert all(len(station["series"]) == 24 for station in payload["stations"])
    assert all(point["actual"] is None for station in payload["stations"] for point in station["series"])
    assert payload["current"] is not None
    assert len(payload["current"]["stations"]) == 12
    print(json.dumps({
        "mode": payload["mode"],
        "stations": len(payload["stations"]),
        "horizons": len(payload["stations"][0]["series"]),
        "issueTime": payload["issueTime"],
        "peakTime": payload["peakTime"],
        "defaultStation": payload["defaultStation"],
        "currentObservedAt": payload["current"]["observedAt"],
        "currentNetworkMean": payload["current"]["networkMean"],
        "quality": payload["dataQuality"],
    }, indent=2))


if __name__ == "__main__":
    main()
