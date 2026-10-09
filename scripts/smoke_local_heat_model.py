#!/usr/bin/env python3
"""Offline smoke checks for the local environment heat model."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.local_environment import LocalHeatIndexModel  # noqa: E402


def main() -> None:
    model_path = ROOT / "backend" / "model" / "local_heat_model.joblib"
    predictor = LocalHeatIndexModel(model_path, ROOT / "dist" / "data" / "spatial")
    examples = [
        (103.851959, 1.290270, 250),
        (103.851959, 1.290270, 500),
        (103.938800, 1.334300, 1_000),
    ]
    for longitude, latitude, radius in examples:
        result = predictor.predict(longitude, latitude, radius, 34.0)
        assert math.isfinite(result["estimate_c"])
        assert 20.0 <= result["estimate_c"] <= 55.0
        assert result["interval_c"][0] <= result["estimate_c"] <= result["interval_c"][1]
        assert 0.0 <= result["spatial_support"]["weight"] <= 1.0
        assert result["spatial"]["radius_m"] == radius

    published = json.loads((ROOT / "dist" / "data" / "local-heat-results.json").read_text(encoding="utf-8"))
    assert published["model"] == "local-environment-v1"
    assert published["radii_m"] == [250, 500, 1000]
    assert len(published["areas"]) == 55
    assert all(set(area["estimates"]) == {"250", "500", "1000"} for area in published["areas"])
    values = [
        estimate["estimate_c"]
        for area in published["areas"]
        for estimate in area["estimates"].values()
    ]
    assert min(values) >= 20.0 and max(values) <= 55.0
    print(json.dumps({
        "status": "ok",
        "queries": len(examples),
        "published_areas": len(published["areas"]),
        "published_range_c": [min(values), max(values)],
    }, indent=2))


if __name__ == "__main__":
    main()
