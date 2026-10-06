#!/usr/bin/env python3
"""Export the exact preprocessing statistics needed by the live V4 service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VALIDATION_START = pd.Timestamp("2023-10-01 00:00:00", tz="Asia/Singapore").tz_convert("UTC")
BASE_FEATURES = [
    "temperature_mean_c", "temperature_min_c", "temperature_max_c",
    "humidity_mean_pct", "humidity_min_pct", "humidity_max_pct",
    "rainfall_total_mm", "heat_index_c", "temperature_coverage",
    "humidity_coverage", "rainfall_coverage",
]
DYNAMIC_NAMES = (
    BASE_FEATURES
    + ["temperature_observed", "humidity_observed", "rainfall_observed", "heat_index_observed"]
    + ["island_temperature", "island_humidity", "island_rainfall", "island_heat_index"]
    + ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
)

# These are the training-set normalization statistics for the five deterministic
# future features selected by V4 from future_known_24h_calendar_only.npz.
FUTURE_NAMES = ["lead_hours", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
FUTURE_MEAN = [12.435276985168457, -2.708851831354025e-13, 1.551904235766108e-13,
               -0.018535718321800232, -0.03219233825802803]
FUTURE_STD = [6.919174671173096, 0.7071067690849304, 0.7071067690849304,
              0.7029017210006714, 0.7097682356834412]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weather_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()

    usecols = ["hour_sgt", "station_id", "station_name", "longitude", "latitude", *BASE_FEATURES]
    raw = pd.read_csv(args.weather_csv, usecols=usecols, low_memory=False)
    raw["hour_utc"] = pd.to_datetime(raw["hour_sgt"], utc=True)
    stations = np.array(sorted(raw["station_id"].dropna().unique()), dtype="U16")
    station_to_index = {station: index for index, station in enumerate(stations)}
    hours = pd.date_range(raw.hour_utc.min().floor("h"), raw.hour_utc.max().ceil("h"), freq="h")
    hour_to_index = pd.Series(np.arange(len(hours)), index=hours)
    raw["t"] = raw["hour_utc"].map(hour_to_index).astype(int)
    raw["s"] = raw["station_id"].map(station_to_index).astype(int)

    base = np.full((len(hours), len(stations), len(BASE_FEATURES)), np.nan, dtype=np.float32)
    for feature_index, feature in enumerate(BASE_FEATURES):
        base[raw["t"].to_numpy(), raw["s"].to_numpy(), feature_index] = (
            pd.to_numeric(raw[feature], errors="coerce").to_numpy(dtype=np.float32)
        )
    primary_indices = [
        BASE_FEATURES.index("temperature_mean_c"),
        BASE_FEATURES.index("humidity_mean_pct"),
        BASE_FEATURES.index("rainfall_total_mm"),
        BASE_FEATURES.index("heat_index_c"),
    ]
    primary_mask = np.isfinite(base[:, :, primary_indices]).astype(np.float32)
    island = []
    for feature in ("temperature_mean_c", "humidity_mean_pct", "rainfall_total_mm", "heat_index_c"):
        with np.errstate(invalid="ignore"):
            island.append(np.nanmean(base[:, :, BASE_FEATURES.index(feature)], axis=1))
    island = np.stack(island, axis=-1).astype(np.float32)
    island_broadcast = np.repeat(island[:, None, :], len(stations), axis=1)

    hour = hours.hour.to_numpy()
    doy = hours.dayofyear.to_numpy()
    calendar = np.column_stack([
        np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
    ]).astype(np.float32)
    calendar_broadcast = np.repeat(calendar[:, None, :], len(stations), axis=1)
    dynamic = np.concatenate([base, primary_mask, island_broadcast, calendar_broadcast], axis=-1)
    train_mask = hours < VALIDATION_START
    means = np.nanmean(dynamic[train_mask], axis=(0, 1))
    stds = np.nanstd(dynamic[train_mask], axis=(0, 1))
    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
    stds = np.where(np.isfinite(stds) & (stds > 1e-6), stds, 1.0).astype(np.float32)

    target = base[:, :, BASE_FEATURES.index("heat_index_c")]
    target_mean = float(np.nanmean(target[train_mask]))
    target_std = float(np.nanstd(target[train_mask]))
    grouped = raw.groupby("station_id", observed=True)
    station_meta = grouped[["longitude", "latitude"]].median().reindex(stations)
    station_names = grouped["station_name"].first().reindex(stations)
    coordinates = station_meta.to_numpy(np.float32)
    static_mean = np.nanmean(coordinates, axis=0)
    static_std = np.nanstd(coordinates, axis=0)
    station_static = np.nan_to_num((coordinates - static_mean) / static_std).astype(np.float32)

    payload = {
        "schema_version": 1,
        "model_version": "V4-history-only",
        "history_hours": 168,
        "horizon_hours": 24,
        "validation_start_utc": VALIDATION_START.isoformat(),
        "stations": stations.tolist(),
        "station_names": {station: str(station_names.loc[station]) for station in stations},
        "station_coordinates": {
            station: {"longitude": float(coordinates[i, 0]), "latitude": float(coordinates[i, 1])}
            for i, station in enumerate(stations)
        },
        "station_static": station_static.tolist(),
        "station_static_mean": static_mean.tolist(),
        "station_static_std": static_std.tolist(),
        "dynamic_names": DYNAMIC_NAMES,
        "dynamic_mean": means.tolist(),
        "dynamic_std": stds.tolist(),
        "future_real_names": FUTURE_NAMES,
        "future_real_mean": FUTURE_MEAN,
        "future_real_std": FUTURE_STD,
        "target_mean": target_mean,
        "target_std": target_std,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_json), "stations": len(stations),
        "dynamic_features": len(DYNAMIC_NAMES), "target_mean": target_mean,
        "target_std": target_std,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
