"""Build the 168-hour V4 tensor and produce the live 1--24 hour forecast."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .tft_model import FullTemporalFusionTransformer


SGT = timezone(timedelta(hours=8))
WEBSITE_STATIONS = ["S104", "S108", "S109", "S111", "S115", "S117", "S121", "S24", "S43", "S44", "S50", "S60"]
BASE_FEATURES = [
    "temperature_mean_c", "temperature_min_c", "temperature_max_c",
    "humidity_mean_pct", "humidity_min_pct", "humidity_max_pct",
    "rainfall_total_mm", "heat_index_c", "temperature_coverage",
    "humidity_coverage", "rainfall_coverage",
]


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc)


def floor_hour(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def heat_index_celsius(temperature_c: float, relative_humidity: float) -> float:
    """NOAA Rothfusz heat index, matching the training-data implementation."""
    if not math.isfinite(temperature_c) or not math.isfinite(relative_humidity):
        return float("nan")
    temperature_f = temperature_c * 9.0 / 5.0 + 32.0
    rh = relative_humidity
    hi_f = (
        -42.379 + 2.04901523 * temperature_f + 10.14333127 * rh
        - 0.22475541 * temperature_f * rh - 0.00683783 * temperature_f**2
        - 0.05481717 * rh**2 + 0.00122874 * temperature_f**2 * rh
        + 0.00085282 * temperature_f * rh**2
        - 0.00000199 * temperature_f**2 * rh**2
    )
    if rh < 13 and 80 <= temperature_f <= 112:
        hi_f -= ((13 - rh) / 4) * math.sqrt(max(0.0, (17 - abs(temperature_f - 95)) / 17))
    elif rh > 85 and 80 <= temperature_f <= 87:
        hi_f += ((rh - 85) / 10) * ((87 - temperature_f) / 5)
    if temperature_f < 80 or rh < 40:
        hi_f = temperature_f
    return (hi_f - 32.0) * 5.0 / 9.0


def risk_label(value: float) -> str:
    if value >= 40:
        return "Very high"
    if value >= 37:
        return "High"
    if value >= 34:
        return "Elevated"
    return "Watch"


class V4Predictor:
    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"V4 checkpoint is missing: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"V4 runtime metadata is missing: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.stations: list[str] = self.metadata["stations"]
        self.station_to_index = {station: index for index, station in enumerate(self.stations)}
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        checkpoint_args = checkpoint.get("arguments", {})
        hidden_size = int(checkpoint_args.get("hidden_size", 64))
        heads = int(checkpoint_args.get("heads", 4))
        dropout = float(checkpoint_args.get("dropout", 0.1))
        self.model = FullTemporalFusionTransformer(
            history_variables=len(self.metadata["dynamic_names"]),
            future_real_variables=len(self.metadata["future_real_names"]),
            future_cat_cardinalities=[],
            stations=len(self.stations),
            hidden_size=hidden_size,
            heads=heads,
            dropout=dropout,
            cooling_gate_enabled=False,
        )
        state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def current_conditions(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Pair each station's freshest temperature and humidity into a live heat index."""
        latest: dict[tuple[str, str], tuple[datetime, float]] = {}
        station_set = set(WEBSITE_STATIONS)
        for row in rows:
            metric = str(row["metric"])
            station = str(row["station_id"])
            if metric not in {"temperature", "humidity"} or station not in station_set:
                continue
            timestamp = parse_timestamp(row["timestamp"])
            key = (station, metric)
            if key not in latest or timestamp > latest[key][0]:
                latest[key] = (timestamp, float(row["value"]))

        candidates: list[dict[str, Any]] = []
        for station in WEBSITE_STATIONS:
            temperature = latest.get((station, "temperature"))
            humidity = latest.get((station, "humidity"))
            if not temperature or not humidity:
                continue
            if abs((temperature[0] - humidity[0]).total_seconds()) > 60 * 60:
                continue
            observed_at = max(temperature[0], humidity[0])
            heat_index = heat_index_celsius(temperature[1], humidity[1])
            if not math.isfinite(heat_index):
                continue
            candidates.append({
                "id": station,
                "time": observed_at.astimezone(SGT).isoformat(),
                "temperature": round(temperature[1], 2),
                "humidity": round(humidity[1], 2),
                "heatIndex": round(heat_index, 2),
                "risk": risk_label(heat_index),
            })
        if not candidates:
            return None

        freshest = max(parse_timestamp(row["time"]) for row in candidates)
        stations = [
            row for row in candidates
            if (freshest - parse_timestamp(row["time"])) <= timedelta(hours=2)
        ]
        if not stations:
            return None
        mean = float(np.mean([row["heatIndex"] for row in stations]))
        return {
            "source": "data.gov.sg real-time weather readings",
            "observedAt": freshest.astimezone(SGT).isoformat(),
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "networkMean": round(mean, 2),
            "highCount": sum(row["heatIndex"] >= 37 for row in stations),
            "elevatedCount": sum(34 <= row["heatIndex"] < 37 for row in stations),
            "stations": sorted(stations, key=lambda row: row["heatIndex"], reverse=True),
        }

    def _aggregate(self, rows: list[dict[str, Any]], issue_time: datetime) -> tuple[np.ndarray, dict[str, Any]]:
        history_hours = int(self.metadata["history_hours"])
        start = issue_time - timedelta(hours=history_hours - 1)
        values: dict[tuple[str, str, datetime], list[float]] = defaultdict(list)
        rain_coordinates: dict[str, tuple[float, float]] = {}
        latest_observation: datetime | None = None
        for row in rows:
            timestamp = parse_timestamp(row["timestamp"])
            if timestamp < start or timestamp >= issue_time + timedelta(hours=1):
                continue
            latest_observation = max(latest_observation or timestamp, timestamp)
            metric = str(row["metric"])
            station = str(row["station_id"])
            values[(metric, station, floor_hour(timestamp))].append(float(row["value"]))
            if metric == "rainfall" and row.get("latitude") is not None and row.get("longitude") is not None:
                rain_coordinates[station] = (float(row["latitude"]), float(row["longitude"]))

        raw = np.full((history_hours, len(self.stations), len(BASE_FEATURES)), np.nan, dtype=np.float32)
        training_coordinates = self.metadata["station_coordinates"]
        rain_choice_cache: dict[tuple[str, datetime], str | None] = {}
        available_rain_by_hour: dict[datetime, set[str]] = defaultdict(set)
        for (metric, station, bucket), samples in values.items():
            if metric == "rainfall" and samples:
                available_rain_by_hour[bucket].add(station)
        for time_index in range(history_hours):
            hour = start + timedelta(hours=time_index)
            available_rain = available_rain_by_hour.get(hour, set())
            for station_index, station in enumerate(self.stations):
                temperatures = values.get(("temperature", station, hour), [])
                humidities = values.get(("humidity", station, hour), [])
                if temperatures:
                    raw[time_index, station_index, 0:3] = (
                        float(np.mean(temperatures)), float(np.min(temperatures)), float(np.max(temperatures))
                    )
                    raw[time_index, station_index, 8] = min(1.0, len(temperatures) / 60.0)
                if humidities:
                    raw[time_index, station_index, 3:6] = (
                        float(np.mean(humidities)), float(np.min(humidities)), float(np.max(humidities))
                    )
                    raw[time_index, station_index, 9] = min(1.0, len(humidities) / 60.0)
                cache_key = (station, hour)
                if cache_key not in rain_choice_cache:
                    target = training_coordinates[station]
                    rain_choice_cache[cache_key] = min(
                        available_rain,
                        key=lambda candidate: (
                            (rain_coordinates.get(candidate, (90.0, 180.0))[0] - target["latitude"]) ** 2
                            + (rain_coordinates.get(candidate, (90.0, 180.0))[1] - target["longitude"]) ** 2
                        ),
                        default=None,
                    )
                rain_station = rain_choice_cache[cache_key]
                rainfall = values.get(("rainfall", rain_station, hour), []) if rain_station else []
                if rainfall:
                    raw[time_index, station_index, 6] = float(np.sum(rainfall))
                    raw[time_index, station_index, 10] = min(1.0, len(rainfall) / 12.0)
                raw[time_index, station_index, 7] = heat_index_celsius(
                    float(raw[time_index, station_index, 0]), float(raw[time_index, station_index, 3])
                )

        primary_indices = [0, 3, 6, 7]
        primary_mask = np.isfinite(raw[:, :, primary_indices]).astype(np.float32)
        with np.errstate(invalid="ignore"):
            island = np.stack([
                np.nanmean(raw[:, :, index], axis=1) for index in primary_indices
            ], axis=-1).astype(np.float32)
        island_broadcast = np.repeat(island[:, None, :], len(self.stations), axis=1)
        hours = [start + timedelta(hours=index) for index in range(history_hours)]
        calendar = np.asarray([
            [
                math.sin(2 * math.pi * hour.hour / 24.0),
                math.cos(2 * math.pi * hour.hour / 24.0),
                math.sin(2 * math.pi * hour.timetuple().tm_yday / 365.25),
                math.cos(2 * math.pi * hour.timetuple().tm_yday / 365.25),
            ] for hour in hours
        ], dtype=np.float32)
        calendar_broadcast = np.repeat(calendar[:, None, :], len(self.stations), axis=1)
        dynamic = np.concatenate([raw, primary_mask, island_broadcast, calendar_broadcast], axis=-1)
        means = np.asarray(self.metadata["dynamic_mean"], dtype=np.float32)
        stds = np.asarray(self.metadata["dynamic_std"], dtype=np.float32)
        dynamic = np.nan_to_num((dynamic - means) / stds, nan=0.0, posinf=0.0, neginf=0.0)
        expected = (history_hours, len(self.stations), len(self.metadata["dynamic_names"]))
        if dynamic.shape != expected:
            raise ValueError(f"V4 history tensor has shape {dynamic.shape}; expected {expected}")
        quality = {
            "historyStart": start.astimezone(SGT).isoformat(),
            "historyEnd": issue_time.astimezone(SGT).isoformat(),
            "latestObservation": latest_observation.astimezone(SGT).isoformat() if latest_observation else None,
            "temperatureCoverage": round(float(np.nan_to_num(raw[:, :, 8], nan=0.0).mean()), 4),
            "humidityCoverage": round(float(np.nan_to_num(raw[:, :, 9], nan=0.0).mean()), 4),
            "rainfallCoverage": round(float(np.nan_to_num(raw[:, :, 10], nan=0.0).mean()), 4),
        }
        return dynamic, quality

    def predict(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            raise ValueError("No cached weather observations are available")
        latest = max(parse_timestamp(row["timestamp"]) for row in rows)
        issue_time = min(
            floor_hour(datetime.now(timezone.utc)) - timedelta(hours=1),
            floor_hour(latest) - timedelta(hours=1),
        )
        history, quality = self._aggregate(rows, issue_time)
        horizon = int(self.metadata["horizon_hours"])
        target_times = [issue_time + timedelta(hours=lead) for lead in range(1, horizon + 1)]
        future_raw = np.asarray([
            [
                lead,
                math.sin(2 * math.pi * target.hour / 24.0),
                math.cos(2 * math.pi * target.hour / 24.0),
                math.sin(2 * math.pi * target.timetuple().tm_yday / 365.25),
                math.cos(2 * math.pi * target.timetuple().tm_yday / 365.25),
            ] for lead, target in enumerate(target_times, start=1)
        ], dtype=np.float32)
        future_mean = np.asarray(self.metadata["future_real_mean"], dtype=np.float32)
        future_std = np.asarray(self.metadata["future_real_std"], dtype=np.float32)
        future = (future_raw - future_mean) / future_std
        history_batch = torch.from_numpy(np.transpose(history, (1, 0, 2))).float()
        future_batch = torch.from_numpy(np.repeat(future[None, :, :], len(self.stations), axis=0)).float()
        station_tensor = torch.arange(len(self.stations), dtype=torch.long)
        static_tensor = torch.tensor(self.metadata["station_static"], dtype=torch.float32)
        empty_categories = torch.empty((len(self.stations), horizon, 0), dtype=torch.long)
        with torch.inference_mode():
            normalized = self.model(
                history_batch, future_batch, empty_categories, station_tensor, static_tensor
            ).cpu().numpy()
        predictions = normalized * float(self.metadata["target_std"]) + float(self.metadata["target_mean"])
        output_stations = [station for station in WEBSITE_STATIONS if station in self.station_to_index]
        stations_payload = []
        for station in output_stations:
            index = self.station_to_index[station]
            series = []
            for lead_index, target in enumerate(target_times):
                lower, median, upper = [float(value) for value in predictions[index, lead_index]]
                series.append({
                    "time": target.astimezone(SGT).isoformat(),
                    "predicted": round(median, 2),
                    "actual": None,
                    "lower": round(min(lower, median), 2),
                    "upper": round(max(upper, median), 2),
                })
            stations_payload.append({"id": station, "rmse": None, "mae": None, "series": series})
        network_means = [
            float(np.mean([station["series"][lead]["predicted"] for station in stations_payload]))
            for lead in range(horizon)
        ]
        peak_index = int(np.argmax(network_means))
        peak_time = stations_payload[0]["series"][peak_index]["time"]
        peak_rows = sorted([
            {
                "station": station["id"],
                "predicted": station["series"][peak_index]["predicted"],
                "actual": None,
                "risk": risk_label(station["series"][peak_index]["predicted"]),
            } for station in stations_payload
        ], key=lambda row: row["predicted"], reverse=True)
        return {
            "mode": "live forecast",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "issueTime": issue_time.astimezone(SGT).isoformat(),
            "model": {
                "name": "High-focus full Temporal Fusion Transformer",
                "version": "V4-history-only",
                "singleModel": True,
                "historyHours": 168,
                "forecastHours": 24,
                "parameters": 906723,
                "overallRmse": 2.8069,
                "overallMae": 1.8487,
                "high37Rmse": 1.3637,
                "high37Mae": 1.0232,
                "high37Recall": 0.8264,
                "high40Rmse": 1.5826,
            },
            "timezone": "Asia/Singapore",
            "peakTime": peak_time,
            "windowStart": stations_payload[0]["series"][0]["time"],
            "windowEnd": stations_payload[0]["series"][-1]["time"],
            "defaultStation": peak_rows[0]["station"],
            "networkMean": round(network_means[peak_index], 2),
            "networkActualMean": None,
            "highCount": sum(row["predicted"] >= 37 for row in peak_rows),
            "elevatedCount": sum(34 <= row["predicted"] < 37 for row in peak_rows),
            "current": self.current_conditions(rows),
            "dataQuality": quality,
            "peakRows": peak_rows,
            "stations": stations_payload,
        }
