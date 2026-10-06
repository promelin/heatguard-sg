#!/usr/bin/env python3
"""Build the compact HeatGuard replay JSON from TFT prediction outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def risk_band(value: float) -> str:
    if value >= 37:
        return "High"
    if value >= 34:
        return "Elevated"
    return "Watch"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path,
                        help="CSV or CSV.GZ with station_id, target_time_utc, actual_heat_index_c and predicted_heat_index_c")
    parser.add_argument("--metrics", required=True, type=Path,
                        help="Full TFT metrics.json")
    parser.add_argument("--output", default=Path("dist/heatguard-data.json"), type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions, parse_dates=["target_time_utc"])
    frame["target_time_sgt"] = frame["target_time_utc"].dt.tz_convert("Asia/Singapore")
    frame["error"] = frame["predicted_heat_index_c"] - frame["actual_heat_index_c"]
    network = frame.groupby("target_time_sgt").agg(
        predicted=("predicted_heat_index_c", "mean"),
        actual=("actual_heat_index_c", "mean"),
        count=("station_id", "size"),
    )
    peak_time = network.loc[network["count"] >= 10, "actual"].idxmax()
    window_start = peak_time - pd.Timedelta(days=3)
    window_end = peak_time + pd.Timedelta(days=3)
    replay = frame[(frame["target_time_sgt"] >= window_start) & (frame["target_time_sgt"] <= window_end)].copy()
    station_metrics = frame.groupby("station_id").agg(
        rmse=("error", lambda values: float((values.pow(2).mean()) ** 0.5)),
        mae=("error", lambda values: float(values.abs().mean())),
    )
    model_metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    peak = frame[frame["target_time_sgt"] == peak_time].sort_values("predicted_heat_index_c", ascending=False)
    default_station = str(peak.iloc[0]["station_id"])

    stations = []
    for station_id, group in replay.groupby("station_id", sort=True):
        group = group.sort_values("target_time_sgt")
        metric = station_metrics.loc[station_id]
        stations.append({
            "id": str(station_id),
            "rmse": round(float(metric.rmse), 3),
            "mae": round(float(metric.mae), 3),
            "series": [{
                "time": row.target_time_sgt.isoformat(),
                "predicted": round(float(row.predicted_heat_index_c), 2),
                "actual": round(float(row.actual_heat_index_c), 2),
            } for row in group.itertuples()],
        })

    payload = {
        "mode": "2024 held-out replay",
        "model": {
            "name": "High-focus full Temporal Fusion Transformer",
            "version": "high-focus-v1",
            "singleModel": True,
            "historyHours": model_metrics["architecture"]["history_hours"],
            "forecastHours": 24,
            "parameters": model_metrics["architecture"]["parameters"],
            "overallRmse": round(model_metrics["test_h24"]["rmse_c"], 3),
            "overallMae": round(model_metrics["test_h24"]["mae_c"], 3),
            "high37Rmse": round(model_metrics["test_high_temperature"]["37.0"]["rmse_c"], 3),
            "high37Mae": round(model_metrics["test_high_temperature"]["37.0"]["mae_c"], 3),
            "high37Recall": round(model_metrics["test_high_temperature"]["37.0"]["recall"], 4),
            "high40Rmse": round(model_metrics["test_high_temperature"]["40.0"]["rmse_c"], 3),
        },
        "timezone": "Asia/Singapore",
        "peakTime": peak_time.isoformat(),
        "windowStart": window_start.isoformat(),
        "windowEnd": window_end.isoformat(),
        "defaultStation": default_station,
        "networkMean": round(float(peak.predicted_heat_index_c.mean()), 2),
        "networkActualMean": round(float(peak.actual_heat_index_c.mean()), 2),
        "highCount": int((peak.predicted_heat_index_c >= 37).sum()),
        "elevatedCount": int(((peak.predicted_heat_index_c >= 34) & (peak.predicted_heat_index_c < 37)).sum()),
        "peakRows": [{
            "station": str(row.station_id),
            "predicted": round(float(row.predicted_heat_index_c), 2),
            "actual": round(float(row.actual_heat_index_c), 2),
            "risk": risk_band(float(row.predicted_heat_index_c)),
        } for row in peak.itertuples()],
        "stations": stations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "stations": len(stations), "peak_time": peak_time.isoformat()}, indent=2))


if __name__ == "__main__":
    main()
