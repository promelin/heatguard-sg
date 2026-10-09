#!/usr/bin/env python3
"""Train the HeatGuard local micro-environment heat-index model.

The target is observed station heat index.  For every timestamp, the mean of
the other stations supplies the island-wide background condition.  Validation
holds out complete stations, so repeated hours from a location never leak into
that location's test fold.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.local_environment import FEATURE_NAMES, LocalFeatureExtractor, LocalHeatIndexModel  # noqa: E402


ENVIRONMENT_FEATURES = [name for name in FEATURE_NAMES if name != "background_heat_index_c"]
SUPPORT_DECAY_SCALE = 3.0


def heat_index_celsius(temperature_c: np.ndarray, relative_humidity: np.ndarray) -> np.ndarray:
    temperature_f = temperature_c * 9.0 / 5.0 + 32.0
    rh = relative_humidity
    value = (
        -42.379 + 2.04901523 * temperature_f + 10.14333127 * rh
        - 0.22475541 * temperature_f * rh - 0.00683783 * temperature_f**2
        - 0.05481717 * rh**2 + 0.00122874 * temperature_f**2 * rh
        + 0.00085282 * temperature_f * rh**2 - 0.00000199 * temperature_f**2 * rh**2
    )
    value = np.where((temperature_f < 80) | (rh < 40), temperature_f, value)
    return (value - 32.0) * 5.0 / 9.0


def restore_seed_cache(project_root: Path) -> Path:
    target = project_root / ".cache" / "local-heat-training.sqlite3"
    if target.exists():
        return target
    source = project_root / "backend" / "data" / "seed-weather.sqlite3.gz"
    if not source.exists():
        raise FileNotFoundError(f"Training replay and weather seed are unavailable: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rb") as compressed, target.open("wb") as database:
        shutil.copyfileobj(compressed, database)
    return target


def load_seed_observations(project_root: Path, station_ids: set[str]) -> pd.DataFrame:
    connection = sqlite3.connect(restore_seed_cache(project_root))
    try:
        hourly = pd.read_sql_query(
            """
            SELECT substr(timestamp, 1, 13) || ':00:00+00:00' AS time,
                   station_id,
                   AVG(CASE WHEN metric = 'temperature' THEN value END) AS temperature_c,
                   AVG(CASE WHEN metric = 'humidity' THEN value END) AS humidity_pct,
                   COUNT(CASE WHEN metric = 'temperature' THEN 1 END) AS temperature_readings,
                   COUNT(CASE WHEN metric = 'humidity' THEN 1 END) AS humidity_readings
            FROM observations
            WHERE metric IN ('temperature', 'humidity')
            GROUP BY time, station_id
            HAVING temperature_c IS NOT NULL AND humidity_pct IS NOT NULL
            ORDER BY time, station_id
            """,
            connection,
        )
    finally:
        connection.close()
    hourly = hourly[
        hourly.station_id.isin(station_ids)
        & (hourly.temperature_readings >= 6)
        & (hourly.humidity_readings >= 6)
    ].copy()
    hourly["actual"] = heat_index_celsius(
        hourly.temperature_c.to_numpy(dtype=float), hourly.humidity_pct.to_numpy(dtype=float)
    )
    return hourly[np.isfinite(hourly.actual)].copy()


def build_training_frame(project_root: Path, radius_m: int) -> pd.DataFrame:
    replay = json.loads((project_root / "dist" / "heatguard-data.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (project_root / "backend" / "model" / "v4_runtime_metadata.json").read_text(encoding="utf-8")
    )
    extractor = LocalFeatureExtractor(project_root / "dist" / "data" / "spatial")
    station_rows: dict[str, list[dict[str, Any]]] = {}
    for station in replay["stations"]:
        coordinates = metadata["station_coordinates"].get(station["id"])
        if not coordinates:
            continue
        rows = [
            {
                "time": point["time"],
                "actual": point.get("actual"),
                "coordinates": coordinates,
            }
            for point in station["series"]
            if point.get("actual") is not None and math.isfinite(float(point["actual"]))
        ]
        if rows:
            station_rows[station["id"]] = rows

    observations_by_time: dict[str, dict[str, float]] = {}
    if len(station_rows) >= 8:
        for station_id, rows in station_rows.items():
            for row in rows:
                observations_by_time.setdefault(row["time"], {})[station_id] = float(row["actual"])
    else:
        station_rows = {
            station["id"]: [{"coordinates": metadata["station_coordinates"][station["id"]]}]
            for station in replay["stations"]
            if station["id"] in metadata["station_coordinates"]
        }
        seed = load_seed_observations(project_root, set(station_rows))
        for row in seed.itertuples(index=False):
            observations_by_time.setdefault(str(row.time), {})[str(row.station_id)] = float(row.actual)

    spatial_by_station = {}
    for station_id, rows in station_rows.items():
        coordinates = rows[0]["coordinates"]
        spatial_by_station[station_id] = extractor.extract(
            float(coordinates["longitude"]), float(coordinates["latitude"]), radius_m
        )

    records: list[dict[str, Any]] = []
    for time, observations in sorted(observations_by_time.items()):
        if len(observations) < 6:
            continue
        for station_id, observed in observations.items():
            peers = [value for other, value in observations.items() if other != station_id]
            if len(peers) < 5:
                continue
            spatial = spatial_by_station[station_id]
            records.append({
                "time": time,
                "station_id": station_id,
                "observed_heat_index_c": observed,
                "background_heat_index_c": float(np.mean(peers)),
                **{name: spatial[name] for name in FEATURE_NAMES if name != "background_heat_index_c"},
            })
    frame = pd.DataFrame.from_records(records)
    if frame.empty or frame.station_id.nunique() < 8:
        raise RuntimeError("Not enough station observations to train the local heat model")
    frame["local_delta_c"] = frame["observed_heat_index_c"] - frame["background_heat_index_c"]
    return frame


def candidate_models() -> dict[str, Any]:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=350,
            min_samples_leaf=8,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=350,
            min_samples_leaf=8,
            max_features=0.9,
            random_state=42,
            n_jobs=-1,
        ),
    }


def grouped_validation(frame: pd.DataFrame, model: Any) -> tuple[np.ndarray, list[dict[str, Any]]]:
    matrix = frame[FEATURE_NAMES].to_numpy(dtype=float)
    target = frame["observed_heat_index_c"].to_numpy(dtype=float)
    delta = frame["local_delta_c"].to_numpy(dtype=float)
    background = frame["background_heat_index_c"].to_numpy(dtype=float)
    groups = frame["station_id"].to_numpy()
    prediction = np.full(len(frame), np.nan, dtype=float)
    folds = []
    splitter = LeaveOneGroupOut()
    for train_index, test_index in splitter.split(matrix, target, groups):
        fitted = clone(model).fit(matrix[train_index], delta[train_index])
        lower, upper = np.quantile(delta[train_index], [0.01, 0.99])
        predicted_delta = np.clip(fitted.predict(matrix[test_index]), lower, upper)
        training_profiles = frame.iloc[train_index].drop_duplicates("station_id")[ENVIRONMENT_FEATURES].to_numpy(dtype=float)
        test_profiles = frame.iloc[test_index][ENVIRONMENT_FEATURES].to_numpy(dtype=float)
        profile_mean = training_profiles.mean(axis=0)
        profile_std = np.where(training_profiles.std(axis=0) < 1e-6, 1.0, training_profiles.std(axis=0))
        distances = np.sqrt(np.sum(
            ((test_profiles[:, None, :] - training_profiles[None, :, :]) / profile_std[None, None, :]) ** 2,
            axis=2,
        )).min(axis=1)
        predicted_delta *= np.exp(-distances / SUPPORT_DECAY_SCALE)
        fold_prediction = background[test_index] + predicted_delta
        prediction[test_index] = fold_prediction
        folds.append({
            "station": str(groups[test_index][0]),
            "rows": int(len(test_index)),
            "mae_c": round(float(mean_absolute_error(target[test_index], fold_prediction)), 4),
            "rmse_c": round(float(mean_squared_error(target[test_index], fold_prediction) ** 0.5), 4),
        })
    return prediction, folds


def validation_summary(frame: pd.DataFrame, prediction: np.ndarray, folds: list[dict[str, Any]]) -> dict[str, Any]:
    target = frame["observed_heat_index_c"].to_numpy(dtype=float)
    residual = target - prediction
    return {
        "grouped_mae_c": round(float(mean_absolute_error(target, prediction)), 4),
        "grouped_rmse_c": round(float(mean_squared_error(target, prediction) ** 0.5), 4),
        "grouped_r2": round(float(r2_score(target, prediction)), 4),
        "residual_p10_c": round(float(np.quantile(residual, 0.10)), 4),
        "residual_p90_c": round(float(np.quantile(residual, 0.90)), 4),
        "folds": folds,
    }


def feature_importance(model: Any) -> dict[str, float]:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    else:
        fitted = model[-1]
        values = np.abs(np.asarray(fitted.coef_, dtype=float))
    total = float(values.sum()) or 1.0
    return {
        name: round(float(value / total), 4)
        for name, value in sorted(zip(FEATURE_NAMES, values), key=lambda item: item[1], reverse=True)
    }


def publish_results(
    project_root: Path,
    model_path: Path,
    background_heat_index_c: float,
    radii: list[int],
) -> None:
    predictor = LocalHeatIndexModel(model_path, project_root / "dist" / "data" / "spatial")
    areas = json.loads(
        (project_root / "dist" / "data" / "spatial" / "areas.geojson").read_text(encoding="utf-8")
    )["features"]
    subzones = json.loads(
        (project_root / "dist" / "data" / "spatial" / "subzones.geojson").read_text(encoding="utf-8")
    )["features"]

    def estimate_features(features: list[dict[str, Any]], level: str) -> list[dict[str, Any]]:
        rows = []
        for feature in features:
            properties = feature["properties"]
            longitude, latitude = properties["centroid"]
            estimates = {}
            for radius in radii:
                result = predictor.predict(longitude, latitude, radius, background_heat_index_c)
                estimates[str(radius)] = {
                    "estimate_c": result["estimate_c"],
                    "interval_c": result["interval_c"],
                    "risk": result["risk"],
                    "local_adjustment_c": result["local_adjustment_c"],
                    "spatial_support": result["spatial_support"],
                    "spatial": result["spatial"],
                }
            row = {
                "id": properties["id"],
                "name": properties["display_name"],
                "centroid": properties["centroid"],
                "estimates": estimates,
            }
            if level == "subzone":
                row.update({
                    "planning_area_id": properties["planning_area_id"],
                    "planning_area": str(properties["planning_area"]).title(),
                })
            rows.append(row)
        return rows

    rows = estimate_features(areas, "planning_area")
    subzone_rows = estimate_features(subzones, "subzone")
    output = {
        "schema_version": 1,
        "model": "local-environment-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "background_heat_index_c": background_heat_index_c,
        "radii_m": radii,
        "areas": rows,
        "subzones": subzone_rows,
        "notice": "Illustrative local-environment estimates under a fixed background heat condition; not observations, forecasts or official alerts.",
    }
    path = project_root / "dist" / "data" / "local-heat-results.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(project_root: Path, report: dict[str, Any]) -> None:
    reports = project_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "local-heat-model-validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selected = report["selected_model"]
    metrics = report["candidates"][selected]
    lines = [
        "# Local environment heat-index model validation",
        "",
        "This model spatially downscales a supplied Singapore-wide background heat index. It does not predict tomorrow's weather.",
        "",
        f"- Selected model: `{selected}`",
        f"- Training rows: {report['training']['rows']:,}",
        f"- Weather stations: {report['training']['stations']}",
        f"- Search radius: {report['training']['radius_m']} m",
        f"- Leave-one-station-out MAE: {metrics['grouped_mae_c']:.3f} °C",
        f"- Leave-one-station-out RMSE: {metrics['grouped_rmse_c']:.3f} °C",
        f"- Leave-one-station-out R²: {metrics['grouped_r2']:.3f}",
        "",
        "## Inputs",
        "",
        "- Background heat index from the station network",
        "- Nearby park-cover proxy",
        "- HDB building footprint, density, floor-area ratio and height proxy",
        "- Coast distance and coastal exposure proxy",
        "- Latitude, longitude and distance to the CBD",
        "",
        "## Validation design",
        "",
        "Every fold holds out one complete weather station. This prevents repeated hourly observations from the same location appearing in both training and validation.",
        "For feature profiles far from any training station, the local adjustment is conservatively shrunk toward the network background instead of extrapolating without bound.",
        "Published map results evaluate the centroids of 55 planning areas and 332 official subzones at 250 m, 500 m and 1 km analysis radii.",
        "",
        "## Limitations",
        "",
        "The available labels cover a small station network and a short held-out replay window. Park coverage is a centroid-and-distance proxy; building coverage currently represents mapped HDB footprints; water exposure represents coastline proximity, not all inland water bodies. The result is a research estimate, not an observation, causal cooling effect, weather forecast or official alert.",
    ]
    (reports / "local-heat-model-validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--radius-m", type=int, default=500)
    parser.add_argument("--background-heat-index-c", type=float, default=34.0)
    parser.add_argument("--radii", type=int, nargs="+", default=[250, 500, 1000])
    parser.add_argument(
        "--output-model",
        type=Path,
        default=PROJECT_ROOT / "backend" / "model" / "local_heat_model.joblib",
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    frame = build_training_frame(project_root, args.radius_m)
    evaluations = {}
    predictions = {}
    fold_records = {}
    models = candidate_models()
    for name, model in models.items():
        prediction, folds = grouped_validation(frame, model)
        predictions[name] = prediction
        fold_records[name] = folds
        evaluations[name] = validation_summary(frame, prediction, folds)
    selected = min(evaluations, key=lambda name: evaluations[name]["grouped_mae_c"])
    fitted = clone(models[selected]).fit(
        frame[FEATURE_NAMES].to_numpy(dtype=float), frame["local_delta_c"].to_numpy(dtype=float)
    )
    delta_bounds = np.quantile(frame["local_delta_c"].to_numpy(dtype=float), [0.01, 0.99])
    station_profiles = frame.drop_duplicates("station_id")[ENVIRONMENT_FEATURES].to_numpy(dtype=float)
    profile_mean = station_profiles.mean(axis=0)
    profile_std = np.where(station_profiles.std(axis=0) < 1e-6, 1.0, station_profiles.std(axis=0))
    training = {
        "rows": int(len(frame)),
        "stations": int(frame.station_id.nunique()),
        "start": str(frame.time.min()),
        "end": str(frame.time.max()),
        "radius_m": int(args.radius_m),
        "target": "observed station heat index",
        "model_target": "observed heat index minus same-hour peer-station background",
        "background": "mean heat index of all other stations at the same timestamp",
        "delta_bounds_c": [round(float(delta_bounds[0]), 4), round(float(delta_bounds[1]), 4)],
    }
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected,
        "training": training,
        "candidates": evaluations,
        "feature_importance": feature_importance(fitted),
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "schema_version": 1,
        "model": fitted,
        "feature_names": FEATURE_NAMES,
        "validation": evaluations[selected],
        "training": training,
        "spatial_support": {
            "feature_names": ENVIRONMENT_FEATURES,
            "mean": profile_mean.tolist(),
            "std": profile_std.tolist(),
            "station_profiles": station_profiles.tolist(),
            "decay_scale": SUPPORT_DECAY_SCALE,
        },
    }, args.output_model)
    write_report(project_root, report)
    publish_results(project_root, args.output_model, args.background_heat_index_c, args.radii)
    print(json.dumps({
        "selected_model": selected,
        "validation": evaluations[selected],
        "model_path": str(args.output_model),
        "public_results": str(project_root / "dist" / "data" / "local-heat-results.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
