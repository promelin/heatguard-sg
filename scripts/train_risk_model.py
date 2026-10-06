#!/usr/bin/env python3
"""Train and export the HeatGuard planning-area risk classifier.

The project does not contain historical health outcomes.  This model is therefore
trained as a policy-calibrated classifier: observed station heat index is combined
with the official 2026 age structure and mapped eldercare access, and labels are
generated from the documented HeatGuard response matrix.  Validation is strictly
time ordered so that all planning areas from a held-out hour stay out of training.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)


FEATURES = [
    "heat_index_c",
    "elderly_density_per_km2",
    "elderly_share_pct",
    "elderly_75plus_share_pct",
    "eldercare_per_10k_seniors",
]
CLASS_NAMES = ["Low", "Medium", "High"]


def heat_index_celsius(temperature_c: np.ndarray, relative_humidity: np.ndarray) -> np.ndarray:
    """Vectorised NOAA Rothfusz heat index, matching backend/predictor.py."""
    temperature_f = temperature_c * 9.0 / 5.0 + 32.0
    rh = relative_humidity
    hi_f = (
        -42.379
        + 2.04901523 * temperature_f
        + 10.14333127 * rh
        - 0.22475541 * temperature_f * rh
        - 0.00683783 * temperature_f**2
        - 0.05481717 * rh**2
        + 0.00122874 * temperature_f**2 * rh
        + 0.00085282 * temperature_f * rh**2
        - 0.00000199 * temperature_f**2 * rh**2
    )
    low_adjustment = ((13 - rh) / 4) * np.sqrt(np.maximum(0.0, (17 - np.abs(temperature_f - 95)) / 17))
    hi_f = np.where((rh < 13) & (temperature_f >= 80) & (temperature_f <= 112), hi_f - low_adjustment, hi_f)
    high_adjustment = ((rh - 85) / 10) * ((87 - temperature_f) / 5)
    hi_f = np.where((rh > 85) & (temperature_f >= 80) & (temperature_f <= 87), hi_f + high_adjustment, hi_f)
    hi_f = np.where((temperature_f < 80) | (rh < 40), temperature_f, hi_f)
    return (hi_f - 32.0) * 5.0 / 9.0


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy() * 100.0


def load_area_features(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for feature in payload["features"]:
        p = feature["properties"]
        seniors = float(p.get("elderly_65plus_2026") or 0)
        residents = float(p.get("population_2026") or 0)
        elderly75 = float(p.get("elderly_75plus_2026") or 0)
        area_km2 = max(float(p.get("area_km2") or 0), 0.01)
        centres = float(p.get("cooling_centres") or 0)
        rows.append(
            {
                "area_id": p["id"],
                "area_name": p["display_name"],
                "longitude": float(p["centroid"][0]),
                "latitude": float(p["centroid"][1]),
                "residents": residents,
                "seniors": seniors,
                "elderly75": elderly75,
                "elderly_density_per_km2": seniors / area_km2,
                "elderly_share_pct": 100.0 * seniors / residents if residents else 0.0,
                "elderly_75plus_share_pct": 100.0 * elderly75 / residents if residents else 0.0,
                "eldercare_per_10k_seniors": 10_000.0 * centres / seniors if seniors else 0.0,
            }
        )
    areas = pd.DataFrame(rows)
    areas["community_vulnerability_score"] = (
        0.50 * percentile_rank(areas["seniors"].to_numpy())
        + 0.30 * percentile_rank(areas["elderly_share_pct"].to_numpy())
        + 0.20 * percentile_rank(areas["elderly75"].to_numpy())
    )
    access = areas["eldercare_per_10k_seniors"].to_numpy()
    areas["access_gap_score"] = np.where(
        areas["seniors"].to_numpy() > 0,
        100.0 - np.clip(access / 2.2 * 100.0, 0.0, 100.0),
        25.0,
    )
    return areas


def load_hourly_heat(cache_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    connection = sqlite3.connect(cache_path)
    try:
        hourly = pd.read_sql_query(
            """
            SELECT substr(timestamp, 1, 13) || ':00:00+00:00' AS hour,
                   station_id,
                   AVG(CASE WHEN metric = 'temperature' THEN value END) AS temperature_c,
                   AVG(CASE WHEN metric = 'humidity' THEN value END) AS humidity_pct,
                   COUNT(CASE WHEN metric = 'temperature' THEN 1 END) AS temperature_readings,
                   COUNT(CASE WHEN metric = 'humidity' THEN 1 END) AS humidity_readings
            FROM observations
            WHERE metric IN ('temperature', 'humidity')
            GROUP BY hour, station_id
            HAVING temperature_c IS NOT NULL AND humidity_pct IS NOT NULL
            ORDER BY hour, station_id
            """,
            connection,
        )
        stations = pd.read_sql_query(
            """
            SELECT station_id, MAX(name) AS station_name,
                   MAX(latitude) AS latitude, MAX(longitude) AS longitude
            FROM stations
            WHERE metric IN ('temperature', 'humidity')
            GROUP BY station_id
            HAVING latitude IS NOT NULL AND longitude IS NOT NULL
            """,
            connection,
        )
    finally:
        connection.close()
    hourly["timestamp"] = pd.to_datetime(hourly.pop("hour"), utc=True)
    hourly = hourly[(hourly.temperature_readings >= 6) & (hourly.humidity_readings >= 6)].copy()
    hourly["heat_index_c"] = heat_index_celsius(hourly.temperature_c.to_numpy(), hourly.humidity_pct.to_numpy())
    hourly = hourly.merge(stations, on="station_id", how="inner")
    hourly = hourly[np.isfinite(hourly.heat_index_c)].copy()
    return hourly, stations


def inverse_distance(area_lon: float, area_lat: float, station_rows: pd.DataFrame) -> float:
    dx = (area_lon - station_rows.longitude.to_numpy()) * math.cos(area_lat * math.pi / 180.0)
    dy = area_lat - station_rows.latitude.to_numpy()
    weights = 1.0 / (dx * dx + dy * dy + 0.000025)
    return float(np.sum(station_rows.heat_index_c.to_numpy() * weights) / np.sum(weights))


def build_examples(hourly: pd.DataFrame, areas: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for timestamp, station_rows in hourly.groupby("timestamp", sort=True):
        if len(station_rows) < 10:
            continue
        for area in areas.itertuples(index=False):
            records.append(
                {
                    "timestamp": timestamp,
                    "area_id": area.area_id,
                    "area_name": area.area_name,
                    "heat_index_c": inverse_distance(area.longitude, area.latitude, station_rows),
                    "elderly_density_per_km2": area.elderly_density_per_km2,
                    "elderly_share_pct": area.elderly_share_pct,
                    "elderly_75plus_share_pct": area.elderly_75plus_share_pct,
                    "eldercare_per_10k_seniors": area.eldercare_per_10k_seniors,
                    "community_vulnerability_score": area.community_vulnerability_score,
                    "access_gap_score": area.access_gap_score,
                }
            )
    frame = pd.DataFrame.from_records(records)
    return frame.sort_values(["timestamp", "area_id"]).reset_index(drop=True)


def policy_labels(frame: pd.DataFrame) -> np.ndarray:
    """Translate the documented response matrix into Low/Medium/High labels.

    Heat index (not WBGT) uses the website's existing 34/37/40 C response bands.
    Heat >=34 C is at least Medium. High is reserved for a joint heat/community
    condition so that an island-wide hot hour still produces a useful priority
    ranking rather than classifying all 55 planning areas identically.
    """
    heat = frame.heat_index_c.to_numpy()
    vulnerability = frame.community_vulnerability_score.to_numpy()
    elevated = vulnerability >= 60.0
    priority = vulnerability >= 80.0
    access_gap = frame.access_gap_score.to_numpy() >= 60.0
    labels = np.zeros(len(frame), dtype=np.int64)
    labels[heat >= 34.0] = 1
    high = (
        ((heat >= 34.0) & (heat < 37.0) & priority & access_gap)
        | ((heat >= 37.0) & (heat < 40.0) & (priority | (elevated & access_gap)))
        | (heat >= 40.0)
    )
    labels[high] = 2
    return labels


def predict_from_thresholds(scores: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    low_medium, medium_high = thresholds
    return np.where(scores >= medium_high, 2, np.where(scores >= low_medium, 1, 0)).astype(np.int64)


def optimise_thresholds(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    best: tuple[float, float, float] | None = None
    best_key: tuple[float, float, float] | None = None
    for low_medium in np.arange(10.0, 61.0, 1.0):
        for medium_high in np.arange(max(low_medium + 8.0, 45.0), 96.0, 1.0):
            prediction = predict_from_thresholds(scores, (low_medium, medium_high))
            matrix = np.bincount(y_true * 3 + prediction, minlength=9).reshape(3, 3)
            true_positive = np.diag(matrix).astype(float)
            precision = np.divide(true_positive, matrix.sum(axis=0), out=np.zeros(3), where=matrix.sum(axis=0) > 0)
            recall = np.divide(true_positive, matrix.sum(axis=1), out=np.zeros(3), where=matrix.sum(axis=1) > 0)
            class_f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=(precision + recall) > 0)
            macro_f1 = float(class_f1.mean())
            high_recall = float(recall[2])
            high_false_alarm = np.mean((prediction == 2) & (y_true != 2))
            objective = macro_f1 + 0.20 * high_recall - 0.08 * high_false_alarm
            # When a whole interval is equally accurate, choose the point with
            # the largest gap to observed validation scores instead of the
            # first/lowest grid value.  This makes the cut-off less brittle.
            decision_margin = float(
                np.min(np.abs(scores - low_medium))
                + np.min(np.abs(scores - medium_high))
            )
            candidate = (round(objective, 12), decision_margin, -abs(low_medium - 25.0) - abs(medium_high - 75.0))
            if best_key is None or candidate > best_key:
                best = (float(low_medium), float(medium_high), float(objective))
                best_key = candidate
    assert best is not None
    return best


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "classes": {
            CLASS_NAMES[index]: {
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "support": int(support[index]),
            }
            for index in range(3)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).astype(int).tolist(),
    }


def bootstrap_thresholds(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    repeats: int = 80,
) -> dict[str, object]:
    rng = np.random.default_rng(42)
    unique_hours = np.unique(timestamps)
    estimates = []
    for _ in range(repeats):
        sampled = rng.choice(unique_hours, size=len(unique_hours), replace=True)
        indices = np.concatenate([np.flatnonzero(timestamps == hour) for hour in sampled])
        if len(np.unique(y_true[indices])) < 3:
            continue
        low_medium, medium_high, _ = optimise_thresholds(y_true[indices], scores[indices])
        estimates.append((low_medium, medium_high))
    values = np.asarray(estimates, dtype=float)
    return {
        "bootstrap_time_block_repeats": int(len(values)),
        "low_medium_p05_p50_p95": [round(float(v), 1) for v in np.quantile(values[:, 0], [0.05, 0.5, 0.95])],
        "medium_high_p05_p50_p95": [round(float(v), 1) for v in np.quantile(values[:, 1], [0.05, 0.5, 0.95])],
    }


def export_forest(model: RandomForestClassifier) -> list[dict[str, object]]:
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        values = tree.value[:, 0, :].astype(float)
        totals = values.sum(axis=1, keepdims=True)
        values = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
        trees.append(
            {
                "childrenLeft": tree.children_left.astype(int).tolist(),
                "childrenRight": tree.children_right.astype(int).tolist(),
                "feature": tree.feature.astype(int).tolist(),
                "threshold": np.round(tree.threshold.astype(float), 7).tolist(),
                "probabilities": np.round(values, 8).tolist(),
            }
        )
    return trees


def write_markdown(report: dict[str, object], output: Path) -> None:
    test = report["test_metrics"]
    stability = report["threshold_validation"]
    rows = [
        "# HeatGuard risk-model validation",
        "",
        "> This is a policy-calibrated planning classifier, not a medical-outcome model. No historical health-event label is present in the supplied data.",
        "",
        "## Data and split",
        "",
        f"- {report['samples']:,} planning-area-hour samples across {report['hours']} held-out-by-time hours and 55 planning areas.",
        f"- Observed period: {report['observed_period']['start']} to {report['observed_period']['end']}.",
        f"- Split: {report['split']['train_hours']} train / {report['split']['validation_hours']} validation / {report['split']['test_hours']} test hours.",
        "- Inputs: observed heat index, 65+ density/share, 75+ share, and mapped eldercare access per 10,000 older residents.",
        "",
        "## Validated score thresholds",
        "",
        f"- Low: score < {report['thresholds']['low_medium']:.0f}",
        f"- Medium: {report['thresholds']['low_medium']:.0f} <= score < {report['thresholds']['medium_high']:.0f}",
        f"- High: score >= {report['thresholds']['medium_high']:.0f}",
        f"- Time-block bootstrap 5th–95th percentile: Low/Medium {stability['low_medium_p05_p50_p95'][0]:.0f}–{stability['low_medium_p05_p50_p95'][2]:.0f}; Medium/High {stability['medium_high_p05_p50_p95'][0]:.0f}–{stability['medium_high_p05_p50_p95'][2]:.0f}.",
        "",
        "## Held-out test result",
        "",
        f"- Accuracy: {test['accuracy']:.4f}",
        f"- Balanced accuracy: {test['balanced_accuracy']:.4f}",
        f"- Macro-F1: {test['macro_f1']:.4f}",
        f"- High recall: {test['classes']['High']['recall']:.4f}",
        f"- High precision: {test['classes']['High']['precision']:.4f}",
        f"- Confusion matrix (rows actual, columns predicted; Low / Medium / High): `{test['confusion_matrix']}`",
        "",
        "## Interpretation",
        "",
        "The model reproduces the documented HeatGuard planning policy on unseen hours. These metrics validate implementation and score cut-offs against that policy target; they do not validate hospitalisation, mortality, or individual health risk.",
    ]
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--weather-cache", type=Path, required=True)
    parser.add_argument("--areas", type=Path, default=root / "dist/data/spatial/areas.geojson")
    parser.add_argument("--output-model", type=Path, default=root / "backend/model/risk_model.joblib")
    parser.add_argument("--output-web", type=Path, default=root / "dist/data/risk-model.json")
    parser.add_argument("--output-report", type=Path, default=root / "reports/risk-model-validation.json")
    args = parser.parse_args()

    areas = load_area_features(args.areas)
    hourly, stations = load_hourly_heat(args.weather_cache)
    examples = build_examples(hourly, areas)
    examples["label"] = policy_labels(examples)

    hours = np.array(sorted(examples.timestamp.unique()))
    train_end = max(1, int(len(hours) * 0.60))
    validation_end = max(train_end + 1, int(len(hours) * 0.80))
    train_hours = hours[:train_end]
    validation_hours = hours[train_end:validation_end]
    test_hours = hours[validation_end:]
    train = examples[examples.timestamp.isin(train_hours)]
    validation = examples[examples.timestamp.isin(validation_hours)]
    test = examples[examples.timestamp.isin(test_hours)]

    model = RandomForestClassifier(
        n_estimators=240,
        max_depth=9,
        min_samples_leaf=18,
        max_features=0.9,
        class_weight={0: 1.0, 1: 1.15, 2: 1.45},
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[FEATURES], train.label)

    validation_probability = model.predict_proba(validation[FEATURES])
    validation_score = 50.0 * validation_probability[:, 1] + 100.0 * validation_probability[:, 2]
    low_medium, medium_high, objective = optimise_thresholds(validation.label.to_numpy(), validation_score)
    thresholds = (low_medium, medium_high)

    test_probability = model.predict_proba(test[FEATURES])
    test_score = 50.0 * test_probability[:, 1] + 100.0 * test_probability[:, 2]
    test_prediction = predict_from_thresholds(test_score, thresholds)
    validation_prediction = predict_from_thresholds(validation_score, thresholds)

    label_counts = Counter(CLASS_NAMES[int(value)] for value in examples.label)
    train_counts = Counter(CLASS_NAMES[int(value)] for value in train.label)
    test_counts = Counter(CLASS_NAMES[int(value)] for value in test.label)
    stability = bootstrap_thresholds(
        validation.timestamp.astype("int64").to_numpy(),
        validation.label.to_numpy(),
        validation_score,
    )
    report = {
        "model": "RandomForestClassifier",
        "model_role": "Policy-calibrated planning-area heat-risk classifier",
        "label_limitation": "No historical health outcomes were supplied. Labels encode the documented HeatGuard response matrix and are not medical outcomes.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
        "samples": int(len(examples)),
        "hours": int(len(hours)),
        "stations": int(hourly.station_id.nunique()),
        "observed_period": {"start": str(hours[0]), "end": str(hours[-1])},
        "split": {
            "method": "chronological 60/20/20 by complete observation hour",
            "train_hours": int(len(train_hours)),
            "validation_hours": int(len(validation_hours)),
            "test_hours": int(len(test_hours)),
            "train_end": str(train_hours[-1]),
            "validation_end": str(validation_hours[-1]),
            "test_start": str(test_hours[0]),
        },
        "label_policy": {
            "Low": "Observed heat index <34 C",
            "Medium": ">=34 C unless the joint High criteria are met",
            "High": "34 to <37 C: priority+gap; 37 to <40 C: priority or elevated+gap; >=40 C: all areas",
            "community_vulnerability": "0.50*percentile(65+ count) + 0.30*percentile(65+ share) + 0.20*percentile(75+ count)",
        },
        "class_counts": dict(label_counts),
        "train_class_counts": dict(train_counts),
        "test_class_counts": dict(test_counts),
        "thresholds": {"low_medium": low_medium, "medium_high": medium_high, "score": "50*P(Medium) + 100*P(High)"},
        "threshold_objective": round(objective, 6),
        "threshold_validation": stability,
        "validation_metrics": metric_summary(validation.label.to_numpy(), validation_prediction),
        "test_metrics": metric_summary(test.label.to_numpy(), test_prediction),
        "feature_importance": {
            name: round(float(value), 5)
            for name, value in sorted(zip(FEATURES, model.feature_importances_), key=lambda item: item[1], reverse=True)
        },
        "station_rows_after_quality_filter": int(len(hourly)),
        "station_metadata_rows": int(len(stations)),
    }

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_web.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "class_names": CLASS_NAMES,
            "thresholds": report["thresholds"],
            "metadata": report,
        },
        args.output_model,
    )
    web_model = {
        "version": "risk-rf-v1",
        "modelType": "random-forest-classifier",
        "modelRole": report["model_role"],
        "labelLimitation": report["label_limitation"],
        "features": FEATURES,
        "classes": CLASS_NAMES,
        "thresholds": report["thresholds"],
        "metrics": {
            "testMacroF1": report["test_metrics"]["macro_f1"],
            "testBalancedAccuracy": report["test_metrics"]["balanced_accuracy"],
            "testHighRecall": report["test_metrics"]["classes"]["High"]["recall"],
            "testHighPrecision": report["test_metrics"]["classes"]["High"]["precision"],
        },
        "featureImportance": report["feature_importance"],
        "trees": export_forest(model),
    }
    args.output_web.write_text(json.dumps(web_model, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, args.output_report.with_suffix(".md"))

    print(json.dumps({
        "samples": report["samples"],
        "hours": report["hours"],
        "class_counts": report["class_counts"],
        "thresholds": report["thresholds"],
        "validation": report["validation_metrics"],
        "test": report["test_metrics"],
        "threshold_stability": report["threshold_validation"],
        "feature_importance": report["feature_importance"],
    }, indent=2))


if __name__ == "__main__":
    main()
