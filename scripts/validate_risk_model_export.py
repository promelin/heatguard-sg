#!/usr/bin/env python3
"""Confirm that browser JSON inference matches the Python risk model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def browser_predict(model: dict[str, object], values: np.ndarray) -> np.ndarray:
    output = np.zeros((len(values), 3), dtype=float)
    for row_index, row in enumerate(values):
        for tree in model["trees"]:
            node = 0
            while tree["feature"][node] >= 0:
                feature = tree["feature"][node]
                node = tree["childrenLeft"][node] if row[feature] <= tree["threshold"][node] else tree["childrenRight"][node]
            output[row_index] += np.asarray(tree["probabilities"][node], dtype=float)
    return output / len(model["trees"])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-model", type=Path, default=root / "backend/model/risk_model.joblib")
    parser.add_argument("--web-model", type=Path, default=root / "dist/data/risk-model.json")
    args = parser.parse_args()

    package = joblib.load(args.python_model)
    web_model = json.loads(args.web_model.read_text(encoding="utf-8"))
    rng = np.random.default_rng(42)
    values = np.column_stack(
        [
            rng.uniform(28, 45, 500),
            rng.uniform(0, 5_000, 500),
            rng.uniform(0, 35, 500),
            rng.uniform(0, 18, 500),
            rng.uniform(0, 4, 500),
        ]
    )
    frame = pd.DataFrame(values, columns=package["features"])
    python_probability = package["model"].predict_proba(frame)
    browser_probability = browser_predict(web_model, values)
    maximum_error = float(np.max(np.abs(python_probability - browser_probability)))
    if maximum_error > 1e-6:
        raise SystemExit(f"Browser export mismatch: maximum probability error {maximum_error}")
    print(json.dumps({"rows": len(values), "maximum_probability_error": maximum_error, "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
