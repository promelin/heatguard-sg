# HeatGuard risk-model validation

> This is a policy-calibrated planning classifier, not a medical-outcome model. No historical health-event label is present in the supplied data.

## Data and split

- 11,055 planning-area-hour samples across 201 held-out-by-time hours and 55 planning areas.
- Observed period: 2026-09-27 16:00:00+00:00 to 2026-10-06 00:00:00+00:00.
- Split: 120 train / 40 validation / 41 test hours.
- Inputs: observed heat index, 65+ density/share, 75+ share, and mapped eldercare access per 10,000 older residents.

## Validated score thresholds

- Low: score < 29
- Medium: 29 <= score < 76
- High: score >= 76
- Time-block bootstrap 5th–95th percentile: Low/Medium 25–29; Medium/High 76–79.

## Held-out test result

- Accuracy: 0.9987
- Balanced accuracy: 0.9948
- Macro-F1: 0.9970
- High recall: 0.9843
- High precision: 1.0000
- Confusion matrix (rows actual, columns predicted; Low / Medium / High): `[[734, 0, 0], [0, 1330, 0], [0, 3, 188]]`

## Interpretation

The model reproduces the documented HeatGuard planning policy on unseen hours. These metrics validate implementation and score cut-offs against that policy target; they do not validate hospitalisation, mortality, or individual health risk.
