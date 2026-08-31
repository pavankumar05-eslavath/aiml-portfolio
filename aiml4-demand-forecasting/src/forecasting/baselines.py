"""Baseline forecasters.

These are the bar the ML models have to clear. On intermittent retail demand they
are not straw men: seasonal naive exploits the strong day-of-week cycle, and
Croston is the standard method for exactly the sparse series that dominate this
panel. A gradient-booster that cannot beat them has not earned its complexity.

All baselines are computed from the same origin-anchored feature frame the ML
models use, so no baseline can accidentally see data the ML models cannot, and
vice versa. That is what makes the comparison fair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: name -> human description, used in reports.
BASELINE_DESCRIPTIONS: dict[str, str] = {
    "naive": "Repeat the last observed day (demand at the forecast origin).",
    "seasonal_naive": "Repeat the same weekday, stepped back in whole weeks to the "
                      "last one at or before the origin.",
    "moving_average_7": "Mean of the last 7 days at the origin.",
    "moving_average_28": "Mean of the last 28 days at the origin.",
    "croston": "Croston's method: exponentially smoothed non-zero demand size "
               "divided by smoothed inter-arrival interval, updated only when a "
               "sale occurs.",
    "sba": "Syntetos-Boylan Approximation: Croston scaled by (1 - alpha/2) to "
           "correct its known upward bias.",
    "zero": "Always forecast zero. Included because ~59% of observations are zero, "
            "so it exposes any metric that rewards predicting nothing.",
}


def predict(name: str, features: pd.DataFrame) -> np.ndarray:
    """Produce baseline predictions for a supervised frame.

    Args:
        name: baseline identifier from BASELINE_DESCRIPTIONS.
        features: frame from ``build_supervised``.

    Returns:
        Non-negative predictions aligned to ``features``' rows.
    """
    if name == "naive":
        out = features["lag_1"]
    elif name == "seasonal_naive":
        out = features["snaive_lag"]
    elif name == "moving_average_7":
        out = features["rolling_mean_7"]
    elif name == "moving_average_28":
        out = features["rolling_mean_28"]
    elif name == "croston":
        out = features["croston_at_origin"]
    elif name == "sba":
        out = features["sba_at_origin"]
    elif name == "zero":
        out = pd.Series(0.0, index=features.index)
    else:
        raise ValueError(f"unknown baseline {name!r}")

    # Demand cannot be negative, and a missing feature (a series too young to have
    # a 28-day window) is treated as no information rather than as zero demand
    # elsewhere -- but for a point forecast the safest fallback is the series' own
    # recent level, then zero.
    out = out.astype("float64")
    if name != "zero":
        out = out.fillna(features["rolling_mean_7"]).fillna(features["lag_1"]).fillna(0.0)
    return np.clip(out.to_numpy(), 0.0, None)


def all_baselines() -> list[str]:
    return list(BASELINE_DESCRIPTIONS)


def mase_denominator(
    history: pd.DataFrame, series_col: str, date_col: str, target: str,
    season: int = 7,
) -> pd.Series:
    """In-sample seasonal-naive MAE per series, the MASE scaling factor.

    MASE divides forecast error by the error a seasonal-naive model makes on the
    training data. That makes it scale-free and comparable across series, which
    plain MAE is not: an MAE of 2 is excellent for a series selling 100 a day and
    useless for one selling 0.2.

    A series with a zero denominator (never changes between same-weekday
    observations) yields NaN rather than infinity, and is excluded from aggregate
    MASE instead of silently dominating it.
    """
    h = history.sort_values([series_col, date_col])
    diff = h.groupby(series_col, observed=True)[target].diff(season).abs()
    denom = diff.groupby(h[series_col], observed=True).mean()
    return denom.replace(0.0, np.nan)
