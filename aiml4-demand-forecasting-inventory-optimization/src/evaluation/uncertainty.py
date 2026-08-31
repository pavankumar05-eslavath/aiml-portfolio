"""Forecast uncertainty: prediction intervals and whether they can be trusted.

A point forecast is not enough to set inventory. The reorder point depends on how
badly demand could exceed expectation, so the **upper tail** is the quantity that
actually drives the decision. A P90 that is really a P70 causes stockouts at a rate
the service-level target never anticipated.

So intervals are not just produced, they are **validated**: empirical coverage is
measured against nominal, per horizon and per demand segment. Reporting a P10-P90
band without checking that ~80% of actuals fall inside it is the forecasting
equivalent of shipping an uncalibrated classifier.

Two constructions are compared:

* **Quantile regression** -- separate LightGBM models for P10/P50/P90. Naturally
  asymmetric and bounded below, which suits demand.
* **Empirical residual quantiles** -- add quantiles of past forecast errors, by
  horizon, to the point forecast. Distribution-free and cheap, but it assumes the
  error distribution is the same for every series, which for a panel spanning three
  orders of magnitude of volume it is not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import SERIES_ID


def coverage(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray,
) -> float:
    """Share of actuals inside the interval."""
    y, lo, hi = (np.asarray(a, dtype="float64") for a in (y, lower, upper))
    return float(np.mean((y >= lo) & (y <= hi)))


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    """Quantile (pinball) loss: the proper scoring rule for a single quantile.

    Reported because coverage alone can be gamed by an absurdly wide interval.
    Pinball rewards being both calibrated and sharp.
    """
    y, f = np.asarray(y, dtype="float64"), np.asarray(q_pred, dtype="float64")
    d = y - f
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def interval_report(
    predictions: pd.DataFrame, quantile_cols: dict[float, str],
    actual_col: str = "y_true", group: str | None = None,
) -> pd.DataFrame:
    """Coverage, width and pinball loss for the prediction intervals."""
    lo_q, hi_q = min(quantile_cols), max(quantile_cols)
    nominal = hi_q - lo_q

    def one(g: pd.DataFrame) -> pd.Series:
        y = g[actual_col].to_numpy(dtype="float64")
        lo = g[quantile_cols[lo_q]].to_numpy(dtype="float64")
        hi = g[quantile_cols[hi_q]].to_numpy(dtype="float64")
        out = {
            "n": len(g),
            "nominal_coverage": nominal,
            "empirical_coverage": coverage(y, lo, hi),
            "mean_interval_width": float(np.mean(hi - lo)),
            "median_interval_width": float(np.median(hi - lo)),
        }
        out["coverage_gap"] = out["empirical_coverage"] - nominal
        for q, col in quantile_cols.items():
            out[f"pinball_q{int(q * 100)}"] = pinball_loss(
                y, g[col].to_numpy(dtype="float64"), q)
            out[f"exceedance_above_q{int(q * 100)}"] = float(
                np.mean(y > g[col].to_numpy(dtype="float64")))
        return pd.Series(out)

    if group is None:
        return one(predictions).to_frame().T
    return (predictions.groupby(group, observed=True)[predictions.columns.tolist()]
            .apply(one).reset_index())


def empirical_residual_quantiles(
    validation_predictions: pd.DataFrame, point_col: str,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    by_horizon: bool = True,
) -> pd.DataFrame:
    """Quantiles of validation-set forecast errors, optionally per horizon.

    Fitted on VALIDATION folds only. Deriving them from the test set would make the
    intervals trivially well calibrated on the only data used to judge them.
    """
    d = validation_predictions.copy()
    d["error"] = d["y_true"] - d[point_col]
    keys = ["horizon"] if by_horizon else []
    if keys:
        out = d.groupby(keys, observed=True)["error"].quantile(list(quantiles)).unstack()
    else:
        out = d["error"].quantile(list(quantiles)).to_frame().T
    out.columns = [f"resid_q{int(q * 100)}" for q in out.columns]
    return out.reset_index() if keys else out


def apply_residual_intervals(
    predictions: pd.DataFrame, point_col: str, resid: pd.DataFrame,
    by_horizon: bool = True,
) -> pd.DataFrame:
    """Add residual-based interval columns to a predictions frame."""
    out = predictions.copy()
    if by_horizon:
        out = out.merge(resid, on="horizon", how="left")
    else:
        for c in resid.columns:
            out[c] = float(resid[c].iloc[0])
    for c in [c for c in out.columns if c.startswith("resid_q")]:
        q = c.replace("resid_", "")
        # Demand cannot be negative, so the lower bound is floored at zero.
        out[f"emp_{q}"] = np.clip(out[point_col] + out[c], 0.0, None)
    return out


def horizon_uncertainty(
    predictions: pd.DataFrame, quantile_cols: dict[float, str],
) -> pd.DataFrame:
    """How interval width and coverage change with horizon.

    Width should grow with horizon. If it does not, the intervals are not really
    conditioning on distance, and safety stock computed from them will be too tight
    at long lead times.
    """
    return interval_report(predictions, quantile_cols, group="horizon")


def lead_time_demand_distribution(
    predictions: pd.DataFrame, quantile_cols: dict[float, str],
    lead_time: int, point_col: str,
) -> pd.DataFrame:
    """Aggregate the per-day predictive distribution over a lead time, per series.

    Summing daily quantiles is **not** the quantile of the sum -- that would assume
    perfect correlation between days and overstate the spread. The sum of the point
    forecasts is exact for the mean; the spread is taken from the daily interval
    widths combined in quadrature, which assumes independence across days.
    Independence is the standard inventory assumption and is stated as one: real
    demand is positively autocorrelated, so this understates lead-time variance
    somewhat, and the empirical simulation in src/inventory is what checks whether
    the resulting service level actually holds.
    """
    lo_q, hi_q = min(quantile_cols), max(quantile_cols)
    d = predictions[predictions["horizon"] <= lead_time].copy()
    d["half_width"] = (d[quantile_cols[hi_q]] - d[quantile_cols[lo_q]]) / 2.0

    g = d.groupby(SERIES_ID, observed=True)
    out = pd.DataFrame({
        "lt_demand_point": g[point_col].sum(),
        "lt_demand_actual": g["y_true"].sum(),
        "lt_half_width_quadrature": np.sqrt(g["half_width"].apply(lambda s: (s ** 2).sum())),
        "n_days": g.size(),
    })
    out["lt_sigma_implied"] = out["lt_half_width_quadrature"] / 1.2816  # P90 of N(0,1)
    return out.reset_index()
