"""Error analysis: which series the model forecasts badly, and why.

An aggregate WAPE says nothing actionable. What a planner needs is: *which* SKUs to
stop trusting the forecast for, and *what* about them breaks it. This module
attributes error to measurable properties of each series -- intermittency,
volatility, volume, seasonality strength, price movement -- so the explanation is
evidence rather than speculation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import DATE, SERIES_ID, TARGET


def series_errors(
    predictions: pd.DataFrame, model_col: str, actual_col: str = "y_true",
) -> pd.DataFrame:
    """Per-series accuracy for one model."""
    d = predictions.copy()
    d["abs_err"] = (d[actual_col] - d[model_col]).abs()
    d["err"] = d[model_col] - d[actual_col]
    g = d.groupby(SERIES_ID, observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "total_actual": g[actual_col].sum(),
        "total_pred": g[model_col].sum(),
        "mae": g["abs_err"].mean(),
        "rmse": g.apply(lambda x: float(np.sqrt(np.mean((x[actual_col] - x[model_col]) ** 2))),
                        include_groups=False),
        "bias": g["err"].mean(),
    })
    out["wape"] = g["abs_err"].sum() / g[actual_col].sum().replace(0, np.nan)
    return out.reset_index()


def series_characteristics(panel: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Measurable properties of each series, computed from history only."""
    hist = panel[panel[DATE] <= as_of]
    g = hist.groupby(SERIES_ID, observed=True)

    out = pd.DataFrame({
        "mean_demand": g[TARGET].mean(),
        "std_demand": g[TARGET].std(),
        "total_demand": g[TARGET].sum(),
        "zero_share": g[TARGET].apply(lambda s: float((s == 0).mean())),
        "history_days": g[DATE].size(),
        "price_mean": g["sell_price"].mean(),
        "price_cv": g["sell_price"].std() / g["sell_price"].mean().replace(0, np.nan),
        "snap_share": g["snap"].mean(),
    })
    out["cv_demand"] = out["std_demand"] / out["mean_demand"].replace(0, np.nan)

    # Strength of the weekly cycle: how much of the variance day-of-week explains.
    # A high value means the seasonal signal is learnable; a low one means the
    # series is mostly noise and no amount of feature engineering will help.
    def weekly_strength(s: pd.DataFrame) -> float:
        y = s[TARGET].to_numpy(dtype="float64")
        if y.std() < 1e-9:
            return 0.0
        dow = s[DATE].dt.dayofweek.to_numpy()
        means = np.array([y[dow == d].mean() if (dow == d).any() else y.mean()
                          for d in range(7)])
        return float(np.var(means[dow]) / np.var(y))

    out["weekly_strength"] = hist.groupby(SERIES_ID, observed=True)[[DATE, TARGET]] \
        .apply(weekly_strength, include_groups=False)
    return out.reset_index()


def error_attribution(
    predictions: pd.DataFrame, model_col: str, panel: pd.DataFrame,
    as_of: pd.Timestamp, segments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join per-series error to per-series characteristics."""
    err = series_errors(predictions, model_col)
    ch = series_characteristics(panel, as_of)
    out = err.merge(ch, on=SERIES_ID, how="left")
    if segments is not None:
        cols = [c for c in ("demand_class", "volume_class", "segment", "cat_id",
                            "store_id") if c in segments.columns]
        out = out.merge(segments[cols], left_on=SERIES_ID, right_index=True, how="left")
    return out


def worst_series(attribution: pd.DataFrame, n: int = 15,
                 min_actual: float = 10.0) -> pd.DataFrame:
    """Highest-WAPE series, restricted to ones with enough demand to be meaningful.

    The filter matters: a series with 2 units of total demand can have a WAPE of 4.0
    from a single unit of error, which is noise rather than a forecasting failure.
    """
    d = attribution[attribution["total_actual"] >= min_actual]
    cols = [c for c in (SERIES_ID, "demand_class", "volume_class", "cat_id",
                        "total_actual", "wape", "mae", "bias", "zero_share",
                        "cv_demand", "weekly_strength", "price_cv")
            if c in d.columns]
    return d.nlargest(n, "wape").loc[:, cols].reset_index(drop=True)


def error_drivers(attribution: pd.DataFrame, target: str = "wape") -> pd.DataFrame:
    """Correlation between series properties and forecast error.

    Spearman, because the relationships are monotone but not linear and the
    characteristics have long tails.
    """
    cols = ["zero_share", "cv_demand", "mean_demand", "total_demand",
            "weekly_strength", "price_cv", "snap_share", "history_days"]
    d = attribution.replace([np.inf, -np.inf], np.nan)
    d = d[d["total_actual"] >= 10.0]
    rows = []
    for c in cols:
        if c not in d.columns:
            continue
        sub = d[[c, target]].dropna()
        if len(sub) < 10:
            continue
        rho = float(sub[c].corr(sub[target], method="spearman"))
        rows.append({"characteristic": c, "spearman_vs_" + target: rho,
                     "n": len(sub)})
    return (pd.DataFrame(rows)
            .sort_values("spearman_vs_" + target, key=abs, ascending=False)
            .reset_index(drop=True))


def promo_effect_errors(
    predictions: pd.DataFrame, panel: pd.DataFrame, model_col: str,
) -> pd.DataFrame:
    """Accuracy on SNAP vs non-SNAP days, and on discounted vs normal days.

    Promotional periods are where forecast error costs most, because they are when
    demand spikes and a stockout is most visible.
    """
    from src.evaluation.metrics import score

    exog = panel.set_index([SERIES_ID, DATE])[["sell_price", "snap"]]
    keys = pd.MultiIndex.from_arrays(
        [predictions[SERIES_ID], predictions["target_date"]])
    d = predictions.copy()
    d["snap"] = exog["snap"].reindex(keys).to_numpy()

    rows = []
    for label, mask in (("snap_day", d["snap"] == 1), ("non_snap_day", d["snap"] == 0)):
        g = d[mask]
        if g.empty:
            continue
        s = score(g["y_true"], g[model_col], g[SERIES_ID])
        rows.append({"condition": label, "n": s.n, "mean_actual": float(g["y_true"].mean()),
                     "wape": s.wape, "mae": s.mae, "bias": s.bias})
    return pd.DataFrame(rows)
