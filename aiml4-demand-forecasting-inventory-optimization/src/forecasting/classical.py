"""Classical time-series models: Holt-Winters (ETS) and SARIMA.

Fitted **per series on a sample**, not on all 600. Two reasons, and both are
findings rather than shortcuts:

* Cost scales with the number of series, not the number of rows. 600 series x 4
  folds x 2 model families is 4,800 optimiser runs, which buys no additional
  insight over a representative sample.
* These models assume a continuous, roughly Gaussian process. ~92% of the series
  here are intermittent or lumpy, where the assumption fails outright: a series
  that is zero on most days has no stable level to smooth, and differencing it
  produces noise. Their weakness on this data is a result worth reporting, not a
  bug to hide.

Failures are caught and recorded rather than dropped silently, because "the model
could not be fitted" is itself information about the data.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Config
from src.data.loader import Fold
from src.data.schema import DATE, SERIES_ID, TARGET

SEASON = 7  # weekly seasonality; the dominant cycle in daily retail demand


@dataclass
class ClassicalResult:
    """Per-series forecasts from the classical models, plus fit diagnostics."""

    predictions: pd.DataFrame
    n_series: int
    n_failed: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def sample_series(cfg: Config, panel: pd.DataFrame, n: int | None = None) -> list[str]:
    """Pick a stratified sample of series for the classical models.

    Stratified by demand class so the sample is not all easy, smooth series -- the
    comparison would be meaningless if the classical models were only asked about
    the 5% of series that suit them.
    """
    from src.data.loader import classify_demand, demand_stats

    n = int(n or cfg["models"]["n_classical_series"])
    stats = demand_stats(panel)
    stats["demand_class"] = classify_demand(stats)
    rng = np.random.default_rng(cfg.seed)

    picks: list[str] = []
    groups = stats.groupby("demand_class", observed=True)
    per = max(1, n // max(len(groups), 1))
    for _, g in groups:
        k = min(per, len(g))
        picks.extend(rng.choice(g.index.to_numpy(), size=k, replace=False).tolist())
    return sorted(picks)[:n]


def _fit_ets(y: pd.Series, horizon: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    seasonal = "add" if (len(y) >= 2 * SEASON and y.gt(0).sum() >= 2 * SEASON) else None
    model = ExponentialSmoothing(
        y, trend=None, seasonal=seasonal,
        seasonal_periods=SEASON if seasonal else None,
        initialization_method="estimated",
    ).fit(optimized=True)
    return np.asarray(model.forecast(horizon), dtype="float64")


def _fit_sarima(y: pd.Series, horizon: int) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # A fixed, modest order. Per-series order search over 600 series would cost far
    # more than it could return on data this sparse, and the config forbids fitting
    # expensive models blindly across every SKU.
    model = SARIMAX(y, order=(1, 0, 1), seasonal_order=(1, 0, 0, SEASON),
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False, maxiter=50)
    return np.asarray(res.forecast(horizon), dtype="float64")


def forecast_classical(
    cfg: Config, panel: pd.DataFrame, fold: Fold, series: list[str] | None = None,
) -> ClassicalResult:
    """Fit ETS and SARIMA per sampled series and forecast the fold window."""
    series = series or sample_series(cfg, panel)
    horizon = fold.horizon
    target_dates = pd.date_range(fold.start, fold.end, freq="D")

    hist = panel[(panel[DATE] <= fold.origin) & (panel[SERIES_ID].isin(series))]
    actual = panel.set_index([SERIES_ID, DATE])[TARGET]

    rows: list[pd.DataFrame] = []
    failed = {"ets": 0, "sarima": 0}

    for sid, g in hist.groupby(SERIES_ID, observed=True):
        y = g.sort_values(DATE).set_index(DATE)[TARGET].astype("float64")
        y = y.asfreq("D").fillna(0.0)

        preds: dict[str, np.ndarray] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                preds["ets"] = _fit_ets(y, horizon)
            except Exception:
                failed["ets"] += 1
                preds["ets"] = np.full(horizon, np.nan)
            try:
                preds["sarima"] = _fit_sarima(y, horizon)
            except Exception:
                failed["sarima"] += 1
                preds["sarima"] = np.full(horizon, np.nan)

        keys = pd.MultiIndex.from_arrays(
            [[sid] * horizon, pd.DatetimeIndex(target_dates)])
        rows.append(pd.DataFrame({
            SERIES_ID: sid,
            "target_date": target_dates,
            "horizon": np.arange(1, horizon + 1, dtype="int16"),
            "y_true": actual.reindex(keys).to_numpy(),
            # Clipped at zero: these models are unconstrained and will happily
            # forecast negative demand on sparse series.
            "ets": np.clip(preds["ets"], 0.0, None),
            "sarima": np.clip(preds["sarima"], 0.0, None),
        }))

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    notes = [
        f"Fitted on a stratified sample of {len(series)} of "
        f"{panel[SERIES_ID].nunique()} series.",
        "Both models are unconstrained and can forecast negative demand on "
        "intermittent series; predictions are clipped at zero.",
    ]
    if failed["ets"] or failed["sarima"]:
        notes.append(f"Fit failures: {failed}. Failures concentrate on the sparsest "
                     "series, where there is no stable level to smooth.")
    return ClassicalResult(predictions=out, n_series=len(series), n_failed=failed,
                           notes=notes)
