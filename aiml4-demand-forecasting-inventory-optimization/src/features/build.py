"""Supervised feature construction for multi-horizon forecasting.

## The design decision that makes this leak-free

Every target-derived feature is computed **as of the forecast origin**, never as
of the target date.

Concretely: to forecast day `origin + 14`, the model may use demand up to and
including `origin`. It may NOT use `lag_1` measured from the target date, because
that would be `origin + 13` -- a day that has not happened yet at forecast time.

This is the mistake that quietly inflates most published demand-forecast results.
Building `rolling_mean_7` with a groupby-shift over the whole panel produces, for
row `origin + 14`, a mean over days `origin + 7 .. origin + 13`. Those values are
unknown when the forecast is made, and a model given them scores far better than
anything deployable. `src/data/audit.py::leakage_probe` mutates all post-origin
actuals and asserts no feature changes, which is the executable version of this
paragraph.

So: features are computed once per series at the origin and are **shared across
all horizons**, with `horizon` itself as a feature. The model learns how the
relationship between the last known state and future demand decays with distance.

## What is allowed to come from the future

`sell_price`, `snap` and the calendar for the **target date**. A retailer sets
prices and promotion calendars in advance, so these are genuinely known at
forecast time. This is an assumption about the business process rather than a
property of the data, is declared in `configs/config.yaml` under
`features.assume_known_future`, and is stated in the README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.data.loader import Fold
from src.data.schema import DATE, SERIES_ID, STATIC_FEATURES, TARGET
from src.features.calendar import CALENDAR_FEATURES, calendar_frame

#: Columns that identify a row but are never model inputs.
ID_COLUMNS: tuple[str, ...] = (SERIES_ID, "origin", "target_date", "y_true")

#: Categorical inputs, declared so the encoders and LightGBM agree.
CATEGORICAL_FEATURES: tuple[str, ...] = STATIC_FEATURES


def _wide(panel: pd.DataFrame, value: str) -> pd.DataFrame:
    """series x date matrix for fast origin-anchored slicing."""
    return panel.pivot_table(index=SERIES_ID, columns=DATE, values=value,
                             aggfunc="first", observed=True)


def origin_features(
    history: pd.DataFrame, origin: pd.Timestamp, cfg: Config,
) -> pd.DataFrame:
    """Per-series features using ONLY data at or before ``origin``.

    Args:
        history: panel rows with ``date <= origin``.
        origin: last date whose actuals may be used.

    Returns:
        One row per series, indexed by ``series_id``.
    """
    lags = cfg.lags
    windows = cfg.rolling_windows
    max_win = max([*windows, *lags, 28])

    hist = history[history[DATE] <= origin]
    recent = hist[hist[DATE] > origin - pd.Timedelta(days=max_win + 400)]
    y = _wide(recent, TARGET).sort_index(axis=1)
    price = _wide(recent, "sell_price").sort_index(axis=1)

    cols = y.columns
    out = pd.DataFrame(index=y.index)

    # --- lags, measured BACK FROM THE ORIGIN (lag_1 == demand on the origin) ---
    for lag in lags:
        d = origin - pd.Timedelta(days=lag - 1)
        out[f"lag_{lag}"] = y[d] if d in cols else np.nan

    # --- rolling statistics over windows ENDING AT THE ORIGIN ---
    for w in windows:
        lo = origin - pd.Timedelta(days=w - 1)
        win = y.loc[:, (cols >= lo) & (cols <= origin)]
        out[f"rolling_mean_{w}"] = win.mean(axis=1)
        out[f"rolling_std_{w}"] = win.std(axis=1)
        out[f"rolling_max_{w}"] = win.max(axis=1)
        out[f"nonzero_rate_{w}"] = (win > 0).sum(axis=1) / max(win.shape[1], 1)

    # --- intermittency state at the origin ---
    lo28 = origin - pd.Timedelta(days=27)
    w28 = y.loc[:, (cols >= lo28) & (cols <= origin)]
    nz = w28 > 0
    # Days since the last non-zero sale: the key state variable for intermittent
    # demand, and what a Croston-style method is implicitly tracking.
    idx = np.arange(w28.shape[1])[::-1]
    last_nz = np.where(nz.to_numpy(), idx[None, :], np.inf).min(axis=1)
    out["days_since_last_sale"] = np.where(np.isinf(last_nz), 99, last_nz)
    out["mean_nonzero_size_28"] = w28.where(nz).mean(axis=1)

    # --- full-history level, capturing the series' scale ---
    allhist = _wide(hist, TARGET)
    out["hist_mean"] = allhist.mean(axis=1)
    out["hist_std"] = allhist.std(axis=1)
    out["hist_nonzero_rate"] = (allhist > 0).sum(axis=1) / allhist.notna().sum(axis=1)
    out["series_age_days"] = (origin - hist.groupby(SERIES_ID, observed=True)[DATE]
                              .min()).dt.days

    # --- price state at the origin, for a discount feature vs the target price ---
    plo = origin - pd.Timedelta(days=27)
    pw = price.loc[:, (price.columns >= plo) & (price.columns <= origin)]
    out["price_mean_28_at_origin"] = pw.mean(axis=1)
    out["price_at_origin"] = price[origin] if origin in price.columns else np.nan

    out["trend_7_28"] = out["rolling_mean_7"] / out["rolling_mean_28"].replace(0, np.nan)

    # Croston / SBA state at the origin. Provided here because it is an
    # origin-anchored quantity, so it serves both as the Croston baseline and as a
    # feature the ML model can use.
    cro = croston_state(y.loc[:, cols <= origin], alpha=0.1)
    out["croston_at_origin"] = cro["croston"]
    out["sba_at_origin"] = cro["sba"]
    return out.astype("float32", errors="ignore")


def croston_state(
    y_wide: pd.DataFrame, alpha: float = 0.1, lookback: int = 365,
) -> pd.DataFrame:
    """Croston's method and the Syntetos-Boylan Approximation, per series.

    Croston forecasts intermittent demand by smoothing two things separately and
    only when a sale occurs: the **size** of non-zero demand and the **interval**
    between sales. The per-period forecast is ``size / interval``.

    This is NOT the same as a moving average, although a naive implementation
    collapses to one: estimating size as the mean non-zero demand over a window and
    interval as the reciprocal of the non-zero rate gives
    ``(sum_nonzero / n_nonzero) x (n_nonzero / n) = sum / n`` -- algebraically
    identical to the window mean. The difference is that Croston updates only at
    demand occurrences and weights recent occurrences more heavily.

    SBA multiplies by ``1 - alpha/2`` to correct Croston's known upward bias, which
    matters for inventory: an inflated demand estimate inflates safety stock at
    every SKU simultaneously.
    """
    arr = y_wide.to_numpy(dtype="float64")
    if arr.shape[1] > lookback:
        arr = arr[:, -lookback:]
    n_series, n_t = arr.shape

    z = np.full(n_series, np.nan)      # smoothed non-zero size
    p = np.full(n_series, np.nan)      # smoothed inter-arrival interval
    q = np.ones(n_series)              # periods since the last non-zero demand

    for t in range(n_t):
        d = arr[:, t]
        nz = d > 0
        first = nz & ~np.isfinite(z)
        z = np.where(first, d, z)
        p = np.where(first, q, p)
        upd = nz & ~first
        z = np.where(upd, alpha * d + (1 - alpha) * z, z)
        p = np.where(upd, alpha * q + (1 - alpha) * p, p)
        q = np.where(nz, 1.0, q + 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        croston = np.where(np.isfinite(z) & np.isfinite(p) & (p > 0), z / p, 0.0)
    return pd.DataFrame({"croston": croston, "sba": croston * (1 - alpha / 2)},
                        index=y_wide.index)


def build_supervised(
    cfg: Config,
    panel: pd.DataFrame,
    fold: Fold,
    horizons: list[int] | None = None,
    origins: list[pd.Timestamp] | None = None,
    require_actuals: bool = True,
) -> pd.DataFrame:
    """Assemble a supervised frame of (series, origin, horizon) rows.

    Args:
        panel: the full panel. Only rows at or before each origin are used for
            target-derived features; `sell_price`, `snap` and the calendar are read
            at the target date, which the config declares as known in advance.
        fold: supplies the default single origin and the evaluation window.
        horizons: which steps ahead to build. Defaults to 1..fold.horizon.
        origins: build for several origins (training). Defaults to [fold.origin].

    Returns:
        One row per (series, origin, horizon) with features, `y_true` and ids.
    """
    horizons = horizons or list(range(1, fold.horizon + 1))
    origins = origins or [fold.origin]

    static = (panel.drop_duplicates(SERIES_ID)
              .set_index(SERIES_ID)[list(STATIC_FEATURES)])
    exog = panel.set_index([SERIES_ID, DATE])[["sell_price", "snap"]]
    actual = panel.set_index([SERIES_ID, DATE])[TARGET]
    series_start = panel.groupby(SERIES_ID, observed=True)[DATE].min()

    all_targets = pd.DatetimeIndex(sorted({
        o + pd.Timedelta(days=h) for o in origins for h in horizons}))
    cal = calendar_frame(all_targets).set_index("date")

    frames: list[pd.DataFrame] = []
    for origin in origins:
        history = panel[panel[DATE] <= origin]
        if history.empty:
            continue
        feats = origin_features(history, origin, cfg)

        for h in horizons:
            target_date = origin + pd.Timedelta(days=h)
            block = feats.copy()
            block[SERIES_ID] = block.index
            block["origin"] = origin
            block["target_date"] = target_date
            block["horizon"] = np.int16(h)

            keys = pd.MultiIndex.from_arrays(
                [block.index, pd.DatetimeIndex([target_date] * len(block))])
            block["y_true"] = actual.reindex(keys).to_numpy()
            block["sell_price"] = exog["sell_price"].reindex(keys).to_numpy()
            block["snap"] = exog["snap"].reindex(keys).to_numpy()

            # Seasonal-naive reference: same weekday, stepped back in whole weeks
            # until the reference date is at or before the origin. For h in 1..7
            # that is target-7; for h in 8..14 it is target-14. Legal by
            # construction -- the reference is never after the origin -- and it
            # doubles as the strongest single feature for weekly seasonality.
            weeks_back = int(np.ceil(h / 7))
            ref_date = target_date - pd.Timedelta(days=7 * weeks_back)
            ref_keys = pd.MultiIndex.from_arrays(
                [block.index, pd.DatetimeIndex([ref_date] * len(block))])
            block["snaive_lag"] = actual.reindex(ref_keys).to_numpy()
            assert ref_date <= origin, "seasonal-naive reference must not be in the future"

            for c in CALENDAR_FEATURES:
                block[c] = cal.loc[target_date, c]

            block["product_age_days"] = (
                target_date - series_start.reindex(block.index)).dt.days.to_numpy()
            frames.append(block.reset_index(drop=True))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Discount relative to the series' own recent price level. Absolute price says
    # little across products; a 10% cut against this SKU's own norm is comparable
    # across a 2 CU item and a 40 CU one.
    df["discount_vs_28d"] = 1.0 - df["sell_price"] / df["price_mean_28_at_origin"].replace(0, np.nan)
    df["price_ratio_vs_origin"] = df["sell_price"] / df["price_at_origin"].replace(0, np.nan)

    df = df.join(static, on=SERIES_ID)
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype("category")

    # Rows whose target date falls outside the series' life have no actual; they
    # are dropped rather than imputed, because a missing actual is not a zero.
    #
    # `require_actuals=False` is for genuine forecasting past the end of the panel,
    # where no actual can exist yet. Evaluation always keeps the default, so a
    # missing label can never be silently scored as a hit.
    if require_actuals:
        df = df[df["y_true"].notna()]
    return df.reset_index(drop=True)


def training_origins(
    cfg: Config, panel: pd.DataFrame, last_origin: pd.Timestamp,
    n_origins: int = 60, step_days: int = 7, max_horizon: int | None = None,
) -> list[pd.Timestamp]:
    """Origins for building the training set, all ending safely before the fold.

    Spaced a week apart so every origin sees a different day-of-week phase, and
    capped in number to keep the training matrix manageable.

    The subtlety that matters: an origin is only safe if its FURTHEST TARGET is
    still at or before ``last_origin``. A training row is ``(origin, horizon)``
    with a label at ``origin + horizon``, so allowing an origin at the fold
    boundary would put labels from the evaluation window into training -- the model
    would be fitted on the very days it is about to be scored on. The latest usable
    origin is therefore ``last_origin - max_horizon``, not ``last_origin``.
    """
    h_max = int(max_horizon if max_horizon is not None else max(cfg.horizons))
    latest = last_origin - pd.Timedelta(days=h_max)
    first = panel[DATE].min() + pd.Timedelta(days=max(cfg.rolling_windows) + 60)
    origins = [latest - pd.Timedelta(days=i * step_days) for i in range(n_origins)]
    return sorted(o for o in origins if o >= first)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns: everything except identifiers and the target."""
    drop = {*ID_COLUMNS, "origin", "target_date", "y_true", SERIES_ID}
    return [c for c in df.columns if c not in drop]
