"""Inventory policy: safety stock and reorder points derived from forecast error.

## The key methodological point

Safety stock is **not** derived from historical demand variability. It is derived
from **forecast error** variability over the lead time.

The distinction is the whole reason forecasting has any value here. Safety stock
exists to absorb what the forecast got wrong, so the relevant quantity is the
spread of `actual - forecast` over the replenishment lead time, not the spread of
demand itself. A better forecast shrinks that spread, which shrinks safety stock,
which is where the money is. Sizing safety stock from raw demand variability gives
the *same* answer no matter how good the forecast is -- and therefore cannot show
any benefit from improving it.

## Two estimators, deliberately compared

* **Normal approximation:** `SS = z(SL) * sigma_error_LT`. Cheap, standard, and
  assumes symmetric Gaussian errors.
* **Empirical quantile:** take the actual quantile of observed lead-time forecast
  errors. Makes no distributional assumption.

For intermittent demand the two disagree materially, because the error
distribution is right-skewed and lumpy. The normal approximation understates the
upper tail exactly where it matters, and the report quantifies by how much rather
than asserting it.

Nothing here hardcodes a safety-stock number: every value is a function of the
measured error distribution and the chosen service level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.config import Config
from src.data.schema import SERIES_ID


def z_for_service_level(service_level: float) -> float:
    """Standard-normal quantile for a cycle service level."""
    if not 0.0 < service_level < 1.0:
        raise ValueError("service_level must be strictly between 0 and 1")
    return float(stats.norm.ppf(service_level))


@dataclass(frozen=True)
class PolicyParams:
    """Replenishment parameters. Lead time and review period are assumptions."""

    lead_time_days: int
    review_period_days: int
    service_level: float

    @property
    def protection_days(self) -> int:
        """Demand that must be covered by stock on hand when an order is placed.

        Under periodic review the exposure is lead time PLUS the review period: a
        stockout can occur any time before the *next* order arrives, not just before
        this one does. Omitting the review period is a common error that
        systematically under-protects.
        """
        return int(self.lead_time_days + self.review_period_days)

    @classmethod
    def from_config(cls, cfg: Config, service_level: float | None = None) -> PolicyParams:
        inv = cfg["inventory"]
        return cls(
            lead_time_days=int(inv["lead_time_days"]),
            review_period_days=int(inv["review_period_days"]),
            service_level=float(service_level if service_level is not None
                                else inv["default_service_level"]),
        )


def lead_time_forecast_errors(
    predictions: pd.DataFrame, point_col: str, protection_days: int,
) -> pd.DataFrame:
    """Per-series forecast error accumulated over the protection window.

    Errors are summed over the window before being described, because inventory is
    exposed to the *total* shortfall across the lead time, not to each day's error
    independently. Summing first preserves the day-to-day correlation that actually
    occurs, which is what makes this estimate empirical rather than an assumption.
    """
    d = predictions[predictions["horizon"] <= protection_days].copy()
    g = d.groupby([SERIES_ID, "origin"] if "origin" in d.columns else [SERIES_ID],
                  observed=True)
    agg = g.agg(lt_actual=("y_true", "sum"), lt_forecast=(point_col, "sum"),
                n_days=("y_true", "size")).reset_index()
    agg["lt_error"] = agg["lt_actual"] - agg["lt_forecast"]
    return agg


def error_spread(lt_errors: pd.DataFrame) -> pd.DataFrame:
    """Per-series spread of lead-time forecast error, both estimators."""
    g = lt_errors.groupby(SERIES_ID, observed=True)["lt_error"]
    out = pd.DataFrame({
        "n_windows": g.size(),
        "error_mean": g.mean(),
        "error_std": g.std(ddof=1),
        "error_q50": g.quantile(0.50),
        "error_q90": g.quantile(0.90),
        "error_q95": g.quantile(0.95),
        "error_q99": g.quantile(0.99),
    })
    # A series observed in only one window has no spread estimate; fall back to the
    # cross-series median rather than to zero, which would imply no safety stock.
    out["error_std"] = out["error_std"].fillna(out["error_std"].median())
    return out.reset_index()


def safety_stock_normal(error_std: np.ndarray, service_level: float) -> np.ndarray:
    """z * sigma of lead-time forecast error."""
    return np.clip(z_for_service_level(service_level) * np.asarray(error_std,
                                                                  dtype="float64"),
                   0.0, None)


def safety_stock_empirical(
    lt_errors: pd.DataFrame, service_level: float,
) -> pd.Series:
    """Empirical quantile of lead-time forecast error, per series.

    Makes no distributional assumption. The quantile is taken of the *under*-forecast
    side (positive error means demand exceeded the forecast), which is the direction
    safety stock protects against.
    """
    q = lt_errors.groupby(SERIES_ID, observed=True)["lt_error"].quantile(service_level)
    return np.clip(q, 0.0, None).rename("safety_stock_empirical")


def build_policy(
    cfg: Config,
    predictions: pd.DataFrame,
    point_col: str,
    service_level: float | None = None,
    lt_errors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute safety stock and reorder point per series.

    Returns a frame with both safety-stock estimators, expected lead-time demand
    and the resulting reorder points, so the two can be compared rather than one
    being asserted.
    """
    params = PolicyParams.from_config(cfg, service_level)
    pd_days = params.protection_days

    if lt_errors is None:
        lt_errors = lead_time_forecast_errors(predictions, point_col, pd_days)
    spread = error_spread(lt_errors)

    # Expected demand over the protection window, from the most recent origin so the
    # policy reflects the latest forecast rather than an average of stale ones.
    latest = predictions["origin"].max() if "origin" in predictions.columns else None
    horizon_slice = predictions[predictions["horizon"] <= pd_days]
    if latest is not None:
        horizon_slice = horizon_slice[horizon_slice["origin"] == latest]
    expected = (horizon_slice.groupby(SERIES_ID, observed=True)[point_col]
                .sum().rename("expected_lt_demand"))

    out = spread.merge(expected, on=SERIES_ID, how="left")
    out["service_level"] = params.service_level
    out["lead_time_days"] = params.lead_time_days
    out["review_period_days"] = params.review_period_days
    out["protection_days"] = pd_days

    out["safety_stock_normal"] = safety_stock_normal(
        out["error_std"].to_numpy(), params.service_level)
    emp = safety_stock_empirical(lt_errors, params.service_level)
    out = out.merge(emp, on=SERIES_ID, how="left")
    out["safety_stock_empirical"] = out["safety_stock_empirical"].fillna(
        out["safety_stock_normal"])

    for kind in ("normal", "empirical"):
        out[f"reorder_point_{kind}"] = np.ceil(
            out["expected_lt_demand"].fillna(0.0) + out[f"safety_stock_{kind}"])

    out["ss_ratio_emp_over_normal"] = (
        out["safety_stock_empirical"] / out["safety_stock_normal"].replace(0, np.nan))
    return out


def economic_order_quantity(
    annual_demand: np.ndarray, ordering_cost: float, holding_cost_per_unit_year: np.ndarray,
) -> np.ndarray:
    """Classic EOQ, used to set a sensible order size for the simulation.

    EOQ assumes constant demand and no uncertainty, which is false here. It is used
    only to pick a *reasonable* order quantity so the policy comparison is not
    distorted by an arbitrary one; the service-level performance comes from the
    reorder point, which is where the forecast enters.
    """
    d = np.asarray(annual_demand, dtype="float64")
    h = np.asarray(holding_cost_per_unit_year, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.sqrt(2.0 * d * float(ordering_cost) / np.where(h > 0, h, np.nan))
    return np.clip(np.nan_to_num(q, nan=1.0, posinf=1.0), 1.0, None)


def baseline_policy(
    cfg: Config, panel_history: pd.DataFrame, service_level: float | None = None,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """The policy to beat: safety stock from historical DEMAND variability.

    This is what an operation without a forecasting model does -- size stock from
    how much demand has bounced around recently, and cover the lead time with a
    moving average. It is a fair comparator precisely because it is what the
    forecast has to justify replacing.
    """
    params = PolicyParams.from_config(cfg, service_level)
    pd_days = params.protection_days
    as_of = panel_history["date"].max()
    recent = panel_history[panel_history["date"] > as_of - pd.Timedelta(days=lookback_days)]

    g = recent.groupby(SERIES_ID, observed=True)["demand"]
    out = pd.DataFrame({"daily_mean": g.mean(), "daily_std": g.std(ddof=1)}).fillna(0.0)
    out["expected_lt_demand"] = out["daily_mean"] * pd_days
    # sigma over the window scales with sqrt(days) under the independence assumption.
    out["safety_stock_normal"] = safety_stock_normal(
        (out["daily_std"] * np.sqrt(pd_days)).to_numpy(), params.service_level)
    out["reorder_point_normal"] = np.ceil(
        out["expected_lt_demand"] + out["safety_stock_normal"])
    out["service_level"] = params.service_level
    out["protection_days"] = pd_days
    return out.reset_index()
