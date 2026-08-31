"""Inventory cost model and service-level scenario analysis.

READ THIS BEFORE QUOTING ANY CURRENCY FIGURE.

Only one input comes from the data: **unit value**, taken from the observed
``sell_price``. Everything else -- gross margin, holding rate, ordering cost, the
stockout penalty multiplier -- is a **declared assumption** in
``configs/config.yaml``. M5 contains no cost of goods, no carrying cost and no
record of what a stockout cost. Figures are reported in neutral currency units (CU)
rather than dollars, and ``scenario_table`` exists because the economically optimal
service level moves when the assumptions move.

## The trade-off, made explicit

Raising the service level buys fewer stockouts with more inventory. The cost curve
is U-shaped, and its minimum is the economically reasonable service level:

    total cost = holding cost + stockout cost + ordering cost

## The analytical cross-check

Newsvendor theory says the optimal service level is the **critical ratio**:

    SL* = Cu / (Cu + Co)

where ``Cu`` is the cost of being one unit short and ``Co`` the cost of carrying one
unit too many. That closed form is computed alongside the simulated optimum. If the
two disagree wildly, something is wrong with the simulation or with the
assumptions -- so it functions as a check, not decoration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.config import Config
from src.data.schema import SERIES_ID

DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class Economics:
    """Declared cost assumptions. Only unit value is measured from the data."""

    gross_margin: float
    holding_rate_annual: float
    ordering_cost_per_order: float
    stockout_penalty_multiplier: float

    @classmethod
    def from_config(cls, cfg: Config, **overrides) -> Economics:
        e = dict(cfg["inventory"]["economics"])
        e.update(overrides)
        return cls(
            gross_margin=float(e["gross_margin"]),
            holding_rate_annual=float(e["holding_rate_annual"]),
            ordering_cost_per_order=float(e["ordering_cost_per_order"]),
            stockout_penalty_multiplier=float(e["stockout_penalty_multiplier"]),
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    # -- per-unit economics ------------------------------------------------
    def holding_cost_per_unit_day(self, unit_value: np.ndarray) -> np.ndarray:
        """Cost of carrying one unit for one day."""
        return np.asarray(unit_value, dtype="float64") * self.holding_rate_annual / DAYS_PER_YEAR

    def understock_cost_per_unit(self, unit_value: np.ndarray) -> np.ndarray:
        """Cu: lost margin on a unit not supplied, times the goodwill multiplier."""
        return (np.asarray(unit_value, dtype="float64") * self.gross_margin
                * self.stockout_penalty_multiplier)

    def overstock_cost_per_unit(self, unit_value: np.ndarray, days: float) -> np.ndarray:
        """Co: cost of holding one surplus unit across the evaluation window."""
        return self.holding_cost_per_unit_day(unit_value) * float(days)

    def critical_ratio(self, unit_value: np.ndarray, days: float) -> np.ndarray:
        """Newsvendor optimal service level, Cu / (Cu + Co)."""
        cu = self.understock_cost_per_unit(unit_value)
        co = self.overstock_cost_per_unit(unit_value, days)
        return cu / (cu + co)


def unit_values(panel: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.Series:
    """Mean observed selling price per series: the one economic input from the data."""
    d = panel if as_of is None else panel[panel["date"] <= as_of]
    return (d.groupby(SERIES_ID, observed=True)["sell_price"].mean()
            .rename("unit_value"))


def cost_simulation(
    per_series: pd.DataFrame, unit_value: pd.Series, econ: Economics,
    days: int | None = None,
) -> pd.DataFrame:
    """Attach costs to a simulation result, per series."""
    d = per_series.merge(unit_value, left_on=SERIES_ID, right_index=True, how="left")
    d["unit_value"] = d["unit_value"].fillna(unit_value.median())
    n_days = float(days if days is not None else d["days"].max())

    d["holding_cost"] = (econ.holding_cost_per_unit_day(d["unit_value"].to_numpy())
                         * d["avg_inventory"] * n_days)
    d["stockout_cost"] = (econ.understock_cost_per_unit(d["unit_value"].to_numpy())
                          * d["units_short"])
    d["ordering_cost"] = econ.ordering_cost_per_order * d["orders_placed"]
    d["total_cost"] = d["holding_cost"] + d["stockout_cost"] + d["ordering_cost"]
    return d


def cost_summary(costed: pd.DataFrame) -> dict[str, float]:
    """Aggregate cost breakdown."""
    total_demand = float(costed["total_demand"].sum())
    short = float(costed["units_short"].sum())
    return {
        "holding_cost_cu": float(costed["holding_cost"].sum()),
        "stockout_cost_cu": float(costed["stockout_cost"].sum()),
        "ordering_cost_cu": float(costed["ordering_cost"].sum()),
        "total_cost_cu": float(costed["total_cost"].sum()),
        "units_short": short,
        "fill_rate": 1.0 - short / max(total_demand, 1e-9),
        "avg_inventory_units": float(costed["avg_inventory"].sum()),
        "cost_per_unit_demand_cu": float(costed["total_cost"].sum()) / max(total_demand, 1e-9),
    }


def scenario_table(
    cfg: Config,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    point_col: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    service_levels: list[float] | None = None,
    econ_overrides: dict | None = None,
    safety_stock_kind: str = "normal",
    lt_errors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Re-run policy, simulation and costing at each service level.

    This is the table that answers "what service level is economically reasonable":
    the cost-minimising row, under the stated assumptions.
    """
    from src.inventory.policy import PolicyParams, build_policy, z_for_service_level
    from src.inventory.simulate import simulate_policy

    econ = Economics.from_config(cfg, **(econ_overrides or {}))
    levels = service_levels or [float(x) for x in cfg["inventory"]["service_levels"]]
    uv = unit_values(panel, as_of=test_start - pd.Timedelta(days=1))
    n_days = int((test_end - test_start).days) + 1

    if lt_errors is None:
        raise ValueError(
            "lt_errors is required. Recomputing the error spread from a single "
            "forecast origin gives one window per series, and the empirical "
            "quantile of one observation is that observation -- which made the "
            "scenario table identical at every service level. Pass the multi-origin "
            "error windows produced by the train stage.")

    rows = []
    for sl in levels:
        params = PolicyParams.from_config(cfg, sl)
        policy = build_policy(cfg, predictions, point_col, service_level=sl,
                              lt_errors=lt_errors)
        sim = simulate_policy(
            panel, policy, test_start, test_end,
            lead_time=params.lead_time_days, review_period=params.review_period_days,
            reorder_col=f"reorder_point_{safety_stock_kind}",
            policy_name=f"forecast_sl{int(sl * 100)}")
        costed = cost_simulation(sim.per_series, uv, econ, days=n_days)
        summary = cost_summary(costed)
        rows.append({
            "service_level": sl,
            "z": z_for_service_level(sl),
            "mean_safety_stock": float(policy[f"safety_stock_{safety_stock_kind}"].mean()),
            "total_safety_stock": float(policy[f"safety_stock_{safety_stock_kind}"].sum()),
            "achieved_fill_rate": summary["fill_rate"],
            "achieved_cycle_sl": float(sim.per_series["cycle_service_level"].mean()),
            "units_short": summary["units_short"],
            "avg_inventory_units": summary["avg_inventory_units"],
            **{k: summary[k] for k in ("holding_cost_cu", "stockout_cost_cu",
                                       "ordering_cost_cu", "total_cost_cu")},
        })
    out = pd.DataFrame(rows)
    out["is_cost_minimum"] = out["total_cost_cu"] == out["total_cost_cu"].min()
    return out


def newsvendor_optimum(
    cfg: Config, panel: pd.DataFrame, as_of: pd.Timestamp, days: float,
    econ_overrides: dict | None = None,
) -> dict[str, float]:
    """Analytical optimal service level from the critical ratio."""
    econ = Economics.from_config(cfg, **(econ_overrides or {}))
    uv = unit_values(panel, as_of=as_of)
    cr = econ.critical_ratio(uv.to_numpy(), days)
    return {
        "critical_ratio_mean": float(np.mean(cr)),
        "critical_ratio_median": float(np.median(cr)),
        "cu_mean_cu": float(np.mean(econ.understock_cost_per_unit(uv.to_numpy()))),
        "co_mean_cu": float(np.mean(econ.overstock_cost_per_unit(uv.to_numpy(), days))),
        "assumptions": econ.as_dict(),
    }


def sensitivity(
    cfg: Config, panel: pd.DataFrame, predictions: pd.DataFrame, point_col: str,
    test_start: pd.Timestamp, test_end: pd.Timestamp,
    lt_errors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """How the cost-optimal service level moves with the assumptions.

    The point is that the recommendation is a function of numbers the data cannot
    supply. Presenting one service level as "the" answer would hide that.
    """
    grid = cfg["inventory"]["sensitivity"]
    rows = []
    for mult in grid["stockout_penalty_multiplier"]:
        tbl = scenario_table(cfg, panel, predictions, point_col, test_start, test_end,
                            econ_overrides={"stockout_penalty_multiplier": float(mult)},
                            lt_errors=lt_errors)
        best = tbl.loc[tbl["total_cost_cu"].idxmin()]
        rows.append({
            "stockout_penalty_multiplier": float(mult),
            "cost_optimal_service_level": float(best["service_level"]),
            "total_cost_cu": float(best["total_cost_cu"]),
            "achieved_fill_rate": float(best["achieved_fill_rate"]),
            "avg_inventory_units": float(best["avg_inventory_units"]),
        })
    return pd.DataFrame(rows)
