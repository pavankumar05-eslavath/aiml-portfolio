"""Inventory simulation: does the policy actually deliver its service level?

A reorder point computed from a formula is a *claim*. This module tests it by
replaying the held-out demand day by day through a periodic-review (R, s, Q)
policy and measuring what really happened: how many units were short, how often
stock ran out, how much inventory was carried.

That check matters because the safety-stock formula rests on assumptions
(independent daily errors, a symmetric error distribution) that intermittent demand
violates. A policy targeting 95% can easily deliver 85%, and only simulation
reveals the gap.

Two service measures are reported, because they are different questions:

* **Cycle service level** -- the fraction of replenishment cycles with no stockout.
  What the z-based formula targets.
* **Fill rate** -- the fraction of demanded units actually supplied. What a customer
  experiences, and what the cost model charges against.

Unmet demand is treated as **lost sales**, not backorders: in grocery retail a
customer who finds an empty shelf usually buys something else or shops elsewhere.
That choice is stated because it changes the arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.schema import DATE, SERIES_ID, TARGET


@dataclass
class SimulationResult:
    """Per-series and aggregate outcomes of an inventory simulation."""

    per_series: pd.DataFrame
    daily: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    policy_name: str = ""

    def summary(self) -> dict[str, float]:
        d = self.per_series
        total_demand = float(d["total_demand"].sum())
        total_short = float(d["units_short"].sum())
        return {
            "policy": self.policy_name,
            "series": len(d),
            "total_demand": total_demand,
            "units_short": total_short,
            "fill_rate": 1.0 - total_short / max(total_demand, 1e-9),
            "cycle_service_level": float(d["cycle_service_level"].mean()),
            "stockout_day_rate": float(d["stockout_days"].sum() / max(d["days"].sum(), 1)),
            "avg_inventory_units": float(d["avg_inventory"].sum()),
            "orders_placed": float(d["orders_placed"].sum()),
            "series_with_stockout": int((d["units_short"] > 0).sum()),
        }


def simulate_series(
    demand: np.ndarray,
    reorder_point: float,
    order_quantity: float,
    lead_time: int,
    review_period: int,
    initial_stock: float | None = None,
) -> dict[str, float]:
    """Replay one series through a periodic-review (R, s, Q) policy.

    On review days, if inventory position (on hand + on order) is at or below the
    reorder point, order enough multiples of ``order_quantity`` to clear it. Orders
    arrive after ``lead_time`` days. Demand not met from stock on hand is lost.
    """
    n = len(demand)
    on_hand = float(initial_stock if initial_stock is not None
                    else reorder_point + order_quantity)
    pipeline = np.zeros(n + lead_time + 1)

    units_short = 0.0
    stockout_days = 0
    inv_track = np.empty(n)
    orders = 0
    cycles = 0
    cycles_with_stockout = 0
    cycle_short = False

    for t in range(n):
        on_hand += pipeline[t]
        pipeline[t] = 0.0

        d = float(demand[t])
        sold = min(on_hand, d)
        on_hand -= sold
        short = d - sold
        if short > 0:
            units_short += short
            stockout_days += 1
            cycle_short = True

        if t % review_period == 0:
            cycles += 1
            if cycle_short and cycles > 1:
                cycles_with_stockout += 1
            cycle_short = False
            position = on_hand + pipeline[t + 1:t + lead_time + 1].sum()
            if position <= reorder_point:
                gap = reorder_point + order_quantity - position
                q = float(np.ceil(gap / max(order_quantity, 1e-9)) * order_quantity)
                if q > 0:
                    pipeline[min(t + lead_time, n + lead_time)] += q
                    orders += 1

        inv_track[t] = on_hand

    if cycle_short:
        cycles_with_stockout += 1

    total_demand = float(np.sum(demand))
    return {
        "days": n,
        "total_demand": total_demand,
        "units_short": units_short,
        "fill_rate": 1.0 - units_short / max(total_demand, 1e-9),
        "stockout_days": stockout_days,
        "cycle_service_level": 1.0 - cycles_with_stockout / max(cycles, 1),
        "avg_inventory": float(np.mean(inv_track)),
        "max_inventory": float(np.max(inv_track)),
        "ending_inventory": float(on_hand),
        "orders_placed": orders,
    }


def simulate_policy(
    panel: pd.DataFrame,
    policy: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lead_time: int,
    review_period: int,
    reorder_col: str = "reorder_point_normal",
    order_quantity_col: str | None = None,
    policy_name: str = "policy",
) -> SimulationResult:
    """Simulate every series over the evaluation window."""
    window = panel[(panel[DATE] >= start) & (panel[DATE] <= end)]
    pol = policy.set_index(SERIES_ID)

    rows = []
    for sid, g in window.sort_values(DATE).groupby(SERIES_ID, observed=True):
        if sid not in pol.index:
            continue
        p = pol.loc[sid]
        rop = float(p[reorder_col])
        if order_quantity_col and order_quantity_col in pol.columns:
            q = float(p[order_quantity_col])
        else:
            # Default order size: cover one review period of expected demand, at
            # least one unit. Kept simple so the comparison isolates the reorder
            # point, which is the part the forecast determines.
            q = max(1.0, float(p.get("expected_lt_demand", 1.0)) / max(
                float(p.get("protection_days", lead_time + review_period)), 1.0)
                * review_period)
        res = simulate_series(g[TARGET].to_numpy(dtype="float64"), rop, q,
                             lead_time, review_period)
        res[SERIES_ID] = sid
        res["reorder_point"] = rop
        res["order_quantity"] = q
        rows.append(res)

    per_series = pd.DataFrame(rows)
    return SimulationResult(per_series=per_series, policy_name=policy_name)


def compare_policies(results: list[SimulationResult]) -> pd.DataFrame:
    """Side-by-side summary of several simulated policies."""
    return pd.DataFrame([r.summary() for r in results])


def stockout_risk_table(
    per_series: pd.DataFrame, segments: pd.DataFrame | None = None, top: int = 20,
) -> pd.DataFrame:
    """Series ranked by units short: the operational watch list."""
    d = per_series.copy()
    d["shortfall_share"] = d["units_short"] / d["total_demand"].replace(0, np.nan)
    if segments is not None:
        cols = [c for c in ("demand_class", "volume_class", "cat_id", "store_id")
                if c in segments.columns]
        d = d.merge(segments[cols], left_on=SERIES_ID, right_index=True, how="left")
    keep = [c for c in (SERIES_ID, "demand_class", "volume_class", "cat_id",
                        "total_demand", "units_short", "shortfall_share", "fill_rate",
                        "stockout_days", "reorder_point", "avg_inventory")
            if c in d.columns]
    return d.nlargest(top, "units_short").loc[:, keep].reset_index(drop=True)


def coverage_days(per_series: pd.DataFrame) -> pd.DataFrame:
    """Inventory coverage: how many days of demand the average stock represents."""
    d = per_series.copy()
    daily = d["total_demand"] / d["days"].replace(0, np.nan)
    d["daily_demand"] = daily
    d["coverage_days"] = d["avg_inventory"] / daily.replace(0, np.nan)
    return d[[SERIES_ID, "daily_demand", "avg_inventory", "coverage_days"]]
