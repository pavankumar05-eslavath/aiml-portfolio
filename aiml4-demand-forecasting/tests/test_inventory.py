"""Tests for the inventory decision layer.

The arithmetic here turns forecasts into money and stock, so the tests pin the
*definitions* as much as the code: what safety stock is derived from, what the
protection window includes, and that raising the service level cannot reduce stock.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inventory import costs as C
from src.inventory.policy import (
    PolicyParams,
    build_policy,
    economic_order_quantity,
    error_spread,
    lead_time_forecast_errors,
    safety_stock_empirical,
    safety_stock_normal,
    z_for_service_level,
)
from src.inventory.simulate import simulate_series, stockout_risk_table

SERIES_ID = "series_id"


# ===========================================================================
# 1. Service level and safety stock
# ===========================================================================

@pytest.mark.parametrize("sl,expected", [(0.5, 0.0), (0.90, 1.2816), (0.95, 1.6449),
                                         (0.99, 2.3263)])
def test_z_matches_the_standard_normal_quantile(sl, expected):
    assert z_for_service_level(sl) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_invalid_service_level_raises(bad):
    with pytest.raises(ValueError):
        z_for_service_level(bad)


def test_safety_stock_increases_with_service_level():
    sigma = np.array([10.0, 5.0])
    ss90 = safety_stock_normal(sigma, 0.90)
    ss95 = safety_stock_normal(sigma, 0.95)
    ss99 = safety_stock_normal(sigma, 0.99)
    assert (ss90 < ss95).all() and (ss95 < ss99).all()


def test_safety_stock_is_zero_when_the_forecast_is_perfect():
    """The headline methodological point: safety stock absorbs FORECAST ERROR.

    If the forecast never errs, no safety stock is needed -- regardless of how
    variable demand itself is. Sizing from demand variability instead would return
    a large number here, and could never show any benefit from a better forecast.
    """
    assert safety_stock_normal(np.array([0.0]), 0.95)[0] == 0.0


def test_safety_stock_is_never_negative():
    assert (safety_stock_normal(np.array([1.0, 0.0]), 0.51) >= 0).all()


def test_protection_window_includes_the_review_period(cfg):
    """A common under-protection error: covering lead time but not the review gap."""
    p = PolicyParams.from_config(cfg)
    assert p.protection_days == p.lead_time_days + p.review_period_days
    assert p.protection_days > p.lead_time_days


def test_lead_time_errors_sum_over_the_window():
    """Errors must be accumulated before being described, not averaged per day."""
    preds = pd.DataFrame({
        SERIES_ID: ["A"] * 4,
        "origin": [pd.Timestamp("2020-01-01")] * 4,
        "horizon": [1, 2, 3, 4],
        "y_true": [2.0, 3.0, 0.0, 5.0],
        "pred": [1.0, 1.0, 1.0, 1.0],
    })
    lt = lead_time_forecast_errors(preds, "pred", protection_days=3)
    assert len(lt) == 1
    assert lt["lt_actual"].iloc[0] == pytest.approx(5.0)   # 2+3+0, horizon 4 excluded
    assert lt["lt_forecast"].iloc[0] == pytest.approx(3.0)
    assert lt["lt_error"].iloc[0] == pytest.approx(2.0)


def test_error_spread_falls_back_rather_than_reporting_zero_spread():
    """One window gives no std; falling back to zero would mean no safety stock."""
    lt = pd.DataFrame({SERIES_ID: ["A", "B", "B", "B"],
                       "lt_error": [3.0, 1.0, 2.0, 6.0]})
    sp = error_spread(lt).set_index(SERIES_ID)
    assert np.isfinite(sp.loc["A", "error_std"])
    assert sp.loc["A", "error_std"] > 0


def test_empirical_safety_stock_uses_the_undersupply_tail():
    """Positive error means demand exceeded forecast: the direction stock protects."""
    lt = pd.DataFrame({SERIES_ID: ["A"] * 5, "lt_error": [-5.0, -1.0, 0.0, 4.0, 10.0]})
    ss = safety_stock_empirical(lt, 0.90)
    assert ss.loc["A"] > 0
    assert ss.loc["A"] <= 10.0


def test_reorder_point_is_expected_demand_plus_safety_stock(cfg):
    preds = pd.DataFrame({
        SERIES_ID: ["A"] * 6,
        "origin": [pd.Timestamp("2020-01-01")] * 3 + [pd.Timestamp("2020-01-08")] * 3,
        "horizon": [1, 2, 3, 1, 2, 3],
        "y_true": [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        "pred": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    })
    pol = build_policy(cfg, preds, "pred", service_level=0.95)
    r = pol.iloc[0]
    for kind in ("normal", "empirical"):
        assert r[f"reorder_point_{kind}"] == pytest.approx(
            np.ceil(r["expected_lt_demand"] + r[f"safety_stock_{kind}"]))
        assert r[f"reorder_point_{kind}"] >= 0


def test_eoq_matches_the_closed_form():
    q = economic_order_quantity(np.array([1000.0]), 50.0, np.array([2.0]))
    assert q[0] == pytest.approx(np.sqrt(2 * 1000 * 50 / 2))


def test_eoq_handles_zero_holding_cost_without_blowing_up():
    q = economic_order_quantity(np.array([100.0]), 10.0, np.array([0.0]))
    assert np.isfinite(q).all() and (q >= 1).all()


# ===========================================================================
# 2. Simulation
# ===========================================================================

def test_infinite_stock_never_stocks_out():
    demand = np.array([5.0, 3.0, 9.0, 1.0, 0.0, 4.0, 7.0] * 4)
    r = simulate_series(demand, reorder_point=1e6, order_quantity=1e6,
                        lead_time=7, review_period=7)
    assert r["units_short"] == 0.0
    assert r["fill_rate"] == pytest.approx(1.0)
    assert r["cycle_service_level"] == pytest.approx(1.0)


def test_zero_stock_policy_loses_all_demand():
    """No initial stock and no reordering: every unit is short (lost sales)."""
    demand = np.array([2.0, 2.0, 2.0, 2.0])
    r = simulate_series(demand, reorder_point=0.0, order_quantity=0.0,
                        lead_time=99, review_period=99, initial_stock=0.0)
    assert r["units_short"] == pytest.approx(demand.sum())
    assert r["fill_rate"] == pytest.approx(0.0)


def test_unmet_demand_is_lost_not_backordered():
    """Shortfall must not be carried into the next day's demand."""
    demand = np.array([10.0, 1.0])
    r = simulate_series(demand, reorder_point=0.0, order_quantity=0.0,
                        lead_time=99, review_period=99, initial_stock=5.0)
    # 5 sold, 5 short on day 1; day 2 demands 1 with 0 on hand -> 1 more short.
    assert r["units_short"] == pytest.approx(6.0)


def test_fill_rate_and_service_level_are_bounded():
    rng = np.random.default_rng(0)
    demand = rng.poisson(3.0, size=60).astype(float)
    r = simulate_series(demand, reorder_point=15.0, order_quantity=20.0,
                        lead_time=5, review_period=7)
    for k in ("fill_rate", "cycle_service_level"):
        assert 0.0 <= r[k] <= 1.0
    assert r["avg_inventory"] >= 0.0
    assert r["orders_placed"] >= 0


def test_higher_reorder_point_cannot_increase_shortfall():
    """Monotonicity: more stock cannot mean more stockouts."""
    rng = np.random.default_rng(1)
    demand = rng.poisson(4.0, size=90).astype(float)
    short = [simulate_series(demand, rop, 20.0, 7, 7)["units_short"]
             for rop in (5.0, 20.0, 60.0)]
    assert short[0] >= short[1] >= short[2]


def test_stockout_risk_table_ranks_by_units_short():
    per = pd.DataFrame({
        SERIES_ID: ["A", "B", "C"],
        "total_demand": [100.0, 100.0, 100.0],
        "units_short": [1.0, 30.0, 10.0],
        "fill_rate": [0.99, 0.70, 0.90],
        "stockout_days": [1, 9, 3],
        "reorder_point": [10.0, 5.0, 8.0],
        "avg_inventory": [20.0, 5.0, 12.0],
        "days": [28, 28, 28],
    })
    out = stockout_risk_table(per, top=2)
    assert out[SERIES_ID].tolist() == ["B", "C"]


# ===========================================================================
# 3. Cost model
# ===========================================================================

def test_economics_load_from_config(cfg):
    e = C.Economics.from_config(cfg)
    assert 0 < e.gross_margin < 1
    assert e.holding_rate_annual > 0
    assert e.stockout_penalty_multiplier >= 1


def test_holding_cost_scales_with_value_and_time(cfg):
    e = C.Economics.from_config(cfg)
    per_day = e.holding_cost_per_unit_day(np.array([10.0, 20.0]))
    assert per_day[1] == pytest.approx(2 * per_day[0])
    assert e.overstock_cost_per_unit(np.array([10.0]), 28)[0] == pytest.approx(
        per_day[0] * 28)


def test_understock_cost_uses_margin_and_penalty(cfg):
    e = C.Economics.from_config(cfg)
    cu = e.understock_cost_per_unit(np.array([10.0]))[0]
    assert cu == pytest.approx(10.0 * e.gross_margin * e.stockout_penalty_multiplier)


def test_critical_ratio_is_a_valid_probability(cfg):
    e = C.Economics.from_config(cfg)
    cr = e.critical_ratio(np.array([1.0, 5.0, 50.0]), 28)
    assert ((cr > 0) & (cr < 1)).all()


def test_critical_ratio_rises_with_the_stockout_penalty(cfg):
    """Newsvendor logic: a costlier stockout justifies a higher service level."""
    low = C.Economics.from_config(cfg, stockout_penalty_multiplier=1.0)
    high = C.Economics.from_config(cfg, stockout_penalty_multiplier=8.0)
    v = np.array([5.0])
    assert high.critical_ratio(v, 28)[0] > low.critical_ratio(v, 28)[0]


def test_cost_summary_components_add_up(cfg):
    per = pd.DataFrame({
        SERIES_ID: ["A", "B"], "days": [28, 28],
        "total_demand": [100.0, 50.0], "units_short": [5.0, 0.0],
        "avg_inventory": [10.0, 4.0], "orders_placed": [4, 2],
        "fill_rate": [0.95, 1.0], "stockout_days": [2, 0],
        "cycle_service_level": [0.9, 1.0],
    })
    uv = pd.Series([4.0, 8.0], index=["A", "B"], name="unit_value")
    costed = C.cost_simulation(per, uv, C.Economics.from_config(cfg), days=28)
    s = C.cost_summary(costed)
    assert s["total_cost_cu"] == pytest.approx(
        s["holding_cost_cu"] + s["stockout_cost_cu"] + s["ordering_cost_cu"])
    assert s["fill_rate"] == pytest.approx(1 - 5.0 / 150.0)


def test_unit_value_comes_from_the_data(panel):
    """The only economic input that is measured rather than assumed."""
    uv = C.unit_values(panel)
    assert (uv > 0).all()
    assert len(uv) == panel[SERIES_ID].nunique()
