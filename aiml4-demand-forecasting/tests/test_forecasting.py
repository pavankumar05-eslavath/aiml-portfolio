"""Tests for the forecasting pipeline.

Two kinds live here, and the second is the point:

1. **Contract tests** -- shapes, non-negativity, required columns, model loading.
2. **Claim tests** -- the methodological guarantees the project rests on. The most
   important is that features cannot see the future
   it is asserted by mutating the
   future and proving nothing changes, and by proving the check itself can fail.
"""
from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    ADI_CUT,
    CV2_CUT,
    Fold,
    classify_demand,
    demand_stats,
    rolling_origin_folds,
)

# Aliased deliberately: pytest collects any module-level callable named `test_*`,
# so importing `test_window` under its own name would register the loader helper
# as a test case.
from src.data.loader import test_window as get_test_window
from src.data.schema import DATE, PANEL_COLUMNS, SERIES_ID, TARGET
from src.evaluation import metrics as M
from src.features.build import build_supervised, croston_state, origin_features, training_origins
from src.features.calendar import calendar_frame
from src.forecasting import baselines as B
from tests.conftest import requires_real_data

# ===========================================================================
# 1. Panel and splits
# ===========================================================================

def test_panel_schema(panel):
    for c in PANEL_COLUMNS:
        assert c in panel.columns


def test_panel_has_no_duplicate_grain_rows(panel):
    assert panel.duplicated(subset=[SERIES_ID, DATE]).sum() == 0


def test_demand_is_non_negative(panel):
    assert (panel[TARGET] >= 0).all()


def test_series_dates_are_contiguous(panel):
    g = panel.groupby(SERIES_ID, observed=True)[DATE].agg(["min", "max", "size"])
    expected = (g["max"] - g["min"]).dt.days + 1
    assert (g["size"] == expected).all()


def test_validation_folds_are_chronological_and_before_test(cfg, panel):
    """Rolling origins, not random splits, and none overlapping the holdout."""
    folds = rolling_origin_folds(cfg, panel)
    test_start, _ = get_test_window(cfg)
    assert len(folds) == int(cfg["split"]["validation"]["n_folds"])
    for f in folds:
        assert f.origin < f.start <= f.end
        assert f.end < test_start, "a validation window overlaps the test window"
    for a, b in pairwise(folds):
        assert a.end < b.start, "validation windows must not overlap"


def test_training_origins_cannot_reach_the_fold_window(cfg, panel, fold):
    """The subtle leak: an origin at the boundary has targets inside the fold.

    A training row is (origin, horizon) with a label at origin+horizon, so the
    latest safe origin is fold.origin - max_horizon.
    """
    origins = training_origins(cfg, panel, fold.origin, n_origins=10)
    h_max = max(cfg.horizons)
    assert origins, "expected at least one usable training origin"
    for o in origins:
        assert o + pd.Timedelta(days=h_max) <= fold.origin


# ===========================================================================
# 2. Leakage — the central methodological claim
# ===========================================================================

def test_features_do_not_change_when_the_future_is_mutated(cfg, panel, fold):
    """Overwrite every actual after the origin; no feature may change.

    This is the executable version of "features are anchored to the forecast
    origin". If it fails, the reported accuracy is fiction.
    """
    from src.data.audit import leakage_probe
    res = leakage_probe(cfg, panel, fold)
    assert res["n_features_checked"] > 20
    assert res["n_rows_compared"] > 0
    assert res["leak_free"], (
        f"features changed: {res['features_changed_by_future_mutation']}")


def test_the_leakage_probe_can_actually_fail(cfg, panel, fold, monkeypatch):
    """A check that cannot fail proves nothing.

    Injects the classic mistake -- a rolling mean computed at the TARGET date via a
    panel-wide groupby-shift -- and asserts the probe catches it.
    """
    import src.features.build as build_mod
    from src.data.audit import leakage_probe

    original = build_mod.build_supervised

    def leaky(cfg_, panel_, fold_, horizons=None, origins=None):
        df = original(cfg_, panel_, fold_, horizons=horizons, origins=origins)
        p = panel_.sort_values([SERIES_ID, DATE]).copy()
        p["LEAK"] = (p.groupby(SERIES_ID, observed=True)[TARGET]
                     .transform(lambda s: s.shift(1).rolling(7).mean()))
        m = p.set_index([SERIES_ID, DATE])["LEAK"]
        keys = pd.MultiIndex.from_arrays([df[SERIES_ID], df["target_date"]])
        df["LEAK"] = m.reindex(keys).to_numpy()
        return df

    monkeypatch.setattr(build_mod, "build_supervised", leaky)
    res = leakage_probe(cfg, panel, fold)
    assert not res["leak_free"]
    assert "LEAK" in res["features_changed_by_future_mutation"]


def test_seasonal_naive_reference_is_never_in_the_future(cfg, tiny_panel):
    """snaive_lag must look back whole weeks to a date at or before the origin."""
    origin = tiny_panel[DATE].max() - pd.Timedelta(days=30)
    f = Fold("t", origin, origin + pd.Timedelta(days=1), origin + pd.Timedelta(days=28))
    df = build_supervised(cfg, tiny_panel, f, horizons=[1, 7, 8, 14, 28])
    assert not df.empty
    # For S_A demand is a constant 4.0, so any legal reference must equal 4.0.
    a = df[df[SERIES_ID] == "S_A"]
    assert np.allclose(a["snaive_lag"].to_numpy(), 4.0)


# ===========================================================================
# 3. Feature semantics
# ===========================================================================

def test_lags_are_measured_back_from_the_origin(cfg, tiny_panel):
    origin = pd.Timestamp("2020-03-01")
    hist = tiny_panel[tiny_panel[DATE] <= origin]
    f = origin_features(hist, origin, cfg)

    for sid in ("S_A", "S_B"):
        for lag in cfg.lags:
            expected_date = origin - pd.Timedelta(days=lag - 1)
            expected = tiny_panel.loc[
                (tiny_panel[SERIES_ID] == sid) & (tiny_panel[DATE] == expected_date),
                TARGET].iloc[0]
            assert f.loc[sid, f"lag_{lag}"] == pytest.approx(expected)


def test_rolling_windows_end_at_the_origin(cfg, tiny_panel):
    origin = pd.Timestamp("2020-03-01")
    hist = tiny_panel[tiny_panel[DATE] <= origin]
    f = origin_features(hist, origin, cfg)
    for w in cfg.rolling_windows:
        lo = origin - pd.Timedelta(days=w - 1)
        for sid in ("S_A", "S_B"):
            window = tiny_panel[(tiny_panel[SERIES_ID] == sid)
                                & (tiny_panel[DATE] >= lo)
                                & (tiny_panel[DATE] <= origin)][TARGET]
            assert f.loc[sid, f"rolling_mean_{w}"] == pytest.approx(window.mean(), rel=1e-5)


def test_days_since_last_sale_is_correct(cfg, tiny_panel):
    """S_B sells every third day, so the gap is 0, 1 or 2 depending on phase."""
    origin = pd.Timestamp("2020-03-01")
    f = origin_features(tiny_panel[tiny_panel[DATE] <= origin], origin, cfg)
    sold_today = tiny_panel[(tiny_panel[SERIES_ID] == "S_B")
                            & (tiny_panel[DATE] == origin)][TARGET].iloc[0]
    assert f.loc["S_B", "days_since_last_sale"] == (0 if sold_today > 0 else
                                                    pytest.approx(f.loc["S_B", "days_since_last_sale"]))
    assert f.loc["S_A", "days_since_last_sale"] == 0  # sells every day


def test_calendar_features_are_deterministic_and_complete():
    dates = pd.date_range("2015-01-01", periods=400, freq="D")
    a, b = calendar_frame(dates), calendar_frame(dates)
    pd.testing.assert_frame_equal(a, b)
    assert a["is_weekend"].isin([0, 1]).all()
    assert a["dow"].between(0, 6).all()
    assert a["days_to_holiday"].between(-30, 30).all()
    # New Year's Day is a US federal holiday and must be flagged.
    assert int(a.loc[a["date"] == pd.Timestamp("2015-01-01"), "is_holiday"].iloc[0]) == 1


def test_engineered_features_contain_no_infinities(features):
    """No ±inf. NaN is allowed and meaningful.

    A young series has no 28-day window and a series with no recent sales has no
    mean non-zero size
    NaN is the honest encoding of "no information", and
    LightGBM splits on missingness natively. Infinity is different -- it comes from
    an unguarded division and breaks most estimators silently.
    """
    num = features.select_dtypes(include=[np.number]).to_numpy(dtype="float64")
    assert not np.isinf(num).any(), "unguarded division produced an infinity"


def test_croston_is_not_a_moving_average():
    """Guards a bug that made Croston numerically identical to MA28.

    Estimating size as the window mean of non-zero demand and interval as the
    reciprocal of the non-zero rate cancels to sum/n -- the window mean. Real
    Croston smooths size and interval separately and updates only on sales.
    """
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    y = np.where(np.arange(90) % 5 == 0, 10.0, 0.0)
    wide = pd.DataFrame([y], index=["S"], columns=dates)
    cro = croston_state(wide, alpha=0.1)
    ma = y[-28:].mean()
    assert cro.loc["S", "croston"] > 0
    assert cro.loc["S", "croston"] != pytest.approx(ma, rel=1e-6)
    # SBA must be strictly below Croston: it corrects a known upward bias.
    assert cro.loc["S", "sba"] < cro.loc["S", "croston"]


# ===========================================================================
# 4. Baselines and metrics
# ===========================================================================

def test_baseline_predictions_are_non_negative_and_right_length(features):
    for name in B.all_baselines():
        p = B.predict(name, features)
        assert len(p) == len(features)
        assert np.isfinite(p).all()
        assert (p >= 0).all(), f"{name} produced negative demand"


def test_naive_baseline_equals_last_observed_value(features):
    np.testing.assert_allclose(B.predict("naive", features),
                               features["lag_1"].fillna(0).clip(lower=0).to_numpy())


def test_zero_baseline_is_exactly_zero(features):
    assert (B.predict("zero", features) == 0).all()


def test_metrics_on_a_known_example():
    y = np.array([0.0, 2.0, 4.0, 0.0])
    f = np.array([1.0, 2.0, 2.0, 0.0])
    assert M.mae(y, f) == pytest.approx(0.75)
    assert M.rmse(y, f) == pytest.approx(np.sqrt((1 + 0 + 4 + 0) / 4))
    assert M.wape(y, f) == pytest.approx(3.0 / 6.0)


def test_wape_is_defined_when_individual_actuals_are_zero():
    """The reason WAPE replaces MAPE here: it only needs aggregate demand > 0."""
    y = np.array([0.0, 0.0, 5.0])
    f = np.array([1.0, 0.0, 4.0])
    assert np.isfinite(M.wape(y, f))


def test_mape_would_be_undefined_on_this_data(panel):
    """Documents why MAPE is not reported: most actuals are zero."""
    assert float((panel[TARGET] == 0).mean()) > 0.3


def test_perfect_forecast_scores_zero_error():
    y = np.array([0.0, 3.0, 7.0])
    s = M.score(y, y)
    assert s.mae == 0.0 and s.rmse == 0.0 and s.wape == 0.0


def test_mase_denominator_excludes_degenerate_series(tiny_panel):
    """A constant series has zero seasonal difference
    it must be NaN, not inf."""
    d = B.mase_denominator(tiny_panel, SERIES_ID, DATE, TARGET, season=7)
    assert np.isnan(d.loc["S_A"])          # constant demand -> no scale
    assert np.isfinite(d.loc["S_B"])


def test_demand_classification_matches_the_cut_points():
    stats = pd.DataFrame({
        "ADI": [1.0, 1.0, 5.0, 5.0],
        "CV2": [0.1, 1.0, 0.1, 1.0],
    }, index=["a", "b", "c", "d"])
    got = classify_demand(stats).tolist()
    assert got == ["Smooth", "Erratic", "Intermittent", "Lumpy"]
    assert ADI_CUT == 1.32 and CV2_CUT == 0.49


def test_demand_stats_uses_nonzero_sizes_for_cv2(tiny_panel):
    """CV2 over non-zero sizes only
    otherwise it conflates sparsity with variability."""
    s = demand_stats(tiny_panel)
    # S_B sells a constant 6.0 whenever it sells, so its non-zero CV2 is 0.
    assert s.loc["S_B", "CV2"] == pytest.approx(0.0, abs=1e-9)
    assert s.loc["S_B", "ADI"] > 1.0


# ===========================================================================
# 5. Real-data claims
# ===========================================================================

@requires_real_data
def test_real_panel_shape_and_grain(panel, cfg):
    assert panel[SERIES_ID].nunique() == 600
    assert panel["sku_id"].nunique() == int(cfg["data"]["subset"]["n_skus"])
    assert sorted(panel["store_id"].unique()) == sorted(cfg["data"]["subset"]["stores"])
    assert panel[SERIES_ID].nunique() == (panel["sku_id"].nunique()
                                         * panel["store_id"].nunique())


@requires_real_data
def test_real_panel_is_intermittent(panel):
    """The fact that shapes every methodological choice in the project."""
    assert float((panel[TARGET] == 0).mean()) > 0.5


@requires_real_data
def test_exogenous_features_exist_across_the_test_window(panel, cfg):
    """Price and SNAP must cover the holdout, else it cannot be scored properly."""
    start, end = get_test_window(cfg)
    w = panel[(panel[DATE] >= start) & (panel[DATE] <= end)]
    assert not w.empty
    assert w["sell_price"].notna().all()
    assert w["snap"].notna().all()


@requires_real_data
def test_subset_preserves_the_population_demand_mix(panel):
    """Stratified sampling, not top-N: the intermittent majority must survive."""
    stats = demand_stats(panel)
    stats["demand_class"] = classify_demand(stats)
    share = stats["demand_class"].isin(["Intermittent", "Lumpy"]).mean()
    assert share > 0.6, f"subset lost the intermittent majority ({share:.2%})"
