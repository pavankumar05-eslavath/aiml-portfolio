"""Stage runner.

    python -m src.run audit       time-series data quality + leakage audit
    python -m src.run eda         exploratory figures
    python -m src.run baselines   naive / seasonal naive / MA / Croston benchmarks
    python -m src.run train       walk-forward validation, test fold, persist model
    python -m src.run classical   ETS and SARIMA on a stratified sample
    python -m src.run inventory   safety stock, reorder points, policy simulation
    python -m src.run scenarios   service-level scenarios and cost sensitivity
    python -m src.run forecast    example forecast + reorder recommendation
    python -m src.run report      write reports/model_report.md
    python -m src.run all         every stage in order
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.data.loader import is_real_panel, load_panel, test_fold, test_window
from src.data.schema import SERIES_ID

warnings.filterwarnings("ignore")
RESULTS_NAME = "run_results.json"


def _native(o):
    if isinstance(o, dict):
        return {str(k): _native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_native(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _native(o.to_dict(orient="records"))
    if isinstance(o, pd.Series):
        return _native(o.to_dict())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _native(o.tolist())
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _save(cfg: Config, key: str, payload) -> None:
    path = cfg.resolve("paths", "reports_dir") / RESULTS_NAME
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[key] = _native(payload)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_results(cfg: Config) -> dict:
    path = cfg.resolve("paths", "reports_dir") / RESULTS_NAME
    if not path.exists():
        raise FileNotFoundError(f"no {RESULTS_NAME}; run `python -m src.run train` first")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_audit(cfg: Config, args) -> None:
    from src.data.audit import run as run_audit
    r = run_audit(cfg)
    _save(cfg, "audit", {
        "n_rows": r.n_rows, "n_series": r.n_series, "n_days": r.n_days,
        "date_min": r.date_min, "date_max": r.date_max,
        "schema_ok": r.schema_ok, "n_duplicate_rows": r.n_duplicate_rows,
        "n_calendar_gaps": r.n_calendar_gaps, "n_negative_demand": r.n_negative_demand,
        "n_missing_values": r.n_missing_values, "n_invalid_values": r.n_invalid_values,
        "zero_share": r.zero_share, "zeros": r.coverage["zeros"],
        "n_series_launched_late": r.n_series_launched_late,
        "n_series_discontinued": r.n_series_discontinued,
        "n_spike_days": r.n_spike_days, "n_spike_series": r.n_spike_series,
        "demand_class_counts": r.demand_class_counts,
        "intermittent_share": r.intermittent_share,
        "coverage": {k: v for k, v in r.coverage.items() if k != "zeros"},
        "leakage_checks": r.leakage_checks, "notes": r.notes,
    })


def stage_eda(cfg: Config, args) -> None:
    from src import eda
    _save(cfg, "eda", eda.run(cfg))


def stage_baselines(cfg: Config, args) -> None:
    """Baselines alone, on every validation fold. Cheap, and the bar to beat."""
    from src.data.loader import rolling_origin_folds
    from src.evaluation.metrics import skill_vs
    from src.evaluation.validate import aggregate_scores, per_fold_table, walk_forward

    panel = load_panel(cfg)
    print(f"folds: {[f.name for f in rolling_origin_folds(cfg, panel)]}")
    results = walk_forward(cfg, panel, fit_ml=False, verbose=True)
    agg = aggregate_scores(results)
    print("\n--- baselines, mean across validation folds ---")
    print(agg.round(4).to_string(index=False))
    tbl = skill_vs(agg.rename(columns={"model": "model"}), "seasonal_naive")
    _save(cfg, "baselines", {
        "aggregate": tbl,
        "per_fold_wape": per_fold_table(results).reset_index(names="model"),
        "descriptions": __import__(
            "src.forecasting.baselines", fromlist=["BASELINE_DESCRIPTIONS"]
        ).BASELINE_DESCRIPTIONS,
    })


def stage_train(cfg: Config, args) -> None:
    """Walk-forward validation, held-out test, uncertainty, and persistence."""
    from src.evaluation.metrics import compare, score, skill_vs
    from src.evaluation.segmentation import (
        abc_xyz_matrix,
        best_model_per_segment,
        classification_note,
        scores_by_segment,
        segment_series,
        segment_summary,
    )
    from src.evaluation.uncertainty import (
        empirical_residual_quantiles,
        interval_report,
    )
    from src.evaluation.validate import (
        aggregate_scores,
        combine_predictions,
        horizon_breakdown,
        per_fold_table,
        walk_forward,
    )
    from src.forecasting import baselines as B
    from src.forecasting import ml as ML
    from src.forecasting.persist import ModelMetadata, current_versions, save_bundle
    from src.inventory.policy import PolicyParams

    cfg.ensure_dirs()
    panel = load_panel(cfg)
    params = PolicyParams.from_config(cfg)
    test_start, test_end = test_window(cfg)

    print("--- walk-forward validation (refit per fold) ---")
    results = walk_forward(cfg, panel, fit_ml=True, n_origins=int(args.n_origins),
                           include_test=True, quantiles=True,
                           protection_days=params.protection_days)
    val_results = [r for r in results if r.fold != "test"]
    test_result = next(r for r in results if r.fold == "test")

    agg = aggregate_scores(val_results)
    agg = skill_vs(agg, "seasonal_naive")
    print("\n--- model comparison, mean across validation folds ---")
    print(agg.round(4).to_string(index=False))

    pf = per_fold_table(val_results)
    print("\n--- WAPE per fold (a single lucky fold cannot hide here) ---")
    print(pf.round(4).to_string())

    # ---- held-out test fold ----
    tp = test_result.predictions
    hist = panel[panel["date"] <= test_result.origin]
    denom = B.mase_denominator(hist, SERIES_ID, "date", "demand", 7)
    model_cols = [c for c in tp.columns if c not in
                  (SERIES_ID, "origin", "target_date", "horizon", "y_true")
                  and not c.startswith("q")]
    test_scores = {c: score(tp["y_true"], tp[c], tp[SERIES_ID], denom)
                   for c in model_cols}
    test_tbl = skill_vs(compare(test_scores), "seasonal_naive")
    print("\n--- HELD-OUT TEST (28 days, never used for fitting or selection) ---")
    print(test_tbl.round(4).to_string(index=False))

    best_model = str(test_tbl.iloc[0]["model"])
    point_col = "lightgbm" if "lightgbm" in model_cols else best_model

    hb = horizon_breakdown(tp, model_cols, cfg, denom)
    print("\n--- accuracy by horizon (error should grow with distance) ---")
    print(hb.pivot(index="horizon", columns="model", values="wape").round(4).to_string())

    # ---- uncertainty ----
    qcols = {q: f"q{int(q * 100)}" for q in [float(x) for x in cfg["models"]["quantiles"]]
             if f"q{int(q * 100)}" in tp.columns}
    ureport = pd.DataFrame()
    uhorizon = pd.DataFrame()
    if len(qcols) >= 2:
        ureport = interval_report(tp, qcols)
        uhorizon = interval_report(tp, qcols, group="horizon")
        print("\n--- prediction interval calibration (P10-P90, nominal 80%) ---")
        print(ureport.round(4).to_string(index=False))
        print(uhorizon[["horizon", "empirical_coverage", "mean_interval_width"]]
              .round(4).to_string(index=False))

    val_preds = combine_predictions(val_results)
    resid = empirical_residual_quantiles(val_preds, point_col)

    # ---- segmentation ----
    seg = segment_series(cfg, panel, as_of=test_result.origin)
    seg_sum = segment_summary(seg)
    print("\n--- demand segments ---")
    print(seg_sum.drop(columns=["recommended_approach"]).round(4).to_string(index=False))
    seg_scores = scores_by_segment(tp, seg, model_cols, denom)
    best_seg = best_model_per_segment(seg_scores)
    print("\n--- best model per demand class (test fold) ---")
    print(best_seg.drop(columns=["recommended_approach"]).round(4).to_string(index=False))

    # ---- error analysis ----
    from src.evaluation.errors import (
        error_attribution,
        error_drivers,
        promo_effect_errors,
        worst_series,
    )
    attrib = error_attribution(tp, point_col, panel, test_result.origin, seg)
    drivers = error_drivers(attrib)
    worst = worst_series(attrib)
    promo = promo_effect_errors(tp, panel, point_col)
    print("\n--- what drives forecast error (Spearman vs series WAPE) ---")
    print(drivers.round(4).to_string(index=False))
    print("\n--- accuracy on SNAP vs non-SNAP days ---")
    print(promo.round(4).to_string(index=False))

    # ---- persist ----
    from src.features.build import build_supervised, training_origins
    fit_origins = training_origins(cfg, panel, test_result.origin,
                                   n_origins=int(args.n_origins))
    train_df = build_supervised(cfg, panel, test_fold(cfg), origins=fit_origins)
    point_model = ML.fit_lightgbm(cfg, train_df)
    qmodels = ML.fit_quantiles(cfg, train_df)
    importance = ML.feature_importance(point_model)
    print("\n--- top features (LightGBM gain) ---")
    print(importance.head(15).to_string(index=False))

    metadata = ModelMetadata(
        model_name=point_model.name,
        trained_at=pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        bundle_format_version="1.0", seed=cfg.seed,
        grain=list(cfg["data"]["grain"]), target=cfg.target, horizons=cfg.horizons,
        train_origin=str(test_result.origin.date()), n_train_rows=len(train_df),
        n_series=int(panel[SERIES_ID].nunique()),
        features=point_model.features, categorical_features=point_model.categorical,
        quantiles=[float(q) for q in qmodels],
        validation_scores=_native(agg), test_scores=_native(test_tbl),
        inventory_assumptions=dict(cfg["inventory"]["economics"]),
        versions=current_versions(),
        notes=[
            "Features are anchored to the forecast origin; the leakage probe in "
            "src/data/audit.py mutates all post-origin actuals and asserts no "
            "feature changes.",
            "sell_price, snap and the calendar for the TARGET date are treated as "
            "known in advance, which is an assumption about the business process.",
            f"Held-out test window {test_start.date()}..{test_end.date()} was never "
            "used for fitting or model selection.",
        ],
    )
    bundle, meta = save_bundle(cfg, point_model, qmodels, metadata)
    print(f"\n[saved] {bundle}\n[saved] {meta}")

    # ---- error windows for the inventory stage ----
    ew = pd.concat([r.error_windows for r in val_results if not r.error_windows.empty],
                   ignore_index=True) if any(
        not r.error_windows.empty for r in val_results) else pd.DataFrame()
    proc = cfg.resolve("data", "processed_dir")
    if not ew.empty:
        ew.to_parquet(proc / "error_windows.parquet", index=False)
    tp.to_parquet(proc / "test_predictions.parquet", index=False)
    val_preds.to_parquet(proc / "validation_predictions.parquet", index=False)
    seg.to_parquet(proc / "segments.parquet")

    _save(cfg, "train", {
        "point_model": point_col, "best_test_model": best_model,
        "n_train_rows": len(train_df), "train_origin": test_result.origin,
        "validation_aggregate": agg, "per_fold_wape": pf.reset_index(names="model"),
        "test_table": test_tbl, "horizon_breakdown": hb,
        "interval_report": ureport, "interval_by_horizon": uhorizon,
        "residual_quantiles": resid,
        "segment_summary": seg_sum, "abc_xyz": abc_xyz_matrix(seg).reset_index(),
        "segment_scores": seg_scores, "best_per_segment": best_seg,
        "classification_note": classification_note(),
        "error_drivers": drivers, "worst_series": worst, "promo_errors": promo,
        "feature_importance": importance,
        "n_error_windows": 0 if ew.empty else int(
            ew.groupby(SERIES_ID)["origin"].nunique().median()),
        "test_window": [test_start, test_end],
    })


def stage_classical(cfg: Config, args) -> None:
    from src.data.loader import rolling_origin_folds
    from src.evaluation.metrics import compare, score
    from src.forecasting import baselines as B
    from src.forecasting.classical import forecast_classical, sample_series

    panel = load_panel(cfg)
    fold = rolling_origin_folds(cfg, panel)[-1]
    series = sample_series(cfg, panel)
    print(f"fitting ETS and SARIMA on {len(series)} sampled series at origin "
          f"{fold.origin.date()} ...")
    res = forecast_classical(cfg, panel, fold, series)
    if res.predictions.empty:
        print("no classical forecasts produced")
        return

    p = res.predictions
    hist = panel[panel["date"] <= fold.origin]
    denom = B.mase_denominator(hist, SERIES_ID, "date", "demand", 7)

    # Score the baselines and the ML model on the SAME sampled series, otherwise the
    # comparison would confound method with which series each was asked about.
    from src.features.build import build_supervised
    ev = build_supervised(cfg, panel[panel[SERIES_ID].isin(series)], fold)
    scores = {n: score(ev["y_true"], B.predict(n, ev), ev[SERIES_ID], denom)
              for n in ("seasonal_naive", "moving_average_28", "croston", "sba")}
    for m in ("ets", "sarima"):
        sub = p[p[m].notna()]
        if not sub.empty:
            scores[m] = score(sub["y_true"], sub[m], sub[SERIES_ID], denom)

    tbl = compare(scores)
    print("\n--- classical vs baselines on the SAME sampled series ---")
    print(tbl.round(4).to_string(index=False))
    for n in res.notes:
        print(f"  note: {n}")
    _save(cfg, "classical", {"table": tbl, "n_series": res.n_series,
                             "n_failed": res.n_failed, "notes": res.notes})


def stage_inventory(cfg: Config, args) -> None:
    from src.inventory import costs as C
    from src.inventory.policy import (
        PolicyParams,
        baseline_policy,
        build_policy,
        error_spread,
        lead_time_forecast_errors,
    )
    from src.inventory.simulate import (
        compare_policies,
        coverage_days,
        simulate_policy,
        stockout_risk_table,
    )

    panel = load_panel(cfg)
    proc = cfg.resolve("data", "processed_dir")
    ewp = proc / "error_windows.parquet"
    if not ewp.exists():
        raise FileNotFoundError("run `python -m src.run train` first")
    ew = pd.read_parquet(ewp)
    seg = pd.read_parquet(proc / "segments.parquet")

    params = PolicyParams.from_config(cfg)
    tf = test_fold(cfg)
    point_col = "lightgbm" if "lightgbm" in ew.columns else "moving_average_28"

    lt = lead_time_forecast_errors(ew, point_col, params.protection_days)
    spread = error_spread(lt)
    print(f"lead-time error windows per series: median "
          f"{int(spread['n_windows'].median())} "
          f"(protection window = {params.protection_days} days = "
          f"{params.lead_time_days} lead + {params.review_period_days} review)")

    test_preds = pd.read_parquet(proc / "test_predictions.parquet")
    policy = build_policy(cfg, test_preds, point_col,
                          service_level=params.service_level, lt_errors=lt)
    base = baseline_policy(cfg, panel[panel["date"] <= tf.origin],
                           service_level=params.service_level)

    ratio = float(policy["ss_ratio_emp_over_normal"].median())
    print(f"\nsafety stock at {params.service_level:.0%}: "
          f"normal mean {policy['safety_stock_normal'].mean():.2f}, "
          f"empirical mean {policy['safety_stock_empirical'].mean():.2f} "
          f"(median ratio emp/normal {ratio:.3f})")
    # Data-driven, not asserted: whether the two estimators agree depends on how
    # skewed the measured error distribution is and on how many windows there are.
    if abs(ratio - 1.0) < 0.10:
        print(f"  The two agree closely (within {abs(ratio - 1) * 100:.1f}%), so on "
              "this panel the normal approximation is adequate once the error spread "
              f"is estimated from ~{int(spread['n_windows'].median())} windows per "
              "series rather than a handful.")
    else:
        direction = "below" if ratio < 1 else "above"
        print(f"  The empirical quantile sits {abs(ratio - 1) * 100:.1f}% {direction} "
              "the normal approximation, so lead-time forecast error is materially "
              "skewed and the Gaussian assumption misplaces the upper tail.")

    sims = []
    for name, pol, col in (
            ("baseline_demand_variability", base, "reorder_point_normal"),
            ("forecast_normal_ss", policy, "reorder_point_normal"),
            ("forecast_empirical_ss", policy, "reorder_point_empirical")):
        sims.append(simulate_policy(
            panel, pol, tf.start, tf.end, params.lead_time_days,
            params.review_period_days, reorder_col=col, policy_name=name))

    cmp = compare_policies(sims)
    print("\n--- policy simulation on the held-out 28 days ---")
    print(cmp.round(4).to_string(index=False))

    uv = C.unit_values(panel, as_of=tf.origin)
    econ = C.Economics.from_config(cfg)
    n_days = int((tf.end - tf.start).days) + 1
    costed = {s.policy_name: C.cost_simulation(s.per_series, uv, econ, days=n_days)
              for s in sims}
    cost_tbl = pd.DataFrame([{"policy": k, **C.cost_summary(v)}
                             for k, v in costed.items()])
    print("\n--- cost comparison (ASSUMED economics, currency units) ---")
    print(cost_tbl.round(2).to_string(index=False))

    best_sim = min(sims, key=lambda s: float(
        cost_tbl.loc[cost_tbl["policy"] == s.policy_name, "total_cost_cu"].iloc[0]))
    risk = stockout_risk_table(best_sim.per_series, seg)
    cov = coverage_days(best_sim.per_series)
    print(f"\n--- highest-risk series under {best_sim.policy_name} ---")
    print(risk.head(10).round(3).to_string(index=False))
    print(f"\ninventory coverage: median {cov['coverage_days'].median():.1f} days")

    policy.to_parquet(proc / "policy.parquet", index=False)
    _save(cfg, "inventory", {
        "protection_days": params.protection_days,
        "lead_time_days": params.lead_time_days,
        "review_period_days": params.review_period_days,
        "service_level": params.service_level,
        "median_error_windows_per_series": int(spread["n_windows"].median()),
        "mean_ss_normal": float(policy["safety_stock_normal"].mean()),
        "mean_ss_empirical": float(policy["safety_stock_empirical"].mean()),
        "median_ss_ratio": float(policy["ss_ratio_emp_over_normal"].median()),
        "policy_comparison": cmp, "cost_comparison": cost_tbl,
        "best_policy": best_sim.policy_name,
        "risk_table": risk.head(15),
        "median_coverage_days": float(cov["coverage_days"].median()),
        "economics": econ.as_dict(),
    })


def stage_scenarios(cfg: Config, args) -> None:
    from src.inventory import costs as C
    from src.inventory.policy import PolicyParams, lead_time_forecast_errors

    panel = load_panel(cfg)
    proc = cfg.resolve("data", "processed_dir")
    test_preds = pd.read_parquet(proc / "test_predictions.parquet")
    ew = pd.read_parquet(proc / "error_windows.parquet")
    tf = test_fold(cfg)
    params = PolicyParams.from_config(cfg)
    point_col = "lightgbm" if "lightgbm" in test_preds.columns else "moving_average_28"
    n_days = int((tf.end - tf.start).days) + 1

    # The multi-origin error windows are required: one window per series makes the
    # empirical quantile independent of the service level.
    lt = lead_time_forecast_errors(ew, point_col, params.protection_days)
    tbl = C.scenario_table(cfg, panel, test_preds, point_col, tf.start, tf.end,
                           lt_errors=lt)
    print("--- service-level scenarios (ASSUMED economics) ---")
    print(tbl.round(4).to_string(index=False))

    nv = C.newsvendor_optimum(cfg, panel, tf.origin, n_days)
    print(f"\nnewsvendor critical ratio Cu/(Cu+Co) = {nv['critical_ratio_mean']:.4f} "
          f"(Cu {nv['cu_mean_cu']:.3f} CU vs Co {nv['co_mean_cu']:.4f} CU over "
          f"{n_days} days)")
    best = tbl.loc[tbl["total_cost_cu"].idxmin()]
    print(f"simulated cost-minimum service level: {best['service_level']:.0%}")

    sens = C.sensitivity(cfg, panel, test_preds, point_col, tf.start, tf.end,
                         lt_errors=lt)
    print("\n--- sensitivity: how the optimum moves with the stockout penalty ---")
    print(sens.round(4).to_string(index=False))

    _save(cfg, "scenarios", {
        "scenario_table": tbl, "newsvendor": {k: v for k, v in nv.items()
                                              if k != "assumptions"},
        "cost_optimal_service_level": float(best["service_level"]),
        "sensitivity": sens,
    })


def stage_forecast(cfg: Config, args) -> None:
    from src.inference.forecaster import DemandForecaster, format_recommendation

    panel = load_panel(cfg)
    f = DemandForecaster.load(cfg)
    seg = pd.read_parquet(cfg.resolve("data", "processed_dir") / "segments.parquet")
    # Show a high-volume series: the recommendation is most legible where demand is
    # not almost entirely zero. The origin is the test-fold origin, so the 28 days
    # forecast are the genuinely held-out window and the exogenous calendar is real
    # rather than carried forward.
    sid = str(seg.sort_values("total_demand", ascending=False).index[0])
    rec = f.recommend(panel, sid, origin=test_fold(cfg).origin)
    print(format_recommendation(rec))
    print("\nfirst 10 days of the forecast:")
    print(rec.to_frame().head(10).round(3).to_string(index=False))
    _save(cfg, "example_forecast", {
        **rec.as_dict(),
        "forecast_head": rec.to_frame().head(10),
    })


def stage_report(cfg: Config, args) -> None:
    from src.reporting import write_report
    print(f"[written] {write_report(cfg, load_results(cfg))}")


STAGES = {
    "audit": stage_audit, "eda": stage_eda, "baselines": stage_baselines,
    "train": stage_train, "classical": stage_classical,
    "inventory": stage_inventory, "scenarios": stage_scenarios,
    "forecast": stage_forecast, "report": stage_report,
}
ORDER = ["audit", "eda", "baselines", "train", "classical", "inventory",
         "scenarios", "forecast", "report"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=[*ORDER, "all"])
    ap.add_argument("--n-origins", type=int, default=40,
                    help="training origins per fold (more = slower, more data)")
    args = ap.parse_args()

    cfg = load_config()
    cfg.ensure_dirs()
    if not is_real_panel(cfg):
        print("NOTE: running on the SYNTHETIC panel, not the real M5 data.\n"
              "      Figures in reports/ come from the real dataset.\n"
              "      Run `make download && make panel` to reproduce them.\n")

    for name in (ORDER if args.stage == "all" else [args.stage]):
        print(f"\n{'=' * 78}\n{name.upper()}\n{'=' * 78}")
        STAGES[name](cfg, args)


if __name__ == "__main__":
    main()
