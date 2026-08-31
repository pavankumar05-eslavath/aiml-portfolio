"""Stage runner.

    python -m src.run audit      data quality and leakage audit
    python -m src.run eda        exploratory figures
    python -m src.run train      train, tune, calibrate, evaluate, persist
    python -m src.run explain     SHAP global + per-customer explanations
    python -m src.run report     write reports/model_report.md
    python -m src.run predict    score the example customer
    python -m src.run all        every stage in order

Uses the real dataset when present, otherwise the synthetic stand-in.
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.data.loader import is_real_dataset, make_dataset, resolve_data_path

warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS_NAME = "run_results.json"


def _to_native(obj):
    """Make numpy/pandas types JSON-serialisable."""
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _to_native(obj.to_dict(orient="records"))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    return obj


def _save_results(cfg: Config, key: str, payload: dict) -> None:
    """Accumulate stage outputs so `report` can run without retraining."""
    path = cfg.resolve("paths", "reports_dir") / RESULTS_NAME
    existing: dict = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing[key] = _to_native(payload)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def load_results(cfg: Config) -> dict:
    path = cfg.resolve("paths", "reports_dir") / RESULTS_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"no {RESULTS_NAME}; run `python -m src.run train` first.")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_audit(cfg: Config, args) -> None:
    from src.data.audit import run as run_audit
    res = run_audit(cfg, args.data)
    _save_results(cfg, "audit", {
        "n_rows": res.n_rows, "churn_rate": res.churn_rate,
        "missing": res.missing,
        "n_exact_duplicates": res.n_exact_duplicates,
        "n_id_duplicates": res.n_id_duplicates,
        "n_feature_duplicates": res.n_feature_duplicates,
        "n_safe_duplicates_removed": res.n_safe_duplicates_removed,
        "n_conflicting_label_rows": res.n_conflicting_label_rows,
        "n_conflicting_groups": res.n_conflicting_groups,
        "label_noise_rate": res.label_noise_rate,
        "contamination_overlap": res.contamination_overlap,
        "total_charges_r2": res.total_charges_r2,
        "single_feature_auc": res.single_feature_auc,
        "leakage_suspects": res.leakage_suspects,
        "invalid_values": res.invalid_values,
        "invalid_categories": res.invalid_categories,
        "structural_inconsistencies": res.structural_inconsistencies,
    })


def stage_eda(cfg: Config, args) -> None:
    from src import eda
    _save_results(cfg, "eda", eda.run(cfg, args.data))


def stage_train(cfg: Config, args) -> None:
    from src.evaluation import metrics as M
    from src.evaluation import plots
    from src.evaluation.error_analysis import (
        error_costs,
        error_frame,
        hardest_cases,
        label_noise_ceiling,
        profile_errors,
        threshold_recommendation,
    )
    from src.evaluation.segments import (
        performance_disparities,
        segment_summary,
        validate_segments,
    )
    from src.evaluation.threshold import Economics, optimise, sensitivity_analysis
    from src.inference.decision import campaign_summary, decide
    from src.models.persist import build_metadata, save_model
    from src.models.train import (
        comparison_table,
        paired_comparison,
        select_best,
        train_all,
    )

    cfg.ensure_dirs()
    ds = make_dataset(cfg, args.data)
    print(f"train {ds.n_train:,} / test {ds.n_test:,} rows "
          f"(churn {ds.y_train.mean():.2%} / {ds.y_test.mean():.2%})")
    print(f"deduplicated {ds.n_duplicates_removed} rows; "
          f"{ds.n_conflicting_labels} label-conflicting rows retained\n")

    results = train_all(ds, cfg)
    table = comparison_table(results)
    print("\n--- cross-validated comparison (train only) ---")
    print(table.round(4).to_string(index=False))

    best = select_best(results)
    simple = results["logistic_regression_plain"]
    cmp = paired_comparison(best, simple)
    print(f"\n--- is {best.name} really better than untuned logistic regression? ---")
    print(f"  PR-AUC {cmp['mean_a']:.4f} vs {cmp['mean_b']:.4f}  "
          f"diff {cmp['difference']:+.4f} ({cmp['difference_in_sd']:.2f} pooled SD)")
    print(f"  paired t-test p={cmp['ttest_p']:.4f}  wilcoxon p={cmp['wilcoxon_p']:.4f}"
          f"  -> significant: {cmp['significant_at_05']}")

    fitted = best.pipeline.fit(ds.X_train, ds.y_train)

    print("\n--- calibration (fitted on train, judged on test) ---")
    cal_best, cal_all = M.calibrate(fitted, ds, cfg)
    for c in cal_all:
        print(f"  {c.method:9s} brier {c.brier_after:.4f}  ece {c.ece_after:.4f}  "
              f"logloss {c.log_loss_after:.4f}  pr_auc {c.pr_auc_after:.4f}")
    print(f"  selected: {cal_best.method}"
          f"  (min_brier_gain={cfg['calibration']['min_brier_gain']})")

    scorer = cal_best.model
    prob = scorer.predict_proba(ds.X_test)[:, 1]
    y_test = ds.y_test.to_numpy()

    print("\n--- split stability: each fold used as the test set in turn ---")
    stability = M.split_stability(fitted, cfg, path=args.data)
    print(stability.round(4).to_string(index=False))
    print(f"  designated test fold PR-AUC {stability.loc[0, 'pr_auc']:.4f} vs "
          f"across-fold mean {stability['pr_auc'].mean():.4f} "
          f"(sd {stability['pr_auc'].std():.4f})")

    econ = Economics.from_config(cfg)
    value = econ.customer_value(ds.X_test)
    thr = optimise(y_test, prob, value, econ)
    print("\n--- threshold optimisation (ASSUMED economics) ---")
    print(f"  value-optimal threshold : {thr.best_threshold:.2f} "
          f"-> {thr.best_net_value:,.0f} CU")
    print(f"  contact everyone        : {thr.baseline_contact_all:,.0f} CU")
    print(f"  contact nobody          : {thr.baseline_contact_none:,.0f} CU")
    print(f"  per-customer EV rule    : {thr.per_customer_net_value:,.0f} CU "
          f"({thr.per_customer_contacts} contacts)")

    m_default = M.evaluate(y_test, prob, 0.5)
    m_tuned = M.evaluate(y_test, prob, thr.best_threshold)
    print("\n--- test metrics ---")
    print(M.metrics_frame({"threshold_0.50": m_default,
                           f"threshold_{thr.best_threshold:.2f}": m_tuned}).round(4).to_string())

    # ---- evaluation figures ----
    figdir = cfg.resolve("paths", "figures_dir")
    # Compare the served probabilities against the calibrated alternatives, so the
    # reliability diagram shows what was chosen AND what was rejected.
    prob_variants = {c.method if c.method != "none" else "uncalibrated":
                     c.model.predict_proba(ds.X_test)[:, 1] for c in cal_all}
    written = [
        plots.plot_calibration(y_test, prob_variants, figdir / "10_calibration_curve.png"),
        plots.plot_threshold_analysis(thr.curve, thr.best_threshold,
                                     figdir / "11_threshold_analysis.png",
                                     per_customer_value=thr.per_customer_net_value),
        plots.plot_pr_roc(y_test, prob, figdir / "12_pr_roc_curves.png"),
        plots.plot_confusion({"threshold 0.50": m_default,
                              f"threshold {thr.best_threshold:.2f}": m_tuned},
                             figdir / "13_confusion_matrices.png"),
    ]
    print(f"\n[figures] {', '.join(p.name for p in written)}")

    sens = sensitivity_analysis(y_test, prob, value, cfg)
    print("\n--- sensitivity of the optimal threshold to the assumptions ---")
    print(sens.round(3).to_string(index=False))

    seg = segment_summary(prob, y_test, cfg, value=value)
    print("\n--- risk segments (validated against realised churn) ---")
    print(seg.drop(columns=["recommended_action"]).round(4).to_string())
    seg_valid = validate_segments(seg)
    print(f"  monotonic realised churn: {seg_valid['monotonic_realised_churn']}"
          f" | max |predicted-realised| = {seg_valid['max_abs_calibration_gap']:.4f}"
          f" | top/bottom lift = {seg_valid['lift_top_vs_bottom']:.1f}x")

    decisions = decide(prob, ds.X_test, cfg)
    camp = campaign_summary(decisions)
    print("\n--- campaign plan (risk x value) ---")
    print(camp.round(3).to_string(index=False))

    # The operational artefact: one row per scored customer with risk band,
    # expected value and recommended action, ordered so a retention team can work
    # down it. This is what the model is actually for.
    processed = cfg.resolve("data", "processed_dir")
    processed.mkdir(parents=True, exist_ok=True)
    scored = decisions.copy()
    scored.insert(0, cfg.id_column, ds.ids_test.to_numpy())
    scored["actually_churned"] = y_test
    scored = scored.sort_values(["priority", "expected_value_cu"],
                               ascending=[True, False])
    scored_path = processed / "scored_test_customers.csv"
    scored.to_csv(scored_path, index=False)
    print(f"  [saved] {scored_path.relative_to(cfg.resolve('paths', 'models_dir').parent)}"
          f"  ({len(scored):,} rows, ranked call list)")
    print(f"  contacts recommended: {int(decisions['contact_recommended'].sum())} "
          f"of {len(decisions)}; EV-overrides-band: "
          f"{int(decisions['ev_overrides_segment'].sum())}")

    errors = error_frame(ds.X_test, y_test, prob, thr.best_threshold)
    costs = error_costs(errors, cfg)
    print("\n--- error analysis ---")
    print(profile_errors(errors).round(3).to_string(index=False))
    print(f"  FN {costs['n_false_negative']} @ ~{costs['mean_fn_cost_cu']:.0f} CU each; "
          f"FP {costs['n_false_positive']} @ {costs['mean_fp_cost_cu']:.0f} CU each; "
          f"ratio {costs['cost_ratio_fn_to_fp']:.1f}:1")
    ceiling = label_noise_ceiling(errors, ds.n_conflicting_labels,
                                 ds.n_train + ds.n_test)
    print(f"  observed error {ceiling['observed_error_rate']:.4f}; irreducible floor "
          f"~{ceiling['irreducible_floor_estimate']:.4f}")

    disp, spreads = performance_disparities(ds.X_test, y_test, prob, thr.best_threshold)
    print("\n--- segment-level performance ---")
    print(disp.round(3).to_string(index=False))
    print("  worst spreads:", {k: round(v, 3) for k, v in
                               sorted(spreads.items(), key=lambda kv: -kv[1])[:4]})

    metadata = build_metadata(
        model_name=best.name, cfg=cfg, threshold=thr.best_threshold,
        threshold_rationale=(
            "Maximises assumed net retention value on the test set. Depends on the "
            "economics in configs/config.yaml, which are assumptions, not measurements."),
        calibration_method=cal_best.method,
        cv_metrics={k: float(v) for k, v in best.cv_means.items()},
        test_metrics=m_tuned.as_dict(),
        n_train=ds.n_train, n_test=ds.n_test,
        notes=[
            f"Selected over untuned logistic regression by {cmp['difference']:+.4f} PR-AUC, "
            f"which is NOT statistically significant (paired t p={cmp['ttest_p']:.3f}).",
            f"Designated test fold is the weakest of {len(stability)} folds "
            f"(PR-AUC {stability.loc[0, 'pr_auc']:.4f} vs mean "
            f"{stability['pr_auc'].mean():.4f}); the split was not re-drawn.",
            f"{ds.n_conflicting_labels} feature-identical rows carry contradictory labels.",
        ],
    )
    bundle_path, meta_path = save_model(scorer, metadata, cfg,
                                       explainer_pipeline=fitted)
    print(f"\n[saved] {bundle_path}")
    print(f"[saved] {meta_path}")

    _save_results(cfg, "train", {
        "n_train": ds.n_train, "n_test": ds.n_test,
        "churn_train": float(ds.y_train.mean()), "churn_test": float(ds.y_test.mean()),
        "n_duplicates_removed": ds.n_duplicates_removed,
        "n_conflicting_labels": ds.n_conflicting_labels,
        "comparison": table, "best_model": best.name,
        "best_params": best.best_params,
        "paired_comparison": cmp,
        "calibration": [
            {"method": c.method, "brier": c.brier_after, "ece": c.ece_after,
             "log_loss": c.log_loss_after, "pr_auc": c.pr_auc_after,
             "brier_before": c.brier_before, "ece_before": c.ece_before}
            for c in cal_all],
        "calibration_selected": cal_best.method,
        "stability": stability,
        "threshold": {
            "best": thr.best_threshold, "net_value": thr.best_net_value,
            "contact_all": thr.baseline_contact_all,
            "per_customer_net_value": thr.per_customer_net_value,
            "per_customer_contacts": thr.per_customer_contacts,
            "breakeven_min": float(econ.breakeven_probability(value).min()),
            "breakeven_max": float(econ.breakeven_probability(value).max()),
        },
        "metrics_default": m_default.as_dict(),
        "metrics_tuned": m_tuned.as_dict(),
        "sensitivity": sens,
        "segments": seg.reset_index(),
        "segment_validation": seg_valid,
        "campaign": camp,
        "n_contacts": int(decisions["contact_recommended"].sum()),
        "n_ev_overrides": int(decisions["ev_overrides_segment"].sum()),
        "error_costs": costs,
        "error_profile": profile_errors(errors),
        "hardest": dict(hardest_cases(errors)),
        "label_noise": ceiling,
        "threshold_note": threshold_recommendation(errors, cfg, thr.best_threshold),
        "disparities": disp,
        "spreads": spreads,
    })


def stage_explain(cfg: Config, args) -> None:
    from src.data.schema import feature_columns
    from src.explainability.shap_analysis import (
        compute_shap,
        explain_customer,
        representative_customers,
        save_global_plots,
        save_waterfall,
    )
    from src.models.persist import load_model

    cfg.ensure_dirs()
    bundle = load_model(cfg)
    fitted = bundle["explainer_pipeline"]
    scorer = bundle["model"]
    ds = make_dataset(cfg, args.data)
    figdir = cfg.resolve("paths", "figures_dir")

    art = compute_shap(fitted, ds.X_test, max_samples=1000, seed=cfg.seed)
    raw_cols = feature_columns()

    grouped = art.grouped_importance(raw_cols)
    concepts = art.concept_importance(raw_cols)
    print("--- global importance, aggregated to business features (top 12) ---")
    print(grouped.head(12).round(4).to_string(index=False))
    print("\n--- ...and to business concepts, matching the local explanations ---")
    print(concepts.head(10).round(4).to_string(index=False))

    figures = save_global_plots(art, figdir, raw_cols)
    print(f"\n[figures] {', '.join(figures)}")

    prob_all = scorer.predict_proba(ds.X_test)[:, 1]
    sub_prob = prob_all[art.raw.index.to_numpy()]
    picks = representative_customers(sub_prob, art, cfg=cfg)

    explanations = {}
    print("\n--- representative customers ---")
    for band, pos in picks.items():
        exp = explain_customer(art, pos, raw_cols)
        row = art.raw.iloc[pos]
        print(f"\n  [{band}] predicted churn {sub_prob[pos]:.1%} | "
              f"{row['Contract']}, tenure {row['tenure']}m, "
              f"{row['MonthlyCharges']:.2f}/month")
        print("    increases risk:")
        for f in exp["top_risk_factors"]:
            print(f"      + {f['reason']}  ({f['contribution']:+.3f})")
        print("    reduces risk:")
        for f in exp["top_protective_factors"]:
            print(f"      - {f['reason']}  ({f['contribution']:+.3f})")

        fname = f"09_waterfall_{band.split()[0].lower()}.png"
        save_waterfall(art, pos, figdir, fname)
        explanations[band] = {
            "probability": float(sub_prob[pos]),
            "contract": str(row["Contract"]),
            "tenure": int(row["tenure"]),
            "monthly_charges": float(row["MonthlyCharges"]),
            "figure": fname,
            **exp,
        }

    _save_results(cfg, "explain", {
        "grouped_importance": grouped.head(15),
        "concept_importance": concepts.head(10),
        "representatives": explanations,
    })


def stage_predict(cfg: Config, args) -> None:
    from src.inference.predict import EXAMPLE_CUSTOMER, ChurnPredictor, format_result
    predictor = ChurnPredictor.load(cfg)
    result = predictor.predict_one(EXAMPLE_CUSTOMER)
    print(format_result(result))
    _save_results(cfg, "example_prediction", result)


def stage_report(cfg: Config, args) -> None:
    from src.reporting import write_report
    path = write_report(cfg, load_results(cfg))
    print(f"[written] {path}")


STAGES = {
    "audit": stage_audit,
    "eda": stage_eda,
    "train": stage_train,
    "explain": stage_explain,
    "predict": stage_predict,
    "report": stage_report,
}
ORDER = ["audit", "eda", "train", "explain", "predict", "report"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=[*ORDER, "all"])
    ap.add_argument("--data", default=None, help="path to a CSV with the published schema")
    args = ap.parse_args()

    cfg = load_config()
    src = resolve_data_path(cfg, args.data)
    if not src.exists():
        raise SystemExit(
            f"no dataset at {src}.\nRun `make download` for the real data, "
            "or `make sample` for the synthetic stand-in.")
    if not is_real_dataset(cfg, args.data):
        print("NOTE: running on the SYNTHETIC stand-in, not the real dataset.\n"
              "      Reported figures in reports/ come from the real 7,043-row file.\n"
              "      Run `make download` to reproduce them.\n")

    for name in (ORDER if args.stage == "all" else [args.stage]):
        print(f"\n{'=' * 78}\n{name.upper()}   ({src.name})\n{'=' * 78}")
        STAGES[name](cfg, args)


if __name__ == "__main__":
    main()
