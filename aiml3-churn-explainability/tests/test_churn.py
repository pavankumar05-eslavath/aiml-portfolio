"""Tests for the churn system.

Two kinds of test live here, and the second kind is the point:

1. **Contract tests** -- the pipeline accepts the documented input, returns
   probabilities in range, fails loudly on a malformed payload, round-trips
   through joblib.
2. **Claim tests** -- assertions on the conclusions in reports/model_report.md.
   Where a stated finding depends on a property of the data or of the pipeline, a
   test pins it, so a future change that invalidates the write-up fails instead of
   silently making the README wrong.

Tests needing the full dataset carry ``requires_real_data`` and skip on the
synthetic stand-in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data.loader import (
    clean_raw,
    deduplicate,
    feature_group_ids,
    load_raw,
    make_dataset,
)
from src.data.schema import (
    ALLOWED_VALUES,
    DATA_DICTIONARY,
    EXPECTED_COLUMNS,
    ID_COLUMN,
    TARGET,
    VALID_RANGES,
    feature_columns,
)
from src.evaluation.metrics import evaluate, expected_calibration_error
from src.evaluation.segments import assign_risk_segments, segment_summary, validate_segments
from src.evaluation.threshold import Economics, net_value, optimise
from src.features.engineer import FEATURE_DOCS, ChurnFeatureEngineer, _safe_divide
from src.features.preprocess import build_pipeline, numeric_features, output_feature_names
from src.inference.decision import ACTION_MATRIX, decide
from src.models.persist import load_model, validate_input
from tests.conftest import requires_real_data

# ===========================================================================
# 1. Data validation
# ===========================================================================

def test_schema_matches_published_file(cfg):
    raw = load_raw(cfg)
    assert tuple(raw.columns) == EXPECTED_COLUMNS


def test_data_dictionary_covers_every_column():
    documented = {spec.name for spec in DATA_DICTIONARY}
    assert documented == set(EXPECTED_COLUMNS)


def test_categorical_values_are_all_declared(cfg):
    raw = load_raw(cfg)
    for col, allowed in ALLOWED_VALUES.items():
        unexpected = set(raw[col].dropna().astype(str)) - set(allowed)
        assert not unexpected, f"{col} has undeclared values {unexpected}"


def test_numeric_values_within_declared_ranges(cfg):
    raw = clean_raw(load_raw(cfg))
    for col, (lo, hi) in VALID_RANGES.items():
        s = pd.to_numeric(raw[col], errors="coerce").dropna()
        assert s.between(lo, hi).all(), f"{col} has out-of-range values"


def test_total_charges_blanks_are_structural_not_random(cfg):
    """The 11 blanks are never-billed customers, so 0.0 is the true value.

    Pinned because the tempting fix -- median imputation -- invents a billing
    history for customers who have none, and does so at the exact tenure where
    churn risk is highest.
    """
    raw = load_raw(cfg)
    blanks = raw[raw["TotalCharges"].isna()]
    if len(blanks) == 0:
        pytest.skip("no blank TotalCharges in this dataset")
    assert (blanks["tenure"] == 0).all(), "blanks must coincide with tenure == 0"
    cleaned = clean_raw(raw)
    assert cleaned["TotalCharges"].isna().sum() == 0
    assert (cleaned.loc[blanks.index, "TotalCharges"] == 0.0).all()


def test_no_duplicate_customer_ids(cfg):
    raw = load_raw(cfg)
    assert raw[ID_COLUMN].duplicated().sum() == 0


def test_expected_columns_are_order_independent_of_the_file(cfg):
    """Regression guard: EXPECTED_COLUMNS is grouped by TYPE, not file order.

    The published file interleaves SeniorCitizen and tenure among the categorical
    columns, while EXPECTED_COLUMNS groups identifier / categorical / numeric /
    target for readability. `load_raw` selects by name so order is irrelevant --
    but `download.py` once compared the header as an ordered tuple and rejected
    the correct file. Anything comparing these must use sets.
    """
    raw = pd.read_csv(cfg.resolve("data", "raw_file")
                      if cfg.resolve("data", "raw_file").exists()
                      else cfg.resolve("data", "sample_file"), nrows=0)
    assert set(raw.columns) == set(EXPECTED_COLUMNS)


def test_identifier_is_excluded_from_features():
    """An id lets a tree memorise individuals and generalise to nobody."""
    assert ID_COLUMN not in feature_columns()
    assert TARGET not in feature_columns()


# ===========================================================================
# 2. Deduplication and contamination
# ===========================================================================

def test_label_conflicting_groups_are_retained_whole(cfg):
    """Conflicting groups are irreducible noise and must be flagged in full.

    Comparing ``duplicated(feats)`` with ``duplicated(feats + target)`` marks only
    the odd row out, so a Yes/No/No group counts as one conflict instead of three.
    """
    cleaned = clean_raw(load_raw(cfg))
    feats = feature_columns()
    _, n_removed, n_conflicting = deduplicate(cleaned)

    per_group = cleaned.groupby(feats, dropna=False, observed=True)[TARGET].nunique()
    expected_rows = int(cleaned.groupby(feats, dropna=False, observed=True)[TARGET]
                        .transform("nunique").gt(1).sum())
    assert n_conflicting == expected_rows
    if (per_group > 1).any():
        assert n_conflicting >= 2 * int((per_group > 1).sum())
    assert n_removed >= 0


def test_deduplication_removes_only_safe_duplicates(cfg):
    cleaned = clean_raw(load_raw(cfg))
    deduped, _, _ = deduplicate(cleaned)
    feats = feature_columns()
    remaining = deduped[deduped.duplicated(subset=feats, keep=False)]
    if len(remaining):
        # anything left must be a genuine label conflict
        assert (remaining.groupby(feats, dropna=False, observed=True)[TARGET]
                .transform("nunique") > 1).all()


def test_split_has_zero_feature_overlap(dataset):
    """Group-aware splitting must leave no identical feature vector on both sides.

    Deduplication alone does not achieve this, because label-conflicting rows are
    deliberately kept and are duplicates by construction. A plain stratified
    ``train_test_split`` left 5 overlapping rows.
    """
    feats = feature_columns()
    train_keys = set(map(tuple, dataset.X_train[feats].astype(str).to_numpy()))
    test_keys = map(tuple, dataset.X_test[feats].astype(str).to_numpy())
    assert sum(1 for k in test_keys if k in train_keys) == 0


def test_split_is_stratified(dataset):
    assert abs(dataset.y_train.mean() - dataset.y_test.mean()) < 0.02


def test_split_is_reproducible(cfg):
    a, b = make_dataset(cfg), make_dataset(cfg)
    pd.testing.assert_frame_equal(a.X_test, b.X_test)
    pd.testing.assert_series_equal(a.y_test, b.y_test)


def test_feature_group_ids_are_stable_and_group_identical_rows():
    X = pd.DataFrame({"a": ["x", "x", "y"], "b": [1, 1, 1]})
    g = feature_group_ids(X)
    assert g.iloc[0] == g.iloc[1] != g.iloc[2]


# ===========================================================================
# 3. Feature engineering
# ===========================================================================

def test_every_engineered_feature_is_documented(dataset):
    fe = ChurnFeatureEngineer().fit(dataset.X_train)
    added = set(fe.transform(dataset.X_train).columns) - set(dataset.X_train.columns)
    assert added == set(FEATURE_DOCS), (
        "engineered features and FEATURE_DOCS have diverged")


def test_feature_engineering_is_stateless(dataset):
    """Row-wise features must not depend on the data used to fit.

    This is what makes them safe to compute before the split. If a target-encoded
    or dataset-mean feature were added, this test would fail -- which is the alarm.
    """
    sample = dataset.X_test.head(50)
    fitted_on_train = ChurnFeatureEngineer().fit(dataset.X_train).transform(sample)
    fitted_on_sample = ChurnFeatureEngineer().fit(sample).transform(sample)
    pd.testing.assert_frame_equal(fitted_on_train, fitted_on_sample)


def test_engineered_features_are_finite(dataset):
    out = ChurnFeatureEngineer().fit_transform(dataset.X_train)
    numeric = out.select_dtypes(include=[np.number]).to_numpy()
    assert np.isfinite(numeric).all()


def test_safe_divide_handles_zero_denominator():
    num = pd.Series([10.0, 5.0, 0.0])
    den = pd.Series([2.0, 0.0, 0.0])
    out = _safe_divide(num, den, fill=1.0)
    assert out.tolist() == [5.0, 1.0, 1.0]
    assert np.isfinite(out.to_numpy()).all()


def test_never_billed_customer_gets_neutral_price_ratio():
    """price_vs_history must be 1.0, not 0 or inf, for a never-billed customer.

    Filling with 0 would read as a total price collapse; inf would break the
    estimator. 1.0 means 'no change versus history', which is the truth when there
    is no history.
    """
    row = pd.DataFrame([{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 0, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 45.0, "TotalCharges": 0.0,
    }])
    out = ChurnFeatureEngineer().fit_transform(row)
    assert out["realized_arpu"].iloc[0] == 0.0
    assert out["price_vs_history"].iloc[0] == 1.0
    assert out["is_new_customer"].iloc[0] == 1


def test_addon_count_respects_structural_categories():
    """'No internet service' is not a declined add-on."""
    base = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "No", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check", "MonthlyCharges": 20.0, "TotalCharges": 240.0,
    }
    for c in ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
              "StreamingTV", "StreamingMovies"):
        base[c] = "No internet service"
    out = ChurnFeatureEngineer().fit_transform(pd.DataFrame([base]))
    assert out["n_addons"].iloc[0] == 0
    assert out["has_internet"].iloc[0] == 0
    assert out["addon_adoption_rate"].iloc[0] == 0.0


def test_interaction_feature_excluded_from_tree_pipelines():
    """Measured: it does not help the tree (0.6753 -> 0.6736, p=0.396) and it
    dominated SHAP output with an uninterpretable term."""
    assert "tenure_x_month_to_month" not in numeric_features(include_interactions=False)
    assert "tenure_x_month_to_month" in numeric_features(include_interactions=True)


# ===========================================================================
# 4. Preprocessing
# ===========================================================================

def test_preprocessing_is_fitted_inside_the_pipeline(dataset):
    """Scaler statistics must come from training data only."""
    pipe = build_pipeline(LogisticRegression(max_iter=500), scale=True)
    pipe.fit(dataset.X_train, dataset.y_train)
    scaler = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scale"]

    engineered = pipe.named_steps["features"].transform(dataset.X_train)
    cols = numeric_features()
    np.testing.assert_allclose(scaler.mean_, engineered[cols].mean().to_numpy(),
                               rtol=1e-6)


def test_unknown_category_does_not_raise(dataset, fast_pipeline):
    """handle_unknown='ignore' keeps one bad field from taking the service down."""
    row = dataset.X_test.head(1).copy()
    row.loc[:, "PaymentMethod"] = "Crypto wallet"
    proba = fast_pipeline.predict_proba(row)[:, 1]
    assert 0.0 <= float(proba[0]) <= 1.0


def test_encoded_matrix_is_finite_and_named(dataset, fast_pipeline):
    engineered = fast_pipeline.named_steps["features"].transform(dataset.X_test)
    encoded = fast_pipeline.named_steps["prep"].transform(engineered)
    names = output_feature_names(fast_pipeline)
    assert encoded.shape[1] == len(names)
    assert np.isfinite(encoded).all()


# ===========================================================================
# 5. Prediction contract
# ===========================================================================

def test_probabilities_within_unit_interval(dataset, fast_pipeline):
    proba = fast_pipeline.predict_proba(dataset.X_test)[:, 1]
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0
    assert np.isfinite(proba).all()


def test_single_row_matches_batch_prediction(dataset, fast_pipeline):
    """Guards against train/serve skew.

    Scoring one customer must give exactly what scoring the batch gave. Any
    row-order or aggregate dependence in feature engineering breaks this, and it is
    the failure that shows up only in production.
    """
    batch = fast_pipeline.predict_proba(dataset.X_test.head(10))[:, 1]
    singles = [float(fast_pipeline.predict_proba(dataset.X_test.iloc[[i]])[:, 1][0])
               for i in range(10)]
    np.testing.assert_allclose(batch, singles, rtol=1e-10)


def test_validate_input_rejects_missing_columns():
    meta = {"required_columns": feature_columns()}
    with pytest.raises(ValueError, match="missing required columns"):
        validate_input(pd.DataFrame([{"tenure": 1}]), meta)


def test_validate_input_rejects_empty_frame():
    meta = {"required_columns": feature_columns()}
    empty = pd.DataFrame(columns=feature_columns())
    with pytest.raises(ValueError, match="no rows"):
        validate_input(empty, meta)


# ===========================================================================
# 6. Model persistence
# ===========================================================================

def test_bundle_round_trips(bundle_path, dataset):
    bundle = load_model(path=bundle_path)
    for key in ("model", "metadata", "format_version", "config"):
        assert key in bundle
    proba = bundle["model"].predict_proba(dataset.X_test.head(5))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_bundle_metadata_carries_the_operating_contract(bundle_path):
    """A threshold without its assumptions is a magic number."""
    meta = load_model(path=bundle_path)["metadata"]
    for key in ("required_columns", "threshold", "threshold_rationale", "seed",
                "economics", "risk_segments", "versions", "engineered_features",
                "calibration_method", "trained_at"):
        assert key in meta, f"metadata missing {key}"
    assert meta["required_columns"] == feature_columns()
    assert 0.0 < float(meta["threshold"]) < 1.0
    assert "scikit-learn" in meta["versions"]


def test_loading_a_missing_bundle_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(path=tmp_path / "nope.joblib")


# ===========================================================================
# 7. Metrics, calibration, thresholds
# ===========================================================================

def test_evaluate_confusion_matrix_is_consistent(dataset, fast_pipeline):
    prob = fast_pipeline.predict_proba(dataset.X_test)[:, 1]
    m = evaluate(dataset.y_test, prob, 0.5)
    assert m.tn + m.fp + m.fn + m.tp == len(dataset.y_test)
    assert m.tp + m.fn == int(dataset.y_test.sum())
    assert 0.0 <= m.roc_auc <= 1.0
    assert 0.0 <= m.pr_auc <= 1.0


def test_perfect_predictions_score_perfectly():
    y = np.array([0, 0, 1, 1])
    m = evaluate(y, np.array([0.01, 0.02, 0.98, 0.99]), 0.5)
    assert m.roc_auc == 1.0
    assert m.recall == 1.0
    assert m.brier < 0.01


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    rng = np.random.default_rng(0)
    p = np.full(10_000, 0.3)
    y = (rng.random(10_000) < 0.3).astype(int)
    assert expected_calibration_error(y, p) < 0.02


def test_threshold_sweep_endpoints_are_correct(dataset, fast_pipeline, cfg):
    """Threshold 0 contacts everyone; threshold above 1 contacts nobody."""
    prob = fast_pipeline.predict_proba(dataset.X_test)[:, 1]
    econ = Economics.from_config(cfg)
    value = econ.customer_value(dataset.X_test)
    y = dataset.y_test.to_numpy()

    assert net_value(y, prob, value, econ, 1.01) == 0.0
    everyone = net_value(y, prob, value, econ, 0.0)
    expected = (value[y == 1] * econ.success_rate).sum() - len(y) * econ.intervention_cost
    assert everyone == pytest.approx(expected)


def test_optimal_threshold_beats_contacting_everyone(dataset, fast_pipeline, cfg):
    prob = fast_pipeline.predict_proba(dataset.X_test)[:, 1]
    econ = Economics.from_config(cfg)
    value = econ.customer_value(dataset.X_test)
    res = optimise(dataset.y_test.to_numpy(), prob, value, econ)
    assert res.best_net_value >= res.baseline_contact_all
    assert res.best_net_value >= res.baseline_contact_none


def test_breakeven_probability_matches_the_ev_definition(cfg):
    """At the break-even probability, expected value is exactly zero."""
    econ = Economics.from_config(cfg)
    value = np.array([600.0, 1200.0])
    be = econ.breakeven_probability(value)
    ev = be * econ.success_rate * value - econ.intervention_cost
    np.testing.assert_allclose(ev, 0.0, atol=1e-9)


def test_higher_value_customers_have_lower_breakeven(cfg):
    """The reason a per-customer rule beats a single global cut."""
    econ = Economics.from_config(cfg)
    be = econ.breakeven_probability(np.array([300.0, 3000.0]))
    assert be[0] > be[1]


# ===========================================================================
# 8. Segmentation and decisions
# ===========================================================================

def test_risk_segments_cover_the_unit_interval(cfg):
    probs = np.array([0.0, 0.05, 0.10, 0.34, 0.35, 0.64, 0.65, 0.99, 1.0])
    seg = assign_risk_segments(probs, cfg)
    assert seg.notna().all()
    assert set(seg) <= {"Low Risk", "Medium Risk", "High Risk", "Critical Risk"}
    assert seg.iloc[0] == "Low Risk"
    assert seg.iloc[-1] == "Critical Risk"


def test_action_matrix_covers_every_combination(cfg):
    from src.evaluation.segments import SEGMENT_ORDER
    tiers = ("High value", "Standard value")
    for band in SEGMENT_ORDER:
        for tier in tiers:
            assert (band, tier) in ACTION_MATRIX


def test_decision_table_is_internally_consistent(dataset, fast_pipeline, cfg):
    prob = fast_pipeline.predict_proba(dataset.X_test)[:, 1]
    table = decide(prob, dataset.X_test, cfg)
    assert len(table) == len(dataset.X_test)
    # contact_recommended must agree with the sign of expected value
    assert (table["contact_recommended"] == (table["expected_value_cu"] > 0)).all()
    # and with the break-even definition
    assert (table["contact_recommended"]
            == (table["churn_probability"] > table["breakeven_probability"])).all()


def test_low_risk_high_value_can_still_be_contacted(cfg):
    """The value dimension must be able to override the risk band.

    A one-dimensional 'contact high risk' rule cannot express this, which is the
    reason the decision layer is two-dimensional.
    """
    econ = Economics.from_config(cfg)
    # value high enough that a modest probability still clears the cost
    huge_value_monthly = econ.intervention_cost / (econ.success_rate * 0.09) / \
        econ.horizon_months
    row = pd.DataFrame([{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "Fiber optic", "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes",
        "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Two year",
        "PaperlessBilling": "No", "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": float(huge_value_monthly) * 1.2, "TotalCharges": 10_000.0,
    }])
    table = decide(np.array([0.09]), row, cfg)
    assert table["risk_segment"].iloc[0] == "Low Risk"
    assert bool(table["contact_recommended"].iloc[0])
    assert bool(table["ev_overrides_segment"].iloc[0])


# ===========================================================================
# 9. Claims that the report depends on (real data only)
# ===========================================================================

@requires_real_data
def test_dataset_shape_and_churn_rate(cfg):
    raw = load_raw(cfg)
    assert raw.shape == (7043, 21)
    assert (raw[TARGET] == "Yes").mean() == pytest.approx(0.2654, abs=0.001)


@requires_real_data
def test_reported_audit_counts(cfg):
    raw = load_raw(cfg)
    assert raw["TotalCharges"].isna().sum() == 11
    assert raw.duplicated().sum() == 0
    _, n_removed, n_conflicting = deduplicate(clean_raw(raw))
    assert n_removed == 16
    assert n_conflicting == 42


@requires_real_data
def test_total_charges_is_redundant_not_leaky(cfg):
    """R^2 ~0.999 against tenure x MonthlyCharges is collinearity, not leakage."""
    from src.data.audit import total_charges_redundancy
    r2 = total_charges_redundancy(clean_raw(load_raw(cfg)))
    assert r2 > 0.99


@requires_real_data
def test_no_single_feature_separates_the_target(cfg):
    """The leakage probe. A feature near AUC 1.0 would be the target in disguise."""
    from src.data.audit import LEAKAGE_AUC_THRESHOLD, single_feature_auc
    cleaned = clean_raw(load_raw(cfg))
    y = (cleaned[TARGET] == cfg.positive_label).astype(int)
    aucs = single_feature_auc(cleaned[feature_columns()], y, cfg)
    strongest = max(aucs.values())
    assert strongest < LEAKAGE_AUC_THRESHOLD
    assert max(aucs, key=aucs.get) == "Contract"


@requires_real_data
def test_churn_is_front_loaded_in_tenure(dataset):
    """The finding that motivates tenure_band and is_new_customer."""
    df = dataset.X_train.copy()
    df["churn"] = dataset.y_train.to_numpy()
    early = df.loc[df["tenure"] <= 6, "churn"].mean()
    late = df.loc[df["tenure"] >= 49, "churn"].mean()
    assert early > 0.45
    assert late < 0.15
    assert early > 3 * late


@requires_real_data
def test_addon_adoption_is_monotonically_protective(dataset):
    """The finding that motivates n_addons; churn falls at every step."""
    from src.data.schema import INTERNET_ADDON_COLUMNS
    df = dataset.X_train.copy()
    df["churn"] = dataset.y_train.to_numpy()
    df = df[df["InternetService"] != "No"]
    df["n"] = sum((df[c] == "Yes").astype(int) for c in INTERNET_ADDON_COLUMNS)
    rates = df.groupby("n")["churn"].mean()
    assert (np.diff(rates.to_numpy()) < 0).all(), f"not monotonic: {rates.to_dict()}"


@requires_real_data
def test_contract_risk_ordering(dataset):
    df = dataset.X_train.copy()
    df["churn"] = dataset.y_train.to_numpy()
    r = df.groupby("Contract")["churn"].mean()
    assert r["Month-to-month"] > r["One year"] > r["Two year"]
    assert r["Month-to-month"] > 0.40
    assert r["Two year"] < 0.05


@requires_real_data
def test_risk_bands_are_validated_against_realised_churn(cfg, dataset):
    """Bands must behave as their labels claim, not merely look tidy."""
    from xgboost import XGBClassifier
    pipe = build_pipeline(
        XGBClassifier(random_state=cfg.seed, n_jobs=1, tree_method="hist",
                      eval_metric="logloss", n_estimators=300, max_depth=4,
                      learning_rate=0.02, subsample=1.0, colsample_bytree=0.6,
                      min_child_weight=1, reg_lambda=0.5),
        scale=False)
    pipe.fit(dataset.X_train, dataset.y_train)
    prob = pipe.predict_proba(dataset.X_test)[:, 1]

    summary = segment_summary(prob, dataset.y_test.to_numpy(), cfg)
    checks = validate_segments(summary)
    assert checks["monotonic_realised_churn"], summary["realised_churn_rate"].to_dict()
    assert checks["max_abs_calibration_gap"] < 0.10
    assert checks["lift_top_vs_bottom"] > 5.0


@requires_real_data
def test_model_beats_the_dummy_baseline_substantially(cfg, dataset):
    """A model that cannot beat the base rate has earned nothing."""
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import average_precision_score

    dummy = build_pipeline(DummyClassifier(strategy="prior"), scale=False)
    dummy.fit(dataset.X_train, dataset.y_train)
    dummy_ap = average_precision_score(
        dataset.y_test, dummy.predict_proba(dataset.X_test)[:, 1])

    lr = build_pipeline(LogisticRegression(max_iter=2000, random_state=cfg.seed),
                        scale=True).fit(dataset.X_train, dataset.y_train)
    lr_ap = average_precision_score(
        dataset.y_test, lr.predict_proba(dataset.X_test)[:, 1])

    assert dummy_ap == pytest.approx(dataset.y_test.mean(), abs=0.02)
    assert lr_ap > dummy_ap + 0.20


@requires_real_data
def test_accuracy_is_a_misleading_headline(dataset):
    """Predicting 'nobody churns' scores ~73.5% accuracy and finds no churners."""
    majority_accuracy = 1 - dataset.y_test.mean()
    assert majority_accuracy > 0.70
