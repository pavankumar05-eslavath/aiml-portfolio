"""
Guards on the claims this project makes.

The important ones are the leakage guards. If a future change makes the
three-clause rule stop working, or lets raw `step` back into the feature matrix,
or quietly reverts to a random split, the conclusions in INSIGHTS.md no longer
hold and these tests should fail.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data.generate_paysim import generate
from src.dataset import SCHEMA, add_features, design_matrix, leak_rule, temporal_split
from src.xgb_diagnosis import used_base_score


@pytest.fixture(scope="module")
def df():
    return generate(n_rows=200_000, seed=7)


# ------------------------------------------------------------------ schema
def test_schema_matches_paysim(df):
    assert list(df.columns) == SCHEMA


def test_no_nulls(df):
    assert df.isna().sum().sum() == 0


def test_balances_never_negative(df):
    for c in ("oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "amount"):
        assert (df[c] >= 0).all(), c


# ------------------------------------------- the article's claim is false
def test_classes_are_severely_imbalanced(df):
    """The article states the classes are already balanced and no sampling is
    needed. They are not: the real file is 1:774."""
    ratio = (df.isFraud == 0).sum() / df.isFraud.sum()
    assert ratio > 100, f"expected severe imbalance, got 1:{ratio:.0f}"


def test_accuracy_is_a_useless_metric_here(df):
    """A model that always predicts 'legit' scores >99% accuracy and catches
    nothing, which is why ROC AUC must not be labelled 'Accuracy'."""
    assert (1 - df.isFraud.mean()) > 0.99


def test_pr_auc_of_a_constant_predictor_equals_base_rate(df):
    y = df.isFraud.to_numpy()
    ap = average_precision_score(y, np.ones(len(y)))
    assert ap == pytest.approx(y.mean(), rel=1e-6)


# ------------------------------------------------------- fraud behaviour
def test_fraud_only_in_transfer_and_cash_out(df):
    assert set(df.loc[df.isFraud == 1, "type"].unique()) <= {"TRANSFER", "CASH_OUT"}


def test_fraud_drains_the_sender(df):
    f = df[df.isFraud == 1]
    assert np.allclose(f.amount, f.oldbalanceOrg)
    assert (f.newbalanceOrig == 0).all()


def test_sender_residual_reconciles_for_fraud(df):
    """errBalOrig = newbalanceOrig + amount - oldbalanceOrg is ~0 on the
    fraudulent leg because the agent takes exactly the balance."""
    f = add_features(df).query("isFraud == 1")
    assert np.allclose(f.errBalOrig, 0, atol=0.01)


# --------------------------------------------------- THE LEAKAGE GUARDS
def test_hand_written_rule_beats_needing_a_model(df):
    """The headline finding: three clauses, no ML, near-perfect precision.

    On the real 6.36M-row CSV this is precision 0.9999 / recall 0.9750 with a
    single false positive. If this ever stops holding, the project's conclusion
    has changed.
    """
    y = df.isFraud.to_numpy().astype(bool)
    rule = leak_rule(df)
    tp = (rule & y).sum()
    fp = (rule & ~y).sum()
    fn = (~rule & y).sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    assert precision > 0.90, f"rule precision collapsed to {precision:.4f}"
    assert recall > 0.90, f"rule recall collapsed to {recall:.4f}"


def test_a_single_boolean_feature_is_nearly_sufficient(df):
    """`amount == oldbalanceOrg` alone scores ROC-AUC 0.9881 on the real data.
    That is the leak, stated as compactly as possible."""
    y = df.isFraud.to_numpy()
    exact = (np.isclose(df.amount, df.oldbalanceOrg) & (df.oldbalanceOrg > 0)).astype(float)
    assert roc_auc_score(y, exact) > 0.90


# ------------------------------------------------------ split integrity
def test_temporal_split_is_disjoint_and_ordered(df):
    sp = temporal_split(df)
    step = df.step.to_numpy()
    tr, va, te = sp["train"], sp["val"], sp["test"]

    assert not (tr & va).any() and not (va & te).any() and not (tr & te).any()
    assert (tr | va | te).all()
    # No test transaction may precede a training transaction.
    assert step[tr].max() < step[va].min()
    assert step[va].max() < step[te].min()


def test_every_split_contains_fraud(df):
    sp = temporal_split(df)
    y = df.isFraud.to_numpy()
    for name in ("train", "val", "test"):
        assert y[sp[name]].sum() > 0, f"{name} split has no positives"


# ------------------------------------------------- feature-matrix hygiene
def test_raw_step_is_excluded(df):
    """Under a temporal split every test `step` exceeds every training `step`,
    so trees can only extrapolate from it. Only hour-of-day survives."""
    X, _ = design_matrix(df)
    assert "step" not in X.columns
    assert "hour" in X.columns


def test_identifiers_are_excluded(df):
    X, _ = design_matrix(df)
    for c in ("nameOrig", "nameDest"):
        assert c not in X.columns


def test_design_matrix_is_finite_and_numeric(df):
    X, y = design_matrix(df)
    assert len(X) == len(y)
    assert np.isfinite(X.to_numpy(dtype=float)).all()
    assert set(np.unique(y)) <= {0, 1}


# ---------------------------------- the XGBoost reproducibility finding
def test_base_score_defaults_to_the_class_prior(df):
    """The mechanism behind the article's unreproducible 0.9992.

    XGBoost >=2.0 estimates base_score from the data instead of using 0.5. At a
    ~0.13% prior that starts every prediction deep in the negative margin.
    """
    X, y = design_matrix(df)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.3, random_state=42)

    auto = XGBClassifier(n_estimators=10).fit(X_tr, y_tr)
    assert used_base_score(auto) == pytest.approx(y_tr.mean(), rel=0.05)

    forced = XGBClassifier(n_estimators=10, base_score=0.5).fit(X_tr, y_tr)
    assert used_base_score(forced) == pytest.approx(0.5)


def test_scale_pos_weight_does_not_hurt_ranking(df):
    """The fix the article omits, because it believes the data is balanced."""
    X, y = design_matrix(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    default = XGBClassifier(n_estimators=50).fit(X_tr, y_tr)
    weighted = XGBClassifier(n_estimators=50, scale_pos_weight=spw).fit(X_tr, y_tr)

    ap_d = average_precision_score(y_te, default.predict_proba(X_te)[:, 1])
    ap_w = average_precision_score(y_te, weighted.predict_proba(X_te)[:, 1])
    assert ap_w > ap_d - 0.05


def test_base_score_is_reported_as_json_parseable(df):
    X, y = design_matrix(df)
    m = XGBClassifier(n_estimators=5).fit(X, y)
    json.loads(m.get_booster().save_config())
