"""
Stage 3 -- why the article's XGBoost number no longer reproduces.

On the real CSV the article reports 0.9992 test ROC AUC for `XGBClassifier()`.
Today the same call returns ~0.712. This isolates the cause.

XGBoost 2.0 changed `base_score` from a fixed 0.5 to an estimate of the class
prior. At a 0.13% prior every prediction starts at a raw margin near -6.6, and
boosting drives probabilities into the floating-point floor, destroying the
ranking. Forcing `base_score=0.5` restores the published figure.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.article_pipeline import build_xy
from src.dataset import load

warnings.filterwarnings("ignore")


def used_base_score(model: XGBClassifier) -> float:
    cfg = json.loads(model.get_booster().save_config())
    raw = cfg["learner"]["learner_model_param"]["base_score"]
    return float(str(raw).strip("[]"))


def run(path: str | None = None) -> dict:
    df = load(path)
    X, y = build_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    prior = float(y_train.mean())
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"training-set fraud prior : {prior:.8f}")
    print(f"scale_pos_weight would be: {spw:.1f}")

    print("\n--- base_score: auto (>=2.0) versus the old fixed 0.5 ---")
    out = {}
    for label, kw in [("auto", {}), ("forced 0.5", {"base_score": 0.5})]:
        m = XGBClassifier(**kw).fit(X_train, y_train)
        p = m.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, p)
        buried = int((p[y_test.to_numpy() == 1] < 1e-4).sum())
        print(
            f"  {label:11s} used={used_base_score(m):.8f}  ROC AUC={auc:.6f}  "
            f"min_p={p.min():.2e}  frauds ranked below 1e-4: {buried}/{int(y_test.sum())}"
        )
        out[label] = auc

    # Saturation, not learning: a model that is genuinely improving does not lose
    # 0.3 AUC between round 5 and round 50 and then win it back by round 300.
    print("\n--- ROC AUC versus boosting rounds (non-monotonic => saturation) ---")
    m = XGBClassifier(n_estimators=300).fit(X_train, y_train)
    curve = {}
    for r in (1, 5, 10, 25, 50, 100, 200, 300):
        p = m.predict_proba(X_test, iteration_range=(0, r))[:, 1]
        curve[r] = roc_auc_score(y_test, p)
        print(f"  rounds={r:>3}  ROC AUC={curve[r]:.6f}")
    out["curve"] = curve
    out["non_monotonic"] = bool(min(curve.values()) < curve[max(curve)] - 0.02)

    print("\n--- the two fixes, and what the article's metric hides ---")
    for label, kw in [
        ("default", {}),
        ("scale_pos_weight", {"scale_pos_weight": spw}),
        ("base_score=0.5", {"base_score": 0.5}),
    ]:
        m = XGBClassifier(**kw).fit(X_train, y_train)
        p = m.predict_proba(X_test)[:, 1]
        print(
            f"  {label:17s} ROC AUC={roc_auc_score(y_test, p):.6f}  "
            f"PR AUC={average_precision_score(y_test, p):.6f}"
        )
    print(
        "\nROC AUC near 0.71 with PR AUC near 0.32 is the tell: at a 0.13% base\n"
        "rate ROC AUC is dominated by easy negatives and reads far kinder than\n"
        "the model deserves."
    )
    return out


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    run()
