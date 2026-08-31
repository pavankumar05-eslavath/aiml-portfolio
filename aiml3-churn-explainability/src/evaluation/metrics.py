"""Evaluation metrics and calibration.

Why accuracy is not reported as a headline: the base rate is 26.5%, so predicting
"nobody churns" scores 73.5% accuracy while identifying zero churners and
supporting zero retention actions. Accuracy also depends on an arbitrary 0.5
cutoff, which is not the operating point any retention campaign would choose.

The metrics that matter here split into two groups:

* **Ranking quality** (ROC-AUC, PR-AUC) -- threshold-free, answers "can the model
  order customers by risk?". PR-AUC is primary because the positive class is the
  one being acted on.
* **Probability quality** (Brier score, calibration curve) -- answers "is a
  predicted 0.7 actually a 70% chance?". This is separate from ranking: a model
  can rank perfectly and still be badly calibrated, and the expected-value
  arithmetic in the decision layer depends on the probabilities being real.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from src.config import Config
from src.data.loader import Dataset


@dataclass(frozen=True)
class MetricSet:
    """Metrics for one model at one threshold."""

    threshold: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    brier: float
    log_loss: float
    tn: int
    fp: int
    fn: int
    tp: int

    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom else 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray, threshold: float = 0.5,
) -> MetricSet:
    """Compute the full metric set at a given decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return MetricSet(
        threshold=float(threshold),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        accuracy=float((y_pred == y_true).mean()),
        brier=float(brier_score_loss(y_true, y_prob)),
        log_loss=float(log_loss(y_true, np.clip(y_prob, 1e-9, 1 - 1e-9))),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


@dataclass
class CalibrationResult:
    """Outcome of comparing calibration methods."""

    method: str
    brier_before: float
    brier_after: float
    log_loss_before: float
    log_loss_after: float
    ece_before: float
    ece_after: float
    pr_auc_before: float
    pr_auc_after: float
    model: Pipeline

    @property
    def improved(self) -> bool:
        return self.brier_after < self.brier_before


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10,
) -> float:
    """Expected calibration error: mean |confidence - accuracy| weighted by bin size.

    Reported alongside the Brier score because Brier mixes calibration and
    discrimination into one number, so it can improve for the wrong reason. ECE
    isolates the calibration component.
    """
    y_true = np.asarray(y_true).astype(int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y_prob[m].mean() - y_true[m].mean())
    return float(total)


def calibrate(
    pipeline: Pipeline, ds: Dataset, cfg: Config,
) -> tuple[CalibrationResult, list[CalibrationResult]]:
    """Fit and compare calibration methods; return the best plus all candidates.

    The calibrator is fitted with internal cross-validation on the TRAINING data
    only (``CalibratedClassifierCV`` with ``cv=StratifiedGroupKFold``), then judged
    on the untouched test set. Fitting a calibrator on the test set would
    guarantee a flattering reliability curve and a model that is miscalibrated in
    production.

    Why this matters commercially: the decision layer multiplies probability by
    customer value. If the model says 0.8 for a group that churns 40% of the time,
    every expected-value calculation downstream is inflated twofold, and the
    campaign spends real money against an imagined return.
    """
    y_test = ds.y_test.to_numpy()
    base_prob = pipeline.predict_proba(ds.X_test)[:, 1]
    brier_before = float(brier_score_loss(y_test, base_prob))
    ll_before = float(log_loss(y_test, np.clip(base_prob, 1e-9, 1 - 1e-9)))
    ece_before = expected_calibration_error(y_test, base_prob)
    ap_before = float(average_precision_score(y_test, base_prob))

    cv = StratifiedGroupKFold(n_splits=int(cfg["calibration"]["cv_folds"]),
                              shuffle=True, random_state=cfg.seed)
    splits = list(cv.split(ds.X_train, ds.y_train, groups=ds.groups_train))

    # The uncalibrated model competes as a candidate in its own right. A gradient
    # booster trained on logloss is often already well calibrated, and wrapping it
    # in a calibrator that does not measurably help adds a component to maintain,
    # an extra 5-fold fit, and a layer between SHAP and the served probability.
    candidates: list[CalibrationResult] = [CalibrationResult(
        method="none", brier_before=brier_before, brier_after=brier_before,
        log_loss_before=ll_before, log_loss_after=ll_before,
        ece_before=ece_before, ece_after=ece_before,
        pr_auc_before=ap_before, pr_auc_after=ap_before, model=pipeline,
    )]

    for method in cfg["calibration"]["methods"]:
        calibrated = CalibratedClassifierCV(
            clone(pipeline), method=method, cv=splits, ensemble=True,
        ).fit(ds.X_train, ds.y_train)

        prob = calibrated.predict_proba(ds.X_test)[:, 1]
        candidates.append(CalibrationResult(
            method=method,
            brier_before=brier_before, brier_after=float(brier_score_loss(y_test, prob)),
            log_loss_before=ll_before,
            log_loss_after=float(log_loss(y_test, np.clip(prob, 1e-9, 1 - 1e-9))),
            ece_before=ece_before, ece_after=expected_calibration_error(y_test, prob),
            pr_auc_before=ap_before,
            pr_auc_after=float(average_precision_score(y_test, prob)),
            model=calibrated,
        ))

    # Parsimony rule, declared in config BEFORE seeing the numbers: adopt a
    # calibrator only if it improves the Brier score by at least min_brier_gain.
    # Without such a rule, "pick the lowest Brier" ships a calibrator for a
    # 0.0002 improvement, which is indistinguishable from noise.
    min_gain = float(cfg["calibration"].get("min_brier_gain", 0.0))
    uncal = candidates[0]
    fitted = [c for c in candidates if c.method != "none"]
    challenger = min(fitted, key=lambda c: c.brier_after) if fitted else uncal
    best = challenger if (uncal.brier_after - challenger.brier_after) >= min_gain else uncal
    return best, candidates


def split_stability(
    pipeline: Pipeline, cfg: Config, n_splits: int = 5, path: str | None = None,
) -> pd.DataFrame:
    """Score the model with each CV fold used in turn as the held-out test set.

    A single 80/20 split of ~7k rows is a noisy estimate, and reporting one number
    from one split invites two errors: mistaking split luck for model quality, and
    quietly re-drawing the split until the number improves (test-set shopping).

    This returns the full distribution so the designated test score can be placed
    within it. The designated split is fold 0 and is NOT re-chosen on the basis of
    its score.
    """
    from src.data.loader import (
        clean_raw,
        deduplicate,
        feature_group_ids,
        load_raw,
    )
    from src.data.schema import TARGET, feature_columns

    df, _, _ = deduplicate(clean_raw(load_raw(cfg, path)))
    X = df.loc[:, feature_columns()]
    y = (df[TARGET] == cfg.positive_label).astype(int)
    groups = feature_group_ids(X)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=cfg.seed)
    rows = []
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups=groups)):
        est = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
        prob = est.predict_proba(X.iloc[te])[:, 1]
        rows.append({
            "fold": fold,
            "is_designated_test": fold == 0,
            "n": len(te),
            "churn_rate": float(y.iloc[te].mean()),
            "pr_auc": float(average_precision_score(y.iloc[te], prob)),
            "roc_auc": float(roc_auc_score(y.iloc[te], prob)),
            "brier": float(brier_score_loss(y.iloc[te], prob)),
        })
    return pd.DataFrame(rows)


def reliability_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Points for a calibration curve."""
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins,
                                            strategy="uniform")
    return mean_pred, frac_pos


def metrics_frame(rows: dict[str, MetricSet]) -> pd.DataFrame:
    """Tabulate several metric sets for side-by-side reporting."""
    return pd.DataFrame({k: v.as_dict() for k, v in rows.items()}).T
