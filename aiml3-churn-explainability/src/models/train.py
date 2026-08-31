"""Model training, tuning and comparison.

Ordering is deliberate: a DummyClassifier and a plain logistic regression are
scored first and reported alongside the tuned models. A gradient-boosted model
that cannot beat regularised logistic regression on this data has not earned its
complexity, and on tabular churn data with ~5.6k rows that is a real possibility
rather than a rhetorical one.

Two choices worth stating:

* **Tuning targets PR-AUC (``average_precision``), not accuracy or ROC-AUC.**
  The positive class is the one we spend money on, and PR-AUC is the metric that
  tracks performance on it under class imbalance.
* **Cross-validation is group-aware.** Folds are split with
  ``StratifiedGroupKFold`` on the feature-vector group id, so the 42 retained
  label-conflicting rows cannot appear in both a training fold and its validation
  fold.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import Config
from src.data.loader import Dataset
from src.features.preprocess import build_pipeline

#: Metrics computed for every candidate during cross-validation.
CV_SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "brier": "neg_brier_score",
}


@dataclass
class CandidateResult:
    """Cross-validated result for one candidate model."""

    name: str
    pipeline: Pipeline
    cv_means: dict[str, float]
    cv_stds: dict[str, float]
    #: Per-fold scores, retained so candidates can be compared with a PAIRED test.
    #: Comparing only the means invites the classic error of reading a 0.004
    #: difference as an improvement when the fold-to-fold spread is 0.025.
    cv_folds: dict[str, np.ndarray] = field(default_factory=dict)
    best_params: dict[str, Any] = field(default_factory=dict)
    fit_seconds: float = 0.0
    tuned: bool = False

    def row(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "tuned": self.tuned,
            "pr_auc": self.cv_means["pr_auc"],
            "pr_auc_std": self.cv_stds["pr_auc"],
            "roc_auc": self.cv_means["roc_auc"],
            "f1": self.cv_means["f1"],
            "recall": self.cv_means["recall"],
            "precision": self.cv_means["precision"],
            "brier": self.cv_means["brier"],
            "fit_seconds": self.fit_seconds,
        }


def make_cv(cfg: Config) -> StratifiedGroupKFold:
    """Group-aware stratified CV splitter."""
    return StratifiedGroupKFold(
        n_splits=int(cfg["split"]["cv_folds"]), shuffle=True, random_state=cfg.seed)


def baseline_candidates(cfg: Config) -> dict[str, tuple[Pipeline, dict[str, list]]]:
    """Untuned baselines. Establish the floor before anything is optimised."""
    seed = cfg.seed
    return {
        # 'prior' always predicts the majority class with the base-rate probability:
        # the honest definition of "no model".
        "dummy_prior": (
            build_pipeline(DummyClassifier(strategy="prior"), scale=False), {}),
        "logistic_regression_plain": (
            build_pipeline(
                LogisticRegression(max_iter=2000, random_state=seed), scale=True), {}),
    }


def tuned_candidates(cfg: Config) -> dict[str, tuple[Pipeline, dict[str, list]]]:
    """Candidates with search spaces, read from the config."""
    seed = cfg.seed
    m = cfg["models"]

    lr_space = {f"model__{k}": v for k, v in m["logistic_regression"].items()}
    rf_space = {f"model__{k}": v for k, v in m["random_forest"].items()}
    xgb_space = {f"model__{k}": v for k, v in m["xgboost"].items()}

    return {
        "logistic_regression": (
            # drop_total_charges: R^2 0.9991 against tenure x MonthlyCharges makes the
            # raw column actively harmful to coefficient stability in a linear model.
            # billing_discrepancy retains its residual information.
            # include_interactions: a linear model cannot discover an interaction,
            # so it is given the tenure x contract term explicitly. Trees get it
            # for free and measurably do not benefit, so they omit it.
            build_pipeline(LogisticRegression(max_iter=2000, random_state=seed),
                           scale=True, drop_total_charges=True,
                           include_interactions=True),
            lr_space,
        ),
        "random_forest": (
            build_pipeline(RandomForestClassifier(random_state=seed, n_jobs=1),
                           scale=False),
            rf_space,
        ),
        "xgboost": (
            build_pipeline(
                XGBClassifier(
                    random_state=seed, n_jobs=1, tree_method="hist",
                    eval_metric="logloss",
                ),
                scale=False,
            ),
            xgb_space,
        ),
    }


def _cv_evaluate(
    pipeline: Pipeline, ds: Dataset, cfg: Config, name: str, *,
    best_params: dict[str, Any] | None = None, tuned: bool = False,
    fit_seconds: float = 0.0,
) -> CandidateResult:
    """Cross-validate a fitted-configuration pipeline on the training split."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_validate(
            pipeline, ds.X_train, ds.y_train,
            groups=ds.groups_train, cv=make_cv(cfg),
            scoring=CV_SCORING, n_jobs=-1, error_score="raise",
        )
    folds = {k: np.asarray(scores[f"test_{k}"], dtype=float) for k in CV_SCORING}
    folds["brier"] = -folds["brier"]  # undo sklearn's neg_ convention
    means = {k: float(v.mean()) for k, v in folds.items()}
    stds = {k: float(v.std()) for k, v in folds.items()}
    return CandidateResult(
        name=name, pipeline=pipeline, cv_means=means, cv_stds=stds, cv_folds=folds,
        best_params=best_params or {}, fit_seconds=fit_seconds, tuned=tuned,
    )


def tune(
    name: str, pipeline: Pipeline, space: dict[str, list], ds: Dataset, cfg: Config,
) -> CandidateResult:
    """Randomised search over ``space``, then cross-validate the winner."""
    if not space:
        t0 = time.perf_counter()
        return _cv_evaluate(pipeline, ds, cfg, name,
                            fit_seconds=time.perf_counter() - t0)

    search = RandomizedSearchCV(
        pipeline, space,
        n_iter=int(cfg["models"]["n_iter"]),
        scoring=str(cfg["models"]["scoring"]),
        cv=make_cv(cfg),
        random_state=cfg.seed,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(ds.X_train, ds.y_train, groups=ds.groups_train)
    elapsed = time.perf_counter() - t0

    best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return _cv_evaluate(search.best_estimator_, ds, cfg, name,
                        best_params=best, tuned=True, fit_seconds=elapsed)


def train_all(ds: Dataset, cfg: Config) -> dict[str, CandidateResult]:
    """Score baselines, then tune the advanced models. Returns every result."""
    results: dict[str, CandidateResult] = {}

    print("--- baselines (untuned) ---")
    for name, (pipe, space) in baseline_candidates(cfg).items():
        res = tune(name, pipe, space, ds, cfg)
        results[name] = res
        print(f"  {name:28s} PR-AUC {res.cv_means['pr_auc']:.4f}  "
              f"ROC-AUC {res.cv_means['roc_auc']:.4f}  recall {res.cv_means['recall']:.4f}")

    print(f"\n--- tuned models (RandomizedSearchCV, n_iter="
          f"{cfg['models']['n_iter']}, scoring={cfg['models']['scoring']}) ---")
    for name, (pipe, space) in tuned_candidates(cfg).items():
        res = tune(name, pipe, space, ds, cfg)
        results[name] = res
        print(f"  {name:28s} PR-AUC {res.cv_means['pr_auc']:.4f} "
              f"(+/-{res.cv_stds['pr_auc']:.4f})  ROC-AUC {res.cv_means['roc_auc']:.4f}  "
              f"[{res.fit_seconds:.0f}s]")
        if res.best_params:
            print(f"      best: {res.best_params}")

    return results


def comparison_table(results: dict[str, CandidateResult]) -> pd.DataFrame:
    """Model comparison sorted by cross-validated PR-AUC."""
    df = pd.DataFrame([r.row() for r in results.values()])
    return df.sort_values("pr_auc", ascending=False).reset_index(drop=True)


def select_best(results: dict[str, CandidateResult]) -> CandidateResult:
    """Pick the highest cross-validated PR-AUC among non-dummy candidates."""
    real = {k: v for k, v in results.items() if not k.startswith("dummy")}
    return max(real.values(), key=lambda r: r.cv_means["pr_auc"])


def paired_comparison(
    a: CandidateResult, b: CandidateResult, metric: str = "pr_auc",
) -> dict[str, float | bool | str]:
    """Paired comparison of two candidates across identical CV folds.

    Both candidates are scored on the same folds with the same seed, so the fold
    scores are paired and a paired test is the correct instrument. A Wilcoxon
    signed-rank test is used alongside the paired t-test because five folds is far
    too few to assume normal differences.

    Returns the difference, both p-values, and a verdict. With five folds the test
    has very low power, so a non-significant result means "this data cannot
    distinguish them", not "they are proven equal" -- which is itself the useful
    conclusion when the simpler model is the one at risk of being discarded.
    """
    from scipy import stats

    x, y = a.cv_folds[metric], b.cv_folds[metric]
    diff = float(x.mean() - y.mean())

    # Both tests are undefined when every paired difference is identical (or all
    # zero); scipy emits a divide warning and returns nan rather than raising, so
    # the degenerate case is handled before calling them.
    differences = x - y
    degenerate = bool(np.allclose(differences, 0.0)) or float(np.ptp(differences)) == 0.0
    if degenerate:
        t_p = w_p = 1.0
    else:
        t_p = float(stats.ttest_rel(x, y).pvalue)
        try:
            w_p = float(stats.wilcoxon(x, y).pvalue)
        except ValueError:
            w_p = 1.0
    t_p = 1.0 if np.isnan(t_p) else t_p
    w_p = 1.0 if np.isnan(w_p) else w_p

    pooled_sd = float(np.std(np.concatenate([x, y])))
    return {
        "metric": metric,
        "model_a": a.name,
        "model_b": b.name,
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "difference": diff,
        "difference_in_sd": diff / pooled_sd if pooled_sd else 0.0,
        "ttest_p": t_p,
        "wilcoxon_p": w_p,
        "significant_at_05": bool(min(t_p, w_p) < 0.05),
    }
