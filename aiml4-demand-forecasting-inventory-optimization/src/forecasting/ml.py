"""Global machine-learning forecasters.

**One model across all series**, with series identity and attributes as features,
rather than one model per SKU. Three reasons:

* 600 series x 4 folds is 2,400 model fits for a per-series approach, and the
  config explicitly forbids blindly fitting expensive models on every SKU.
* Most series here are intermittent, with few non-zero observations each. A global
  model pools that sparse signal; a per-series model sees almost nothing.
* New products have no history at all. A global model can forecast them from their
  category, store and price; a per-series model cannot exist for them.

**Objective: Tweedie.** Demand is a non-negative, zero-inflated count. Squared
error treats it as symmetric and unbounded, which pushes predictions negative and
under-weights the multiplicative nature of demand. Tweedie with power ~1.1 sits
between Poisson and Gamma and handles the zero mass directly, which is why it was
the common choice among strong M5 entries.

**Uncertainty via quantile regression.** Separate models for P10/P50/P90 rather
than a normal interval around a point forecast: demand is skewed and bounded below
at zero, so a symmetric interval puts probability mass on impossible values and
understates the upper tail that safety stock has to cover.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

from src.config import Config
from src.features.build import CATEGORICAL_FEATURES, feature_columns


@dataclass
class FittedModel:
    """A fitted global model plus what is needed to reproduce its inputs."""

    name: str
    model: Any
    features: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    quantile: float | None = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict, clipped at zero because demand cannot be negative."""
        Xf = prepare_matrix(X, self.features, self.categorical)
        return np.clip(np.asarray(self.model.predict(Xf), dtype="float64"), 0.0, None)


def prepare_matrix(
    df: pd.DataFrame, features: list[str], categorical: list[str],
) -> pd.DataFrame:
    """Select and type the model matrix.

    Categoricals are kept as pandas ``category`` so LightGBM splits on them
    natively instead of imposing a false ordering, and unseen levels at inference
    become NaN rather than raising.
    """
    X = df.loc[:, features].copy()
    for c in categorical:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def _numeric_matrix(df: pd.DataFrame, features: list[str],
                    categorical: list[str]) -> pd.DataFrame:
    """Integer-coded matrix for estimators without native categorical support."""
    X = df.loc[:, features].copy()
    for c in categorical:
        if c in X.columns:
            X[c] = X[c].astype("category").cat.codes.astype("int32")
    return X.fillna(-999.0)


def fit_lightgbm(
    cfg: Config, train: pd.DataFrame, quantile: float | None = None,
    name: str | None = None,
) -> FittedModel:
    """Fit the global LightGBM model, optionally as a quantile regressor."""
    params = dict(cfg["models"]["lightgbm"])
    feats = feature_columns(train)
    cats = [c for c in CATEGORICAL_FEATURES if c in feats]

    if quantile is not None:
        # Quantile loss replaces Tweedie: the two objectives answer different
        # questions and cannot be combined in one model.
        params.pop("tweedie_variance_power", None)
        params["objective"] = "quantile"
        params["alpha"] = float(quantile)

    model = LGBMRegressor(random_state=cfg.seed, verbose=-1, **params)
    X = prepare_matrix(train, feats, cats)
    model.fit(X, train["y_true"].to_numpy(dtype="float64"),
              categorical_feature=cats or "auto")
    label = name or (f"lightgbm_q{int(quantile * 100)}" if quantile is not None
                     else "lightgbm")
    return FittedModel(name=label, model=model, features=feats, categorical=cats,
                       quantile=quantile)


def fit_random_forest(cfg: Config, train: pd.DataFrame) -> FittedModel:
    """Fit a random forest as a second, structurally different ML comparator."""
    params = dict(cfg["models"]["random_forest"])
    feats = feature_columns(train)
    cats = [c for c in CATEGORICAL_FEATURES if c in feats]
    model = RandomForestRegressor(random_state=cfg.seed, **params)
    X = _numeric_matrix(train, feats, cats)
    model.fit(X, train["y_true"].to_numpy(dtype="float64"))

    fitted = FittedModel(name="random_forest", model=model, features=feats,
                         categorical=cats)
    # Wrap prediction so the integer coding used at fit time is reused.
    def _predict(X_new: pd.DataFrame, _m=model, _f=feats, _c=cats) -> np.ndarray:
        return np.clip(_m.predict(_numeric_matrix(X_new, _f, _c)), 0.0, None)
    fitted.predict = _predict  # type: ignore[method-assign]
    return fitted


def feature_importance(fitted: FittedModel, top: int = 25) -> pd.DataFrame:
    """Gain-based importance, for sanity-checking what the model actually uses."""
    m = fitted.model
    if hasattr(m, "booster_"):
        imp = m.booster_.feature_importance(importance_type="gain")
    elif hasattr(m, "feature_importances_"):
        imp = m.feature_importances_
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return (pd.DataFrame({"feature": fitted.features, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(top).reset_index(drop=True))


def fit_quantiles(
    cfg: Config, train: pd.DataFrame, quantiles: list[float] | None = None,
) -> dict[float, FittedModel]:
    """Fit one model per requested quantile."""
    qs = quantiles or [float(q) for q in cfg["models"]["quantiles"]]
    return {q: fit_lightgbm(cfg, train, quantile=q) for q in qs}


def enforce_monotone_quantiles(preds: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """Sort quantile predictions so P10 <= P50 <= P90.

    Independently fitted quantile models can cross, which produces a negative
    interval width and a nonsensical safety stock. Sorting is the standard,
    distribution-free repair and cannot make calibration worse.
    """
    qs = sorted(preds)
    stacked = np.vstack([preds[q] for q in qs])
    stacked = np.sort(stacked, axis=0)
    return {q: stacked[i] for i, q in enumerate(qs)}
