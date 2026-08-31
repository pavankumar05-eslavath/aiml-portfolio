"""Preprocessing pipeline construction.

All preprocessing lives inside an sklearn ``Pipeline``, so it is fitted on
training folds only. Scaling or encoding the full dataset before splitting is the
most common quiet leak in tutorial churn code: the scaler's mean and the
encoder's category list both carry information from the test rows.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.schema import (
    BINARY_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)
from src.features.engineer import (
    ENGINEERED_BINARY,
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    ChurnFeatureEngineer,
)

#: Explicit interaction terms. A tree ensemble discovers interactions by
#: construction, so these exist for the LINEAR model only. Measured on the tuned
#: XGBoost: including tenure_x_month_to_month moved CV PR-AUC from 0.6753 to
#: 0.6736 -- i.e. very slightly worse, paired t p=0.396. It also dominated the
#: SHAP output with an uninterpretable "tenure x contract" term that absorbed
#: credit belonging to tenure. No measurable gain plus a real interpretability
#: cost, so tree pipelines exclude it.
INTERACTION_FEATURES: tuple[str, ...] = ("tenure_x_month_to_month",)


def numeric_features(
    drop_total_charges: bool = False, include_interactions: bool = False,
) -> list[str]:
    """Numeric columns entering the model, raw plus engineered."""
    base = [c for c in NUMERIC_COLUMNS
            if not (drop_total_charges and c == "TotalCharges")]
    engineered = [c for c in ENGINEERED_NUMERIC
                  if include_interactions or c not in INTERACTION_FEATURES]
    return [*base, *BINARY_NUMERIC_COLUMNS, *engineered, *ENGINEERED_BINARY]


def categorical_features() -> list[str]:
    """Categorical columns entering the model, raw plus engineered."""
    return [*CATEGORICAL_COLUMNS, *ENGINEERED_CATEGORICAL]


def build_preprocessor(
    *, scale: bool, drop_total_charges: bool = False,
    include_interactions: bool = False,
) -> ColumnTransformer:
    """Assemble the column-wise preprocessor.

    Args:
        scale: Standardise numeric columns. Required for regularised logistic
            regression, where an unscaled feature is penalised by its units
            rather than its importance. Unnecessary for trees, which are
            invariant to monotone rescaling, so it is off for them to keep the
            persisted pipeline minimal.
        drop_total_charges: Remove ``TotalCharges``, whose R^2 against
            ``tenure x MonthlyCharges`` is 0.9991.

    Note:
        ``handle_unknown='ignore'`` matters at inference: a category absent from
        training (or a typo in an API payload) encodes to all-zeros instead of
        raising, so a single bad field cannot take the service down.
    """
    num_cols = numeric_features(drop_total_charges, include_interactions)
    cat_cols = categorical_features()

    num_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    estimator, *, scale: bool, drop_total_charges: bool = False,
    include_interactions: bool = False,
) -> Pipeline:
    """Full pipeline: feature engineering -> preprocessing -> estimator.

    Feature engineering is the first step so that a caller only ever has to supply
    the 19 raw published columns, at training time and at inference time alike.
    """
    return Pipeline([
        ("features", ChurnFeatureEngineer(drop_total_charges=drop_total_charges)),
        ("prep", build_preprocessor(scale=scale, drop_total_charges=drop_total_charges,
                                    include_interactions=include_interactions)),
        ("model", estimator),
    ])


def output_feature_names(pipeline: Pipeline) -> list[str]:
    """Feature names after preprocessing, aligned to the model's input matrix.

    Needed to label SHAP values, which are computed on the encoded matrix and are
    meaningless without the mapping back to column names.
    """
    prep = pipeline.named_steps["prep"]
    return [str(n) for n in np.asarray(prep.get_feature_names_out())]
