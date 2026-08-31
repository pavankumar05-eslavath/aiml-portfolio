"""Model persistence.

Saving the estimator alone is not enough to reproduce a prediction. The bundle
carries everything needed to score a new customer identically months later, and
to detect when that is no longer safe:

* the fitted pipeline (feature engineering + preprocessing + model), so inference
  cannot re-implement training's transformations and drift from them
* the operating threshold and the economic assumptions it was derived from -- a
  threshold without its assumptions is a magic number
* the input contract (required raw columns) so a malformed payload fails loudly
* library versions, because a pickle silently loading under a different sklearn or
  xgboost is a real source of wrong answers
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.pipeline import Pipeline

from src.config import Config
from src.data.schema import feature_columns

BUNDLE_NAME = "churn_model.joblib"
METADATA_NAME = "model_metadata.json"
#: Bumped when the bundle layout changes in a way old loaders cannot read.
BUNDLE_FORMAT_VERSION = "1.0"


@dataclass
class ModelMetadata:
    """Everything about a trained model except the weights."""

    model_name: str
    trained_at: str
    bundle_format_version: str
    seed: int
    threshold: float
    threshold_rationale: str
    calibration_method: str
    required_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    engineered_features: list[str]
    risk_segments: dict[str, list[float]]
    economics: dict[str, float]
    cv_metrics: dict[str, float]
    test_metrics: dict[str, float]
    n_train: int
    n_test: int
    versions: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def current_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }


def save_model(
    model: Any, metadata: ModelMetadata, cfg: Config,
    explainer_pipeline: Pipeline | None = None,
) -> tuple[Path, Path]:
    """Persist the bundle and a human-readable metadata sidecar.

    ``explainer_pipeline`` is the uncalibrated pipeline SHAP runs on. It is stored
    alongside the scoring model so explanations remain available at inference
    without refitting.
    """
    models_dir = cfg.resolve("paths", "models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "model": model,
        "explainer_pipeline": explainer_pipeline,
        "metadata": asdict(metadata),
        "config": cfg.raw,
    }
    bundle_path = models_dir / BUNDLE_NAME
    joblib.dump(bundle, bundle_path, compress=3)

    meta_path = models_dir / METADATA_NAME
    meta_path.write_text(metadata.to_json(), encoding="utf-8")
    return bundle_path, meta_path


def load_model(cfg: Config | None = None, path: str | Path | None = None) -> dict[str, Any]:
    """Load a bundle and verify it is usable.

    Version mismatches warn rather than raise: refusing to load is worse than
    scoring with a loud warning when a model is needed in production. A missing or
    malformed bundle does raise.
    """
    if path is None:
        if cfg is None:
            raise ValueError("provide either cfg or path")
        path = cfg.resolve("paths", "models_dir") / BUNDLE_NAME
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no model bundle at {path}. Train one with `python -m src.run train`.")

    bundle = joblib.load(path)
    for key in ("model", "metadata", "format_version"):
        if key not in bundle:
            raise ValueError(f"bundle at {path} is missing '{key}'")

    saved = bundle["metadata"].get("versions", {})
    for lib, current in current_versions().items():
        was = saved.get(lib)
        if was and was != current:
            print(f"WARNING: {lib} {was} at training vs {current} now; "
                  "predictions may differ.")
    return bundle


def validate_input(df: pd.DataFrame, metadata: dict[str, Any]) -> None:
    """Check an inference payload against the stored input contract."""
    required = list(metadata["required_columns"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if df.empty:
        raise ValueError("no rows to score")


def build_metadata(
    *, model_name: str, cfg: Config, threshold: float, threshold_rationale: str,
    calibration_method: str, cv_metrics: dict[str, float],
    test_metrics: dict[str, float], n_train: int, n_test: int,
    notes: list[str] | None = None,
) -> ModelMetadata:
    """Assemble metadata from a completed training run."""
    from src.features.engineer import FEATURE_DOCS
    from src.features.preprocess import categorical_features, numeric_features

    return ModelMetadata(
        model_name=model_name,
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        bundle_format_version=BUNDLE_FORMAT_VERSION,
        seed=cfg.seed,
        threshold=float(threshold),
        threshold_rationale=threshold_rationale,
        calibration_method=calibration_method,
        required_columns=feature_columns(),
        numeric_columns=numeric_features(),
        categorical_columns=categorical_features(),
        engineered_features=list(FEATURE_DOCS),
        risk_segments={k: list(v) for k, v in cfg["risk_segments"].items()},
        economics={k: v for k, v in cfg["economics"].items()
                   if isinstance(v, (int, float))},
        cv_metrics=cv_metrics,
        test_metrics=test_metrics,
        n_train=n_train,
        n_test=n_test,
        versions=current_versions(),
        notes=notes or [],
    )
