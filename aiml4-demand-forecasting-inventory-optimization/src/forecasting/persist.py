"""Model persistence.

The estimator alone is not enough to reproduce a forecast. The bundle carries
everything needed to score a new day identically months later, and to detect when
that is no longer safe:

* the fitted global model and the quantile models
* the feature list and categorical declarations, so the matrix is rebuilt the same way
* the config, including the inventory assumptions the reorder points depend on
* library versions, because a pickle loading under a different LightGBM is a real
  source of silently different numbers
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn

from src.config import Config

BUNDLE_NAME = "forecaster.joblib"
METADATA_NAME = "model_metadata.json"
BUNDLE_FORMAT_VERSION = "1.0"


@dataclass
class ModelMetadata:
    """Everything about a trained forecaster except the weights."""

    model_name: str
    trained_at: str
    bundle_format_version: str
    seed: int
    grain: list[str]
    target: str
    horizons: list[int]
    train_origin: str
    n_train_rows: int
    n_series: int
    features: list[str]
    categorical_features: list[str]
    quantiles: list[float]
    validation_scores: dict[str, Any]
    test_scores: dict[str, Any]
    inventory_assumptions: dict[str, Any]
    versions: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def current_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "lightgbm": lightgbm.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }


def save_bundle(
    cfg: Config, point_model, quantile_models: dict[float, Any],
    metadata: ModelMetadata, policy: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    """Persist the forecaster and a human-readable metadata sidecar."""
    models_dir = cfg.resolve("paths", "models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "point_model": point_model,
        "quantile_models": quantile_models,
        "metadata": asdict(metadata),
        "config": cfg.raw,
        # The policy table is stored so inference can return a reorder point without
        # re-running the whole validation loop to re-estimate error spread.
        "policy": policy,
    }
    bundle_path = models_dir / BUNDLE_NAME
    joblib.dump(bundle, bundle_path, compress=3)

    meta_path = models_dir / METADATA_NAME
    meta_path.write_text(metadata.to_json(), encoding="utf-8")
    return bundle_path, meta_path


def load_bundle(cfg: Config | None = None, path: str | Path | None = None) -> dict[str, Any]:
    """Load a bundle and warn on version drift."""
    if path is None:
        if cfg is None:
            raise ValueError("provide either cfg or path")
        path = cfg.resolve("paths", "models_dir") / BUNDLE_NAME
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no model bundle at {path}. Train one with `python -m src.run train`.")

    bundle = joblib.load(path)
    for key in ("point_model", "metadata", "format_version"):
        if key not in bundle:
            raise ValueError(f"bundle at {path} is missing '{key}'")

    saved = bundle["metadata"].get("versions", {})
    for lib, now in current_versions().items():
        was = saved.get(lib)
        if was and was != now:
            print(f"WARNING: {lib} {was} at training vs {now} now; "
                  "forecasts may differ.")
    return bundle
