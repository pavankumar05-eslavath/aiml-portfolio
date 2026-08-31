"""Shared fixtures.

Fixtures build their own model rather than depending on a bundle left behind by a
previous run, so the suite is independent of stage order and of whether anyone has
run training. A fast logistic-regression pipeline is used where the test is about
the *contract* (probability range, required columns, persistence) rather than about
model quality.
"""
from __future__ import annotations

import subprocess
import sys

import pytest
from sklearn.linear_model import LogisticRegression

from src.config import load_config
from src.data.loader import is_real_dataset, make_dataset, resolve_data_path
from src.features.preprocess import build_pipeline

#: Skip marker for assertions that only hold on the full published dataset.
requires_real_data = pytest.mark.skipif(
    not is_real_dataset(load_config()),
    reason="needs the real 7,043-row Telco dataset (`make download`)",
)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session", autouse=True)
def _ensure_data(cfg):
    """Generate the synthetic stand-in when no dataset is present."""
    if not resolve_data_path(cfg).exists():
        subprocess.run([sys.executable, "-m", "src.data.generate_sample"], check=True)


@pytest.fixture(scope="session")
def dataset(cfg):
    return make_dataset(cfg)


@pytest.fixture(scope="session")
def real_data(cfg) -> bool:
    return is_real_dataset(cfg)


@pytest.fixture(scope="session")
def fast_pipeline(dataset):
    """A cheap fitted pipeline for contract tests."""
    pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=42), scale=True)
    return pipe.fit(dataset.X_train, dataset.y_train)


@pytest.fixture(scope="session")
def bundle_path(tmp_path_factory, cfg, dataset, fast_pipeline):
    """Persist a bundle to a temp directory and return its path."""
    from src.evaluation.metrics import evaluate
    from src.models.persist import build_metadata, save_model

    prob = fast_pipeline.predict_proba(dataset.X_test)[:, 1]
    m = evaluate(dataset.y_test, prob, 0.5)

    tmp = tmp_path_factory.mktemp("models")
    raw = dict(cfg.raw)
    raw["paths"] = dict(raw["paths"])
    raw["paths"]["models_dir"] = str(tmp)
    tmp_cfg = type(cfg)(raw=raw, path=cfg.path)

    metadata = build_metadata(
        model_name="logistic_regression_test", cfg=tmp_cfg, threshold=0.5,
        threshold_rationale="fixed 0.5 for contract testing",
        calibration_method="none",
        cv_metrics={"pr_auc": 0.0}, test_metrics=m.as_dict(),
        n_train=dataset.n_train, n_test=dataset.n_test,
    )
    path, _ = save_model(fast_pipeline, metadata, tmp_cfg,
                         explainer_pipeline=None)
    return path
