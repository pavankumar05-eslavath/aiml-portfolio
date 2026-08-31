"""Shared fixtures.

Fixtures build what they need rather than depending on artefacts left behind by a
previous stage, so the suite is independent of run order. A small synthetic panel is
used wherever the test is about *behaviour* (leakage, feature semantics, inventory
arithmetic) rather than about the real dataset's values.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.loader import is_real_panel, load_panel, rolling_origin_folds

requires_real_data = pytest.mark.skipif(
    not is_real_panel(load_config()),
    reason="needs the real M5 panel (`make download && make panel`)",
)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session", autouse=True)
def _ensure_panel(cfg):
    """Generate the synthetic panel when nothing is on disk."""
    dest = cfg.resolve("data", "processed_dir") / "panel.parquet"
    if not dest.exists() and not is_real_panel(cfg):
        subprocess.run([sys.executable, "-m", "src.data.generate_sample"], check=True)


@pytest.fixture(scope="session")
def panel(cfg):
    return load_panel(cfg)


@pytest.fixture(scope="session")
def fold(cfg, panel):
    return rolling_origin_folds(cfg, panel)[-1]


@pytest.fixture(scope="session")
def features(cfg, panel, fold):
    from src.features.build import build_supervised
    return build_supervised(cfg, panel, fold, horizons=[1, 7, 14, 28])


@pytest.fixture
def tiny_panel() -> pd.DataFrame:
    """A two-series panel with known values, for exact arithmetic assertions."""
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    rows = []
    for i, sid in enumerate(("S_A", "S_B")):
        # S_A: steady demand. S_B: intermittent, every third day.
        y = (np.full(len(dates), 4.0) if i == 0
             else np.where(np.arange(len(dates)) % 3 == 0, 6.0, 0.0))
        rows.append(pd.DataFrame({
            "series_id": sid, "date": dates, "demand": y.astype("float32"),
            "sku_id": f"SKU_{i}", "dept_id": "D1", "cat_id": "C1",
            "store_id": "ST_1", "state_id": "CA",
            "sell_price": np.full(len(dates), 2.0, dtype="float32"),
            "snap": np.zeros(len(dates), dtype="float32"),
        }))
    return pd.concat(rows, ignore_index=True)
