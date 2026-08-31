"""Build the modelling panel and the chronological splits.

Three design decisions here determine whether everything downstream is valid.

**1. The grain is SKU x store x day.** The source series key already encodes
SKU x store; it is renamed to ``series_id`` and the product and location are kept
as separate columns so the model can pool across stores.

**2. Subsetting is stratified, not top-N.** 30,490 series is more than this
project needs, and ranking by volume would quietly drop the intermittent series
that make demand forecasting hard -- which are ~72% of the data. The subset spans
every category and every Syntetos-Boylan demand class, chosen with a fixed seed.

**3. Splits are chronological, never random.** The test window is the official M5
28-day evaluation period. Validation uses rolling origins inside the training
period. A random split would let the model interpolate between neighbouring days
of the same series and would score far better than any deployed forecaster can,
because at forecast time the future genuinely does not exist yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.data.schema import (
    DATE,
    PANEL_COLUMNS,
    SERIES_ID,
    SRC_DATE,
    SRC_DEMAND,
    SRC_SERIES,
    TARGET,
)

PANEL_NAME = "panel.parquet"


# ---------------------------------------------------------------------------
# demand character, used for stratified sampling and for segmentation later
# ---------------------------------------------------------------------------

def demand_stats(panel: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Per-series intermittency and variability statistics.

    ``ADI`` (average demand interval) is periods per non-zero period, and ``CV2``
    is the squared coefficient of variation of the NON-ZERO demand sizes. Using
    non-zero sizes is what makes the pair separable: computing CV2 over all
    periods conflates "varies a lot when it sells" with "rarely sells", which is
    exactly the distinction the classification exists to draw.
    """
    g = panel.groupby(SERIES_ID, observed=True)[target]
    n = g.size().rename("n_periods")
    nz = g.apply(lambda s: int((s > 0).sum())).rename("n_nonzero")
    total = g.sum().rename("total_demand")
    mean = g.mean().rename("mean_demand")

    nonzero = panel.loc[panel[target] > 0]
    gnz = nonzero.groupby(SERIES_ID, observed=True)[target]
    nz_mean = gnz.mean().rename("nonzero_mean")
    nz_std = gnz.std().rename("nonzero_std")

    out = pd.concat([n, nz, total, mean, nz_mean, nz_std], axis=1)
    out["zero_share"] = 1.0 - out["n_nonzero"] / out["n_periods"]
    out["ADI"] = out["n_periods"] / out["n_nonzero"].replace(0, np.nan)
    out["CV2"] = (out["nonzero_std"] / out["nonzero_mean"].replace(0, np.nan)) ** 2
    return out


#: Syntetos-Boylan cut points. ADI 1.32 and CV2 0.49 are the standard values from
#: the intermittent-demand literature, not tuned here.
ADI_CUT = 1.32
CV2_CUT = 0.49


def classify_demand(stats: pd.DataFrame) -> pd.Series:
    """Syntetos-Boylan demand classification."""
    adi = stats["ADI"].fillna(np.inf)
    cv2 = stats["CV2"].fillna(0.0)
    labels = np.where(
        adi < ADI_CUT,
        np.where(cv2 < CV2_CUT, "Smooth", "Erratic"),
        np.where(cv2 < CV2_CUT, "Intermittent", "Lumpy"),
    )
    return pd.Series(labels, index=stats.index, name="demand_class")


# ---------------------------------------------------------------------------
# panel construction
# ---------------------------------------------------------------------------

def _read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def select_series(cfg: Config, static: pd.DataFrame, target: pd.DataFrame) -> list[str]:
    """Choose the subset of series, stratified over category x demand class.

    Sampling is at SKU level so that a chosen SKU is present in every selected
    store; that keeps store comparisons like-for-like instead of comparing
    different product mixes.
    """
    sub = cfg["data"]["subset"]
    stores = list(sub["stores"])
    n_skus = int(sub["n_skus"])
    min_days = int(sub["min_series_days"])

    static = static[static["store_id"].isin(stores)]
    keep_ids = set(static[SRC_SERIES])
    t = target[target[SERIES_ID].isin(keep_ids)]

    # Require enough history to judge annual seasonality.
    span = t.groupby(SERIES_ID, observed=True)[DATE].agg(["min", "max", "size"])
    long_enough = span[span["size"] >= min_days].index
    t = t[t[SERIES_ID].isin(long_enough)]

    stats = demand_stats(t)
    stats["demand_class"] = classify_demand(stats)
    meta = static.set_index(SRC_SERIES).loc[
        stats.index.intersection(static[SRC_SERIES]), ["sku_id", "cat_id"]]
    stats = stats.join(meta, how="inner")

    # A SKU is eligible only if it survived the history filter in every store.
    per_sku = stats.groupby("sku_id", observed=True).agg(
        n_stores=("total_demand", "size"),
        cat_id=("cat_id", "first"),
        total_demand=("total_demand", "sum"),
        demand_class=("demand_class", lambda s: s.value_counts().index[0]),
    )
    per_sku = per_sku[per_sku["n_stores"] == len(stores)]

    rng = np.random.default_rng(cfg.seed)
    strata = per_sku.groupby(["cat_id", "demand_class"], observed=True)
    # Allocate proportionally to stratum size, so the subset mirrors the population
    # mix of demand classes rather than over-weighting the easy ones.
    sizes = strata.size()
    alloc = (sizes / sizes.sum() * n_skus).round().astype(int).clip(lower=1)

    chosen: list[str] = []
    for key, group in strata:
        k = int(min(alloc.get(key, 0), len(group)))
        if k <= 0:
            continue
        picks = rng.choice(group.index.to_numpy(), size=k, replace=False)
        chosen.extend(picks.tolist())

    chosen = sorted(set(chosen))[:n_skus]
    keep = static[static["sku_id"].isin(chosen)][SRC_SERIES].tolist()
    return sorted(keep)


def build_panel(cfg: Config, force: bool = False) -> pd.DataFrame:
    """Join the three source files into one panel and cache it as parquet."""
    cfg.ensure_dirs()
    dest = cfg.resolve("data", "processed_dir") / PANEL_NAME
    if dest.exists() and not force:
        return pd.read_parquet(dest)

    sample = cfg.resolve("data", "sample_file")
    if not cfg.raw_file("train_target").exists() and sample.exists():
        panel = pd.read_parquet(sample)
        panel.to_parquet(dest, index=False)
        return panel

    tgt = _read(cfg.raw_file("train_target"))
    tst = _read(cfg.raw_file("test_target"))
    static = _read(cfg.raw_file("train_static"))
    temporal = _read(cfg.raw_file("train_temporal"))

    for df in (tgt, tst, temporal):
        df[SRC_DATE] = pd.to_datetime(df[SRC_DATE])

    # The official 28-day window is concatenated onto the training target so the
    # panel is one continuous series per SKU-store; the split is applied by DATE,
    # never by which file a row came from.
    target = pd.concat([tgt, tst], ignore_index=True)
    target = target.rename(columns={SRC_SERIES: SERIES_ID, SRC_DATE: DATE,
                                    SRC_DEMAND: TARGET})
    target[SERIES_ID] = target[SERIES_ID].astype(str)
    target = target.sort_values([SERIES_ID, DATE], ignore_index=True)

    keep = select_series(cfg, static, target)
    target = target[target[SERIES_ID].isin(keep)]

    static = static.rename(columns={SRC_SERIES: SERIES_ID})
    static[SERIES_ID] = static[SERIES_ID].astype(str)
    static = static[static[SERIES_ID].isin(keep)]

    temporal = temporal.rename(columns={SRC_SERIES: SERIES_ID, SRC_DATE: DATE})
    temporal[SERIES_ID] = temporal[SERIES_ID].astype(str)
    temporal = temporal[temporal[SERIES_ID].isin(keep)]

    panel = target.merge(static, on=SERIES_ID, how="left", validate="many_to_one")
    panel = panel.merge(temporal, on=[SERIES_ID, DATE], how="left",
                        validate="one_to_one")

    # SNAP is published as three state columns; collapse to the one that applies.
    state = panel["state_id"].astype(str)
    snap = np.select(
        [state == "CA", state == "TX", state == "WI"],
        [panel["snap_CA"], panel["snap_TX"], panel["snap_WI"]],
        default=np.nan,
    )
    panel["snap"] = pd.Series(snap, index=panel.index).astype("float32")
    panel = panel.drop(columns=["snap_CA", "snap_TX", "snap_WI"])

    for c in ("sku_id", "dept_id", "cat_id", "store_id", "state_id"):
        panel[c] = panel[c].astype(str)

    panel = panel.loc[:, list(PANEL_COLUMNS)].sort_values(
        [SERIES_ID, DATE], ignore_index=True)
    panel.to_parquet(dest, index=False)
    return panel


def load_panel(cfg: Config) -> pd.DataFrame:
    """Load the cached panel, building it if necessary."""
    dest = cfg.resolve("data", "processed_dir") / PANEL_NAME
    if dest.exists():
        return pd.read_parquet(dest)
    return build_panel(cfg)


def is_real_panel(cfg: Config) -> bool:
    """True when the panel came from the real M5 files rather than the stand-in."""
    return cfg.raw_file("train_target").exists()


# ---------------------------------------------------------------------------
# chronological splits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    """One rolling-origin fold.

    ``origin`` is the last date whose actuals may be used to build features. The
    fold predicts ``origin + 1 .. origin + horizon``. Keeping the origin explicit
    (rather than only the window bounds) is what makes leakage checkable: every
    feature must be a function of data at or before this date.
    """

    name: str
    origin: pd.Timestamp
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def horizon(self) -> int:
        return int((self.end - self.start).days) + 1


def test_window(cfg: Config) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (pd.Timestamp(cfg["split"]["test_start"]),
            pd.Timestamp(cfg["split"]["test_end"]))


def rolling_origin_folds(cfg: Config, panel: pd.DataFrame) -> list[Fold]:
    """Validation folds strictly before the test window, most recent last.

    Windows are consecutive and non-overlapping, and every fold's evaluation
    window sits entirely before the test window, so no validation decision is made
    on data the test set also contains.
    """
    v = cfg["split"]["validation"]
    n_folds, horizon, step = int(v["n_folds"]), int(v["horizon_days"]), int(v["step_days"])
    test_start, _ = test_window(cfg)

    # Last validation window ends the day before the test window opens.
    last_end = test_start - pd.Timedelta(days=1)
    folds: list[Fold] = []
    for i in range(n_folds):
        end = last_end - pd.Timedelta(days=i * step)
        start = end - pd.Timedelta(days=horizon - 1)
        folds.append(Fold(name=f"fold_{n_folds - i}", origin=start - pd.Timedelta(days=1),
                          start=start, end=end))
    return list(reversed(folds))


def train_slice(cfg: Config, panel: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    """Rows usable for training at ``fold.origin``, honouring the split scheme."""
    v = cfg["split"]["validation"]
    hist = panel[panel[DATE] <= fold.origin]
    if str(v.get("scheme", "expanding")) == "sliding":
        cutoff = fold.origin - pd.Timedelta(days=int(v["sliding_train_days"]))
        hist = hist[hist[DATE] > cutoff]
    return hist


def test_fold(cfg: Config) -> Fold:
    """The final holdout as a fold, so it flows through the same code path."""
    start, end = test_window(cfg)
    return Fold(name="test", origin=start - pd.Timedelta(days=1), start=start, end=end)
