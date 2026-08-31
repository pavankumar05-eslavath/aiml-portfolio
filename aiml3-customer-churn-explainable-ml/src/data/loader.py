"""Reproducible data loading and splitting.

Two decisions here materially affect every downstream result, so both are
implemented explicitly rather than left to a default:

1. ``TotalCharges`` is published as text with 11 blank cells. Those 11 rows all
   have ``tenure == 0``, so the value is not missing at random -- it is
   structurally absent because the customer has never been billed. It is filled
   with 0.0, which is the true amount billed, not an imputation.

2. Feature-identical duplicate rows are removed BEFORE splitting. Splitting
   first would place identical feature vectors on both sides of the boundary and
   inflate test scores.

3. The split is GROUP-AWARE on the feature vector. Deduplication alone does not
   reach zero contamination, because label-conflicting rows are deliberately
   retained and those rows are duplicates by definition. A plain stratified
   ``train_test_split`` left 5 feature-identical rows straddling the boundary.
   Grouping identical feature vectors and keeping each group on one side removes
   the remaining overlap.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import Config
from src.data.schema import EXPECTED_COLUMNS, ID_COLUMN, TARGET, feature_columns


@dataclass(frozen=True)
class Dataset:
    """A split dataset plus the metadata needed to reproduce it."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    ids_train: pd.Series
    ids_test: pd.Series
    #: Feature-vector group id per training row. Passed to the CV splitter so
    #: identical feature vectors never span a fold boundary either.
    groups_train: pd.Series
    n_duplicates_removed: int
    n_conflicting_labels: int

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)


def resolve_data_path(cfg: Config, explicit: str | Path | None = None) -> Path:
    """Prefer the real dataset, fall back to the generated sample."""
    if explicit:
        return Path(explicit)
    real = cfg.resolve("data", "raw_file")
    return real if real.exists() else cfg.resolve("data", "sample_file")


def is_real_dataset(cfg: Config, explicit: str | Path | None = None) -> bool:
    """True when the file on disk is the full published dataset."""
    p = resolve_data_path(cfg, explicit)
    if not p.exists():
        return False
    n = sum(1 for _ in p.open(encoding="utf-8")) - 1
    return n >= int(cfg["data"]["expected_rows"]) - 1


def load_raw(cfg: Config, path: str | Path | None = None) -> pd.DataFrame:
    """Read the CSV and coerce declared types, without dropping anything.

    ``TotalCharges`` is read as text and converted here so the 11 blanks are
    visible to the audit as real NaNs rather than hiding inside an object column.
    """
    df = pd.read_csv(resolve_data_path(cfg, path))

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing expected columns: {missing}")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df.loc[:, list(EXPECTED_COLUMNS)]


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the two structural fixes. Deterministic and order-independent."""
    out = df.copy()
    # Structurally absent, not missing at random: never-billed customers.
    out["TotalCharges"] = out["TotalCharges"].fillna(0.0)
    return out


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop feature-identical rows, and report label conflicts among them.

    Returns ``(deduplicated, n_removed, n_conflicting)``.

    A feature-identical pair with the SAME label is a harmless repeat and one
    copy is kept. A feature-identical pair with DIFFERENT labels is irreducible
    label noise: no model can separate them, so they cap achievable performance.
    Both copies of a conflicting pair are kept, because dropping one would
    silently delete a real churn outcome.
    """
    feats = feature_columns()

    # A group is conflicting when identical features carry more than one distinct
    # label. Flag the WHOLE group: comparing `duplicated(feats)` against
    # `duplicated(feats + target)` marks only the odd row out, so a Yes/No/No
    # group would be counted as one conflict instead of three.
    distinct_labels = df.groupby(feats, dropna=False, observed=True)[TARGET].transform("nunique")
    conflict_mask = distinct_labels > 1
    n_conflicting = int(conflict_mask.sum())

    safe = df.loc[~conflict_mask]
    deduped = safe.drop_duplicates(subset=feats, keep="first")
    n_removed = len(safe) - len(deduped)

    out = pd.concat([deduped, df.loc[conflict_mask]]).sort_index()
    return out, n_removed, n_conflicting


def feature_group_ids(X: pd.DataFrame) -> pd.Series:
    """Assign an integer id to each distinct feature vector.

    Used as the grouping key so identical feature vectors cannot be split across
    train/test or across CV folds.
    """
    key = X.astype(str).agg("\x1f".join, axis=1)
    return pd.Series(pd.factorize(key)[0], index=X.index, name="feature_group")


def make_dataset(cfg: Config, path: str | Path | None = None) -> Dataset:
    """Load, clean, deduplicate and split. The single entry point for modelling."""
    df = clean_raw(load_raw(cfg, path))
    df, n_removed, n_conflicting = deduplicate(df)

    y = (df[TARGET] == cfg.positive_label).astype(int)
    X = df.loc[:, feature_columns()]
    ids = df[ID_COLUMN]
    groups = feature_group_ids(X)

    # StratifiedGroupKFold gives a stratified split that also respects groups.
    # n_splits = 1/test_size, and the first fold is taken as the test set.
    n_splits = max(2, round(1.0 / float(cfg["split"]["test_size"])))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=cfg.seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    return Dataset(
        X_train=X.iloc[train_idx].reset_index(drop=True),
        X_test=X.iloc[test_idx].reset_index(drop=True),
        y_train=y.iloc[train_idx].reset_index(drop=True),
        y_test=y.iloc[test_idx].reset_index(drop=True),
        ids_train=ids.iloc[train_idx].reset_index(drop=True),
        ids_test=ids.iloc[test_idx].reset_index(drop=True),
        groups_train=groups.iloc[train_idx].reset_index(drop=True),
        n_duplicates_removed=n_removed,
        n_conflicting_labels=n_conflicting,
    )
