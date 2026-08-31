"""Stage 1 -- profile the data and check the article's factual claims."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.dataset import load, resolve_path


def run(path: str | None = None) -> dict:
    df = load(path)
    print(f"source : {resolve_path(path)}")
    print(f"shape  : {df.shape}")
    print(f"\ndtypes:\n{df.dtypes}")

    n_fraud = int(df["isFraud"].sum())
    n_legit = int((df["isFraud"] == 0).sum())
    ratio = n_legit / max(n_fraud, 1)

    print("\nisFraud value_counts():")
    print(df["isFraud"].value_counts().to_string())
    print(f"\nfraud rate : {df['isFraud'].mean():.4%}")
    print(f"imbalance  : 1:{ratio:.0f}")
    print(
        "\nCLAIM CHECK -- the article states the classes are already balanced and\n"
        f"no sampling is needed. At 1:{ratio:.0f} that is false, and it is the\n"
        "central difficulty of the task."
    )

    print("\nfraud by transaction type:")
    by_type = df.groupby("type")["isFraud"].agg(n="size", frauds="sum", rate="mean")
    print(by_type.sort_values("frauds", ascending=False).to_string())
    silent = sorted(by_type.index[by_type["frauds"] == 0])
    print(f"\nzero frauds ever recorded in: {silent}")

    print(f"\nstep range : {df['step'].min()} -> {df['step'].max()}")
    print(f"nulls      : {int(df.isna().sum().sum())}")
    print(
        f"nameOrig   : {df['nameOrig'].nunique():,} distinct across {len(df):,} rows "
        "-- an identifier, not a feature"
    )

    # A majority-class predictor, to show why 'accuracy' is the wrong word.
    print(
        f"\nA model that always predicts 'legit' scores {1 - df['isFraud'].mean():.4%} "
        "accuracy\nand catches zero fraud. The article prints ROC AUC under the "
        "label 'Accuracy'."
    )

    return {
        "rows": len(df),
        "frauds": n_fraud,
        "imbalance": ratio,
        "fraud_types": sorted(by_type.index[by_type["frauds"] > 0]),
        "step_max": int(df["step"].max()),
    }


def correlation_note(df: pd.DataFrame) -> pd.DataFrame:
    """The article correlates pd.factorize() codes, including on continuous
    columns, which replaces amounts with order-of-first-appearance integers.
    Returned here only so the two heatmaps can be compared."""
    return df.apply(lambda x: pd.factorize(x)[0]).corr()


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    run()
