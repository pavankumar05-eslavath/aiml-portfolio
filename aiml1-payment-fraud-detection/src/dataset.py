"""Loading, feature engineering and the temporal split. Shared by every stage."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REAL = Path("data/onlinefraud.csv")
SYNTHETIC = Path("data/paysim_synthetic.csv")

SCHEMA = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud",
]
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]
EPS = 1e-9


def resolve_path(path: str | Path | None = None) -> Path:
    """Prefer the real PaySim CSV; fall back to the generated one."""
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    if REAL.exists():
        return REAL
    if SYNTHETIC.exists():
        return SYNTHETIC
    raise FileNotFoundError(
        "No dataset found. Run `make download` for the real 480 MB PaySim CSV, "
        "or `make data` to generate a small stand-in."
    )


def load(path: str | Path | None = None) -> pd.DataFrame:
    p = resolve_path(path)
    df = pd.read_csv(p)
    missing = set(SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"{p} is missing columns: {sorted(missing)}")
    return df


def is_real(path: str | Path | None = None) -> bool:
    return resolve_path(path).name == REAL.name


# --------------------------------------------------------------- features
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered features. The balance residuals carry most of the signal."""
    out = df.copy()
    out["hour"] = out["step"] % 24

    # What the balance *should* be, minus what it is. Legitimate transactions
    # reconcile to ~0 on the sender side; PaySim's fraud leg does too, which is
    # why this pairs with the drain flags rather than replacing them.
    out["errBalOrig"] = out["newbalanceOrig"] + out["amount"] - out["oldbalanceOrg"]
    out["errBalDest"] = out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]

    out["drainedOrig"] = (
        (out["newbalanceOrig"] == 0) & (out["oldbalanceOrg"] > 0)
    ).astype("int8")
    out["amtRatioOrig"] = out["amount"] / (out["oldbalanceOrg"] + EPS)
    out["origZeroBefore"] = (out["oldbalanceOrg"] == 0).astype("int8")
    out["destZeroBefore"] = (out["oldbalanceDest"] == 0).astype("int8")
    out["destZeroAfter"] = (out["newbalanceDest"] == 0).astype("int8")
    out["amountLog"] = np.log1p(out["amount"])
    out["destIsMerchant"] = out["nameDest"].astype(str).str.startswith("M").astype("int8")
    return out


FEATURES = [
    "hour", "amount", "amountLog", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "errBalOrig", "errBalDest",
    "drainedOrig", "amtRatioOrig", "origZeroBefore", "destZeroBefore",
    "destZeroAfter", "destIsMerchant",
]


def design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature matrix + target.

    Raw `step` is deliberately excluded. Under a temporal split every test value
    exceeds every training value, so a tree can only extrapolate from it; hour of
    day carries the usable part of the timestamp.
    """
    feat = add_features(df)
    dummies = pd.get_dummies(feat["type"], prefix="type", drop_first=True).astype("int8")
    X = pd.concat([feat[FEATURES], dummies], axis=1)
    return X, feat["isFraud"].to_numpy()


# ------------------------------------------------------------------ split
def temporal_split(
    df: pd.DataFrame, train_q: float = 0.60, val_q: float = 0.75
) -> dict[str, np.ndarray]:
    """Split on `step`, not at random.

    `step` is an hour counter. A random split trains on hour 700 and tests on
    hour 300, which cannot happen in production and inflates every metric.
    """
    step = df["step"].to_numpy()
    cut_tr = int(np.quantile(step, train_q))
    cut_va = int(np.quantile(step, val_q))
    return {
        "train": step <= cut_tr,
        "val": (step > cut_tr) & (step <= cut_va),
        "test": step > cut_va,
        "cut_train": cut_tr,
        "cut_val": cut_va,
    }


def leak_rule(df: pd.DataFrame) -> np.ndarray:
    """The three-clause rule that beats the models.

    PaySim's fraud agent empties the victim's account, leaving an exact
    arithmetic signature that essentially never occurs in legitimate traffic.
    """
    return (
        (df["newbalanceOrig"] == 0)
        & (df["oldbalanceOrg"] > 0)
        & df["type"].isin(FRAUD_TYPES)
        & np.isclose(df["amount"], df["oldbalanceOrg"])
    ).to_numpy()
