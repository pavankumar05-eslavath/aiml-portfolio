"""Risk segmentation and segment-level performance.

Two distinct jobs:

* ``assign_risk_segments`` turns a continuous probability into operational bands.
  The bands are *validated*, not just declared: the realised churn rate inside
  each band on the test set is reported next to the predicted risk. A band whose
  realised rate does not match its label is a broken band, however tidy the cut
  points look.

* ``segment_performance`` asks whether the model works equally well for different
  kinds of customer. A single aggregate AUC can hide a model that is strong on
  month-to-month customers and near-useless on two-year contracts -- which
  matters, because the second group is where a retention budget is wasted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.config import Config
from src.features.engineer import TENURE_BINS, TENURE_LABELS

#: Order used for reporting, lowest risk first.
SEGMENT_ORDER: tuple[str, ...] = ("Low Risk", "Medium Risk", "High Risk", "Critical Risk")

#: Segment -> the retention posture it implies. Cost rises with risk.
SEGMENT_ACTIONS: dict[str, str] = {
    "Low Risk": "Normal engagement. No retention spend; include in standard lifecycle "
                "comms and monitor for band changes.",
    "Medium Risk": "Low-cost nurture. Service review, add-on education, or a bundle "
                   "recommendation. Automated channels only.",
    "High Risk": "Proactive outreach. Targeted offer or contract-term incentive, "
                 "prioritised by customer value.",
    "Critical Risk": "Priority intervention. Human contact with retention authority, "
                     "subject to the customer clearing the expected-value test.",
}


def _band_edges(cfg: Config) -> list[tuple[str, float, float]]:
    """Config bands as (label, lower, upper), ordered."""
    raw = cfg["risk_segments"]
    mapping = {
        "low": "Low Risk", "medium": "Medium Risk",
        "high": "High Risk", "critical": "Critical Risk",
    }
    bands = [(mapping[k], float(v[0]), float(v[1])) for k, v in raw.items()]
    return sorted(bands, key=lambda b: b[1])


def assign_risk_segments(probabilities: np.ndarray, cfg: Config) -> pd.Series:
    """Map probabilities to named risk bands."""
    bands = _band_edges(cfg)
    edges = [b[1] for b in bands] + [bands[-1][2]]
    labels = [b[0] for b in bands]
    seg = pd.cut(probabilities, bins=edges, labels=labels, right=False,
                 include_lowest=True)
    return pd.Series(seg, name="risk_segment").astype("string").fillna(labels[-1])


def segment_summary(
    probabilities: np.ndarray, y_true: np.ndarray, cfg: Config,
    value: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-band size, predicted risk, realised churn rate and value at risk.

    ``realised_churn_rate`` is the validation: it should rise monotonically across
    the bands and sit inside each band's probability range.
    """
    seg = assign_risk_segments(probabilities, cfg)
    df = pd.DataFrame({
        "risk_segment": seg.to_numpy(),
        "probability": probabilities,
        "churned": np.asarray(y_true).astype(int),
    })
    if value is not None:
        df["value"] = value

    agg: dict[str, tuple] = {
        "customers": ("probability", "size"),
        "avg_predicted_risk": ("probability", "mean"),
        "realised_churn_rate": ("churned", "mean"),
        "churners": ("churned", "sum"),
    }
    if value is not None:
        agg["avg_customer_value"] = ("value", "mean")
        agg["value_at_risk"] = ("value", "sum")

    out = df.groupby("risk_segment", observed=True).agg(**agg)
    out = out.reindex([s for s in SEGMENT_ORDER if s in out.index])
    out["share_of_customers"] = out["customers"] / len(df)
    out["share_of_churners"] = out["churners"] / max(int(df["churned"].sum()), 1)
    out["recommended_action"] = [SEGMENT_ACTIONS[s] for s in out.index]
    return out


def validate_segments(summary: pd.DataFrame) -> dict[str, bool | float]:
    """Check the bands behave as their labels claim.

    Returns whether realised churn rises monotonically across bands, and the worst
    absolute gap between average predicted risk and realised churn rate.
    """
    realised = summary["realised_churn_rate"].to_numpy(dtype=float)
    predicted = summary["avg_predicted_risk"].to_numpy(dtype=float)
    return {
        "monotonic_realised_churn": bool(np.all(np.diff(realised) > 0)),
        "max_abs_calibration_gap": float(np.max(np.abs(predicted - realised))),
        "lowest_band_churn": float(realised[0]),
        "highest_band_churn": float(realised[-1]),
        "lift_top_vs_bottom": float(realised[-1] / realised[0]) if realised[0] > 0 else float("inf"),
    }


def _safe_metric(fn, y: np.ndarray, p: np.ndarray) -> float:
    """Metric that returns NaN when a segment has only one class present."""
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(fn(y, p))


def segment_performance(
    X: pd.DataFrame, y_true: np.ndarray, probabilities: np.ndarray,
    threshold: float, by: str, min_size: int = 50,
) -> pd.DataFrame:
    """Model performance within each level of a customer attribute."""
    y_true = np.asarray(y_true).astype(int)
    pred = (probabilities >= threshold).astype(int)

    if by == "tenure_band":
        keys = pd.cut(X["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    else:
        keys = X[by].astype(str)

    rows = []
    for level, idx in pd.Series(range(len(keys)), index=keys.to_numpy()).groupby(level=0):
        i = idx.to_numpy()
        if len(i) < min_size:
            continue
        yt, pp, pd_ = y_true[i], probabilities[i], pred[i]
        tp = int(((pd_ == 1) & (yt == 1)).sum())
        rows.append({
            "segment": str(level),
            "n": len(i),
            "churn_rate": float(yt.mean()),
            "roc_auc": _safe_metric(roc_auc_score, yt, pp),
            "pr_auc": _safe_metric(average_precision_score, yt, pp),
            "brier": float(brier_score_loss(yt, pp)),
            "mean_predicted": float(pp.mean()),
            "calibration_gap": float(pp.mean() - yt.mean()),
            "recall": tp / max(int(yt.sum()), 1),
            "precision": tp / max(int(pd_.sum()), 1),
            "flag_rate": float(pd_.mean()),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def performance_disparities(
    X: pd.DataFrame, y_true: np.ndarray, probabilities: np.ndarray, threshold: float,
    attributes: tuple[str, ...] = (
        "Contract", "tenure_band", "InternetService", "PaymentMethod",
        "SeniorCitizen", "gender",
    ),
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Segment performance across several attributes, plus the worst spreads."""
    frames = []
    for attr in attributes:
        f = segment_performance(X, y_true, probabilities, threshold, attr)
        if not f.empty:
            f.insert(0, "attribute", attr)
            frames.append(f)
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    spreads: dict[str, float] = {}
    if not table.empty:
        for attr, grp in table.groupby("attribute"):
            auc = grp["roc_auc"].dropna()
            if len(auc) > 1:
                spreads[f"{attr}_roc_auc_spread"] = float(auc.max() - auc.min())
            rec = grp["recall"].dropna()
            if len(rec) > 1:
                spreads[f"{attr}_recall_spread"] = float(rec.max() - rec.min())
    return table, spreads
