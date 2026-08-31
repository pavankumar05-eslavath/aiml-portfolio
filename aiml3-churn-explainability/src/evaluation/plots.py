"""Evaluation figures: calibration, thresholds, PR/ROC, confusion.

Kept separate from metrics.py so the metric functions stay importable without
pulling in matplotlib, and so a headless run can skip plotting entirely.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot; keeps a headless run from needing a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.evaluation.metrics import MetricSet, expected_calibration_error, reliability_data

CHURN_COLOR = "#c53030"
STAY_COLOR = "#2b6cb0"


def plot_calibration(
    y_true: np.ndarray, probabilities: dict[str, np.ndarray], dest: Path,
    n_bins: int = 10,
) -> Path:
    """Reliability diagram comparing raw and calibrated probabilities.

    The diagonal is perfect calibration. A curve below it means the model is
    overconfident (it says 0.8 for groups that churn less often), which inflates
    every expected-value calculation in the decision layer.

    The histogram matters as much as the curve: a bin holding 5 customers can sit
    far off the diagonal through noise alone, so the curve should not be read
    without the counts underneath it.
    """
    y_true = np.asarray(y_true).astype(int)
    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(7.5, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})

    ax.plot([0, 1], [0, 1], "--", color="black", lw=1, label="perfectly calibrated")
    for name, prob in probabilities.items():
        mean_pred, frac_pos = reliability_data(y_true, prob, n_bins=n_bins)
        ece = expected_calibration_error(y_true, prob, n_bins=n_bins)
        ax.plot(mean_pred, frac_pos, "o-", lw=1.8, ms=5,
                label=f"{name} (ECE {ece:.4f})")
        ax_hist.hist(prob, bins=n_bins, range=(0, 1), alpha=0.5, label=name)

    ax.set_ylabel("observed churn rate")
    ax.set_title("Calibration: does a predicted 0.7 mean a 70% chance?", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    ax_hist.set_xlabel("predicted probability")
    ax_hist.set_ylabel("customers")
    ax_hist.set_yscale("log")
    ax_hist.set_title("Prediction distribution (bin counts behind the curve above)",
                      fontsize=10)
    ax_hist.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)
    return dest


def plot_threshold_analysis(
    curve: pd.DataFrame, best_threshold: float, dest: Path,
    per_customer_value: float | None = None,
) -> Path:
    """Net value, precision/recall and contact volume against the threshold.

    Shows why 0.5 is an arbitrary choice: the value-optimal point is wherever the
    net-value curve peaks, and that depends on the assumed economics rather than
    on anything in the data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    ax = axes[0]
    ax.plot(curve["threshold"], curve["net_value"], color="#276749", lw=2)
    ax.axvline(best_threshold, ls="--", color=CHURN_COLOR, lw=1.5,
               label=f"optimal {best_threshold:.2f}")
    ax.axvline(0.5, ls=":", color="black", lw=1.2, label="default 0.5")
    if per_customer_value is not None:
        ax.axhline(per_customer_value, ls="-.", color="#553c9a", lw=1.4,
                   label="per-customer EV rule")
    ax.set_title("Net retention value by threshold\n(assumed economics)", fontsize=10)
    ax.set_xlabel("threshold")
    ax.set_ylabel("net value (CU)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(curve["threshold"], curve["precision"], label="precision", color=STAY_COLOR)
    ax.plot(curve["threshold"], curve["recall"], label="recall", color=CHURN_COLOR)
    ax.axvline(best_threshold, ls="--", color=CHURN_COLOR, lw=1.2)
    ax.axvline(0.5, ls=":", color="black", lw=1.2)
    ax.set_title("Precision and recall trade-off", fontsize=10)
    ax.set_xlabel("threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(curve["threshold"], curve["contact_rate"], color="#975a16", lw=2)
    ax.axvline(best_threshold, ls="--", color=CHURN_COLOR, lw=1.2)
    ax.set_title("Share of customers contacted", fontsize=10)
    ax.set_xlabel("threshold")
    ax.set_ylabel("contact rate")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)
    return dest


def plot_pr_roc(
    y_true: np.ndarray, prob: np.ndarray, dest: Path, base_rate: float | None = None,
) -> Path:
    """PR and ROC curves side by side.

    The PR baseline is the base rate, not 0.5 -- which is why PR-AUC of 0.61 at a
    26.5% base rate is a real result rather than a poor one.
    """
    y_true = np.asarray(y_true).astype(int)
    base = float(y_true.mean()) if base_rate is None else base_rate

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    precision, recall, _ = precision_recall_curve(y_true, prob)
    axes[0].plot(recall, precision, color=CHURN_COLOR, lw=2)
    axes[0].axhline(base, ls="--", color="black", lw=1,
                    label=f"no-skill baseline = base rate {base:.3f}")
    axes[0].set_xlabel("recall")
    axes[0].set_ylabel("precision")
    axes[0].set_title("Precision-Recall curve (primary metric)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    fpr, tpr, _ = roc_curve(y_true, prob)
    axes[1].plot(fpr, tpr, color=STAY_COLOR, lw=2)
    axes[1].plot([0, 1], [0, 1], "--", color="black", lw=1, label="no skill")
    axes[1].set_xlabel("false positive rate")
    axes[1].set_ylabel("true positive rate")
    axes[1].set_title("ROC curve")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)
    return dest


def plot_confusion(metrics: dict[str, MetricSet], dest: Path) -> Path:
    """Confusion matrices at the default and value-optimal thresholds."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4.4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (name, m) in zip(axes, metrics.items(), strict=True):
        grid = np.array([[m.tn, m.fp], [m.fn, m.tp]])
        ax.imshow(grid, cmap="Blues")
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=14,
                    color="white" if v > grid.max() / 2 else "black")
        ax.set_xticks([0, 1], ["predicted stay", "predicted churn"])
        ax.set_yticks([0, 1], ["actually stayed", "actually churned"])
        ax.set_title(f"{name}\nprecision {m.precision:.3f}  recall {m.recall:.3f}",
                     fontsize=10)

    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)
    return dest
