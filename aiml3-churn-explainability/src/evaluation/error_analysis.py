"""Error analysis: who the model gets wrong, and what it costs.

False positives and false negatives are not symmetric here, and the asymmetry
runs the opposite way to the fraud-detection case:

* **False negative** -- a churner the model missed. The customer leaves and their
  remaining value is lost. Cost ~ customer value x success_rate (the saving that
  was available and was not attempted).
* **False positive** -- a loyal customer offered a retention incentive. Cost ~
  contact_cost + incentive_cost, plus the harder-to-quantify risk of teaching
  customers that threatening to leave earns a discount.

Under the assumed economics a false negative is roughly an order of magnitude
more expensive than a false positive, which is why the value-optimal threshold
sits far below 0.5 and buys recall with precision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.evaluation.threshold import Economics
from src.features.engineer import TENURE_BINS, TENURE_LABELS

#: Attributes profiled when characterising errors.
PROFILE_CATEGORICAL: tuple[str, ...] = (
    "Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity",
)
PROFILE_NUMERIC: tuple[str, ...] = ("tenure", "MonthlyCharges", "TotalCharges")


def error_frame(
    X: pd.DataFrame, y_true: np.ndarray, probabilities: np.ndarray, threshold: float,
) -> pd.DataFrame:
    """Attach outcome, prediction and error class to each test row."""
    y_true = np.asarray(y_true).astype(int)
    pred = (probabilities >= threshold).astype(int)

    # default must be a string: np.select cannot promote the implicit int 0
    # against string choices, and raises rather than coercing.
    kind = np.select(
        [(pred == 1) & (y_true == 1), (pred == 0) & (y_true == 0),
         (pred == 1) & (y_true == 0), (pred == 0) & (y_true == 1)],
        ["true_positive", "true_negative", "false_positive", "false_negative"],
        default="unclassified",
    )
    out = X.copy()
    out["y_true"] = y_true
    out["probability"] = probabilities
    out["predicted"] = pred
    out["error_kind"] = kind
    out["tenure_band"] = pd.cut(out["tenure"], bins=TENURE_BINS,
                                labels=TENURE_LABELS).astype(str)
    return out


def error_costs(
    errors: pd.DataFrame, cfg: Config,
) -> dict[str, float]:
    """Cost of each error class under the declared assumptions."""
    econ = Economics.from_config(cfg)
    value = econ.customer_value(errors)

    fn = errors["error_kind"] == "false_negative"
    fp = errors["error_kind"] == "false_positive"

    fn_cost = float((value[fn.to_numpy()] * econ.success_rate).sum())
    fp_cost = float(int(fp.sum()) * econ.intervention_cost)
    return {
        "n_false_negative": int(fn.sum()),
        "n_false_positive": int(fp.sum()),
        "fn_opportunity_cost_cu": fn_cost,
        "fp_wasted_spend_cu": fp_cost,
        "mean_fn_cost_cu": fn_cost / max(int(fn.sum()), 1),
        "mean_fp_cost_cu": econ.intervention_cost,
        "cost_ratio_fn_to_fp": (fn_cost / max(int(fn.sum()), 1)) / econ.intervention_cost,
    }


def profile_errors(errors: pd.DataFrame) -> pd.DataFrame:
    """Compare numeric feature means across the four outcome classes."""
    rows = []
    for col in PROFILE_NUMERIC:
        r: dict[str, object] = {"feature": col}
        for kind in ("true_positive", "false_negative", "false_positive", "true_negative"):
            sub = errors.loc[errors["error_kind"] == kind, col]
            r[kind] = float(sub.mean()) if len(sub) else float("nan")
        rows.append(r)

    r = {"feature": "predicted probability"}
    for kind in ("true_positive", "false_negative", "false_positive", "true_negative"):
        sub = errors.loc[errors["error_kind"] == kind, "probability"]
        r[kind] = float(sub.mean()) if len(sub) else float("nan")
    rows.append(r)
    return pd.DataFrame(rows)


def categorical_error_profile(
    errors: pd.DataFrame, column: str,
) -> pd.DataFrame:
    """Distribution of a categorical attribute within each error class."""
    ct = pd.crosstab(errors[column], errors["error_kind"], normalize="columns")
    return ct.reindex(columns=[c for c in (
        "true_positive", "false_negative", "false_positive", "true_negative")
        if c in ct.columns])


def hardest_cases(errors: pd.DataFrame, n: int = 5) -> dict[str, pd.DataFrame]:
    """The most confidently wrong predictions in each direction.

    These are the cases worth reading individually: a false negative with a very
    low predicted probability is a customer who looked entirely safe and left.
    """
    cols = ["probability", "tenure", "Contract", "MonthlyCharges", "InternetService",
            "PaymentMethod", "TechSupport"]
    fn = errors[errors["error_kind"] == "false_negative"].nsmallest(n, "probability")
    fp = errors[errors["error_kind"] == "false_positive"].nlargest(n, "probability")
    return {
        "confident_false_negatives": fn.loc[:, cols].reset_index(drop=True),
        "confident_false_positives": fp.loc[:, cols].reset_index(drop=True),
    }


def label_noise_ceiling(errors: pd.DataFrame, n_conflicting: int, n_total: int) -> dict[str, float]:
    """Relate observed error to the irreducible floor found in the audit.

    Puts the error rate in context: some of it is not learnable, because the data
    contains feature-identical rows with contradictory labels.
    """
    err_rate = float((errors["y_true"] != errors["predicted"]).mean())
    floor = n_conflicting / (2 * n_total)
    return {
        "observed_error_rate": err_rate,
        "irreducible_floor_estimate": floor,
        "share_of_error_that_is_irreducible": floor / err_rate if err_rate else 0.0,
    }


def threshold_recommendation(
    errors: pd.DataFrame, cfg: Config, chosen_threshold: float,
) -> str:
    """Narrative on whether the threshold should move, given the cost asymmetry."""
    costs = error_costs(errors, cfg)
    ratio = costs["cost_ratio_fn_to_fp"]
    return (
        f"Under the declared assumptions a missed churner costs about "
        f"{costs['mean_fn_cost_cu']:.0f} CU in forgone saving, against "
        f"{costs['mean_fp_cost_cu']:.0f} CU for a wasted contact -- a ratio of "
        f"{ratio:.1f}:1. That asymmetry is why the value-optimal threshold is "
        f"{chosen_threshold:.2f} rather than 0.5: buying recall with precision is "
        f"correct while a false negative costs {ratio:.0f}x a false positive. The "
        f"ranking of thresholds is only as sound as the assumed success rate, so "
        f"the sensitivity table should be read alongside this."
    )
