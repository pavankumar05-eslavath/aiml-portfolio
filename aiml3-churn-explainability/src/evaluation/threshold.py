"""Threshold selection under a retention cost/value framework.

READ THIS BEFORE QUOTING ANY CURRENCY FIGURE FROM THIS MODULE.

The dataset contains no campaign history: no contact costs, no offer values, no
intervention outcomes. Therefore none of the economic parameters can be estimated
from it. They are **declared assumptions** in configs/config.yaml, and every
figure produced here is conditional on them. They are reported in neutral
"currency units" (CU) rather than dollars to avoid implying a real financial
result, and ``sensitivity_analysis`` exists because the chosen threshold moves
when the assumptions move.

The model:

    A customer has churn probability p and value v = MonthlyCharges x horizon.
    Contacting costs contact_cost + incentive_cost, paid whether or not it works.
    An intervention retains a would-be churner with probability success_rate s.

    EV(contact) = p * s * v - (contact_cost + incentive_cost)

Two policies follow, and they are not equivalent:

* **Global threshold** -- contact everyone above one probability cut. Simple to
  operate, and what almost every churn write-up stops at.
* **Per-customer rule** -- contact when that customer's own EV is positive, i.e.
  p > (contact_cost + incentive_cost) / (s * v). Because v varies across
  customers, the break-even probability varies too: a high-value customer is
  worth contacting at a much lower risk than a low-value one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config


@dataclass(frozen=True)
class Economics:
    """Declared retention-economics assumptions. Not measured from data."""

    contact_cost: float
    incentive_cost: float
    success_rate: float
    horizon_months: int
    value_column: str

    @property
    def intervention_cost(self) -> float:
        """Total cost of contacting one customer."""
        return self.contact_cost + self.incentive_cost

    @classmethod
    def from_config(cls, cfg: Config) -> Economics:
        e = cfg["economics"]
        return cls(
            contact_cost=float(e["contact_cost"]),
            incentive_cost=float(e["incentive_cost"]),
            success_rate=float(e["success_rate"]),
            horizon_months=int(e["horizon_months"]),
            value_column=str(e["value_column"]),
        )

    def customer_value(self, X: pd.DataFrame) -> np.ndarray:
        """Revenue at risk per customer over the horizon.

        Deliberately revenue, not profit: the data has no margin, discount rate or
        cost-to-serve, so calling this 'customer lifetime value' would overstate
        what it is.
        """
        return X[self.value_column].to_numpy(dtype=float) * self.horizon_months

    def breakeven_probability(self, value: np.ndarray) -> np.ndarray:
        """Churn probability at which contacting this customer breaks even."""
        return self.intervention_cost / (self.success_rate * np.maximum(value, 1e-9))


@dataclass(frozen=True)
class ThresholdResult:
    """Outcome of a threshold sweep."""

    best_threshold: float
    best_net_value: float
    curve: pd.DataFrame
    baseline_contact_all: float
    baseline_contact_none: float
    per_customer_net_value: float
    per_customer_contacts: int

    @property
    def uplift_vs_contact_all(self) -> float:
        return self.best_net_value - self.baseline_contact_all


def net_value(
    y_true: np.ndarray, y_prob: np.ndarray, value: np.ndarray, econ: Economics,
    threshold: float,
) -> float:
    """Net value of contacting everyone at or above ``threshold``.

    Only customers who would actually have churned (``y_true == 1``) can be saved,
    and the saving is credited at the assumed success rate. Contact and incentive
    costs are charged for every customer contacted, including the false positives
    -- which is what makes precision economically meaningful here.
    """
    contacted = y_prob >= threshold
    n_contacted = int(contacted.sum())
    saved = float((value[contacted & (y_true == 1)] * econ.success_rate).sum())
    return saved - n_contacted * econ.intervention_cost


def sweep(
    y_true: np.ndarray, y_prob: np.ndarray, value: np.ndarray, econ: Economics,
    n_steps: int = 101,
) -> pd.DataFrame:
    """Evaluate every candidate global threshold."""
    y_true = np.asarray(y_true).astype(int)
    rows = []
    for t in np.linspace(0.0, 1.0, n_steps):
        contacted = y_prob >= t
        n = int(contacted.sum())
        tp = int((contacted & (y_true == 1)).sum())
        fp = int((contacted & (y_true == 0)).sum())
        rows.append({
            "threshold": float(t),
            "contacted": n,
            "contact_rate": n / len(y_true),
            "true_positives": tp,
            "false_positives": fp,
            "precision": tp / n if n else 0.0,
            "recall": tp / max(int((y_true == 1).sum()), 1),
            "net_value": net_value(y_true, y_prob, value, econ, t),
        })
    return pd.DataFrame(rows)


def optimise(
    y_true: np.ndarray, y_prob: np.ndarray, value: np.ndarray, econ: Economics,
) -> ThresholdResult:
    """Find the value-maximising global threshold and compare policies."""
    y_true = np.asarray(y_true).astype(int)
    curve = sweep(y_true, y_prob, value, econ)
    best = curve.loc[curve["net_value"].idxmax()]

    # Per-customer rule: contact where this customer's own EV is positive.
    breakeven = econ.breakeven_probability(value)
    contacted = y_prob > breakeven
    pc_value = float((value[contacted & (y_true == 1)] * econ.success_rate).sum()
                     - contacted.sum() * econ.intervention_cost)

    return ThresholdResult(
        best_threshold=float(best["threshold"]),
        best_net_value=float(best["net_value"]),
        curve=curve,
        baseline_contact_all=net_value(y_true, y_prob, value, econ, 0.0),
        baseline_contact_none=0.0,
        per_customer_net_value=pc_value,
        per_customer_contacts=int(contacted.sum()),
    )


def sensitivity_analysis(
    y_true: np.ndarray, y_prob: np.ndarray, value: np.ndarray, cfg: Config,
) -> pd.DataFrame:
    """Re-optimise the threshold across assumed success rates and incentive costs.

    The point is that the *optimal threshold is a function of assumptions the data
    cannot supply*. Presenting one threshold as "the" answer hides that dependency.
    """
    base = Economics.from_config(cfg)
    grid = cfg["economics"]["sensitivity"]
    rows = []
    for s in grid["success_rate"]:
        for inc in grid["incentive_cost"]:
            econ = Economics(
                contact_cost=base.contact_cost, incentive_cost=float(inc),
                success_rate=float(s), horizon_months=base.horizon_months,
                value_column=base.value_column,
            )
            res = optimise(y_true, y_prob, value, econ)
            row = res.curve.loc[res.curve["net_value"].idxmax()]
            rows.append({
                "success_rate": float(s),
                "incentive_cost": float(inc),
                "best_threshold": res.best_threshold,
                "net_value": res.best_net_value,
                "contact_rate": float(row["contact_rate"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "per_customer_rule_net_value": res.per_customer_net_value,
            })
    return pd.DataFrame(rows)
