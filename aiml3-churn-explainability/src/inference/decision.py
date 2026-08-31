"""Business decision layer.

Turns a probability into a decision, which is a different question from "who is
most likely to churn". "Contact the high-risk customers" is not a decision rule:
it ignores what the customer is worth, what the contact costs, and whether the
intervention is likely to work.

The rule implemented here is a two-dimensional matrix (risk band x value tier)
gated by an expected-value test:

    EV = p * success_rate * value - (contact_cost + incentive_cost)

A customer is only contacted when their own EV is positive. That produces
outcomes a one-dimensional rule cannot reach -- notably a low-value customer at
high risk whose expected saving does not cover the offer, who is deliberately
NOT contacted, and a high-value customer at moderate risk who is.

All currency figures inherit the assumptions in configs/config.yaml. They are
labelled CU (currency units) rather than dollars because they are not measured.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config
from src.evaluation.segments import assign_risk_segments
from src.evaluation.threshold import Economics

#: (risk band, value tier) -> (action, priority, channel).
#: Priority 1 is most urgent. Cost of the action scales with expected return.
ACTION_MATRIX: dict[tuple[str, str], tuple[str, int, str]] = {
    ("Critical Risk", "High value"): (
        "Priority retention call from a senior agent with discretionary discount "
        "authority and a contract-term offer.", 1, "outbound call"),
    ("Critical Risk", "Standard value"): (
        "Automated retention offer: fixed discount or free add-on for a term "
        "commitment. No agent time.", 3, "email + SMS"),
    ("High Risk", "High value"): (
        "Proactive outreach within 7 days: service review plus incentive to move "
        "off month-to-month.", 2, "outbound call"),
    ("High Risk", "Standard value"): (
        "Targeted automated offer, cheapest effective incentive.", 4, "email"),
    ("Medium Risk", "High value"): (
        "Relationship nurture: account review, add-on bundle recommendation to "
        "raise switching cost.", 5, "email + in-app"),
    ("Medium Risk", "Standard value"): (
        "Monitor. Low-cost engagement content only; re-score next cycle.", 6, "in-app"),
    ("Low Risk", "High value"): (
        "Normal engagement. Protect the relationship, no retention spend. Consider "
        "upsell instead.", 7, "standard lifecycle"),
    ("Low Risk", "Standard value"): (
        "Normal engagement. No action.", 8, "standard lifecycle"),
}


@dataclass(frozen=True)
class Decision:
    """A single customer's recommended treatment."""

    churn_probability: float
    risk_segment: str
    value_tier: str
    customer_value: float
    expected_value: float
    breakeven_probability: float
    contact_recommended: bool
    action: str
    priority: int
    channel: str

    def as_dict(self) -> dict[str, object]:
        return {
            "churn_probability": round(self.churn_probability, 4),
            "risk_segment": self.risk_segment,
            "value_tier": self.value_tier,
            "customer_value_cu": round(self.customer_value, 2),
            "expected_value_cu": round(self.expected_value, 2),
            "breakeven_probability": round(self.breakeven_probability, 4),
            "contact_recommended": self.contact_recommended,
            "recommended_action": self.action,
            "priority": self.priority,
            "channel": self.channel,
        }


def value_tier(value: np.ndarray, threshold: float) -> np.ndarray:
    """Split customers into value tiers at a fixed cut."""
    return np.where(value >= threshold, "High value", "Standard value")


def value_threshold(value: np.ndarray, cfg: Config) -> float:
    """Value cut point, as a quantile of the scored population."""
    q = float(cfg["decision"]["high_value_quantile"])
    return float(np.quantile(value, q))


def decide(
    probabilities: np.ndarray, X: pd.DataFrame, cfg: Config,
    value_cut: float | None = None,
) -> pd.DataFrame:
    """Produce a decision table for a scored population."""
    econ = Economics.from_config(cfg)
    value = econ.customer_value(X)
    cut = value_threshold(value, cfg) if value_cut is None else value_cut

    segments = assign_risk_segments(probabilities, cfg).to_numpy()
    tiers = value_tier(value, cut)
    breakeven = econ.breakeven_probability(value)
    expected = probabilities * econ.success_rate * value - econ.intervention_cost

    actions, priorities, channels = [], [], []
    for seg, tier in zip(segments, tiers, strict=True):
        action, priority, channel = ACTION_MATRIX[(seg, tier)]
        actions.append(action)
        priorities.append(priority)
        channels.append(channel)

    out = pd.DataFrame({
        "churn_probability": probabilities,
        "risk_segment": segments,
        "value_tier": tiers,
        "customer_value_cu": value,
        "breakeven_probability": breakeven,
        "expected_value_cu": expected,
        "contact_recommended": expected > 0,
        "recommended_action": actions,
        "priority": priorities,
        "channel": channels,
    })
    # A positive-EV customer in a "no spend" band is a genuine conflict between the
    # two rules. The EV test wins, because it is the one tied to money, but the
    # override is recorded rather than hidden.
    out["ev_overrides_segment"] = out["contact_recommended"] & out["risk_segment"].isin(
        ["Low Risk", "Medium Risk"])
    return out


def decide_one(probability: float, customer: pd.DataFrame, cfg: Config,
               value_cut: float | None = None) -> Decision:
    """Decision for a single customer, used by the inference API."""
    table = decide(np.asarray([probability]), customer, cfg, value_cut=value_cut)
    r = table.iloc[0]
    return Decision(
        churn_probability=float(r["churn_probability"]),
        risk_segment=str(r["risk_segment"]),
        value_tier=str(r["value_tier"]),
        customer_value=float(r["customer_value_cu"]),
        expected_value=float(r["expected_value_cu"]),
        breakeven_probability=float(r["breakeven_probability"]),
        contact_recommended=bool(r["contact_recommended"]),
        action=str(r["recommended_action"]),
        priority=int(r["priority"]),
        channel=str(r["channel"]),
    )


def campaign_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a decision table into a campaign plan."""
    g = decisions.groupby(["risk_segment", "value_tier"], observed=True).agg(
        customers=("churn_probability", "size"),
        avg_risk=("churn_probability", "mean"),
        avg_value_cu=("customer_value_cu", "mean"),
        contacts=("contact_recommended", "sum"),
        expected_value_cu=("expected_value_cu", lambda s: s[s > 0].sum()),
        priority=("priority", "first"),
    )
    return g.sort_values("priority").reset_index()
