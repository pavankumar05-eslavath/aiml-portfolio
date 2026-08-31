"""Inference pipeline.

Scores new customers from the persisted bundle and returns, for each one:
churn probability, risk segment, top contributing factors and a recommended
action.

Usage as a library:

    from src.inference.predict import ChurnPredictor
    predictor = ChurnPredictor.load()
    result = predictor.predict_one({...})

Usage from the command line:

    python -m src.inference.predict --example
    python -m src.inference.predict --input customers.csv --output scored.csv

The predictor accepts the 19 raw published columns. Feature engineering lives
inside the persisted pipeline, so a caller never reimplements it -- the most
common way a served model diverges from the one that was evaluated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.data.schema import feature_columns
from src.explainability.shap_analysis import compute_shap, explain_customer
from src.inference.decision import decide, decide_one
from src.models.persist import load_model, validate_input

#: A complete, valid customer record used by --example and by the tests.
EXAMPLE_CUSTOMER: dict[str, Any] = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 94.40,
    "TotalCharges": 188.80,
}


class ChurnPredictor:
    """Loads a persisted bundle and scores customers."""

    def __init__(self, bundle: dict[str, Any], cfg: Config) -> None:
        self.bundle = bundle
        self.cfg = cfg
        self.model = bundle["model"]
        self.metadata: dict[str, Any] = bundle["metadata"]
        self.explainer_pipeline = bundle.get("explainer_pipeline")
        self.threshold = float(self.metadata["threshold"])

    @classmethod
    def load(cls, cfg: Config | None = None, path: str | Path | None = None) -> ChurnPredictor:
        cfg = cfg or load_config()
        return cls(load_model(cfg, path), cfg)

    # -- core ---------------------------------------------------------------
    def _frame(self, customers: pd.DataFrame | dict | list[dict]) -> pd.DataFrame:
        if isinstance(customers, dict):
            df = pd.DataFrame([customers])
        elif isinstance(customers, list):
            df = pd.DataFrame(customers)
        else:
            df = customers.copy()
        validate_input(df, self.metadata)
        return df.loc[:, feature_columns()]

    def predict_proba(self, customers: pd.DataFrame | dict | list[dict]) -> np.ndarray:
        """Churn probability per customer."""
        df = self._frame(customers)
        proba = self.model.predict_proba(df)[:, 1]
        if not np.all((proba >= 0.0) & (proba <= 1.0)):
            raise ValueError("model returned a probability outside [0, 1]")
        return proba

    def predict(
        self, customers: pd.DataFrame | dict | list[dict], *, explain: bool = False,
        value_cut: float | None = None,
    ) -> pd.DataFrame:
        """Score customers and attach risk band and recommended action."""
        df = self._frame(customers)
        proba = self.predict_proba(df)
        out = decide(proba, df, self.cfg, value_cut=value_cut)
        out.insert(0, "will_churn_at_threshold", proba >= self.threshold)

        if explain:
            out["explanation"] = [
                json.dumps(e) for e in self.explain(df)
            ]
        return out

    def explain(self, customers: pd.DataFrame | dict | list[dict]) -> list[dict[str, Any]]:
        """Per-customer SHAP explanation in plain language."""
        if self.explainer_pipeline is None:
            raise RuntimeError(
                "bundle has no explainer_pipeline; retrain with `python -m src.run train`")
        df = self._frame(customers)
        art = compute_shap(self.explainer_pipeline, df, max_samples=None)
        raw_cols = feature_columns()
        return [explain_customer(art, i, raw_cols) for i in range(len(df))]

    def predict_one(self, customer: dict[str, Any]) -> dict[str, Any]:
        """Full result for a single customer: the shape an API would return."""
        df = self._frame(customer)
        proba = float(self.predict_proba(df)[0])
        decision = decide_one(proba, df, self.cfg)
        explanation = self.explain(df)[0]

        return {
            "churn_probability": round(proba, 4),
            "risk_segment": decision.risk_segment,
            "value_tier": decision.value_tier,
            "customer_value_cu": round(decision.customer_value, 2),
            "expected_value_of_contact_cu": round(decision.expected_value, 2),
            "breakeven_probability": round(decision.breakeven_probability, 4),
            "contact_recommended": decision.contact_recommended,
            "recommended_action": decision.action,
            "priority": decision.priority,
            "channel": decision.channel,
            "top_risk_factors": explanation["top_risk_factors"],
            "top_protective_factors": explanation["top_protective_factors"],
            "model": {
                "name": self.metadata["model_name"],
                "threshold": self.threshold,
                "calibration": self.metadata["calibration_method"],
                "trained_at": self.metadata["trained_at"],
            },
            "caveat": "Currency figures use the assumed retention economics in "
                      "configs/config.yaml; they are not measured business results.",
        }


def format_result(result: dict[str, Any]) -> str:
    """Render a single prediction for a human reader."""
    lines = [
        "=" * 72,
        "CHURN PREDICTION",
        "=" * 72,
        f"  churn probability   : {result['churn_probability']:.1%}",
        f"  risk segment        : {result['risk_segment']}",
        f"  value tier          : {result['value_tier']} "
        f"({result['customer_value_cu']:,.0f} CU over the horizon)",
        f"  break-even risk     : {result['breakeven_probability']:.1%} "
        "(contact pays off above this)",
        f"  expected value      : {result['expected_value_of_contact_cu']:+,.0f} CU",
        f"  contact recommended : {'YES' if result['contact_recommended'] else 'NO'}",
        f"  priority            : {result['priority']}  via {result['channel']}",
        "",
        f"  ACTION: {result['recommended_action']}",
        "",
        "  Why this customer is at risk:",
    ]
    lines.extend(f"    + {f['reason']}" for f in result["top_risk_factors"])
    if result["top_protective_factors"]:
        lines.append("  What is holding them:")
        lines.extend(f"    - {f['reason']}"
                     for f in result["top_protective_factors"])
    lines += ["", f"  note: {result['caveat']}", "=" * 72]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", action="store_true",
                    help="score the built-in example customer")
    ap.add_argument("--input", type=str, help="CSV of customers to score")
    ap.add_argument("--output", type=str, help="where to write scored CSV")
    ap.add_argument("--explain", action="store_true", help="include explanations")
    args = ap.parse_args()

    cfg = load_config()
    predictor = ChurnPredictor.load(cfg)

    if args.example or not args.input:
        result = predictor.predict_one(EXAMPLE_CUSTOMER)
        print(format_result(result))
        return 0

    df = pd.read_csv(args.input)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    scored = predictor.predict(df, explain=args.explain)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.output, index=False)
        print(f"wrote {args.output} ({len(scored):,} rows)")
    else:
        print(scored.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
