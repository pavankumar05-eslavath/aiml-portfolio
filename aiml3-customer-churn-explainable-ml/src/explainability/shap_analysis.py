"""SHAP explainability.

Two audiences, two outputs:

* **Global** -- which features drive the model overall. Used to sanity-check that
  the model learned the relationships the EDA found, rather than an artefact.
* **Local** -- why *this* customer scored what they did, rendered in plain
  language a retention agent can act on. A SHAP force plot is not an explanation
  for a non-technical stakeholder; "on a two-year contract, which lowers risk" is.

Implementation notes:

* SHAP runs on the **uncalibrated** tree model. TreeExplainer needs the tree
  ensemble itself, and a calibrator wrapping the pipeline hides it. This is sound
  because the reported probability comes from a monotone transform of the model
  score, so the sign and ranking of each contribution are unchanged. Where a
  calibrator is in use, the explanation describes the score that the calibrated
  probability is derived from -- stated in the output rather than glossed over.
* Contributions are in **log-odds**, not probability. Summing them to a
  probability is wrong, and the plain-language layer therefore reports direction
  and relative magnitude, not "this feature added 8% to your risk".
* One-hot encoding splits a single business concept across several columns, so
  contributions are aggregated back to the original feature before being shown.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.config import Config
from src.features.preprocess import output_feature_names

#: Minimum |contribution| worth showing a stakeholder. Below this the factor is
#: indistinguishable from noise, and surfacing it invites over-reading -- for
#: example a +0.02 `gender` term, which the segment analysis shows carries no real
#: signal (ROC-AUC spread between genders is 0.001).
MIN_CONTRIBUTION = 0.02

#: Add-on columns share one phrasing rule.
_ADDON_LABELS: dict[str, str] = {
    "OnlineSecurity": "online-security", "OnlineBackup": "online-backup",
    "DeviceProtection": "device-protection", "TechSupport": "tech-support",
    "StreamingTV": "TV-streaming", "StreamingMovies": "movie-streaming",
}


def _addon_phrase(feature: str, value) -> str:
    label = _ADDON_LABELS[feature]
    if value == "Yes":
        return f"has the {label} add-on"
    if value == "No":
        return f"has no {label} add-on"
    return "has no internet service"


def _price_phrase(value: float) -> str:
    if value > 1.02:
        return f"now pays {value:.2f}x their historical average (a recent increase)"
    if value < 0.98:
        return f"now pays {value:.2f}x their historical average (a discount)"
    return "pays in line with their billing history"


#: business feature -> callable(value) -> stakeholder-facing phrase.
#:
#: Phrasing is derived from the customer's ACTUAL value, never from the name of an
#: encoded column. Doing it the other way round inverts the meaning of every
#: one-hot term: a negative SHAP value on `PaymentMethod_Electronic check` for a
#: customer whose value is 0 means "does NOT pay by electronic check, which lowers
#: risk", but naming the column reports it as "pays by electronic check" as a
#: protective factor -- the exact opposite of the truth, and contradicted by the
#: EDA where electronic check is the highest-churn method.
PHRASE_RULES: dict[str, object] = {
    "Contract": lambda v: f"is on a {str(v).lower()} contract",
    "PaymentMethod": lambda v: f"pays by {str(v).lower()}",
    "InternetService": lambda v: ("has no internet service" if v == "No"
                                  else f"has {v} internet"),
    "PaperlessBilling": lambda v: ("uses paperless billing" if v == "Yes"
                                   else "receives paper bills"),
    "Partner": lambda v: "has a partner" if v == "Yes" else "has no partner",
    "Dependents": lambda v: "has dependents" if v == "Yes" else "has no dependents",
    "PhoneService": lambda v: ("has phone service" if v == "Yes"
                               else "has no phone service"),
    "MultipleLines": lambda v: ("has multiple phone lines" if v == "Yes"
                                else "has a single phone line" if v == "No"
                                else "has no phone service"),
    "gender": lambda v: f"is recorded as {str(v).lower()}",
    "SeniorCitizen": lambda v: ("is a senior citizen" if int(v) == 1
                                else "is not a senior citizen"),
    "tenure": lambda v: f"has been a customer for {float(v):.0f} months",
    "MonthlyCharges": lambda v: f"pays {float(v):.2f} per month",
    "TotalCharges": lambda v: f"has been billed {float(v):.2f} to date",
    "tenure_band": lambda v: f"is in the {v} tenure band",
    "n_addons": lambda v: f"holds {float(v):.0f} of 6 add-on services",
    "addon_adoption_rate": lambda v: f"has taken up {float(v):.0%} of available add-ons",
    "has_internet": lambda v: ("has internet service" if int(v) == 1
                               else "has no internet service"),
    "is_new_customer": lambda v: ("joined within the last 6 months" if int(v) == 1
                                  else "is past the first 6 months"),
    "is_month_to_month": lambda v: ("has no contract commitment" if int(v) == 1
                                    else "is on a fixed-term contract"),
    "is_autopay": lambda v: ("pays by automatic method" if int(v) == 1
                             else "does not pay automatically"),
    "realized_arpu": lambda v: f"has averaged {float(v):.2f} per month historically",
    "price_vs_history": _price_phrase,
    "billing_discrepancy": lambda v: f"has a {float(v):+.2f} difference between billed "
                                     "and expected charges",
    "charges_per_addon": lambda v: f"pays {float(v):.2f} per service held",
    "household_size": lambda v: f"has {float(v):.0f} household member(s) on the account",
    "tenure_x_month_to_month": lambda v: (
        f"combines {float(v):.0f} months tenure with month-to-month terms"),
}
PHRASE_RULES.update({c: (lambda v, c=c: _addon_phrase(c, v)) for c in _ADDON_LABELS})


#: Business concept -> (member features, the feature that supplies the wording).
#:
#: Several features encode the SAME fact. `Contract` and `is_month_to_month` are
#: one commercial reality; `tenure`, `tenure_band` and `is_new_customer` are
#: another. Reporting them separately pads the explanation with the same point
#: three times and splits the contribution, understating it. Contributions are
#: therefore summed within a concept and phrased once.
CONCEPT_GROUPS: dict[str, tuple[tuple[str, ...], str]] = {
    "contract": (("Contract", "is_month_to_month", "tenure_x_month_to_month"), "Contract"),
    "tenure": (("tenure", "tenure_band", "is_new_customer"), "tenure"),
    "internet": (("InternetService", "has_internet"), "InternetService"),
    "payment": (("PaymentMethod", "is_autopay"), "PaymentMethod"),
    "billing_channel": (("PaperlessBilling",), "PaperlessBilling"),
    "spend": (("MonthlyCharges", "TotalCharges", "realized_arpu",
               "charges_per_addon"), "MonthlyCharges"),
    "price_change": (("price_vs_history", "billing_discrepancy"), "price_vs_history"),
    "addon_breadth": (("n_addons", "addon_adoption_rate"), "n_addons"),
    "household": (("Partner", "Dependents", "household_size"), "household_size"),
    "phone": (("PhoneService", "MultipleLines"), "MultipleLines"),
    "demographics": (("gender", "SeniorCitizen"), "SeniorCitizen"),
}

_FEATURE_TO_CONCEPT: dict[str, str] = {
    f: concept for concept, (members, _lead) in CONCEPT_GROUPS.items() for f in members
}
_CONCEPT_LEAD: dict[str, str] = {c: lead for c, (_m, lead) in CONCEPT_GROUPS.items()}


def _base_feature(name: str, raw_columns: list[str]) -> str:
    """Map an encoded column back to the business feature it came from."""
    for col in sorted(raw_columns, key=len, reverse=True):
        if name == col or name.startswith(f"{col}_"):
            return col
    return name


def _concept_of(feature: str, row: pd.Series) -> tuple[str, str]:
    """Return (concept, feature supplying the wording) for one feature.

    Add-on columns are normally their own concept, because "has no tech support"
    is individually actionable. But when their value is the structural
    'No internet service' they carry no independent information, so they are folded
    into the internet concept instead of repeating that phrase six times.
    """
    if feature in _ADDON_LABELS:
        if str(row.get(feature)) == "No internet service":
            return "internet", "InternetService"
        return feature, feature
    concept = _FEATURE_TO_CONCEPT.get(feature)
    if concept is None:
        return feature, feature
    return concept, _CONCEPT_LEAD[concept]


def describe(feature: str, value) -> str:
    """Render one feature/value pair as a stakeholder-facing phrase."""
    rule = PHRASE_RULES.get(feature)
    if rule is None:
        return f"{feature} = {value}"
    try:
        return rule(value)
    except (TypeError, ValueError):
        return f"{feature} = {value}"


@dataclass
class ShapArtifacts:
    """Computed SHAP values plus the matrices needed to interpret them."""

    values: np.ndarray            # (n_samples, n_encoded_features), log-odds
    encoded: pd.DataFrame         # preprocessed feature matrix
    feature_names: list[str]
    base_value: float
    raw: pd.DataFrame             # original input rows
    #: Engineered frame (raw + derived columns). Phrases are rendered from this so
    #: derived features such as price_vs_history can be described in real units.
    engineered: pd.DataFrame = None  # type: ignore[assignment]

    def global_importance(self) -> pd.DataFrame:
        """Mean |SHAP| per encoded feature."""
        imp = np.abs(self.values).mean(axis=0)
        return pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": imp}) \
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def grouped_importance(self, raw_columns: list[str]) -> pd.DataFrame:
        """Importance aggregated to the original business features.

        One-hot encoding spreads one concept over several columns, so per-column
        importance understates it. Contract split three ways looks less important
        than it is until the parts are summed.
        """
        df = self.global_importance()
        df["base_feature"] = [_base_feature(f, raw_columns) for f in df["feature"]]
        return df.groupby("base_feature", as_index=False)["mean_abs_shap"].sum() \
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def concept_importance(self, raw_columns: list[str]) -> pd.DataFrame:
        """Importance aggregated to business CONCEPTS.

        Uses the same grouping as the per-customer explanations, so the global and
        local views agree. Without this the global table reports `Contract` and
        `is_month_to_month` as two separate drivers when they are one commercial
        fact, and each looks weaker than the concept actually is.
        """
        df = self.grouped_importance(raw_columns)
        df["concept"] = [_FEATURE_TO_CONCEPT.get(f, f) for f in df["base_feature"]]
        return df.groupby("concept", as_index=False)["mean_abs_shap"].sum() \
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def base_estimator(pipeline: Pipeline):
    """Return the estimator inside the pipeline."""
    return pipeline.named_steps["model"]


def build_explainer(model, background: pd.DataFrame):
    """Pick the right SHAP explainer for the estimator.

    Model selection is data-dependent -- on this dataset the boosted tree and
    logistic regression are statistically indistinguishable, so a different split
    or dataset can legitimately select the linear model. Hard-coding
    ``TreeExplainer`` makes explainability crash exactly when that happens, so the
    explainer is chosen from the fitted estimator instead.

    Both are exact for their model class; no sampling approximation is used.
    """
    if hasattr(model, "get_booster") or hasattr(model, "estimators_"):
        return shap.TreeExplainer(model)
    if hasattr(model, "coef_"):
        return shap.LinearExplainer(model, background)
    raise TypeError(
        f"no exact SHAP explainer for {type(model).__name__}; add a case here "
        "rather than falling back to a sampling explainer silently")


def compute_shap(
    pipeline: Pipeline, X: pd.DataFrame, max_samples: int | None = 1000,
    seed: int = 42,
) -> ShapArtifacts:
    """Compute SHAP values for ``X`` under the pipeline's tree model."""
    rows = X
    if max_samples is not None and len(X) > max_samples:
        rows = X.sample(max_samples, random_state=seed)

    engineered = pipeline.named_steps["features"].transform(rows)
    encoded = pipeline.named_steps["prep"].transform(engineered)
    names = output_feature_names(pipeline)
    encoded_df = pd.DataFrame(encoded, columns=names, index=rows.index)

    explainer = build_explainer(base_estimator(pipeline), encoded_df)
    values = explainer.shap_values(encoded_df)
    if isinstance(values, list):          # older API returns one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:                  # (n, features, classes) -> positive class
        values = values[:, :, -1]

    base = explainer.expected_value
    base = float(np.ravel(base)[0]) if np.ndim(base) else float(base)

    return ShapArtifacts(values=np.asarray(values), encoded=encoded_df,
                         feature_names=names, base_value=base, raw=rows,
                         engineered=engineered)


def explain_customer(
    art: ShapArtifacts, position: int, raw_columns: list[str], top_n: int = 4,
    min_contribution: float = MIN_CONTRIBUTION,
) -> dict[str, object]:
    """Plain-language explanation for one customer.

    Contributions are summed to the business feature (so a three-way one-hot
    Contract is one factor, not three) and each factor is described using the
    customer's own value, which is what keeps the direction correct.
    """
    source = art.engineered if art.engineered is not None else art.raw
    row = source.iloc[position]

    contrib = pd.DataFrame({
        "feature": art.feature_names,
        "shap": art.values[position],
    })
    contrib["base_feature"] = [_base_feature(f, raw_columns) for f in contrib["feature"]]

    concepts = [_concept_of(f, row) for f in contrib["base_feature"]]
    contrib["concept"] = [c for c, _ in concepts]
    contrib["lead"] = [lead for _, lead in concepts]

    agg = contrib.groupby("concept", as_index=False).agg(
        shap=("shap", "sum"), lead=("lead", "first"))

    agg["reason"] = [
        describe(r.lead, row[r.lead]) if r.lead in source.columns else r.concept
        for r in agg.itertuples()
    ]
    agg = agg[agg["shap"].abs() >= min_contribution]

    increasing = agg[agg["shap"] > 0].nlargest(top_n, "shap")
    reducing = agg[agg["shap"] < 0].nsmallest(top_n, "shap")

    def pack(frame: pd.DataFrame) -> list[dict[str, object]]:
        return [{"feature": r.concept, "reason": r.reason,
                 "contribution": round(float(r.shap), 4)}
                for r in frame.itertuples()]

    return {
        "top_risk_factors": pack(increasing),
        "top_protective_factors": pack(reducing),
        "note": "Contributions are in log-odds and are not additive in probability; "
                "they rank why this customer scored as they did. Factors below "
                f"|{min_contribution}| are omitted as indistinguishable from noise.",
    }


def save_global_plots(art: ShapArtifacts, figures_dir, raw_columns: list[str]) -> list[str]:
    """Write the SHAP summary (beeswarm) and grouped-importance figures."""
    written: list[str] = []

    plt.figure()
    shap.summary_plot(art.values, art.encoded, show=False, max_display=15)
    plt.title("SHAP summary: how each feature moves churn risk", fontsize=11)
    plt.tight_layout()
    p = figures_dir / "07_shap_summary.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    written.append(str(p))

    grouped = art.grouped_importance(raw_columns).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(grouped["base_feature"], grouped["mean_abs_shap"], color="#2b6cb0")
    ax.set_xlabel("mean |SHAP| (log-odds), summed over encoded columns")
    ax.set_title("Global feature importance, aggregated to business features")
    fig.tight_layout()
    p = figures_dir / "08_shap_importance_grouped.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(str(p))
    return written


def save_waterfall(
    art: ShapArtifacts, position: int, figures_dir, filename: str,
) -> str:
    """Write a waterfall plot for one customer."""
    exp = shap.Explanation(
        values=art.values[position],
        base_values=art.base_value,
        data=art.encoded.iloc[position].to_numpy(),
        feature_names=art.feature_names,
    )
    plt.figure()
    shap.plots.waterfall(exp, max_display=12, show=False)
    plt.tight_layout()
    p = figures_dir / filename
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    return str(p)


def representative_customers(
    probabilities: np.ndarray, art: ShapArtifacts, n_per_band: int = 1,
    cfg: Config | None = None,
) -> dict[str, int]:
    """Pick one representative customer per risk band, by position in ``art``.

    Chooses the customer closest to each band's midpoint so the example is typical
    of the band rather than an extreme case.
    """
    from src.evaluation.segments import SEGMENT_ORDER, assign_risk_segments

    probs = probabilities[art.raw.index.to_numpy()] if len(probabilities) != len(art.raw) \
        else probabilities
    if cfg is None:
        raise ValueError("cfg is required to resolve risk bands")

    seg = assign_risk_segments(probs, cfg).to_numpy()
    picks: dict[str, int] = {}
    for band in SEGMENT_ORDER:
        idx = np.where(seg == band)[0]
        if len(idx) == 0:
            continue
        mid = float(np.median(probs[idx]))
        picks[band] = int(idx[np.argmin(np.abs(probs[idx] - mid))])
    return picks
