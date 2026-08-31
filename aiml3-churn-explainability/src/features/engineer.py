"""Feature engineering.

Every feature here is **row-wise**: it depends only on the values in its own row,
never on an aggregate over the dataset. That property is what makes it safe to
apply before the split, and it is why this transformer is stateless — ``fit`` has
nothing to learn. A target-encoded or dataset-mean feature would need to be
fitted inside CV folds to avoid leakage; none is used.

The transformer is still placed inside the sklearn pipeline rather than applied
as a one-off script, so training and inference cannot drift apart.

Each feature is documented in FEATURE_DOCS with the evidence that motivated it.
Nothing here is generated speculatively: the EDA numbers quoted in the rationales
are measured on the training split (see reports/figures).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.data.schema import AUTOPAY_METHODS, INTERNET_ADDON_COLUMNS

TENURE_BINS = [-0.1, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"]

#: name -> (kind, rationale). Rendered into reports/model_report.md.
FEATURE_DOCS: dict[str, tuple[str, str]] = {
    "tenure_band": (
        "categorical",
        "Churn is heavily front-loaded: 53.3% in the first 6 months against 9.2% at "
        "49-72 months. Banding gives the linear model the step change it cannot "
        "express from raw tenure, and makes the risk legible to a retention team.",
    ),
    "is_new_customer": (
        "binary",
        "Isolates the 0-6 month window, the single riskiest period. Retention playbooks "
        "for new customers (onboarding) differ from those for established ones.",
    ),
    "n_addons": (
        "numeric",
        "Count of the six internet add-ons held. Churn falls monotonically with it: "
        "50.2% at zero add-ons to 4.9% at six. This is the clearest stickiness signal "
        "in the data — each additional service is another switching cost.",
    ),
    "has_internet": (
        "binary",
        "Separates 'declined an add-on' from 'cannot have one'. The add-on columns use a "
        "structural 'No internet service' category, so without this flag n_addons=0 "
        "conflates two different customers.",
    ),
    "addon_adoption_rate": (
        "numeric",
        "n_addons as a share of those available to that customer (0 when no internet). "
        "Normalises adoption so internet and non-internet customers are comparable.",
    ),
    "is_month_to_month": (
        "binary",
        "Contract is the strongest single predictor (AUC 0.739 alone); month-to-month "
        "churns at 43.1% versus 2.5% on two-year terms. An explicit flag lets the linear "
        "model use it directly and makes the SHAP output readable.",
    ),
    "is_autopay": (
        "binary",
        "Electronic check churns at 45.5%, far above the automatic methods. An automatic "
        "payment instruction is both a friction reducer and a commitment signal.",
    ),
    "realized_arpu": (
        "numeric",
        "TotalCharges / tenure: what the customer has actually averaged per month, as "
        "opposed to their current list price. Captures the effect of past discounts.",
    ),
    "price_vs_history": (
        "numeric",
        "MonthlyCharges / realized_arpu. Above 1 means the customer now pays more than "
        "their historical average — a recent price rise or expiring promotion, which is a "
        "recognised churn trigger. This is the feature a raw price level cannot express.",
    ),
    "billing_discrepancy": (
        "numeric",
        "TotalCharges - (tenure x MonthlyCharges). The audit found TotalCharges is 99.91% "
        "explained by that product, so the residual is the only genuinely new information "
        "in the column. Using the residual instead of the raw value avoids feeding the "
        "model a third copy of the same signal.",
    ),
    "charges_per_addon": (
        "numeric",
        "MonthlyCharges / (n_addons + 1): price paid per unit of service received. A "
        "value-for-money proxy — high spend with few services is a plausible grievance.",
    ),
    "household_size": (
        "numeric",
        "Partner + Dependents (0-2). A household account is harder to move than a single "
        "line, so it proxies switching cost.",
    ),
    "tenure_x_month_to_month": (
        "numeric",
        "Interaction between tenure and the month-to-month flag. Included because the "
        "tenure effect is contract-dependent (the heatmap shows month-to-month churn "
        "falling 56%->26% across tenure while two-year stays near 0%). Trees can discover "
        "this; the linear model cannot without being given it.",
    ),
}

#: Engineered columns by type, consumed by the preprocessor.
ENGINEERED_NUMERIC: tuple[str, ...] = (
    "n_addons", "addon_adoption_rate", "realized_arpu", "price_vs_history",
    "billing_discrepancy", "charges_per_addon", "household_size",
    "tenure_x_month_to_month",
)
ENGINEERED_BINARY: tuple[str, ...] = (
    "is_new_customer", "has_internet", "is_month_to_month", "is_autopay",
)
ENGINEERED_CATEGORICAL: tuple[str, ...] = ("tenure_band",)


def _safe_divide(num: pd.Series, den: pd.Series, fill: float = 0.0) -> pd.Series:
    """Element-wise division that never yields inf or NaN.

    Both zero denominators (tenure 0, never-billed customers) and zero numerators
    occur in this dataset, so an unguarded division silently injects inf into the
    feature matrix, which most estimators reject and XGBoost quietly accepts.
    """
    out = num.astype(float).divide(den.astype(float).replace(0.0, np.nan))
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add domain features to the raw churn frame.

    Stateless by design: ``fit`` records only the output column order, so the same
    transformation is applied to a training set of 5,621 rows and to a single
    customer at inference time.
    """

    def __init__(self, *, drop_total_charges: bool = False) -> None:
        #: TotalCharges is 99.91% explained by tenure x MonthlyCharges. Dropping it
        #: is offered as an option for the linear model, where the collinearity
        #: destabilises coefficients. Off by default; trees are unaffected.
        self.drop_total_charges = drop_total_charges

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ChurnFeatureEngineer:
        self.feature_names_in_ = list(X.columns)
        self.feature_names_out_ = list(self.transform(X.head(1)).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the frame with engineered columns appended."""
        df = X.copy()

        tenure = df["tenure"].astype(float)
        monthly = df["MonthlyCharges"].astype(float)
        total = df["TotalCharges"].astype(float)

        df["tenure_band"] = pd.cut(tenure, bins=TENURE_BINS,
                                   labels=TENURE_LABELS).astype(str)
        df["is_new_customer"] = (tenure <= 6).astype(int)

        has_internet = df["InternetService"].ne("No")
        df["has_internet"] = has_internet.astype(int)
        n_addons = sum((df[c] == "Yes").astype(int) for c in INTERNET_ADDON_COLUMNS)
        df["n_addons"] = n_addons
        df["addon_adoption_rate"] = np.where(
            has_internet, n_addons / len(INTERNET_ADDON_COLUMNS), 0.0)

        df["is_month_to_month"] = df["Contract"].eq("Month-to-month").astype(int)
        df["is_autopay"] = df["PaymentMethod"].isin(AUTOPAY_METHODS).astype(int)

        # Realised average monthly spend. Zero for never-billed customers, which is
        # correct: they have no billing history, and the has_internet /
        # is_new_customer flags carry that fact.
        df["realized_arpu"] = _safe_divide(total, tenure, fill=0.0)
        # Fill 1.0 = "no change vs history", the neutral value, so never-billed
        # customers are not misread as having had a price cut.
        df["price_vs_history"] = _safe_divide(monthly, df["realized_arpu"], fill=1.0)
        df["billing_discrepancy"] = total - tenure * monthly
        df["charges_per_addon"] = monthly / (n_addons + 1)

        df["household_size"] = (df["Partner"].eq("Yes").astype(int)
                                + df["Dependents"].eq("Yes").astype(int))
        df["tenure_x_month_to_month"] = tenure * df["is_month_to_month"]

        if self.drop_total_charges:
            df = df.drop(columns=["TotalCharges"])

        if not np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError("engineered features contain inf or NaN")
        return df

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)
