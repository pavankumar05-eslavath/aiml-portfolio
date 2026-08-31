"""Dataset schema and data dictionary.

Declaring the schema in code (rather than inferring it at runtime) means a
change in the source file fails loudly instead of silently altering the model's
input space.
"""
from __future__ import annotations

from dataclasses import dataclass

ID_COLUMN = "customerID"
TARGET = "Churn"

#: Continuous inputs, as published.
NUMERIC_COLUMNS: tuple[str, ...] = ("tenure", "MonthlyCharges", "TotalCharges")

#: SeniorCitizen ships as int64 0/1 but is categorical in meaning. Kept binary
#: numeric because one-hot encoding a 2-level flag adds a redundant column.
BINARY_NUMERIC_COLUMNS: tuple[str, ...] = ("SeniorCitizen",)

#: Categorical inputs, as published.
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
)

#: The six add-on services, each conditional on InternetService != 'No'.
#: Used to build a service-adoption count.
INTERNET_ADDON_COLUMNS: tuple[str, ...] = (
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
)

#: Categories that encode "the parent service is absent" rather than a customer
#: choice. They are structurally determined by PhoneService / InternetService,
#: which is why a naive "count of Yes" over these columns is not the same as
#: "count of services the customer declined".
STRUCTURAL_ABSENT_CATEGORIES: tuple[str, ...] = (
    "No internet service", "No phone service",
)

#: Automatic payment methods, used to derive an autopay flag.
AUTOPAY_METHODS: tuple[str, ...] = (
    "Bank transfer (automatic)", "Credit card (automatic)",
)

EXPECTED_COLUMNS: tuple[str, ...] = (
    ID_COLUMN, *CATEGORICAL_COLUMNS, *BINARY_NUMERIC_COLUMNS,
    *NUMERIC_COLUMNS, TARGET,
)

#: Allowed values per categorical column. Anything else is an invalid category.
ALLOWED_VALUES: dict[str, tuple[str, ...]] = {
    "gender": ("Female", "Male"),
    "Partner": ("Yes", "No"),
    "Dependents": ("Yes", "No"),
    "PhoneService": ("Yes", "No"),
    "MultipleLines": ("Yes", "No", "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "OnlineSecurity": ("Yes", "No", "No internet service"),
    "OnlineBackup": ("Yes", "No", "No internet service"),
    "DeviceProtection": ("Yes", "No", "No internet service"),
    "TechSupport": ("Yes", "No", "No internet service"),
    "StreamingTV": ("Yes", "No", "No internet service"),
    "StreamingMovies": ("Yes", "No", "No internet service"),
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": ("Yes", "No"),
    "PaymentMethod": (
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ),
    TARGET: ("Yes", "No"),
}

#: Physically impossible ranges. tenure of 0 is legal (a new customer) but
#: negative tenure or negative charges are not.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "tenure": (0, 72),
    "MonthlyCharges": (0, 1_000),
    "TotalCharges": (0, 100_000),
    "SeniorCitizen": (0, 1),
}


@dataclass(frozen=True)
class ColumnSpec:
    """One row of the data dictionary."""

    name: str
    kind: str
    description: str
    notes: str = ""


#: Human-readable data dictionary, rendered by src/data/audit.py.
DATA_DICTIONARY: tuple[ColumnSpec, ...] = (
    ColumnSpec(ID_COLUMN, "identifier",
               "Unique customer reference.",
               "Dropped before modelling: an identifier carries no generalisable signal "
               "and lets a tree memorise individuals."),
    ColumnSpec("gender", "categorical", "Female / Male."),
    ColumnSpec("SeniorCitizen", "binary", "1 if the customer is a senior citizen."),
    ColumnSpec("Partner", "categorical", "Has a partner."),
    ColumnSpec("Dependents", "categorical", "Has dependents."),
    ColumnSpec("tenure", "numeric", "Months the customer has been with the company.",
               "0 for 11 never-billed customers; see the missing-value analysis."),
    ColumnSpec("PhoneService", "categorical", "Subscribes to phone service."),
    ColumnSpec("MultipleLines", "categorical", "Multiple phone lines.",
               "'No phone service' is structural, set by PhoneService."),
    ColumnSpec("InternetService", "categorical", "DSL / Fiber optic / No."),
    ColumnSpec("OnlineSecurity", "categorical", "Online-security add-on.",
               "'No internet service' is structural."),
    ColumnSpec("OnlineBackup", "categorical", "Online-backup add-on.",
               "'No internet service' is structural."),
    ColumnSpec("DeviceProtection", "categorical", "Device-protection add-on.",
               "'No internet service' is structural."),
    ColumnSpec("TechSupport", "categorical", "Tech-support add-on.",
               "'No internet service' is structural."),
    ColumnSpec("StreamingTV", "categorical", "TV streaming add-on.",
               "'No internet service' is structural."),
    ColumnSpec("StreamingMovies", "categorical", "Movie streaming add-on.",
               "'No internet service' is structural."),
    ColumnSpec("Contract", "categorical", "Month-to-month / One year / Two year.",
               "Strongest single predictor in this dataset."),
    ColumnSpec("PaperlessBilling", "categorical", "Paperless billing enabled."),
    ColumnSpec("PaymentMethod", "categorical", "How the customer pays.",
               "Electronic check is associated with markedly higher churn."),
    ColumnSpec("MonthlyCharges", "numeric", "Current monthly charge.",
               "Used as the revenue-at-risk rate in the decision layer."),
    ColumnSpec("TotalCharges", "numeric", "Lifetime amount billed to date.",
               "Published as text; 11 blanks. Correlates 0.9996 with "
               "tenure x MonthlyCharges -- redundant, not leaky."),
    ColumnSpec(TARGET, "target", "Yes if the customer left within the last month.",
               "Snapshot label with no timestamp, so no out-of-time split is possible."),
)


def feature_columns() -> list[str]:
    """Model input columns, i.e. everything except the identifier and target."""
    return [c for c in EXPECTED_COLUMNS if c not in (ID_COLUMN, TARGET)]
