"""Generate a schema-faithful synthetic stand-in for the churn dataset.

    python -m src.data.generate_sample

The repository gitignores ``*.csv``, so CI has no data. This writes a file with
the published schema AND the published quirks, so the audit, pipeline and tests
are genuinely exercised offline:

  * ``TotalCharges`` as text, including blank cells on ``tenure == 0`` rows
  * structural ``'No internet service'`` / ``'No phone service'`` categories that
    stay consistent with their parent service column
  * feature-identical duplicate rows, including one label-conflicting pair
  * a churn signal that depends on contract, tenure and payment method, so the
    models have something learnable and metrics are non-degenerate

It is NOT a substitute for the real data. Reported metrics come from the real
file; tests that assert real-data figures are marked and skipped here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.schema import EXPECTED_COLUMNS

N_ROWS = 1200


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build(n: int = N_ROWS, seed: int = 7) -> pd.DataFrame:
    """Build a synthetic frame matching the published schema."""
    rng = np.random.default_rng(seed)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    tenure = np.where(
        contract == "Month-to-month",
        rng.integers(0, 40, n),
        rng.integers(12, 73, n),
    ).astype(int)

    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"], n, p=[0.34, 0.23, 0.22, 0.21])

    def addon() -> np.ndarray:
        """Add-on that respects the structural 'No internet service' category."""
        v = rng.choice(["Yes", "No"], n, p=[0.42, 0.58])
        return np.where(internet == "No", "No internet service", v)

    df = pd.DataFrame({
        "customerID": [f"{i:04d}-SYNTH" for i in range(n)],
        "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": rng.choice(["Yes", "No"], n, p=[0.48, 0.52]),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service",
                                  rng.choice(["Yes", "No"], n)),
        "InternetService": internet,
        "OnlineSecurity": addon(),
        "OnlineBackup": addon(),
        "DeviceProtection": addon(),
        "TechSupport": addon(),
        "StreamingTV": addon(),
        "StreamingMovies": addon(),
        "Contract": contract,
        "PaperlessBilling": rng.choice(["Yes", "No"], n, p=[0.59, 0.41]),
        "PaymentMethod": payment,
    })

    base = np.where(internet == "Fiber optic", 70.0,
                    np.where(internet == "DSL", 45.0, 20.0))
    addons = sum((df[c] == "Yes").to_numpy(dtype=float) for c in (
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"))
    monthly = base + 5.0 * addons + (phone == "Yes") * 8.0 + rng.normal(0, 3, n)
    df["MonthlyCharges"] = np.round(np.clip(monthly, 18.0, None), 2)

    # Lifetime billed, with drift so it is not an exact product of the two.
    total = df["MonthlyCharges"].to_numpy() * tenure * rng.normal(1.0, 0.02, n)
    df["TotalCharges"] = np.round(np.clip(total, 0, None), 2).astype(str)

    # A learnable churn signal: contract dominates, then tenure, then payment.
    logit = (
        -1.2
        + 1.9 * (contract == "Month-to-month")
        - 1.1 * (contract == "Two year")
        - 0.045 * tenure
        + 0.7 * (payment == "Electronic check")
        + 0.5 * (internet == "Fiber optic")
        - 0.4 * (df["TechSupport"] == "Yes").to_numpy()
        + 0.010 * (df["MonthlyCharges"].to_numpy() - 65.0)
        + rng.normal(0, 0.45, n)
    )
    df["Churn"] = np.where(rng.random(n) < _sigmoid(logit), "Yes", "No")

    # Quirk 1: never-billed customers -- tenure 0, blank TotalCharges, no churn.
    new_idx = rng.choice(n, 6, replace=False)
    df.loc[new_idx, "tenure"] = 0
    df.loc[new_idx, "TotalCharges"] = " "
    df.loc[new_idx, "Churn"] = "No"

    # Quirk 2: feature-identical duplicates, and one label-conflicting pair.
    dup = df.iloc[[3, 11, 27, 44]].copy()
    dup["customerID"] = [f"D{i:03d}-SYNTH" for i in range(len(dup))]
    conflict = df.iloc[[5]].copy()
    conflict["customerID"] = ["C000-SYNTH"]
    conflict["Churn"] = np.where(conflict["Churn"] == "Yes", "No", "Yes")

    out = pd.concat([df, dup, conflict], ignore_index=True)
    return out.loc[:, list(EXPECTED_COLUMNS)]


def main() -> int:
    cfg = load_config()
    dest = cfg.resolve("data", "sample_file")
    dest.parent.mkdir(parents=True, exist_ok=True)

    df = build()
    df.to_csv(dest, index=False)

    churn_rate = df["Churn"].eq("Yes").mean()
    print(f"wrote {dest} ({len(df)} rows, churn rate {churn_rate:.1%})")
    print("reproduces: text TotalCharges with blanks on tenure=0, structural service")
    print("            categories, duplicate rows incl. one label conflict.")
    print("NOTE: synthetic. Reported metrics come from the real dataset "
          "(`make download`).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
