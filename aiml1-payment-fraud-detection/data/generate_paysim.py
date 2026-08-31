"""
Generate a small PaySim-shaped dataset.

The real PaySim CSV is 480 MB, so it cannot live in git or run in CI. This
generator reproduces the properties that the analysis depends on, at ~1/30th the
size:

  * identical schema and transaction-type mix
  * fraud confined to TRANSFER and CASH_OUT
  * a ~0.13% fraud rate (1:774 imbalance)
  * and, critically, PaySim's *fraud script*: the fraud agent empties the
    victim's account, so `amount == oldbalanceOrg` and `newbalanceOrig == 0`

That last property is the whole point of this project -- it is the leak that lets
a three-clause rule beat a gradient-boosted ensemble. Regenerating it here means
the tests and CI exercise the finding rather than trusting a stored number.

Numbers in INSIGHTS.md come from the real 6.36M-row CSV (`make download`); this
file exists so `make all` and `pytest` work with no download.

What it does NOT reproduce, and you should not read conclusions off it:

  * The legitimate traffic here is cleaner than PaySim's, so the problem is
    easier. Logistic regression reaches ~1.000 PR AUC on this stand-in but only
    0.8455 on the real file -- the real data has messier balance behaviour that a
    linear model genuinely struggles with.
  * The `base_score` saturation that breaks the article's XGBoost result needs the
    real data's scale and noise to show up as a 0.29 AUC drop. The tests here pin
    the *mechanism* (that base_score resolves to the class prior), not the
    magnitude.
  * Fraud counts per type follow from random assignment, so the TRANSFER/CASH_OUT
    split will not match the real file's near-even 4,097/4,116.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Type mix measured on the real 6,362,620-row PaySim CSV.
TYPE_MIX = {
    "CASH_OUT": 0.3517,
    "PAYMENT": 0.3381,
    "CASH_IN": 0.2199,
    "TRANSFER": 0.0838,
    "DEBIT": 0.0065,
}
FRAUD_RATE = 0.00129
MAX_STEP = 743


def generate(n_rows: int = 200_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    types = rng.choice(list(TYPE_MIX), size=n_rows, p=list(TYPE_MIX.values()))

    # PaySim's step is an hour counter over ~31 days with a strong diurnal cycle;
    # most volume falls in the first ~400 steps.
    step = np.clip(
        (rng.beta(2.2, 2.6, n_rows) * MAX_STEP).astype(int) + 1, 1, MAX_STEP
    )

    amount = np.round(rng.lognormal(mean=9.4, sigma=1.35, size=n_rows), 2)

    # Sender balances. A large share of accounts sit at zero, as in the real data.
    oldbalanceOrg = np.where(
        rng.random(n_rows) < 0.35,
        0.0,
        np.round(rng.lognormal(mean=10.2, sigma=1.7, size=n_rows), 2),
    )

    # Legitimate senders move a fraction of what they hold, so their books
    # reconcile: newbalanceOrig == oldbalanceOrg - amount, floored at zero.
    newbalanceOrig = np.maximum(oldbalanceOrg - amount, 0.0).round(2)

    oldbalanceDest = np.where(
        rng.random(n_rows) < 0.28,
        0.0,
        np.round(rng.lognormal(mean=11.0, sigma=1.9, size=n_rows), 2),
    )
    newbalanceDest = (oldbalanceDest + amount).round(2)

    # PAYMENT goes to a merchant, whose balances PaySim leaves at zero.
    is_payment = types == "PAYMENT"
    oldbalanceDest = np.where(is_payment, 0.0, oldbalanceDest)
    newbalanceDest = np.where(is_payment, 0.0, newbalanceDest)

    isFraud = np.zeros(n_rows, dtype=np.int8)

    # ---- the fraud script -------------------------------------------------
    # Fraud can only be a TRANSFER or a CASH_OUT, and only from a funded
    # account. The agent takes the entire balance.
    eligible = np.isin(types, ["TRANSFER", "CASH_OUT"]) & (oldbalanceOrg > 0)
    n_fraud = max(round(n_rows * FRAUD_RATE), 1)
    idx = rng.choice(np.flatnonzero(eligible), size=n_fraud, replace=False)

    isFraud[idx] = 1
    amount[idx] = oldbalanceOrg[idx]      # takes exactly the whole balance
    newbalanceOrig[idx] = 0.0            # and leaves the account empty
    # PaySim does not credit the mule account on the fraudulent leg, which is
    # why errBalDest is uninformative but errBalOrig is not.
    oldbalanceDest[idx] = 0.0
    newbalanceDest[idx] = 0.0

    # ---- near-misses ------------------------------------------------------
    # A handful of legitimate transactions also empty an account. Without these
    # the rule would be trivially perfect; the real data has exactly 1 such
    # false positive in 6.36M rows, so keep this population very small.
    legit_drain = rng.choice(
        np.flatnonzero(eligible & (isFraud == 0)),
        size=max(int(n_rows * 0.00002), 1),
        replace=False,
    )
    amount[legit_drain] = oldbalanceOrg[legit_drain]
    newbalanceOrig[legit_drain] = 0.0

    df = pd.DataFrame(
        {
            "step": step,
            "type": types,
            "amount": amount.round(2),
            "nameOrig": [f"C{i}" for i in rng.integers(1e8, 1e9, n_rows)],
            "oldbalanceOrg": oldbalanceOrg.round(2),
            "newbalanceOrig": newbalanceOrig.round(2),
            "nameDest": [
                ("M" if p else "C") + str(i)
                for p, i in zip(is_payment, rng.integers(1e8, 1e9, n_rows), strict=True)
            ],
            "oldbalanceDest": oldbalanceDest.round(2),
            "newbalanceDest": newbalanceDest.round(2),
            "isFraud": isFraud,
        }
    )
    return df.sort_values("step", ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/paysim_synthetic.csv")
    args = ap.parse_args()

    df = generate(args.rows, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"wrote {args.out}: {len(df):,} rows")
    print(f"  frauds      : {df.isFraud.sum():,} ({df.isFraud.mean():.4%})")
    print(f"  imbalance   : 1:{(df.isFraud == 0).sum() / max(df.isFraud.sum(), 1):.0f}")
    print("  frauds by type:")
    print(
        df.groupby("type")["isFraud"]
        .agg(n="size", frauds="sum")
        .to_string()
        .replace("\n", "\n    ")
    )


if __name__ == "__main__":
    main()
