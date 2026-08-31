"""
Stage 5 -- the audit, and the point of the project.

Stage 4 produced ~1.000 PR AUC with zero false positives. Near-perfect scores
mean leakage until proven otherwise, so this goes looking for the shortcut.

It finds it: PaySim scripts its fraud agents to empty the victim's account,
leaving an exact arithmetic signature. A three-clause rule matches a
gradient-boosted ensemble.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.dataset import leak_rule, load

warnings.filterwarnings("ignore")


def run(path: str | None = None) -> dict:
    df = load(path)
    y = df["isFraud"].to_numpy()
    fraud = y == 1
    print(f"frauds={fraud.sum():,}  legit={(~fraud).sum():,}")

    exact = (np.isclose(df["amount"], df["oldbalanceOrg"]) & (df["oldbalanceOrg"] > 0)).to_numpy()
    drained = ((df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)).to_numpy()
    err = (df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]).to_numpy()

    print("\n--- does fraud always move the sender's entire balance? ---")
    print(pd.crosstab(exact, y, rownames=["amount==oldbalanceOrg"], colnames=["isFraud"]))
    print(f"  P(fraud | amount==oldbalanceOrg) = {y[exact].mean():.4f}")
    print(f"  P(amount==oldbalanceOrg | fraud) = {exact[fraud].mean():.4f}")

    print("\n--- the sender-side balance residual ---")
    print(pd.crosstab(np.isclose(err, 0), y, rownames=["errBalOrig==0"], colnames=["isFraud"]))

    print("\n--- the rule ---")
    rule = leak_rule(df)
    tp = int((rule & fraud).sum())
    fp = int((rule & ~fraud).sum())
    fn = int((~rule & fraud).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print("  newbalanceOrig == 0  AND  oldbalanceOrg > 0")
    print("  AND type in (TRANSFER, CASH_OUT)  AND  amount == oldbalanceOrg")
    print(f"\n  precision={precision:.4f}  recall={recall:.4f}")
    print(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}   -- no model involved")

    print("\n--- single-signal rankings ---")
    scores = {}
    for nm, v in {
        "amount == oldbalanceOrg": exact.astype(float),
        "drainedOrig": drained.astype(float),
        "the 3-clause rule": rule.astype(float),
        "-errBalOrig": -err,
        "amount": df["amount"].to_numpy(float),
    }.items():
        v = np.nan_to_num(v)
        scores[nm] = {"roc_auc": roc_auc_score(y, v), "pr_auc": average_precision_score(y, v)}
        print(f"  {nm:26s} ROC-AUC={scores[nm]['roc_auc']:.4f}  PR-AUC={scores[nm]['pr_auc']:.4f}")

    print(
        "\nCONCLUSION\n"
        "----------\n"
        "The ~0.999 headline scores -- the article's and the corrected pipeline's\n"
        "alike -- measure how well a model rediscovers the simulator's fraud\n"
        "script, not how well it would detect real payment fraud. Real fraud does\n"
        "not announce itself with exact-balance arithmetic.\n\n"
        "Treat PaySim as an exercise in pipeline mechanics at realistic data\n"
        "volume. Do not cite its scores as evidence a model is production-ready."
    )
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn,
            "signals": scores}


if __name__ == "__main__":
    run()
