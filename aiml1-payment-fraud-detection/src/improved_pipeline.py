"""
Stage 4 -- the corrected pipeline.

Fixes, each of which the article gets wrong:
  temporal split rather than random; class imbalance actually handled; PR AUC as
  the headline metric; engineered balance residuals; features scaled for the
  linear model; a threshold tuned on validation and reported as an analyst
  review budget rather than left at 0.5.

The result is ~1.000 PR AUC, which is not a success. Stage 5 explains why.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.dataset import add_features, design_matrix, load, temporal_split

warnings.filterwarnings("ignore")
OUT = Path("outputs")


def run(path: str | None = None, save_figures: bool = False) -> dict:
    df = load(path)
    X, y = design_matrix(df)
    print(f"{X.shape[1]} features: {list(X.columns)}")

    sp = temporal_split(df)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    step = df["step"].to_numpy()
    print(f"\ntemporal split (train <= step {sp['cut_train']}, val <= {sp['cut_val']}):")
    for nm, m in (("train", tr), ("val", va), ("test", te)):
        print(
            f"  {nm:5s} n={m.sum():>9,}  frauds={int(y[m].sum()):>5,}  "
            f"rate={y[m].mean():.4%}  steps {step[m].min()}-{step[m].max()}"
        )

    X_tr, X_va, X_te = X[tr], X[va], X[te]
    y_tr, y_va, y_te = y[tr], y[va], y[te]
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"\nscale_pos_weight = {spw:.1f}")

    models = {
        "LogisticRegression(scaled, balanced)": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "RandomForest(balanced)": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=7,
        ),
        "XGBClassifier(scale_pos_weight)": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric="aucpr", n_jobs=-1, random_state=7,
        ),
    }

    rows, probs = [], {}
    print()
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        fit_s = time.time() - t0
        p_va = model.predict_proba(X_va)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]
        probs[name] = (p_va, p_te)
        rows.append({
            "model": name,
            "test_pr_auc": average_precision_score(y_te, p_te),
            "test_roc_auc": roc_auc_score(y_te, p_te),
            "val_pr_auc": average_precision_score(y_va, p_va),
            "fit_s": round(fit_s, 1),
        })
        print(f"  {name:38s} PR-AUC={rows[-1]['test_pr_auc']:.6f} "
              f"ROC-AUC={rows[-1]['test_roc_auc']:.6f}  ({fit_s:.0f}s)")

    # A trivial ranker, so the headline numbers have something to be compared to.
    naive = np.nan_to_num(add_features(df).loc[te, "amtRatioOrig"].to_numpy())
    print(f"  {'baseline: rank by amount/oldbalance':38s} "
          f"PR-AUC={average_precision_score(y_te, naive):.6f} "
          f"ROC-AUC={roc_auc_score(y_te, naive):.6f}")

    summary = pd.DataFrame(rows).sort_values("test_pr_auc", ascending=False)
    best = str(summary.iloc[0]["model"])
    p_va, p_te = probs[best]
    print(f"\nbest by PR-AUC: {best}")

    # ---- threshold chosen on validation, never on test ----
    prec, rec, thr = precision_recall_curve(y_va, p_va)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    t_best = float(thr[int(np.nanargmax(f1[:-1]))]) if len(thr) else 0.5
    print(f"\nvalidation-tuned threshold: {t_best:.6f}")

    ops = []
    for tag, t in (("tuned", t_best), ("article's default 0.5", 0.5)):
        tn, fp, fn, tp = confusion_matrix(y_te, (p_te >= t).astype(int)).ravel()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        print(f"\n{tag} (threshold={t:.6f})")
        print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
        print(f"  precision={p:.4f}  recall={r:.4f}")
        ops.append({"tag": tag, "threshold": t, "tn": int(tn), "fp": int(fp),
                    "fn": int(fn), "tp": int(tp), "precision": p, "recall": r})

    if ops[0]["fp"] == 0:
        print(
            "\nZero false positives across the whole test set is not a result to\n"
            "celebrate -- it means a near-deterministic rule exists. See stage 5."
        )

    # ---- what a fraud team actually asks ----
    print("\nrecall under a fixed analyst review budget:")
    order = np.argsort(-p_te)
    budget = []
    for frac in (0.0001, 0.0005, 0.001, 0.005, 0.01):
        k = max(int(frac * len(y_te)), 1)
        caught = int(y_te[order[:k]].sum())
        budget.append({"budget_frac": frac, "alerts": k, "frauds_caught": caught,
                       "recall": caught / max(y_te.sum(), 1), "precision": caught / k})
        print(f"  top {frac:.2%} ({k:>7,} alerts) -> {caught:>5,}/{int(y_te.sum()):,} "
              f"frauds  recall={caught / max(y_te.sum(), 1):.4f}  precision={caught / k:.4f}")

    OUT.mkdir(exist_ok=True)
    summary.to_csv(OUT / "improved_results.csv", index=False)
    pd.DataFrame(budget).to_csv(OUT / "alert_budget.csv", index=False)
    result = {"best_model": best, "scale_pos_weight": spw,
              "cut_train": sp["cut_train"], "cut_val": sp["cut_val"],
              "operating_points": ops}
    (OUT / "improved_summary.json").write_text(json.dumps(result, indent=2))

    if save_figures:
        _figures(y_te, probs, models, X.columns, t_best, best)

    print(f"\nwrote metrics to {OUT}/")
    return {**result, "summary": summary, "budget": budget}


def _figures(y_te, probs, models, columns, t_best, best) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 6))
    for name, (_, pt) in probs.items():
        pr, rc, _ = precision_recall_curve(y_te, pt)
        plt.plot(rc, pr, label=f"{name.split('(')[0]} (AP={average_precision_score(y_te, pt):.3f})")
    plt.axhline(y_te.mean(), ls="--", c="grey", label=f"random (AP={y_te.mean():.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall on the temporal test set")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "pr_curves.png", dpi=110)
    plt.close()

    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay(
        confusion_matrix(y_te, (probs[best][1] >= t_best).astype(int)),
        display_labels=["legit", "fraud"],
    ).plot(cmap="Blues", values_format=",d")
    plt.title(f"{best.split('(')[0]} at the tuned threshold")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_tuned.png", dpi=110)
    plt.close()

    xgb = models["XGBClassifier(scale_pos_weight)"]
    pd.Series(xgb.feature_importances_, index=columns).sort_values().plot.barh(figsize=(8, 7))
    plt.title("XGBoost feature importance (gain)")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=110)
    plt.close()
    print(f"wrote figures to {OUT}/")


if __name__ == "__main__":
    run(save_figures=True)
