"""
Stage 2 -- the GeeksforGeeks pipeline as published.

Reproduced so its numbers can be checked rather than trusted. Deviations are
limited to what will not run at all on current library versions, and each is
marked.
"""
from __future__ import annotations

import time
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.metrics import roc_auc_score as ras
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.dataset import load

warnings.filterwarnings("ignore")


def build_xy(df: pd.DataFrame):
    """Exactly the article's preprocessing: one-hot `type`, drop the identifiers."""
    type_new = pd.get_dummies(df["type"], drop_first=True)
    data_new = pd.concat([df, type_new], axis=1)
    X = data_new.drop(["isFraud", "type", "nameOrig", "nameDest"], axis=1)
    y = data_new["isFraud"]
    return X, y


def run(path: str | None = None) -> pd.DataFrame:
    df = load(path)

    X, y = build_xy(df)
    print(f"X.shape, y.shape -> {(X.shape, y.shape)}")
    print(f"features: {list(X.columns)}")

    # The article's random split. See src/improved_pipeline.py for why this is
    # the wrong split for a column that counts hours.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"train {X_train.shape}  test {X_test.shape}")

    # The article also lists SVC but never fits it -- and could not: a kernel SVC
    # is at least O(n^2), which is hopeless at this row count.
    models = [
        LogisticRegression(),
        XGBClassifier(),
        RandomForestClassifier(n_estimators=7, criterion="entropy", random_state=7),
    ]

    rows = []
    for model in models:
        name = type(model).__name__
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_s = time.time() - t0

        tr = ras(y_train, model.predict_proba(X_train)[:, 1])
        p_te = model.predict_proba(X_test)[:, 1]
        te = ras(y_test, p_te)
        ap = average_precision_score(y_test, p_te)

        # The article labels these "Accuracy". They are ROC AUC.
        print(f"\n{name}  (fit {fit_s:.1f}s)")
        print(f"  train ROC AUC : {tr}")
        print(f"  test  ROC AUC : {te}")
        print(f"  test  PR  AUC : {ap:.6f}   <- omitted by the article")
        rows.append(
            {"model": name, "train_roc_auc": tr, "test_roc_auc": te,
             "test_pr_auc": ap, "fit_s": round(fit_s, 1)}
        )

    tn, fp, fn, tp = confusion_matrix(y_test, models[1].predict(X_test)).ravel()
    print(f"\nXGBClassifier @ threshold 0.5: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
    if tp + fp:
        print(f"  precision={tp / (tp + fp):.4f}  recall={tp / (tp + fn):.4f}")

    out = pd.DataFrame(rows)
    print(f"\n{out.to_string(index=False)}")
    return out


if __name__ == "__main__":
    run()
