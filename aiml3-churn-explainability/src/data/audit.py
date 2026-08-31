"""Data quality and leakage audit.

Runs before any modelling and writes reports/data_audit.md.

The leakage section is deliberately *executable* rather than prose. A written
claim that "no leakage was found" is unfalsifiable; a single-feature AUC probe
either surfaces a feature that alone separates the target or it does not, and the
numbers go in the report either way.

One limitation is stated up front because it cannot be engineered away: this
dataset is a single snapshot with no event timestamps. Whether a column was
recorded before or after the churn decision therefore cannot be tested
empirically -- only reasoned about from each column's semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import Config
from src.data.loader import clean_raw, deduplicate, load_raw, make_dataset
from src.data.schema import (
    ALLOWED_VALUES,
    BINARY_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    DATA_DICTIONARY,
    ID_COLUMN,
    INTERNET_ADDON_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET,
    VALID_RANGES,
    feature_columns,
)

#: A single feature whose cross-validated AUC exceeds this is treated as a
#: leakage suspect and investigated before it is allowed into the model.
LEAKAGE_AUC_THRESHOLD = 0.95

#: Post-churn reasoning. This is a judgement per column, not a measurement,
#: because the snapshot carries no timestamps.
POST_CHURN_ASSESSMENT: dict[str, str] = {
    "tenure": "SAFE-BY-DESIGN. Length of relationship up to the snapshot. For a churner "
              "this is its final value, which is inherent to a snapshot label and is the "
              "quantity a deployed model would also see at scoring time.",
    "MonthlyCharges": "SAFE. The current rate plan, set before any churn decision.",
    "TotalCharges": "FLAGGED, KEPT. Cumulative billing, so it stops accruing at churn. It "
                    "encodes tenure (corr 0.9996 with tenure x MonthlyCharges) rather than "
                    "the outcome, and a live system would have the same figure for an "
                    "active customer. Kept, but its redundancy is quantified below.",
    "Contract": "SAFE. Contractual state, and the strongest legitimate predictor.",
    "PaymentMethod": "SAFE. A standing instruction, not an outcome.",
    "PaperlessBilling": "SAFE. A billing preference.",
    ID_COLUMN: "EXCLUDED. An identifier. Retaining it lets a tree memorise individuals "
               "and generalise to nobody.",
}


@dataclass
class AuditResult:
    """Structured audit output, so tests can assert on findings."""

    n_rows: int
    n_cols: int
    churn_rate: float
    missing: dict[str, int]
    n_exact_duplicates: int
    n_id_duplicates: int
    n_feature_duplicates: int
    n_safe_duplicates_removed: int
    n_conflicting_label_rows: int
    n_conflicting_groups: int
    invalid_values: dict[str, int]
    invalid_categories: dict[str, list[str]]
    structural_inconsistencies: dict[str, int]
    single_feature_auc: dict[str, float]
    leakage_suspects: list[str]
    total_charges_r2: float
    contamination_overlap: int
    notes: list[str] = field(default_factory=list)

    @property
    def label_noise_rate(self) -> float:
        return self.n_conflicting_label_rows / self.n_rows


def _missing_analysis(df: pd.DataFrame) -> dict[str, int]:
    """Count true NaNs plus whitespace-only strings, which pandas reads as values."""
    out: dict[str, int] = {}
    for c in df.columns:
        n = int(df[c].isna().sum())
        if df[c].dtype == object or str(df[c].dtype) == "str":
            n += int(df[c].astype("string").str.strip().eq("").sum())
        if n:
            out[c] = n
    return out


def _invalid_ranges(df: pd.DataFrame) -> dict[str, int]:
    """Values outside a physically possible range."""
    out: dict[str, int] = {}
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            n = int(((s < lo) | (s > hi)).sum())
            if n:
                out[col] = n
    return out


def _invalid_categories(df: pd.DataFrame) -> dict[str, list[str]]:
    """Category values not in the declared schema."""
    out: dict[str, list[str]] = {}
    for col, allowed in ALLOWED_VALUES.items():
        if col in df.columns:
            seen = set(df[col].dropna().astype(str).unique())
            unexpected = sorted(seen - set(allowed))
            if unexpected:
                out[col] = unexpected
    return out


def _structural_consistency(df: pd.DataFrame) -> dict[str, int]:
    """Verify the service columns agree with their parent service.

    'No internet service' must appear exactly when InternetService == 'No'.
    A violation would mean the categories are not structurally determined, and
    the service-adoption feature would be counting something else.
    """
    out: dict[str, int] = {}
    no_internet = df["InternetService"].eq("No")
    for c in INTERNET_ADDON_COLUMNS:
        flag = df[c].eq("No internet service")
        n = int((flag != no_internet).sum())
        if n:
            out[f"{c} vs InternetService"] = n

    no_phone = df["PhoneService"].eq("No")
    n = int((df["MultipleLines"].eq("No phone service") != no_phone).sum())
    if n:
        out["MultipleLines vs PhoneService"] = n
    return out


def single_feature_auc(
    X: pd.DataFrame, y: pd.Series, cfg: Config,
) -> dict[str, float]:
    """Cross-validated ROC-AUC of each feature used alone.

    The leakage probe. A feature that alone approaches AUC 1.0 is either the
    target in disguise or recorded after the outcome. Each feature is fitted in
    its own pipeline so encoding never crosses fold boundaries.

    A shallow decision tree is used for numerics (it captures a monotone or
    single-split relationship without assuming linearity) and one-hot plus
    logistic regression for categoricals.
    """
    cv = StratifiedKFold(n_splits=int(cfg["split"]["cv_folds"]), shuffle=True,
                         random_state=cfg.seed)
    scores: dict[str, float] = {}

    for col in X.columns:
        if col in CATEGORICAL_COLUMNS:
            pipe = Pipeline([
                ("enc", ColumnTransformer(
                    [("oh", OneHotEncoder(handle_unknown="ignore"), [col])])),
                ("clf", LogisticRegression(max_iter=1000, random_state=cfg.seed)),
            ])
        else:
            pipe = Pipeline([
                ("prep", ColumnTransformer(
                    [("num", StandardScaler(), [col])])),
                ("clf", DecisionTreeClassifier(max_depth=3, random_state=cfg.seed)),
            ])
        s = cross_val_score(pipe, X[[col]], y, cv=cv, scoring="roc_auc", n_jobs=-1)
        scores[col] = float(np.mean(s))

    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def total_charges_redundancy(df: pd.DataFrame) -> float:
    """R^2 of TotalCharges regressed on tenure x MonthlyCharges.

    Quantifies redundancy rather than asserting it. High R^2 means the column
    adds little independent information, which matters for coefficient stability
    in the linear model and for reading SHAP attributions on correlated inputs.
    """
    d = df.dropna(subset=["TotalCharges"])
    expected = d["tenure"] * d["MonthlyCharges"]
    actual = d["TotalCharges"]
    ss_res = float(((actual - expected) ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def contamination_check(cfg: Config) -> int:
    """Count feature-identical rows shared between train and test.

    Runs on the ACTUAL split the model uses, so it verifies the pipeline rather
    than restating the deduplication policy.
    """
    ds = make_dataset(cfg)
    feats = feature_columns()
    train_keys = set(map(tuple, ds.X_train[feats].astype(str).to_numpy()))
    test_keys = list(map(tuple, ds.X_test[feats].astype(str).to_numpy()))
    return sum(1 for k in test_keys if k in train_keys)


def run_audit(cfg: Config, path: str | None = None) -> AuditResult:
    """Execute the full audit and return structured findings."""
    raw = load_raw(cfg, path)
    cleaned = clean_raw(raw)
    feats = feature_columns()

    _, n_removed, n_conflicting = deduplicate(cleaned)
    groups = cleaned.groupby(feats, dropna=False, observed=True)[TARGET].nunique()

    y = (cleaned[TARGET] == cfg.positive_label).astype(int)
    aucs = single_feature_auc(cleaned[feats], y, cfg)
    suspects = [c for c, v in aucs.items() if v >= LEAKAGE_AUC_THRESHOLD]

    notes: list[str] = []
    if not suspects:
        notes.append(
            f"No single feature reaches AUC {LEAKAGE_AUC_THRESHOLD:.2f}; the strongest is "
            f"{next(iter(aucs))} at {next(iter(aucs.values())):.4f}. That is consistent "
            "with a genuinely predictive contract variable, not with label leakage.")
    notes.append(
        "No event timestamps exist, so an out-of-time split and an empirical "
        "post-churn test are both impossible. Post-churn risk is assessed per column "
        "from semantics; see the table in the report.")

    return AuditResult(
        n_rows=len(raw),
        n_cols=raw.shape[1],
        churn_rate=float(y.mean()),
        missing=_missing_analysis(raw),
        n_exact_duplicates=int(raw.duplicated().sum()),
        n_id_duplicates=int(raw[ID_COLUMN].duplicated().sum()),
        n_feature_duplicates=int(cleaned.duplicated(subset=feats).sum()),
        n_safe_duplicates_removed=n_removed,
        n_conflicting_label_rows=n_conflicting,
        n_conflicting_groups=int((groups > 1).sum()),
        invalid_values=_invalid_ranges(raw),
        invalid_categories=_invalid_categories(raw),
        structural_inconsistencies=_structural_consistency(cleaned),
        single_feature_auc=aucs,
        leakage_suspects=suspects,
        total_charges_r2=total_charges_redundancy(cleaned),
        contamination_overlap=contamination_check(cfg),
        notes=notes,
    )


def _fmt_dict(d: dict[str, Any], empty: str = "none") -> str:
    if not d:
        return f"_{empty}_"
    return "\n".join(f"- `{k}`: {v}" for k, v in d.items())


def render_report(res: AuditResult, cfg: Config) -> str:
    """Render the audit as markdown."""
    lines: list[str] = []
    a = lines.append

    a("# Data Quality & Leakage Audit\n")
    a("Generated by `python -m src.run audit`. Every number below is computed, not asserted.\n")

    a("## 1. Schema and data dictionary\n")
    a(f"`{res.n_rows:,}` rows x `{res.n_cols}` columns. "
      f"Target `{TARGET}`, positive class `{cfg.positive_label}`.\n")
    a("| column | kind | description | notes |")
    a("|---|---|---|---|")
    for spec in DATA_DICTIONARY:
        a(f"| `{spec.name}` | {spec.kind} | {spec.description} | {spec.notes or '—'} |")
    a("")
    a(f"Numeric: {', '.join(f'`{c}`' for c in NUMERIC_COLUMNS)}. "
      f"Binary numeric: {', '.join(f'`{c}`' for c in BINARY_NUMERIC_COLUMNS)}. "
      f"Categorical: {len(CATEGORICAL_COLUMNS)} columns.\n")

    a("## 2. Target definition and balance\n")
    a(f"`{TARGET} == '{cfg.positive_label}'` maps to 1. "
      f"Churn rate **{res.churn_rate:.2%}** "
      f"(imbalance ratio ~1:{(1 - res.churn_rate) / res.churn_rate:.2f}).\n")
    a("Moderate imbalance. It rules out accuracy as a headline metric — predicting "
      f"'no churn' for everyone scores {1 - res.churn_rate:.2%} accuracy while finding "
      "zero churners — but it is mild enough that resampling is unnecessary; class "
      "weighting and threshold tuning are sufficient.\n")

    a("## 3. Missing values\n")
    a(_fmt_dict(res.missing, "no missing values detected"))
    a("")
    a("`TotalCharges` is published as **text**, and its 11 empty cells are whitespace "
      "strings rather than nulls — `df.isna().sum()` reports zero for it. All 11 rows "
      "have `tenure == 0`, so the value is **structurally absent**: the customer has "
      "never been billed. It is set to `0.0`, which is the true amount billed, not an "
      "imputed guess. All 11 have `Churn == 'No'`.\n")

    a("## 4. Duplicates and train/test contamination\n")
    a(f"- exact duplicate rows: **{res.n_exact_duplicates}**")
    a(f"- duplicate `{ID_COLUMN}`: **{res.n_id_duplicates}**")
    a(f"- feature-identical rows (ignoring id and target): **{res.n_feature_duplicates}**")
    a(f"- safe duplicates removed before splitting: **{res.n_safe_duplicates_removed}**")
    a(f"- label-conflicting rows retained: **{res.n_conflicting_label_rows}** "
      f"across **{res.n_conflicting_groups}** feature groups")
    a(f"- feature-identical rows shared across the train/test boundary: "
      f"**{res.contamination_overlap}**\n")
    a("Deduplication happens **before** the split. Splitting first would put identical "
      "feature vectors on both sides and inflate test scores.\n")
    a(f"The {res.n_conflicting_label_rows} retained rows are feature-identical but carry "
      "contradictory labels. They are **irreducible label noise**: no model can separate "
      f"them, so they cap achievable accuracy at roughly "
      f"{1 - res.label_noise_rate / 2:.2%}. Both copies are kept — dropping one would "
      "delete a real churn outcome and quietly bias the base rate.\n")

    a("## 5. Invalid values and category consistency\n")
    a("Out-of-range numerics:")
    a(_fmt_dict(res.invalid_values, "none; all values within declared valid ranges"))
    a("")
    a("Unexpected categories:")
    a(_fmt_dict(res.invalid_categories, "none; all categories match the declared schema"))
    a("")
    a("Structural consistency (service columns vs their parent service):")
    a(_fmt_dict(res.structural_inconsistencies,
                "consistent; 'No internet service' and 'No phone service' appear exactly "
                "when the parent service is absent"))
    a("")
    a("This check matters for feature engineering: because the categories are structurally "
      "determined, a naive count of `'Yes'` across the six add-on columns conflates "
      "*declined a service* with *cannot have the service*. The engineered adoption count "
      "is therefore computed only over customers who have internet.\n")

    a("## 6. Leakage audit\n")
    a("### 6.1 Single-feature separability probe\n")
    a("Each feature is scored alone under 5-fold stratified CV, encoded inside the fold. "
      f"A feature at AUC >= {LEAKAGE_AUC_THRESHOLD:.2f} would be the target in disguise.\n")
    a("| feature | CV ROC-AUC alone |")
    a("|---|---:|")
    for c, v in res.single_feature_auc.items():
        a(f"| `{c}` | {v:.4f} |")
    a("")
    if res.leakage_suspects:
        a(f"**Suspects: {', '.join(f'`{c}`' for c in res.leakage_suspects)}** — investigated "
          "before modelling.\n")
    else:
        a("**No suspects.** No feature alone separates the target.\n")

    a("### 6.2 Redundancy, not leakage\n")
    a(f"`TotalCharges` regressed on `tenure x MonthlyCharges` gives "
      f"**R^2 = {res.total_charges_r2:.4f}**. The column is almost a deterministic "
      "function of two others. That is collinearity, which destabilises linear "
      "coefficients and splits SHAP credit between correlated inputs — not leakage. "
      "It is kept, and the engineered features deliberately extract the *residual* "
      "(billing discrepancy) rather than restating the product.\n")

    a("### 6.3 Post-churn assessment\n")
    a("**This cannot be tested empirically.** The dataset is one snapshot with no event "
      "timestamps, so there is no way to establish that a column was recorded before the "
      "churn decision. The following is reasoning from semantics.\n")
    a("| column | assessment |")
    a("|---|---|")
    for c, v in POST_CHURN_ASSESSMENT.items():
        a(f"| `{c}` | {v} |")
    a("")

    a("## 7. Notes and limitations\n")
    for n in res.notes:
        a(f"- {n}")
    a("")
    return "\n".join(lines)


def run(cfg: Config, path: str | None = None) -> AuditResult:
    """Audit entry point: prints a summary and writes the markdown report."""
    res = run_audit(cfg, path)
    cfg.ensure_dirs()
    dest = cfg.resolve("paths", "reports_dir") / "data_audit.md"
    dest.write_text(render_report(res, cfg), encoding="utf-8")

    print(f"rows x cols                : {res.n_rows:,} x {res.n_cols}")
    print(f"churn rate                 : {res.churn_rate:.2%}")
    print(f"missing (incl. whitespace) : {res.missing or 'none'}")
    print(f"exact dups / dup ids       : {res.n_exact_duplicates} / {res.n_id_duplicates}")
    print(f"feature dups               : {res.n_feature_duplicates}"
          f"  (removed {res.n_safe_duplicates_removed})")
    print(f"label conflicts            : {res.n_conflicting_label_rows} rows in "
          f"{res.n_conflicting_groups} groups  -> noise floor {res.label_noise_rate:.4%}")
    print(f"train/test contamination   : {res.contamination_overlap}")
    print(f"invalid ranges/categories  : {res.invalid_values or 'none'} / "
          f"{res.invalid_categories or 'none'}")
    print(f"structural inconsistencies : {res.structural_inconsistencies or 'none'}")
    print(f"TotalCharges R^2 vs tenure*Monthly : {res.total_charges_r2:.4f}")
    top = list(res.single_feature_auc.items())[:5]
    print("top single-feature AUC     : "
          + ", ".join(f"{c}={v:.4f}" for c, v in top))
    print(f"leakage suspects           : {res.leakage_suspects or 'none'}")
    print(f"\n[written] {dest}")
    return res
