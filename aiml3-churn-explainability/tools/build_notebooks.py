"""Generate the six analysis notebooks.

The notebooks are generated from this single source so they stay consistent with
`src/` and with each other. They deliberately contain no implementation: each cell
calls into `src/`, which is what stops the notebook version of the analysis from
drifting away from the one that is tested and served.

    python tools/build_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks"

PREAMBLE = [
    "import sys, warnings",
    "from pathlib import Path",
    "sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()))",
    "warnings.filterwarnings('ignore')",
    "",
    "from src.config import load_config",
    "cfg = load_config()",
]

NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "01_data_audit": [
        ("md", "# 01 — Data Audit\n\n"
               "Schema, data dictionary, quality checks and the leakage probe.\n\n"
               "The audit runs **before** any modelling. Its purpose is to find the "
               "problems that a `df.info()` pass does not: this dataset reports zero "
               "nulls while `TotalCharges` has 11 blank strings, and it has zero "
               "duplicate ids while carrying feature-identical rows with "
               "contradictory labels."),
        ("code", "\n".join(PREAMBLE)),
        ("md", "## Raw schema and data dictionary"),
        ("code", "from src.data.loader import load_raw, clean_raw, deduplicate\n"
                 "from src.data.schema import DATA_DICTIONARY, TARGET\n"
                 "import pandas as pd\n\n"
                 "raw = load_raw(cfg)\n"
                 "print(raw.shape)\n"
                 "raw.head()"),
        ("code", "pd.DataFrame([{'column': s.name, 'kind': s.kind,\n"
                 "               'description': s.description, 'notes': s.notes}\n"
                 "              for s in DATA_DICTIONARY])"),
        ("md", "## Target definition and balance\n\n"
               "`Churn == 'Yes'` is the positive class. Note what the base rate implies: "
               "a model predicting 'nobody churns' scores that accuracy while finding "
               "no churners at all."),
        ("code", "y = (raw[TARGET] == cfg.positive_label).astype(int)\n"
                 "print(f'churn rate           : {y.mean():.4f}')\n"
                 "print(f'majority-class acc.  : {1 - y.mean():.4f}')\n"
                 "print(f'imbalance ratio      : 1:{(1-y.mean())/y.mean():.2f}')"),
        ("md", "## The missing values that `isna()` does not show\n\n"
               "`TotalCharges` ships as text. Its blanks are whitespace, and they are "
               "**not** missing at random."),
        ("code", "print('pandas-visible nulls in the raw frame:')\n"
                 "print(raw.isna().sum()[lambda s: s > 0])\n\n"
                 "blank = raw[raw['TotalCharges'].isna()]\n"
                 "print(f'\\nblank TotalCharges: {len(blank)}')\n"
                 "print(f'their tenure values: {sorted(blank[\"tenure\"].unique())}')\n"
                 "print(f'their churn values : {blank[TARGET].unique().tolist()}')\n"
                 "print('\\n=> never-billed customers. 0.0 is the true amount billed, '\n"
                 "      'not an imputed guess.')"),
        ("md", "## Duplicates, label conflicts and contamination"),
        ("code", "cleaned = clean_raw(raw)\n"
                 "deduped, n_removed, n_conflicting = deduplicate(cleaned)\n"
                 "print(f'exact duplicate rows        : {raw.duplicated().sum()}')\n"
                 "print(f'duplicate customerIDs       : {raw[cfg.id_column].duplicated().sum()}')\n"
                 "print(f'safe duplicates removed     : {n_removed}')\n"
                 "print(f'label-conflicting rows kept : {n_conflicting}')\n"
                 "print(f'=> irreducible noise floor  : {n_conflicting/len(cleaned):.4%} of rows')"),
        ("md", "## The full audit, including the executable leakage probe\n\n"
               "Every feature is scored **alone** under cross-validation. A feature "
               "approaching AUC 1.0 would be the target in disguise. This is the "
               "difference between claiming there is no leakage and testing for it."),
        ("code", "from src.data import audit\n"
                 "res = audit.run(cfg)"),
        ("code", "import pandas as pd\n"
                 "pd.Series(res.single_feature_auc, name='cv_roc_auc_alone').head(10)"),
        ("md", "### What cannot be tested\n\n"
               "The dataset is a single snapshot with **no event timestamps**. So there "
               "is no way to verify empirically that a column was recorded before the "
               "churn decision, and no out-of-time split is possible. Post-churn risk "
               "is assessed per column from semantics — see "
               "`reports/data_audit.md`."),
    ],
    "02_eda": [
        ("md", "# 02 — Exploratory Data Analysis\n\n"
               "Six figures, each answering one business question. Deliberately not "
               "thirty: a correlation heatmap of one-hot columns looks industrious and "
               "says nothing actionable.\n\n"
               "**All of this runs on the training split only.** Exploring the test set "
               "is a soft leak — every decision made after seeing it (which bands to "
               "cut, which features to build) is fitted to it."),
        ("code", "\n".join(PREAMBLE)),
        ("code", "from src import eda\n"
                 "findings = eda.run(cfg)"),
        ("md", "## Q1–Q2: how much churn, and when?"),
        ("code", "from IPython.display import Image, display\n"
                 "figdir = cfg.resolve('paths', 'figures_dir')\n"
                 "display(Image(str(figdir / '01_churn_overview.png')))\n"
                 "display(Image(str(figdir / '02_churn_by_tenure.png')))"),
        ("md", "Churn is heavily **front-loaded**. That is what motivates `tenure_band` "
               "and `is_new_customer` in the feature set, and it means a retention "
               "programme aimed at long-tenured customers is aimed at the wrong group."),
        ("md", "## Q3: which commercial terms carry the risk?"),
        ("code", "display(Image(str(figdir / '03_churn_by_contract_payment.png')))"),
        ("md", "## Q4: does product adoption protect against churn?"),
        ("code", "display(Image(str(figdir / '04_churn_by_services.png')))"),
        ("code", "import pandas as pd\n"
                 "pd.Series(findings['churn_by_n_addons'], name='churn_rate')"),
        ("md", "Churn falls **monotonically** with add-on count. Each extra service is "
               "another switching cost. This is the clearest stickiness signal in the "
               "data and justifies `n_addons` / `addon_adoption_rate`."),
        ("md", "## Q5–Q6: the money variables, and where risk concentrates"),
        ("code", "display(Image(str(figdir / '05_numeric_distributions.png')))\n"
                 "display(Image(str(figdir / '06_risk_concentration.png')))"),
        ("md", "The heatmap pair matters more than either alone: the highest-risk cell "
               "is also the **highest-volume** one, so it is where retention effort has "
               "the most to work with."),
    ],
    "03_feature_engineering": [
        ("md", "# 03 — Feature Engineering\n\n"
               "Every feature is **row-wise**: it depends only on the customer's own "
               "values, never on a dataset aggregate. That property is what makes it "
               "safe to compute before the split, and it is why the transformer is "
               "stateless."),
        ("code", "\n".join(PREAMBLE)),
        ("code", "from src.data.loader import make_dataset\n"
                 "from src.features.engineer import ChurnFeatureEngineer, FEATURE_DOCS\n"
                 "import pandas as pd\n\n"
                 "ds = make_dataset(cfg)\n"
                 "fe = ChurnFeatureEngineer().fit(ds.X_train)\n"
                 "out = fe.transform(ds.X_train)\n"
                 "print(f'{ds.X_train.shape[1]} raw columns -> {out.shape[1]} columns')"),
        ("md", "## Every feature, and why it should help"),
        ("code", "pd.DataFrame([{'feature': k, 'type': v[0], 'rationale': v[1]}\n"
                 "              for k, v in FEATURE_DOCS.items()])"),
        ("md", "## The features are stateless\n\n"
               "Fitting on the training set and fitting on the rows themselves give "
               "identical output. If a target-encoded feature were added, this check "
               "would fail — which is the alarm."),
        ("code", "sample = ds.X_test.head(50)\n"
                 "a = ChurnFeatureEngineer().fit(ds.X_train).transform(sample)\n"
                 "b = ChurnFeatureEngineer().fit(sample).transform(sample)\n"
                 "print('identical:', a.equals(b))"),
        ("md", "## The awkward cases\n\n"
               "Never-billed customers (`tenure == 0`) would produce division by zero. "
               "`price_vs_history` fills with **1.0** — 'no change versus history' — "
               "because filling with 0 would read as a total price collapse at exactly "
               "the tenure where churn risk peaks."),
        ("code", "cols = ['tenure','TotalCharges','realized_arpu','price_vs_history',\n"
                 "        'billing_discrepancy','is_new_customer']\n"
                 "out.loc[out['tenure'] == 0, cols].head()"),
        ("code", "out[['n_addons','addon_adoption_rate','price_vs_history',\n"
                 "     'billing_discrepancy','charges_per_addon']].describe().round(3)"),
        ("md", "## `billing_discrepancy` extracts the residual, not a third copy\n\n"
               "The audit found `TotalCharges` is 99.91% explained by "
               "`tenure x MonthlyCharges`. So the *residual* is the only genuinely new "
               "information in that column."),
        ("code", "import numpy as np\n"
                 "expected = out['tenure'] * out['MonthlyCharges']\n"
                 "print('corr(TotalCharges, tenure*Monthly):',\n"
                 "      round(float(np.corrcoef(out['TotalCharges'], expected)[0,1]), 6))\n"
                 "out['billing_discrepancy'].describe().round(2)"),
        ("md", "## The encoded matrix\n\n"
               "Preprocessing lives inside the pipeline, so scaler statistics and "
               "encoder categories are fitted on training folds only."),
        ("code", "from sklearn.linear_model import LogisticRegression\n"
                 "from src.features.preprocess import build_pipeline, output_feature_names\n\n"
                 "pipe = build_pipeline(LogisticRegression(max_iter=1000), scale=True)\n"
                 "pipe.fit(ds.X_train, ds.y_train)\n"
                 "names = output_feature_names(pipe)\n"
                 "print(f'encoded width: {len(names)}')\n"
                 "names[:12]"),
    ],
    "04_modeling": [
        ("md", "# 04 — Modelling\n\n"
               "Baselines first, then tuned models. A gradient-boosted model that "
               "cannot beat regularised logistic regression has not earned its "
               "complexity, and on tabular churn data with ~5.6k rows that is a real "
               "possibility rather than a rhetorical one.\n\n"
               "Cross-validation is **group-aware**, so feature-identical rows cannot "
               "span a fold boundary. Tuning targets **PR-AUC**, because the positive "
               "class is the one we spend money on."),
        ("code", "\n".join(PREAMBLE)),
        ("code", "from src.data.loader import make_dataset\n"
                 "ds = make_dataset(cfg)\n"
                 "print(f'train {ds.n_train:,} / test {ds.n_test:,}')\n"
                 "print(f'churn {ds.y_train.mean():.4f} / {ds.y_test.mean():.4f}')\n"
                 "print(f'deduplicated {ds.n_duplicates_removed}, '\n"
                 "      f'{ds.n_conflicting_labels} conflicting rows retained')"),
        ("md", "## Train baselines and tuned candidates\n\n"
               "This cell runs the randomised searches and takes a couple of minutes."),
        ("code", "from src.models.train import (train_all, comparison_table,\n"
                 "                              select_best, paired_comparison)\n"
                 "results = train_all(ds, cfg)"),
        ("code", "comparison_table(results).round(4)"),
        ("md", "## Does the complex model earn its place?\n\n"
               "The candidates are scored on identical folds, so the fold scores are "
               "paired and a **paired** test is the right instrument. Comparing means "
               "alone invites reading a 0.004 difference as an improvement when the "
               "fold-to-fold spread is 0.025."),
        ("code", "best = select_best(results)\n"
                 "cmp = paired_comparison(best, results['logistic_regression_plain'])\n"
                 "for k, v in cmp.items():\n"
                 "    print(f'{k:22s} {v}')"),
        ("md", "With five folds the test has very low power, so a non-significant "
               "result means **the data cannot distinguish them** — not that they are "
               "proven equal. That is the useful conclusion when the simpler model is "
               "the one at risk of being discarded for no measured gain."),
        ("code", "import pandas as pd\n"
                 "pd.DataFrame({name: r.cv_folds['pr_auc'] for name, r in results.items()\n"
                 "              if not name.startswith('dummy')}).round(4)"),
    ],
    "05_model_evaluation": [
        ("md", "# 05 — Model Evaluation\n\n"
               "Metrics split into two questions that are often conflated:\n\n"
               "* **Ranking** (ROC-AUC, PR-AUC) — can the model order customers by risk?\n"
               "* **Probability quality** (Brier, ECE, calibration curve) — is a "
               "predicted 0.7 actually a 70% chance?\n\n"
               "The second matters because the decision layer multiplies probability by "
               "customer value. A model that ranks well but is miscalibrated by 2x "
               "inflates every expected-value calculation."),
        ("code", "\n".join(PREAMBLE)),
        ("code", "from src.data.loader import make_dataset\n"
                 "from src.features.preprocess import build_pipeline\n"
                 "from xgboost import XGBClassifier\n\n"
                 "ds = make_dataset(cfg)\n"
                 "pipe = build_pipeline(XGBClassifier(\n"
                 "    random_state=cfg.seed, n_jobs=1, tree_method='hist',\n"
                 "    eval_metric='logloss', n_estimators=300, max_depth=4,\n"
                 "    learning_rate=0.02, subsample=1.0, colsample_bytree=0.6,\n"
                 "    min_child_weight=1, reg_lambda=0.5), scale=False)\n"
                 "pipe.fit(ds.X_train, ds.y_train)\n"
                 "prob = pipe.predict_proba(ds.X_test)[:, 1]"),
        ("md", "## Why accuracy is not the headline"),
        ("code", "from src.evaluation.metrics import evaluate, metrics_frame\n"
                 "m = evaluate(ds.y_test, prob, 0.5)\n"
                 "print(f'model accuracy at 0.5        : {m.accuracy:.4f}')\n"
                 "print(f'\"nobody churns\" accuracy      : {1 - ds.y_test.mean():.4f}')\n"
                 "print(f'...and it finds {0} churners of {int(ds.y_test.sum())}')"),
        ("md", "## Calibration\n\n"
               "Calibrators are fitted with internal CV on the **training** split and "
               "judged on the untouched test set. The uncalibrated model competes as a "
               "candidate: a booster trained on logloss is often already calibrated, "
               "and a calibrator that does not measurably help is a component to "
               "maintain for nothing."),
        ("code", "from src.evaluation import metrics as M\n"
                 "best_cal, all_cal = M.calibrate(pipe, ds, cfg)\n"
                 "import pandas as pd\n"
                 "pd.DataFrame([{'method': c.method, 'brier': c.brier_after,\n"
                 "               'ece': c.ece_after, 'log_loss': c.log_loss_after,\n"
                 "               'pr_auc': c.pr_auc_after} for c in all_cal]).round(4)"),
        ("code", "print('selected:', best_cal.method)\n"
                 "print('parsimony guard (min Brier gain):', cfg['calibration']['min_brier_gain'])"),
        ("md", "## Split stability\n\n"
               "A single 80/20 split of ~7k rows is a noisy estimate. Reporting one "
               "number from one split invites two errors: mistaking split luck for "
               "model quality, and quietly re-drawing the split until the number "
               "improves."),
        ("code", "stability = M.split_stability(pipe, cfg)\n"
                 "stability.round(4)"),
        ("code", "print(f\"designated test fold : {stability.loc[0,'pr_auc']:.4f}\")\n"
                 "print(f\"across-fold mean     : {stability['pr_auc'].mean():.4f} \"\n"
                 "      f\"(sd {stability['pr_auc'].std():.4f})\")"),
        ("md", "## Threshold selection on a cost/value framework\n\n"
               "> The economic parameters are **assumptions** declared in "
               "`configs/config.yaml`. The dataset has no campaign costs, offer values "
               "or intervention outcomes, so they cannot be estimated from it. Figures "
               "are in neutral currency units (CU)."),
        ("code", "from src.evaluation.threshold import Economics, optimise, sensitivity_analysis\n"
                 "econ = Economics.from_config(cfg)\n"
                 "value = econ.customer_value(ds.X_test)\n"
                 "res = optimise(ds.y_test.to_numpy(), prob, value, econ)\n"
                 "print(f'best global threshold : {res.best_threshold:.2f} -> {res.best_net_value:,.0f} CU')\n"
                 "print(f'contact everyone      : {res.baseline_contact_all:,.0f} CU')\n"
                 "print(f'contact nobody        : 0 CU')\n"
                 "print(f'per-customer EV rule  : {res.per_customer_net_value:,.0f} CU '\n"
                 "      f'({res.per_customer_contacts} contacts)')"),
        ("md", "The per-customer rule wins because the break-even probability varies "
               "with customer value. A single global cut cannot express that a "
               "high-value customer is worth contacting at much lower risk."),
        ("code", "be = econ.breakeven_probability(value)\n"
                 "print(f'break-even probability ranges {be.min():.1%} to {be.max():.1%}')"),
        ("code", "metrics_frame({'threshold_0.50': evaluate(ds.y_test, prob, 0.5),\n"
                 "               f'threshold_{res.best_threshold:.2f}':\n"
                 "                   evaluate(ds.y_test, prob, res.best_threshold)}).round(4)"),
        ("md", "## Sensitivity: the threshold is a function of the assumptions"),
        ("code", "sensitivity_analysis(ds.y_test.to_numpy(), prob, value, cfg).round(3)"),
        ("md", "## Error analysis and segment-level performance"),
        ("code", "from src.evaluation.error_analysis import (error_frame, error_costs,\n"
                 "                                           profile_errors, hardest_cases)\n"
                 "errors = error_frame(ds.X_test, ds.y_test.to_numpy(), prob, res.best_threshold)\n"
                 "profile_errors(errors).round(2)"),
        ("code", "costs = error_costs(errors, cfg)\n"
                 "for k, v in costs.items():\n"
                 "    print(f'{k:28s} {v:,.2f}' if isinstance(v, float) else f'{k:28s} {v}')"),
        ("code", "hardest_cases(errors)['confident_false_negatives']"),
        ("code", "from src.evaluation.segments import performance_disparities\n"
                 "disp, spreads = performance_disparities(\n"
                 "    ds.X_test, ds.y_test.to_numpy(), prob, res.best_threshold)\n"
                 "disp.round(3)"),
        ("code", "sorted(spreads.items(), key=lambda kv: -kv[1])[:6]"),
        ("md", "Read these as **stability and coverage checks**, not a fairness "
               "certification. A large spread on `gender` would be a warning sign; a "
               "small one is not evidence of fairness in any broader sense."),
    ],
    "06_explainability": [
        ("md", "# 06 — Explainability, Segmentation and Decisions\n\n"
               "Two audiences:\n\n"
               "* **Global** — which features drive the model, to check it learned the "
               "relationships the EDA found rather than an artefact.\n"
               "* **Local** — why *this* customer scored what they did, in language a "
               "retention agent can act on.\n\n"
               "Requires a trained bundle: run `make train` first."),
        ("code", "\n".join(PREAMBLE)),
        ("code", "from src.data.loader import make_dataset\n"
                 "from src.models.persist import load_model\n"
                 "from src.data.schema import feature_columns\n\n"
                 "ds = make_dataset(cfg)\n"
                 "bundle = load_model(cfg)\n"
                 "fitted = bundle['explainer_pipeline']\n"
                 "scorer = bundle['model']\n"
                 "print('model:', bundle['metadata']['model_name'])\n"
                 "print('threshold:', bundle['metadata']['threshold'])"),
        ("md", "## Global importance\n\n"
               "Aggregated back to business features: one-hot encoding splits one "
               "concept across several columns and understates it. `Contract` split "
               "three ways looks less important than it is until the parts are summed."),
        ("code", "from src.explainability.shap_analysis import compute_shap\n"
                 "art = compute_shap(fitted, ds.X_test, max_samples=1000, seed=cfg.seed)\n"
                 "art.grouped_importance(feature_columns()).head(12).round(4)"),
        ("code", "from IPython.display import Image, display\n"
                 "figdir = cfg.resolve('paths', 'figures_dir')\n"
                 "for f in ('07_shap_summary.png', '08_shap_importance_grouped.png'):\n"
                 "    p = figdir / f\n"
                 "    if p.exists():\n"
                 "        display(Image(str(p)))"),
        ("md", "## Per-customer explanations\n\n"
               "Contributions are in **log-odds** and are not additive in probability, "
               "so the plain-language layer reports direction and relative magnitude "
               "rather than 'this added 8% to your risk'.\n\n"
               "Phrasing is derived from the customer's **actual value**, never from the "
               "name of an encoded column — doing it the other way inverts the meaning "
               "of every one-hot term."),
        ("code", "from src.explainability.shap_analysis import (explain_customer,\n"
                 "                                              representative_customers)\n"
                 "prob_all = scorer.predict_proba(ds.X_test)[:, 1]\n"
                 "sub = prob_all[art.raw.index.to_numpy()]\n"
                 "picks = representative_customers(sub, art, cfg=cfg)\n\n"
                 "for band, pos in picks.items():\n"
                 "    row = art.raw.iloc[pos]\n"
                 "    exp = explain_customer(art, pos, feature_columns())\n"
                 "    print(f'\\n[{band}] churn {sub[pos]:.1%} | {row[\"Contract\"]}, '\n"
                 "          f'tenure {row[\"tenure\"]}m, {row[\"MonthlyCharges\"]:.2f}/month')\n"
                 "    for f in exp['top_risk_factors']:\n"
                 "        print(f'   + {f[\"reason\"]}  ({f[\"contribution\"]:+.3f})')\n"
                 "    for f in exp['top_protective_factors']:\n"
                 "        print(f'   - {f[\"reason\"]}  ({f[\"contribution\"]:+.3f})')"),
        ("md", "## Risk segmentation, validated\n\n"
               "The bands are checked against **realised** churn on the test set. A band "
               "whose realised rate does not match its label is a broken band, however "
               "tidy the cut points look."),
        ("code", "from src.evaluation.segments import segment_summary, validate_segments\n"
                 "from src.evaluation.threshold import Economics\n"
                 "econ = Economics.from_config(cfg)\n"
                 "value = econ.customer_value(ds.X_test)\n"
                 "seg = segment_summary(prob_all, ds.y_test.to_numpy(), cfg, value=value)\n"
                 "seg.drop(columns=['recommended_action']).round(4)"),
        ("code", "validate_segments(seg)"),
        ("code", "for band, action in seg['recommended_action'].items():\n"
                 "    print(f'{band:15s} {action}')"),
        ("md", "## The decision layer\n\n"
               "Risk alone is not a decision. Every action is gated on an "
               "expected-value test, so a low-value customer at high risk whose "
               "expected saving does not cover the offer is deliberately not contacted."),
        ("code", "from src.inference.decision import decide, campaign_summary\n"
                 "decisions = decide(prob_all, ds.X_test, cfg)\n"
                 "campaign_summary(decisions).round(2)"),
        ("code", "print(f\"contacts recommended : {int(decisions['contact_recommended'].sum())}\"\n"
                 "      f\" of {len(decisions)}\")\n"
                 "print(f\"EV overrides band    : {int(decisions['ev_overrides_segment'].sum())}\")"),
        ("md", "## End-to-end inference\n\n"
               "The shape an API would return for one customer."),
        ("code", "from src.inference.predict import ChurnPredictor, EXAMPLE_CUSTOMER, format_result\n"
                 "predictor = ChurnPredictor.load(cfg)\n"
                 "print(format_result(predictor.predict_one(EXAMPLE_CUSTOMER)))"),
    ],
}


def build(cells: list[tuple[str, str]]) -> dict:
    out = []
    for kind, body in cells:
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {},
                        "source": body.splitlines(keepends=True)})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                        "outputs": [], "source": body.splitlines(keepends=True)})
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        path = DEST / f"{name}.ipynb"
        path.write_text(json.dumps(build(cells), indent=1), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
