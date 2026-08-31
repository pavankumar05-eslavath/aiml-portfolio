# AI / ML Portfolio

Machine learning projects where the deliverable is a defensible conclusion, not a
leaderboard score. Each one ends in a claim about what the model can and cannot
support, and the evidence for it.

| Project | What it demonstrates |
|---|---|
| **[AIML-1 — Online Payment Fraud Detection, audited](./aiml1-payment-fraud-detection)** | Reproduction and audit of a widely-followed tutorial on 6.36M PaySim transactions. RandomForest matches the published figure to **16 significant figures**; its XGBoost result does not reproduce at all (**0.9992 → 0.7125**), root-caused to XGBoost 2.0 estimating `base_score` from the class prior. Then the real finding: the corrected pipeline scores **1.000 PR AUC with zero false positives**, so I went looking for the leak — a **3-clause rule** gets precision **0.9999** / recall **0.9750** with **one false positive in 6.36M rows**. The tutorial's 99.9% measures recovery of the simulator's fraud script, not fraud detection. |
| **[AIML-3 — Customer Churn Prediction & Explainable ML](./aiml3-customer-churn-explainable-ml)** | End-to-end churn system on 7,043 Telco customers that predicts who leaves, explains why per customer with SHAP, and turns both into a retention decision. The headline is a negative result, reported as such: **XGBoost does not significantly beat untuned logistic regression** (PR-AUC 0.6753 vs 0.6690, paired t **p = 0.469**), and the **designated test fold is the weakest of five** (0.6134 vs 0.6609 mean) — kept rather than re-drawn. Calibration was measured, then **declined** under a parsimony rule set before the numbers were seen. A **per-customer expected-value rule** beats the best global threshold using **half the contacts** (713 of 1,406), because break-even risk varies 8.2% → 51.9% with customer value. Segment analysis exposes that the model is **near-blind on two-year contracts** (recall 0.143), which the aggregate AUC hides. |

## What these are not

They are not analytics projects — there is no dashboarding, no SQL metric
modelling, no stakeholder readout. That work lives in the
[data analyst portfolio](https://github.com/pavankumar05-eslavath/data-analyst-portfolio)
and the
[business analyst portfolio](https://github.com/pavankumar05-eslavath/business-analyst-portfolio).
They are not pipeline projects either; orchestration and warehouse modelling live
in the
[data engineering portfolio](https://github.com/pavankumar05-eslavath/data-engineering-portfolio).

They are also not benchmark chases. A model that cannot beat a stated baseline has
not earned its complexity, and every project here reports that baseline.

## Conventions

Every project ships:

- **`INSIGHTS.md`** — the findings, the recommendation, and what would have to be
  measured to confirm it. Every number is reproducible from the committed code.
- **`LEARN.md`** — how the method works, why each choice was made, and the
  questions the project invites. Written to be argued with.
- **`tests/`** — assertions on the *claims*, not just the code. Where a project's
  conclusion depends on a property of the data, a test pins that property, so a
  future change that invalidates the conclusion fails loudly.
- **A baseline.** Stated before the model, and reported next to it.
- **A caveat section.** What the project does not establish.

## Running any project

```bash
cd aiml1-payment-fraud-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make data      # generate the dataset stand-in
make all
make test
```

Large datasets are never committed. Each project either generates a faithful
stand-in or ships a `make download`, and says which figures require the real file.

## A note on scores

Several projects here report numbers above 0.99. In every case the README says
plainly whether that number means the model is good — and in AIML-1 it does not.
Reporting a high score without establishing that it is *earned* is the failure
mode this portfolio is built to avoid.
