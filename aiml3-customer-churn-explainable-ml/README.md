# AIML-3 — Customer Churn Prediction & Explainable ML

An end-to-end churn system that predicts **who** will leave, explains **why** for each
individual customer, and converts both into a **retention decision** with an
expected-value test attached.

Built on the [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset — 7,043 customers, 26.54% churn.

Every number below is produced by the pipeline and reproducible with `make all`.
Nothing is typed by hand: `reports/model_report.md` is generated from
`reports/run_results.json`, which the run writes.

---

## The three findings that shape this project

**1. Gradient boosting does not beat untuned logistic regression here.**
CV PR-AUC 0.6753 vs 0.6690 — a difference of **+0.0063**, or 0.23 pooled standard
deviations, paired t-test **p = 0.469**. The complexity is not earned on
discrimination. XGBoost is kept mainly because `TreeExplainer` gives exact SHAP
values, and the README says so rather than implying a win.

**2. The designated test split is the weakest of the five folds** — PR-AUC 0.6134
against an across-fold mean of 0.6609 (sd 0.0318). A single 80/20 split of 7k rows
carries ±0.03 of noise. The split was **not re-drawn** to improve the headline; doing
so is test-set shopping. Treat 0.6609 ± 0.0318 as the honest estimate and the
single-split number as conservative.

**3. Calibration was measured and then declined.** XGBoost trained on logloss is
already calibrated (Brier 0.1425, ECE 0.0257). Sigmoid made it *worse* (0.1437 /
0.0461); isotonic improved Brier by 0.0004, below the parsimony threshold declared in
config *before* the numbers were seen. Shipping a calibrator for a 0.0004 gain adds a
component to maintain and a layer between SHAP and the served probability, for nothing.

> **All currency figures are assumptions, not business results.** The dataset contains
> no campaign costs, offer values or intervention outcomes, so the retention economics
> in `configs/config.yaml` are declared inputs. Figures use neutral **currency units
> (CU)**, and a sensitivity table shows how far the conclusions move when the
> assumptions do.

---

## Business problem

A telco wants to reduce voluntary churn. Prediction alone does not reduce churn — a
ranked list of at-risk customers is not a plan. Three things have to be true before a
model changes anything:

1. It must **rank** customers by risk (PR-AUC, ROC-AUC).
2. Its probabilities must be **real**, because the decision multiplies probability by
   customer value. A model that ranks well but is miscalibrated 2× inflates every
   expected-value calculation and spends real budget against an imagined return.
3. Each prediction must be **explainable**, because a retention agent needs a reason to
   open a conversation with, not a score.

This project treats all three as deliverables, and adds a fourth: deciding *who is
worth contacting*, which depends on value and cost, not risk alone.

## Dataset

| | |
|---|---|
| Source | IBM sample dataset (Kaggle: `blastchar/telco-customer-churn`) |
| Rows × columns | 7,043 × 21 |
| Target | `Churn` — `Yes` if the customer left within the last month |
| Base rate | **26.54%** (1,869 / 5,174), ≈1:2.77 |
| Features | demographics, tenure, 9 service columns, contract, billing, payment, charges |

**Why it is appropriate.** It contains exactly the four families a churn model needs:
customer demographics, account/contract information, product usage, and billing
behaviour. It is also small and messy enough that the audit has real work to do, and
its quirks are genuine rather than synthetic.

**Why it is limited.** It is a **single snapshot with no event timestamps**. That rules
out out-of-time validation and survival analysis, and it means "was this column recorded
before the churn decision?" cannot be answered empirically — only reasoned about. This is
the largest caveat in the project and is stated wherever it matters.

## Architecture

```
                 ┌─────────────────────────────────────────────┐
raw CSV  ──────▶ │ src/data/loader.py                          │
                 │  • TotalCharges: 11 blanks -> 0.0           │
                 │  • deduplicate BEFORE splitting             │
                 │  • GROUP-AWARE stratified split             │
                 └─────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        ▼                                                       ▼
┌───────────────────┐                            ┌──────────────────────────┐
│ src/data/audit.py │                            │ sklearn Pipeline         │
│  leakage probe    │                            │  1. ChurnFeatureEngineer │
│  contamination    │                            │  2. ColumnTransformer    │
│  -> data_audit.md │                            │  3. estimator            │
└───────────────────┘                            └──────────────────────────┘
                                                            │
                     ┌──────────────────────────────────────┼───────────────┐
                     ▼                                      ▼               ▼
          ┌────────────────────┐              ┌─────────────────┐  ┌────────────────┐
          │ evaluation/        │              │ explainability/ │  │ models/persist │
          │  metrics           │              │  SHAP global    │  │  bundle +      │
          │  calibration       │              │  + per-customer │  │  metadata      │
          │  threshold + EV    │              │  plain language │  └────────────────┘
          │  segments, errors  │              └─────────────────┘          │
          └────────────────────┘                        │                  ▼
                     └────────────────┬─────────────────┘        ┌──────────────────┐
                                      ▼                          │ inference/       │
                          reports/model_report.md                │  predict +       │
                                                                 │  decision layer  │
                                                                 └──────────────────┘
```

Feature engineering sits **inside** the pipeline, so a caller supplies only the 19 raw
published columns at training and at inference alike. That removes the most common way a
served model diverges from the one that was evaluated.

## Data quality findings

Full detail in **[reports/data_audit.md](./reports/data_audit.md)** (generated).

| check | result |
|---|---|
| Missing values | `TotalCharges` **11** — but `df.isna().sum()` reports **0** |
| Exact duplicate rows | 0 |
| Duplicate `customerID` | 0 |
| Feature-identical rows | 40 |
| Safe duplicates removed pre-split | **16** |
| Label-conflicting rows retained | **42** across **18** groups |
| Train/test contamination | **0** (was 5 before the group-aware split) |
| Out-of-range numerics | none |
| Undeclared categories | none |
| Structural inconsistencies | none |

Four findings that changed the implementation:

**`TotalCharges` is text, and its blanks are not missing at random.** All 11 blank cells
have `tenure == 0`: the customer has never been billed. `0.0` is the *true* amount, not
an imputation. Median imputation would invent a billing history at exactly the tenure
where churn risk peaks.

**42 rows are feature-identical with contradictory labels.** Irreducible label noise
(0.60% of rows) — no model can separate them, so they cap achievable accuracy. Both
copies are kept; dropping one would delete a real churn outcome and bias the base rate.
Detecting them requires grouping by feature vector and counting distinct labels;
comparing `duplicated(feats)` against `duplicated(feats + target)` flags only the odd row
out, so a Yes/No/No group counts as one conflict instead of three.

**Deduplication alone did not reach zero contamination.** Because label-conflicting rows
are deliberately retained, and they are duplicates by definition, a plain stratified
`train_test_split` left **5 identical feature vectors on both sides**. The split is
therefore group-aware (`StratifiedGroupKFold` on the feature-vector hash), which brings
overlap to 0 while holding stratification (churn 0.2645 train / 0.2646 test).

**The service columns are structurally determined.** `'No internet service'` appears
exactly when `InternetService == 'No'`, verified. So a naive count of `'Yes'` across the
six add-on columns conflates *declined a service* with *cannot have one* — which is why
`has_internet` exists and `addon_adoption_rate` is computed only over customers who have
internet.

### Leakage audit

Rather than asserting "no leakage was found", the audit **runs a probe**: each feature is
scored alone under 5-fold stratified CV, encoded inside the fold.

| feature | CV ROC-AUC alone |
|---|---:|
| `Contract` | 0.7391 |
| `tenure` | 0.7319 |
| `OnlineSecurity` | 0.7055 |
| `TechSupport` | 0.7035 |
| `InternetService` | 0.6953 |

**No suspects.** Nothing approaches AUC 1.0; the strongest single feature is a genuine
commercial predictor. Additionally:

- **`customerID` is excluded.** An identifier lets a tree memorise individuals and
  generalise to nobody.
- **`TotalCharges` is redundant, not leaky.** R² = **0.9991** against
  `tenure × MonthlyCharges`. That is collinearity, which destabilises linear coefficients
  and splits SHAP credit — so the linear model drops it, and `billing_discrepancy` feeds
  the model the *residual* instead of a third copy of the same signal.
- **Post-churn variables cannot be tested**, only reasoned about, because there are no
  timestamps. The per-column assessment is in the audit report.

## EDA findings

Six figures, each answering one business question, computed on the **training split
only** — exploring the test set is a soft leak, since every subsequent decision is fitted
to it.

| finding | number |
|---|---|
| Churn in first 6 months | **53.29%** |
| Churn at 49–72 months | **9.24%** |
| Month-to-month contract | **43.07%** |
| Two-year contract | **2.54%** |
| Electronic check | **45.45%** |
| Fibre optic | **41.88%** |
| Median monthly charge, churners vs stayers | **79.85** vs **64.65** |

![tenure](./reports/figures/02_churn_by_tenure.png)

**Churn is heavily front-loaded**, which means a retention programme aimed at
long-tenured customers is aimed at the wrong group.

![risk concentration](./reports/figures/06_risk_concentration.png)

The pair of heatmaps matters more than either alone: **month-to-month + 0–6 months is
both the highest-risk cell (55.85%) and the highest-volume one (1,094 customers)**. That
is where retention effort has the most to work with.

![services](./reports/figures/04_churn_by_services.png)

**Add-on adoption is monotonically protective** — churn falls at every step from 50.18%
(0 add-ons) to 4.89% (6). Each additional service is another switching cost. This is the
clearest stickiness signal in the data and directly motivates `n_addons`.

## Feature engineering

13 features, all **row-wise**: each depends only on its own row, never on a dataset
aggregate. That property is what makes them safe to compute before the split and
identical at inference, and it is why the transformer is stateless. No target encoding is
used.

| feature | why it should help |
|---|---|
| `tenure_band`, `is_new_customer` | Churn is front-loaded (53.3% vs 9.2%); banding gives the linear model the step change it cannot express from raw tenure. |
| `n_addons`, `addon_adoption_rate` | Churn falls monotonically 50.2% → 4.9% with add-on count. |
| `has_internet` | Separates "declined an add-on" from "cannot have one" — the add-on columns use a structural category. |
| `is_month_to_month` | Contract is the strongest single predictor (AUC 0.739 alone). |
| `is_autopay` | Electronic check churns at 45.5%, far above automatic methods. |
| `realized_arpu` | `TotalCharges / tenure` — what the customer actually averaged, not their list price. |
| `price_vs_history` | `MonthlyCharges / realized_arpu`. Above 1 = paying more than their historical average, i.e. a recent rise or expiring promotion. A raw price level cannot express this. |
| `billing_discrepancy` | The residual of `TotalCharges` against `tenure × MonthlyCharges` — the only genuinely new information in a column that is 99.91% explained by the other two. |
| `charges_per_addon` | Price per unit of service received; a value-for-money proxy. |
| `household_size` | Household accounts are harder to move — a switching-cost proxy. |
| `tenure_x_month_to_month` | **Linear model only.** Measured on the tree: including it moved PR-AUC 0.6753 → 0.6736 (p = 0.396) *and* dominated the SHAP output with an uninterpretable term. No gain, real interpretability cost, so tree pipelines exclude it. |

Awkward cases are handled explicitly: never-billed customers (`tenure == 0`) would divide
by zero, so `price_vs_history` fills with **1.0** — "no change versus history" — because
0 would read as a total price collapse at the riskiest tenure.

## Models tested

Baselines are scored **first** and reported alongside, so complexity has to justify
itself. Tuning targets **PR-AUC** (`average_precision`), because the positive class is
the one we spend money on. CV is group-aware.

| model | tuned | PR-AUC | ± sd | ROC-AUC | Brier |
|---|---|---:|---:|---:|---:|
| **xgboost** | yes | **0.6753** | 0.0252 | 0.8512 | 0.1326 |
| random_forest | yes | 0.6722 | 0.0240 | 0.8510 | 0.1330 |
| logistic_regression | yes | 0.6704 | 0.0300 | 0.8524 | 0.1324 |
| logistic_regression_plain | **no** | 0.6690 | 0.0301 | 0.8519 | 0.1325 |
| dummy_prior | no | 0.2645 | 0.0004 | 0.5000 | 0.1946 |

Best XGBoost parameters: `max_depth=4, learning_rate=0.02, n_estimators=300,
subsample=1.0, colsample_bytree=0.6, min_child_weight=1, reg_lambda=0.5`.

### Is the complexity earned? No.

Candidates are scored on identical folds, so the scores are **paired** and a paired test
is the right instrument:

| | |
|---|---|
| xgboost vs untuned logistic regression | 0.6753 vs 0.6690 |
| difference | **+0.0063** (0.23 pooled sd) |
| paired t-test | **p = 0.469** |
| Wilcoxon | p = 0.625 |
| significant at 0.05 | **No** |

With five folds the test has low power, so this means *the data cannot distinguish them*
— not that they are proven equal. The honest reading: **logistic regression is a
legitimate production choice here**, trains in ~2s against ~16s, and is inherently
interpretable.

### Split stability

| fold | PR-AUC | ROC-AUC | note |
|---|---:|---:|---|
| **0** | **0.6134** | 0.8270 | **designated test set** |
| 1 | 0.6868 | 0.8592 | |
| 2 | 0.6436 | 0.8466 | |
| 3 | 0.6855 | 0.8538 | |
| 4 | 0.6752 | 0.8543 | |
| mean | **0.6609** | | sd **0.0318** |

Churn rates are identical across folds, so the spread is composition luck, not
stratification. Fold 0 is the weakest and was kept.

## Evaluation methodology

Accuracy is not the headline: predicting "nobody churns" scores **73.5% accuracy** and
finds **zero** churners, and accuracy presumes a 0.5 cutoff no campaign would choose.

| | threshold 0.50 | threshold 0.13 (value-optimal) |
|---|---:|---:|
| ROC-AUC | 0.8270 | 0.8270 |
| PR-AUC | 0.6134 | 0.6134 |
| Precision | 0.6308 | 0.4186 |
| Recall | 0.4731 | **0.9059** |
| F1 | 0.5407 | 0.5726 |
| Accuracy | 0.7873 | 0.6422 |
| Brier | 0.1425 | 0.1425 |
| TN / FP / FN / TP | 931 / 103 / 196 / 176 | 566 / 468 / 35 / 337 |

### Cost/value framework

```
EV(contact) = p × success_rate × value − (contact_cost + incentive_cost)

contact_cost   = 5 CU     (ASSUMED)
incentive_cost = 30 CU    (ASSUMED)
success_rate   = 0.30     (ASSUMED)
value          = MonthlyCharges × 12 months
```

| policy | net value (CU) | contacts |
|---|---:|---:|
| contact nobody | 0 | 0 |
| contact everyone | 49,916 | all 1,406 |
| best global threshold (0.13) | 63,446 | — |
| **per-customer EV rule** | **63,799** | **713** |

The per-customer rule wins **with roughly half the contacts**, because the break-even
probability varies with customer value — from **8.2% to 51.9%** across the test set. A
single global cut cannot express that a high-value customer is worth contacting at much
lower risk than a low-value one.

**Sensitivity.** The optimal threshold is a function of parameters the data cannot
supply, and it moves from **0.03 to 0.76** across the assumed grid. That dependency is
why one threshold is never presented as *the* answer — the full table is in the
[model report](./reports/model_report.md).

## Calibration

| method | Brier | ECE | log loss | PR-AUC |
|---|---:|---:|---:|---:|
| **none (selected)** | 0.1425 | **0.0257** | 0.4379 | 0.6134 |
| sigmoid | 0.1437 | 0.0461 | 0.4428 | 0.6178 |
| isotonic | **0.1421** | 0.0260 | 0.4384 | 0.6185 |

Calibrators are fitted with internal CV on the **training** split and judged on the
untouched test set. ECE is reported next to Brier because Brier mixes calibration with
discrimination and can improve for the wrong reason.

Isotonic wins on Brier by 0.0004 — below the `min_brier_gain: 0.001` guard set in config
*before* seeing results — so the uncalibrated model ships. The reliability of the
probabilities is what the decision layer depends on, and it already holds: predicted risk
tracks realised churn to within 0.057 across all four risk bands.

## Explainable ML

Global importance, aggregated to business **concepts** so the global and local views use
the same grouping:

| concept | mean \|SHAP\| |
|---|---:|
| contract | 0.8920 |
| tenure | 0.4438 |
| spend | 0.4352 |
| OnlineSecurity | 0.2356 |
| internet | 0.2074 |
| TechSupport | 0.1954 |
| payment | 0.1921 |

This matches the EDA and the leakage probe (`Contract` strongest at AUC 0.739 alone),
which is the point of checking: the model learned the relationships the data actually
contains, not an artefact.

![shap summary](./reports/figures/07_shap_summary.png)

### Per-customer explanations

Rendered for a non-technical reader. Example, Critical Risk (77% predicted):

```
increases risk:
  + is on a month-to-month contract          (+0.575)
  + pays 95.60 per month                     (+0.399)
  + has been a customer for 7 months         (+0.269)
  + has Fiber optic internet                 (+0.238)
reduces risk:
  - pays by bank transfer (automatic)        (-0.136)
```

Three implementation details that make these correct rather than merely plausible:

1. **Phrasing comes from the customer's actual value, never from an encoded column
   name.** Doing it the other way inverts every one-hot term: a negative SHAP on
   `PaymentMethod_Electronic check` for a customer whose value is 0 means "does *not* pay
   by e-check, which lowers risk", but naming the column reports "pays by electronic
   check" as *protective* — the opposite of the truth, and contradicted by the EDA.
2. **Contributions are summed within a business concept.** `Contract` and
   `is_month_to_month` are one commercial fact; without grouping, the explanation repeats
   the same point and splits its magnitude. Add-on columns whose value is the structural
   `'No internet service'` fold into the internet concept instead of repeating that
   phrase six times.
3. **Contributions are log-odds and are not additive in probability**, so the output
   ranks reasons rather than claiming "this added 8% to your risk".

## Risk segmentation

Bands are **validated** against realised churn, not merely declared:

| segment | customers | avg predicted | **realised churn** | share of churners | action |
|---|---:|---:|---:|---:|---|
| Low Risk | 528 | 0.0338 | **0.0455** | 6.5% | Normal engagement, no retention spend |
| Medium Risk | 428 | 0.2062 | **0.2173** | 25.0% | Low-cost nurture, automated only |
| High Risk | 308 | 0.4927 | **0.5032** | 41.7% | Proactive outreach, prioritised by value |
| Critical Risk | 142 | 0.7612 | **0.7042** | 26.9% | Priority human intervention |

Realised churn rises monotonically, the largest predicted-vs-realised gap is **0.0569**,
and the top band churns **15.5×** the bottom. The Critical band is the one mildly
overconfident (0.761 predicted vs 0.704 realised), which is worth knowing before quoting
its probabilities in a business case.

## Business decision framework

Risk alone is not a decision. Actions come from a **risk × value matrix**, and every one
is gated on the expected-value test.

| priority | risk band | value tier | action |
|---:|---|---|---|
| 1 | Critical | High | Senior agent call, discretionary discount, contract-term offer |
| 2 | High | High | Proactive outreach within 7 days, incentive off month-to-month |
| 3 | Critical | Standard | Automated retention offer, no agent time |
| 4 | High | Standard | Targeted automated offer, cheapest effective incentive |
| 5 | Medium | High | Account review, add-on bundle to raise switching cost |
| 6 | Medium | Standard | Monitor, low-cost content, re-score next cycle |
| 7–8 | Low | either | Normal engagement; consider upsell instead |

On the test set this recommends **713 contacts of 1,406**. In **277** cases the EV test
recommends contacting someone in a Low/Medium band — the value dimension overriding the
risk band, which a one-dimensional "contact the high-risk customers" rule cannot express.
Conversely, some High Risk / Standard value customers are **not** contacted, because the
expected saving does not cover the offer.

## Error analysis

At threshold 0.13, mean values by outcome class:

| feature | true positive | **false negative** | false positive | true negative |
|---|---:|---:|---:|---:|
| tenure | 15.7 | **38.3** | 23.9 | 46.3 |
| MonthlyCharges | 76.6 | **57.7** | 71.9 | 54.3 |
| predicted probability | 0.535 | **0.093** | 0.391 | 0.054 |

**The model's blind spot is the long-tenured, low-spending customer who leaves anyway** —
missed churners average 38 months tenure and 57.69/month, and the model gave them 0.093.
They do not look like churners because on every learned signal they are loyal.

| | count | cost each (CU) |
|---|---:|---:|
| False negatives | 35 | ~214 (forgone saving) |
| False positives | 468 | 35 (wasted spend) |

The **6.1:1** asymmetry is why the value-optimal threshold is 0.13 rather than 0.5:
buying recall with precision is correct while a miss costs 6× a wasted contact. Observed
error rate is 0.358 against an irreducible floor of ~0.003 from label conflicts, so
almost all of the error is genuine model limitation, not data noise.

## Segment-level performance

| attribute | segment | n | churn | ROC-AUC | PR-AUC | recall | calib. gap |
|---|---|---:|---:|---:|---:|---:|---:|
| Contract | Month-to-month | 792 | 0.409 | 0.741 | 0.646 | 0.969 | +0.006 |
| Contract | One year | 259 | 0.131 | 0.714 | 0.273 | 0.618 | −0.030 |
| Contract | **Two year** | 355 | 0.039 | **0.699** | **0.108** | **0.143** | −0.009 |
| InternetService | Fibre optic | 617 | 0.415 | 0.778 | 0.682 | 0.965 | −0.007 |
| InternetService | No | 320 | 0.081 | 0.783 | 0.258 | 0.577 | −0.001 |
| SeniorCitizen | 1 | 226 | 0.407 | 0.791 | 0.720 | 0.957 | −0.011 |
| gender | Female | 705 | 0.254 | 0.828 | 0.601 | 0.927 | +0.014 |
| gender | Male | 701 | 0.275 | 0.828 | 0.634 | 0.886 | −0.023 |

**The model is close to blind on two-year contracts**: recall 0.143 and PR-AUC 0.108
across 355 customers. Churn there is rare (3.9%), so the aggregate metric hides it — but
these are long-committed, often high-value customers, and their churn is the kind a
business most wants to catch. Recall spread across contract types is **0.826**
(0.969 → 0.143), the largest disparity in the model.

`gender` shows no meaningful difference (ROC-AUC 0.828 vs 0.828). Read these as
**stability and coverage checks, not a fairness certification**: a large spread on gender
would be a warning sign, but a small one is not evidence of fairness in any broader
sense, and this dataset has no protected attributes beyond gender and senior-citizen
status.

## How to run

```bash
pip install -r requirements.txt

make download     # fetch the real dataset (7,043 rows); or `make sample`
make all          # audit -> eda -> train -> explain -> predict -> report
make test         # 51 tests
make lint         # ruff
```

Individual stages:

```bash
make audit        # -> reports/data_audit.md
make eda          # -> reports/figures/
make train        # train, tune, calibrate, evaluate, persist
make explain      # SHAP global + per-customer
make predict      # score the example customer
make report       # -> reports/model_report.md
make notebooks    # execute all six notebooks, failing on any error
```

Scoring new customers:

```python
from src.inference.predict import ChurnPredictor

predictor = ChurnPredictor.load()
result = predictor.predict_one({
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 94.40, "TotalCharges": 188.80,
})
```

### Example prediction

```
  churn probability   : 79.1%
  risk segment        : Critical Risk
  value tier          : High value (1,133 CU over the horizon)
  break-even risk     : 10.3%  (contact pays off above this)
  expected value      : +234 CU
  contact recommended : YES
  priority            : 1  via outbound call

  ACTION: Priority retention call from a senior agent with discretionary
          discount authority and a contract-term offer.

  Why this customer is at risk:
    + is on a month-to-month contract
    + has been a customer for 2 months
    + pays 94.40 per month
    + has no online-security add-on
  What is holding them:
    - has a single phone line
```

Or from the CLI:

```bash
python -m src.inference.predict --example
python -m src.inference.predict --input customers.csv --output scored.csv --explain
```

## Project structure

```
aiml3-customer-churn-explainable-ml/
├── configs/config.yaml          all tunable values incl. declared economics
├── data/raw/                    gitignored; fetched or generated
├── models/                      bundle + human-readable metadata sidecar
├── notebooks/                   01..06, executed, thin wrappers over src/
├── reports/
│   ├── data_audit.md            generated
│   ├── model_report.md          generated from run_results.json
│   └── figures/                 9 figures
├── src/
│   ├── config.py                path resolution; no hardcoded paths
│   ├── eda.py                   six business questions
│   ├── reporting.py             renders model_report.md
│   ├── run.py                   stage runner
│   ├── data/                    schema, loader, download, sample, audit
│   ├── features/                engineer (row-wise), preprocess
│   ├── models/                  train (+ paired tests), persist
│   ├── evaluation/              metrics, threshold/EV, segments, error_analysis
│   ├── explainability/          shap_analysis
│   └── inference/               predict, decision
├── tests/                       51 tests
└── tools/build_notebooks.py     generates the notebooks from one source
```

## Reproducibility

- Single seed (`project.seed: 42`) threads through the split, CV, tuning and models.
- Every value that affects a result lives in `configs/config.yaml`; a run is defined by
  that file plus the seed.
- Preprocessing is deterministic and fitted inside the pipeline.
- The persisted bundle carries the pipeline, threshold, **the economic assumptions the
  threshold was derived from**, the input contract, and library versions — a version
  mismatch warns on load.
- `reports/model_report.md` is generated from recorded results, so it cannot drift from
  what the code produced.
- CI runs lint, the suite, and the full pipeline on every push.

Verified on Python 3.11.15, pandas 3.0.5, scikit-learn 1.9.0, xgboost 3.2.0, shap 0.51.0.

## Limitations

- **Snapshot data, no timestamps.** No out-of-time validation, so the estimate assumes
  the future resembles a random sample of the past. For churn this is the single largest
  caveat: real deployments decay as pricing, competitors and product mix change.
- **The economics are assumptions.** Every CU figure is conditional on
  `configs/config.yaml`; the optimal threshold moves from 0.03 to 0.76 across the assumed
  grid.
- **The boosted model is not significantly better** than untuned logistic regression
  (p = 0.469).
- **A single 80/20 split is noisy at this size** (fold spread 0.0734 PR-AUC), and the
  designated test fold is the weakest of five.
- **42 rows carry contradictory labels**, so perfect accuracy is unreachable.
- **Near-blind on two-year contracts** (recall 0.143) — a genuinely underserved segment.
- **`Churn` pools voluntary and involuntary churn** with no reason code. They need
  different interventions and the model cannot separate them.
- **SHAP is correlational, not causal.** `Contract` dominating does **not** establish
  that moving a customer onto a two-year contract causes retention — customers who accept
  long contracts differ systematically. Only an experiment answers that, and the decision
  layer should not be read as a causal claim.
- **`customer_value` is revenue, not profit.** No margin, discount rate or cost-to-serve
  exists in the data, so calling it CLV would overstate it.

## Future improvements

- **Event-level history with timestamps**, enabling out-of-time validation and survival
  modelling (time-to-churn rather than a binary snapshot).
- **A randomised retention holdout** to *measure* `success_rate` and offer uplift instead
  of assuming them, turning the EV layer into a real financial model.
- **Uplift modelling.** The decision should target customers whose behaviour the
  intervention *changes*, not those most likely to leave regardless — which is a
  different objective from the one optimised here.
- **A dedicated model or richer features for the two-year segment**, where the current
  model is close to useless.
- **Drift monitoring** on input distributions and calibration, since a churn model's
  economics decay silently.
- **Churn reason codes** to separate voluntary from involuntary churn.

---

Part of the [AI/ML portfolio](../README.md). Companion project:
[AIML-1 — Online Payment Fraud Detection, audited](../aiml1-payment-fraud-detection).
