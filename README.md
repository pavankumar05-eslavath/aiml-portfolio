# AI / ML Portfolio

Machine learning projects where the deliverable is a defensible conclusion, not a
leaderboard score. Each one ends in a claim about what the model can and cannot
support, and the evidence for it.

| Project | What it demonstrates |
|---|---|
| **[AIML-1 — Online Payment Fraud Detection, audited](./aiml1-payment-fraud-detection)** | Reproduction and audit of a widely-followed tutorial on 6.36M PaySim transactions. RandomForest matches the published figure to **16 significant figures**; its XGBoost result does not reproduce at all (**0.9992 → 0.7125**), root-caused to XGBoost 2.0 estimating `base_score` from the class prior. Then the real finding: the corrected pipeline scores **1.000 PR AUC with zero false positives**, so I went looking for the leak — a **3-clause rule** gets precision **0.9999** / recall **0.9750** with **one false positive in 6.36M rows**. The tutorial's 99.9% measures recovery of the simulator's fraud script, not fraud detection. |
| **[AIML-2 — Indian Startup Funding, audited](./aiml2-indian-startup-funding)** | Audit of a widely-used Kaggle dataset that loads cleanly and parses without error. Its largest amount is in the **wrong currency** — ₹390 crore read as $3.9B, making a bike-taxi app out-raise Flipkart, and **one cell is 10.1% of the $38.14B total**. Its escape-text defect is invisible to its own diagnostic: `grep -P '\xc2\xa0'` finds **zero matches** while 92 cells hold the literal 8-character `\\xc2\\xa0`. And its time axis runs backwards — deal counts fall 993 → 111 while, benchmarked against Tracxn and Inc42, coverage drops to **14%** and Indian funding actually hit a **record high** in 2019. |
| **[AIML-3 — Churn Prediction & Explainability](./aiml3-churn-explainability)** | End-to-end churn system on 7,043 Telco customers with SHAP explanations and an expected-value decision layer. The headline is a negative result, reported as one: **XGBoost does not significantly beat untuned logistic regression** (PR-AUC 0.6753 vs 0.6690, paired t **p = 0.469**), and the **designated test fold is the weakest of five** (0.6134 vs 0.6609) — kept rather than re-drawn. Calibration was measured, then **declined** under a parsimony rule set before the numbers were seen. A **per-customer expected-value rule** beats the best global threshold using **half the contacts** (713 of 1,406), because break-even risk varies 8.2% → 51.9% with customer value. Segment analysis exposes the model as **near-blind on two-year contracts** (recall 0.143). |
| **[AIML-4 — Demand Forecasting & Inventory Optimization](./aiml4-demand-forecasting)** | Time-series ML on the M5 panel (1M rows, 600 SKU-store series, daily) turned into reorder points. Leakage is **proven absent**, not asserted: every post-origin actual is overwritten and the 51 features are shown not to change — and a test proves the probe itself *can* fail. The headline is a segmented negative result: LightGBM beats a 28-day moving average by **1.4%** overall and **6.3%** on Smooth series, but on the **92% of series that are Intermittent or Lumpy the SBA baseline wins outright**. Predicting zero everywhere attains the **best MASE**, which is why WAPE is the headline. The forecast-driven inventory policy cuts units short **22%** and total cost **7.7%**, and the cost-optimal 95% service level is independently confirmed by a newsvendor critical ratio of 0.9631. |
| **[AIML-5 — Multimodal Product Search](./aiml5-multimodal-product-search)** | A retrieval system, not a notebook: search a 6,000-product catalogue by image, by natural language, or by both — *"this shoe, but in black"*. Products are indexed as **two named vectors** (image and text) rather than one averaged vector, because averaging destroys the ability to say *why* a result ranked where it did; the system reports each channel's contribution per result. Fusion beats both baselines (**nDCG@10 0.617** vs 0.595 text-only and 0.395 image-only), but the finding that matters is that **the optimal image weight varies 5× with intent** — contradiction queries peak at 0.1 and collapse **7.7×** as image weight rises, agreement queries peak at 0.5 — so no global constant is correct and the weight is exposed per request. My first strategy comparison was **unfair and reached the wrong conclusion**; matching the image share across strategies reversed it. Score normalisation, the design's centrepiece, is reported at its true size: **+0.005 to +0.027** nDCG@10. |

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

- **`INSIGHTS.md`** or **`reports/model_report.md`** — the findings, the
  recommendation, and what would have to be measured to confirm it. Every number is
  reproducible from the committed code, and the reports are *generated* from recorded
  run results so they cannot drift from what the code produced.
- **`LEARN.md`** or a documented `src/` — how the method works, why each choice was
  made, and the questions the project invites. Written to be argued with.
- **`tests/`** — assertions on the *claims*, not just the code. Where a project's
  conclusion depends on a property of the data, a test pins that property, so a
  future change that invalidates the conclusion fails loudly.
- **A baseline.** Stated before the model, and reported next to it.
- **A caveat section.** What the project does not establish.

## Running any project

```bash
cd aiml1-payment-fraud-detection      # or aiml2-… / aiml3-… / aiml4-…
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make data      # download the real dataset, or generate a faithful stand-in
make all
make test
```

Large datasets are never committed. Each project either generates a faithful
stand-in or ships a download step, and says which figures require the real file.

## A note on scores

Several projects here report numbers above 0.99. In every case the README says
plainly whether that number means the model is good — and in AIML-1 it does not.
Reporting a high score without establishing that it is *earned* is the failure
mode this portfolio is built to avoid.

Two projects report the opposite: AIML-3 and AIML-4 both conclude that the
sophisticated model does **not** meaningfully beat a simple baseline, and say so in
the first paragraph rather than burying it.
