# AI / ML Portfolio — Complete Guide

**Author:** Pavan Kumar Eslavath · IIT Madras
**Purpose:** Five end-to-end machine learning projects, each ending in a defensible claim rather than a score
**GitHub:** [`pavankumar05-eslavath`](https://github.com/pavankumar05-eslavath)

This document explains what each project is, what it found, how to run it, and which files
to read. Companion to the guides in the
[data engineering](https://github.com/pavankumar05-eslavath/data-engineering-portfolio),
[data analyst](https://github.com/pavankumar05-eslavath/data-analyst-portfolio) and
[business analyst](https://github.com/pavankumar05-eslavath/business-analyst-portfolio)
repositories.

---

## Table of contents

1. [The short version](#1-the-short-version)
2. [Setup — running any project in 5 minutes](#2-setup--running-any-project-in-5-minutes)
3. [The five projects in detail](#3-the-five-projects-in-detail)
4. [Reading order — understanding a project fast](#4-reading-order--understanding-a-project-fast)
5. [The bugs found and documented](#5-the-bugs-found-and-documented)
6. [Verified numbers reference](#6-verified-numbers-reference)
7. [Concepts glossary](#7-concepts-glossary)
8. [What is deliberately NOT here](#8-what-is-deliberately-not-here)

---

## 1. The short version

| # | Project | Dataset | The claim |
|---|---|---|---|
| **AIML-1** | [Payment Fraud Detection, audited](./aiml1-payment-fraud-detection) | PaySim, 6,362,620 txns | A **3-clause `if` statement** beats the models. The tutorial's 99.9% measures recovery of the simulator's fraud script, not fraud detection. |
| **AIML-2** | [Indian Startup Funding, audited](./aiml2-indian-startup-funding) | Kaggle, 3,044 rounds | The largest "USD" amount is **rupees**, and one cell is **10.1%** of the dataset total. The time axis runs backwards versus reality. |
| **AIML-3** | [Churn Prediction & Explainability](./aiml3-churn-explainability) | Telco, 7,043 customers | XGBoost does **not** significantly beat untuned logistic regression (p = 0.469). A per-customer expected-value rule beats the best global threshold with **half the contacts**. |
| **AIML-4** | [Demand Forecasting & Inventory](./aiml4-demand-forecasting) | M5, 1,003,600 rows | ML wins on **5% of series**; on the 92% that are intermittent, a classical baseline wins outright. Forecast-driven inventory cuts stockouts **22%**. |
| **AIML-5** | [Multimodal Product Search](./aiml5-multimodal-product-search) | Fashion catalogue, 6,000 products | Image+text fusion beats both single modalities (**nDCG@10 0.617** vs 0.595 / 0.395), but the optimal image weight varies **5×** with query intent, so no global value is right. A **7.7× collapse** when the image is over-weighted on "change this attribute" queries. |

The through-line: **every project reports a baseline, and two of the five conclude that the
sophisticated model did not earn its complexity.** A third — AIML-5 — finds that it *does*, and
then finds that the parameter making it work cannot be fixed globally. That is the point. A
portfolio where every model wins is a portfolio where the baselines were chosen to lose.

---

## 2. Setup — running any project in 5 minutes

```bash
git clone https://github.com/pavankumar05-eslavath/aiml-portfolio.git
cd aiml-portfolio/aiml4-demand-forecasting        # or aiml1-… / aiml2-… / aiml3-… / aiml5-…

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make data        # download the real dataset, or generate a faithful stand-in
make all         # run every stage
make test        # the test suite
make lint        # ruff
```

**No dataset is committed.** Each project either downloads the real file or generates a
stand-in that reproduces the schema *and the defects*, so the pipeline and tests run offline.
Every project states which figures require the real data.

| project | data | `make all` runtime |
|---|---|---|
| AIML-1 | 480 MB PaySim CSV (`make download`) or generated stand-in | ~3 min |
| AIML-2 | 416 KB CSV, CC0 mirror | ~10 s |
| AIML-3 | 977 KB CSV, public mirror | ~2 min |
| AIML-4 | 133 MB of M5 parquet, public mirror | **~17 min** (`train` is 15 of it) |
| AIML-5 | 260 MB of Parquet shards, public mirror, + a ~600 MB CLIP checkpoint | **~6 min** (indexing 1,000 products is most of it) |

---

## 3. The five projects in detail

### AIML-1 — Online Payment Fraud Detection, audited

**Question:** does a widely-followed fraud-detection tutorial actually work?

Reproduction of the tutorial on the full 6,362,620-row PaySim dataset. RandomForest matches
the published figure to **16 significant figures** (`0.9650098729693373`), which proves the
data and split are identical and isolates the one result that does not reproduce: the
tutorial's XGBoost drops **0.9992 → 0.7125**. Root cause: XGBoost 2.0 began estimating
`base_score` from the class prior, driving probabilities to `3.01e-39` and burying **904 of
2,435 frauds** below 1e-4.

The corrected pipeline then scored **1.000 PR AUC with zero false positives** — which is a
reason to go looking for a leak, not to celebrate. The leak:

```python
rule = (newbalanceOrig == 0) & (oldbalanceOrg > 0) \
       & type.isin(["TRANSFER", "CASH_OUT"]) & (amount == oldbalanceOrg)
```

**Precision 0.9999, recall 0.9750 — one false positive in 6.36M rows. No model.**

**Read:** `INSIGHTS.md` → `LEARN.md` → `src/leakage_audit.py`

### AIML-2 — Indian Startup Funding, audited

**Question:** is this popular Kaggle dataset safe to draw conclusions from?

Two defects that survive every obvious check:

1. **The largest "USD" amount is rupees.** Row 61 holds `3,90,00,00,000` in `Amount in USD`
   — ₹390 crore ≈ $55M, the round actually reported. Read as USD it is **$3.9B**, making a
   bike-taxi app out-raise Flipkart's genuine $2.5B. **That one cell is 10.1% of the
   dataset's $38.14B total.**
2. **A defect invisible to its own diagnostic.** The escape junk is stored as the literal
   8 characters `\\xc2\\xa0` — two backslashes, not a real U+00A0. `grep -P '\xc2\xa0'`
   returns **zero matches** while 92 cells are affected, and `.strip()` cannot touch them.

Benchmarked against Tracxn and Inc42: 2015–2017 track reality within ~10%, then coverage
collapses to **14% of deals** by 2019 — while Indian funding hit a **record high** that year.
The dataset shows its steepest decline exactly where the market peaked.

**Read:** `INSIGHTS.md` → `LEARN.md` → `clean.py`

### AIML-3 — Churn Prediction & Explainability

**Question:** who will churn, why, and who is worth contacting?

| | |
|---|---|
| Best model | XGBoost, CV PR-AUC **0.6753** |
| Untuned logistic regression | **0.6690** |
| Difference | +0.0063 (0.23 pooled SD), paired t **p = 0.469** — **not significant** |

The designated test fold is the **weakest of five** (0.6134 vs 0.6609 mean) and was **not
re-drawn**; re-drawing is test-set shopping. Calibration was measured and then **declined**:
the booster was already calibrated (ECE 0.0257), sigmoid made it worse, and isotonic gained
0.0004 — below a parsimony threshold set in config before the numbers existed.

The decision layer is the deliverable. A **per-customer expected-value rule** returns more
value from **713 contacts than the best global threshold does**, because break-even risk
varies **8.2% → 51.9%** with customer value. On 277 of 1,406 customers the value dimension
overrides the risk band — something a "contact the high-risk customers" rule cannot express.

Segment analysis exposes the model as **near-blind on two-year contracts** (recall 0.143,
PR-AUC 0.108), which the aggregate AUC hides completely.

**Read:** `README.md` → `reports/model_report.md` → `LEARN.md` → `src/inference/decision.py`

### AIML-4 — Demand Forecasting & Inventory Optimization

**Question:** how much demand, how uncertain, and how much stock?

Grain **SKU × store × day** on the M5 panel — 600 series, 1,969 days, real prices and SNAP
promotion days.

**Leakage is proven absent, not asserted.** Every target-derived feature is anchored to the
forecast origin. The audit overwrites **every actual past the origin** and shows none of the
51 features change — and a test proves the probe itself *can* fail, by injecting the classic
`rolling_mean_7`-at-target-date mistake.

The headline is segmented:

| demand class | series | best model | WAPE | best baseline | improvement |
|---|---:|---|---:|---|---:|
| Smooth | 31 | lightgbm | 0.5175 | MA28 (0.5523) | **+6.30%** |
| Intermittent | 449 | **sba** | 0.9297 | sba | **0.00%** |
| Lumpy | 103 | **sba** | 0.7974 | sba | **0.00%** |

On the **92% of series that are intermittent or lumpy, the classical SBA baseline wins
outright.** Also: **predicting zero everywhere attains the best MASE** (0.9717), because
averaging per-row scaled errors rewards predicting nothing when 55.6% of actuals are zero.
WAPE scores it 1.0000 — which is why WAPE is the headline and MAPE is not reported at all.

Safety stock is derived from **lead-time forecast error**, not demand variability — the
distinction is the entire reason forecasting has value, since sizing from demand variability
returns the same answer no matter how good the forecast is. Result: **22% fewer units short**
and **7.7% lower total cost**. The cost-optimal 95% service level is independently confirmed
by a newsvendor critical ratio of **0.9631**.

**Read:** `README.md` → `reports/model_report.md` → `src/features/build.py` (the leakage
design) → `src/inventory/policy.py`

---

### AIML-5 — Multimodal Product Search

**Question:** can a shopper search with a photo *and* words at the same time — "this shoe, but
in black" — and can the system say why each result ranked where it did?

This is the one project here that is a **running system** rather than an analysis: a FastAPI
service, a Qdrant vector index, a React interface, 444 tests. It is also the only one where a
neural model is load-bearing rather than decorative — see §8.

**The design decision.** The obvious implementation averages the image and text query
embeddings into one vector and does a single lookup. That is implemented and benchmarked here
(`FUSION_STRATEGY=embedding_fusion`), but it is not the default, because averaging destroys
attribution: no result can be traced back to "matched visually" versus "matched the words".
Instead each product carries **two named vectors** in one Qdrant collection, producing up to
five retrieval channels fused at the *score* level, so every response reports each channel's
raw cosine, normalised score, weight and rank.

**Fusion earns its place** — measured on one fixed query set so difficulty is held constant:

| channels used | nDCG@10 | MRR | P@10 |
|---|---:|---:|---:|
| image only | 0.395 | 0.527 | 0.385 |
| text only | 0.595 | 0.804 | 0.550 |
| **both, fused** | **0.617** | 0.797 | **0.580** |

**But the weighting cannot be fixed.** Multimodal queries divide into two kinds with opposite
optima, so they are swept separately:

| query style | best `IMAGE_WEIGHT` | nDCG@10 | at the other's optimum |
|---|---:|---:|---:|
| contradiction — *"but in black"* | **0.1** | 0.429 | 0.279 |
| agreement — text restates the image | **0.5** | 0.831 | 0.805 |

Contradiction queries collapse **7.7×** (0.429 → 0.056) as image weight rises, because
weighting the image drags results back toward the attribute the user asked to *change*. Both
groups still beat their single-modality baselines at their own optimum — so the architecture
holds — but **no single global weight is right**, which is why the weight is a per-request
parameter and a slider in the UI rather than a constant.

**An unfair comparison that reached the wrong conclusion.** The strategy sweep first pitted
`weighted_sum` at its default 50/50 weighting against `embedding_fusion` at a text-heavy
α = 0.3, and "showed" the naive single-vector approach winning. Since half the benchmark
strongly prefers text, that was an artefact of the weighting, not the mechanism. Pinning the
image share equal reversed the ordering (0.598 vs 0.574). The script now enforces the match.
`embedding_fusion` does keep the best **MRR** (0.832) — a real trade-off, reported as one.

**The design's centrepiece, reported at its true size.** Channels sit on genuinely different
scales (image↔image 0.7305, text↔text 0.4499, image↔text 0.3124), so normalising before
combining is sound reasoning. Measured payoff: **+0.005 to +0.027** nDCG@10. It is kept
because it makes the weights *mean what they say*, not because it transforms quality. The
lexical channel measured **worse than off** and ships disabled.

**Read:** `INSIGHTS.md` → `docs/evaluation.md` (methodology and its ten caveats) →
`backend/app/services/fusion.py` (the ranking layer — pure, and the one module at 100%
coverage) → `docs/architecture.md`

---

## 4. Reading order — understanding a project fast

1. **`README.md`** — the claim and the evidence, in the first three paragraphs.
2. **`reports/model_report.md`** or **`INSIGHTS.md`** — generated from recorded run results,
   so it cannot drift from what the code produced.
3. **`LEARN.md`** where present — why each choice was made, written to be argued with.
4. **`tests/`** — the fastest way to see what the project actually guarantees.
5. **The one interesting module** — named at the end of each project section above.

---

## 5. The bugs found and documented

Every one of these was caught by a test, an assertion, or reading output that looked wrong —
not by inspection after the fact. They are documented in place because the *class* of mistake
generalises.

| # | Project | Bug | Why it mattered |
|---|---|---|---|
| 1 | AIML-1 | XGBoost 2.0 estimates `base_score` from the class prior | The tutorial's published figure is unreproducible on any current version |
| 2 | AIML-2 | Nullable-dtype `NA != "x"` returns `NA`, and `.where()` treats it as `False` | Silently divided **all 2,073 amounts** by 71; invisible per row, wrong only in the total |
| 3 | AIML-2 | Same trap inverted `money_subset()` | Returned the 6 *flagged* rows instead of the ~2,000 good ones |
| 4 | AIML-2 | `download.py` compared the CSV header as an **ordered tuple** | Rejected the correct file; only surfaced on a fresh clone |
| 5 | AIML-3 | SHAP phrasing derived from one-hot **column names** | Reported "pays by electronic check" as *protective* — the opposite of the truth |
| 6 | AIML-3 | `fix_amount` returned nullable `Int64` when all values were whole | Worked on the real CSV, crashed on the stand-in — a bug whose existence depended on the data |
| 7 | AIML-3 | `TreeExplainer` hard-coded | Crashed exactly when the linear model legitimately won |
| 8 | AIML-4 | Training origins allowed at the fold boundary | **A real leak** — 28-day labels landed inside the evaluation window |
| 9 | AIML-4 | Croston implemented as size × rate | Algebraically **identical to a moving average**; a duplicate baseline masquerading as a method |
| 10 | AIML-4 | Empirical quantile recomputed from one forecast origin | Scenario table **identical at every service level** — the quantile of one observation is that observation |
| 11 | AIML-4 | A hard-coded narrative sentence | Claimed two estimators "disagree" after a fix made them agree to 1.3%; now computed from the data |
| 12 | AIML-5 | The strategy comparison used **unequal weightings** | Concluded the naive single-vector fusion beat score-level fusion; matching the image share **reversed** it |
| 13 | AIML-5 | `coverage` silently under-reported tested code | SQLAlchemy's async bridge runs the DBAPI in **greenlets**; without `concurrency=[thread,greenlet]` coverage loses its trace after any `await` touching the database — a route at 100% reported as 83% |
| 14 | AIML-5 | `.gitignore` denied `data/full|raw|local` **individually** | A later-added `data/eval/` was trackable; **6,000 images** were one `git add` from being committed |
| 15 | AIML-5 | A bare `models/` ignore pattern | Silently excluded `backend/app/models/` — the **entire ORM package** — from the repository |
| 16 | AIML-5 | `Path.is_symlink()` called **after** `.resolve()` | Always `False`; a security check that looked meaningful and guaranteed nothing |
| 17 | AIML-5 | The sample fetcher downloaded **one of two** dataset shards | Half the catalogue arrived with no image and indexed as text-only; found by cloning the repo and following my own README |

Bugs 2, 3, 10 and 13 share a shape worth internalising: **the code ran, produced plausible
numbers, and was wrong.** Only a total, an inverse, a table that failed to vary, or a coverage
figure that disagreed with a spy test gave them away.

Bugs 12 and 17 share a different shape: **the mistake was in the measurement, not the
system.** An unfair benchmark and a half-downloaded dataset both produce output that looks
entirely reasonable. Neither was found by reading the code — one by asking whether the
comparison was fair, the other by cloning the repository and following the README verbatim.

---

## 6. Verified numbers reference

Every figure below is produced by the committed code.

**AIML-1** · PaySim 6,362,620 txns · fraud rate 0.13% (1:774)
RandomForest ROC-AUC `0.9650098729693373` (exact match to published) · tutorial XGBoost
0.9992 → **0.7125** · 3-clause rule: precision **0.9999**, recall **0.9750**, 8,008 TP /
**1 FP** / 205 FN

**AIML-2** · 3,044 rounds · Jan 2015–Jan 2020
Total as published **$38.14B** → corrected **$34.30B** · one cell = **10.1%** · 92 cells with
literal escape text, **0** real U+00A0 · 42 rows in 18 groups with contradictory labels ·
train/test contamination **5 → 0** after group-aware splitting · coverage 2019: **14%** of
deals

**AIML-3** · 7,043 customers · churn 26.54%
CV PR-AUC: xgboost **0.6753**, RF 0.6722, LR tuned 0.6704, LR plain **0.6690**, dummy 0.2645 ·
paired t **p = 0.469** · fold spread 0.6134–0.6868 · calibration Brier 0.1425 (none) /
0.1437 (sigmoid) / 0.1421 (isotonic) · threshold 0.13 → 63,446 CU vs per-customer rule
**63,799 CU** from **713** contacts · segments validated, lift **15.5×**, max gap 0.0569 ·
two-year contracts recall **0.143**

**AIML-4** · 1,003,600 rows · 600 series · zero share 58.90%
Validation WAPE: lightgbm **0.7397**, RF 0.7473, MA28 0.7503, SBA 0.7901, seasonal naive
0.8668, zero 1.0000 (MASE **0.9420**) · test: RF **0.7726**, lightgbm 0.7766, MA28 0.7870 ·
per class: Smooth **+6.30%**, Erratic +0.52%, Intermittent **0.00%**, Lumpy **0.00%** ·
intervals **75.04%** vs nominal 80% · safety stock normal 11.60 / empirical 11.92 (ratio
0.987, 60 windows) · units short **526.7 vs 675.8** · total cost **3,231.88 vs 3,500.94 CU** ·
cost optimum **95%**, newsvendor ratio **0.9631** · SNAP-day WAPE 0.8033 vs 0.7619

**AIML-5** · 6,000 products · 46 benchmark queries · CLIP ViT-B/32 (512-d), CPU
nDCG@10: text **0.677**, image **0.878**, multimodal **0.617**, all queries **0.694** · MRR
0.891 / **1.000** / 0.797 · ablation on the multimodal set: image-only 0.395 < text-only 0.595
< **both 0.617** · `IMAGE_WEIGHT` optima **0.1** (contradiction, 0.429) vs **0.5** (agreement,
0.831), collapse to **0.056** at 1.0 · strategies at matched image share: weighted_sum
**0.598**, rrf 0.575, embedding_fusion 0.574 (MRR **0.832**) · normalisation zscore **0.682** /
none 0.655–0.677 / minmax 0.646 · cross-modal top-1 **0.812** templated vs **0.771** raw name ·
cosines image↔image **0.7305**, text↔text 0.4499, image↔text **0.3124** (gap **+0.4181**) ·
indexing **19–21 products/s**, re-index unchanged **0.1 s** vs 46.8 s · search latency ~28 ms
text / ~100 ms multimodal · **444 tests, 90% coverage**

---

## 7. Concepts glossary

| term | meaning, and why it appears here |
|---|---|
| **PR-AUC** | Area under precision–recall. Preferred to ROC-AUC under class imbalance because the negative class dominates ROC and flatters the model. |
| **WAPE** | Total absolute error ÷ total actual. The headline in AIML-4 because it stays defined when individual actuals are zero. |
| **MASE** | Error scaled by in-sample seasonal-naive error. Scale-free — but on intermittent demand it *rewards predicting zero*, which AIML-4 demonstrates. |
| **MAPE** | Deliberately absent. Undefined when the actual is zero, which is ~59% of AIML-4's observations. |
| **ADI / CV²** | Average demand interval and squared CV of *non-zero* sizes. Together they give the Syntetos-Boylan classes: Smooth / Erratic / Intermittent / Lumpy. |
| **Croston / SBA** | The standard intermittent-demand methods. SBA multiplies by `1 − α/2` to correct Croston's upward bias — visible in AIML-4 as bias +0.047 → −0.029. |
| **Rolling-origin validation** | Refit at each origin and forecast forward. Replaces random splitting, which lets a model interpolate between days it has already seen. |
| **Forecast origin** | The last date whose actuals may be used. Every AIML-4 feature is anchored to it; that is what makes leakage checkable. |
| **Safety stock** | Buffer sized from **forecast error** spread, not demand variability. The distinction is why a better forecast reduces stock. |
| **Newsvendor critical ratio** | `Cu / (Cu + Co)` — the analytically optimal service level, used in AIML-4 as an independent check on the simulated optimum. |
| **Calibration / ECE** | Whether a predicted 0.7 means a 70% chance. Separate from ranking, and required before multiplying probability by money. |
| **Expected-value threshold** | Act when `p × value × success_rate > cost`. Because value varies per customer, so does the break-even probability. |

---

## 8. What is deliberately NOT here

- **No dashboards or SQL metric modelling.** That is analyst work and lives in the
  [data analyst](https://github.com/pavankumar05-eslavath/data-analyst-portfolio) and
  [business analyst](https://github.com/pavankumar05-eslavath/business-analyst-portfolio)
  repositories.
- **No orchestration or warehouse modelling.** That lives in the
  [data engineering portfolio](https://github.com/pavankumar05-eslavath/data-engineering-portfolio).
- **No deep learning where it would be decoration.** AIML-1 to AIML-4 are tabular or panel
  problems, where gradient boosting is the honest choice and a neural network would add
  nothing. AIML-5 is the exception that states the rule: multimodal retrieval *requires* a
  shared image-text embedding space, which no gradient-boosting model provides — and the
  project verifies that the space is genuinely shared (cross-modal top-1 **0.812**) rather
  than assuming it. The encoder is also **pretrained, not trained**: nothing here fine-tunes a
  large model.
- **No leaderboard chasing.** AIML-1 exists to show a 99.9% score can be meaningless,
  AIML-3 and AIML-4 both conclude the complex model did not earn its place, and AIML-5 reports
  its own centrepiece (score normalisation) as worth only +0.005 to +0.027.
- **No committed datasets.** Every project downloads the real file or generates a
  defect-preserving stand-in.
- **No claimed business results.** Currency figures in AIML-3 and AIML-4 come from **declared
  assumptions** in `configs/config.yaml`, are reported in neutral currency units, and ship with
  sensitivity analysis showing how the recommendation moves when the assumptions do.
