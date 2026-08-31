# AIML-1 — Online Payment Fraud Detection: a tutorial, audited

Reproduction and audit of a widely-followed tutorial,
[Online Payment Fraud Detection using Machine Learning in Python](https://www.geeksforgeeks.org/machine-learning/online-payment-fraud-detection-using-machine-learning-in-python/)
(GeeksforGeeks), on the full PaySim dataset — 6,362,620 transactions.

The pipeline runs. Its numbers mostly reproduce. But one of its three results
cannot be reproduced on a current XGBoost, and the dataset itself turns out to be
solvable by a hand-written `if` statement — which makes the tutorial's headline
99.9% meaningless as evidence of model quality.

The finding, not the model, is the deliverable.

*The article's steps were independently reimplemented and its prose rephrased;
no tutorial text is reproduced here.*

---

## What this demonstrates

| | |
|---|---|
| **Reproduction with a verified anchor** | RandomForest matches the published figure to 16 significant figures (`0.9650098729693373`), which proves the data and split are identical and isolates the one result that does not reproduce. |
| **A real library regression, root-caused** | `XGBClassifier()` scores **0.9992 → 0.7125**. Cause: XGBoost 2.0 began estimating `base_score` from the class prior, driving probabilities to `3.01e-39` and burying **904 of 2,435 frauds** below 1e-4. Proven by a non-monotonic AUC-vs-rounds curve and fixed two ways. |
| **Leakage found by refusing to accept a good score** | The corrected pipeline hit **1.000 PR AUC with zero false positives** in 1.58M transactions. Instead of shipping that, I went looking for the shortcut — and found a 3-clause rule at **precision 0.9999, recall 0.9750, one false positive in 6.36M rows**. |
| **Metric discipline** | The article's claim that the classes are balanced is false (**1:774**), it prints ROC AUC labelled "Accuracy", and never reports PR AUC — which at a 0.13% base rate is where its default XGBoost drops to **0.32**. |
| **Honest evaluation** | Temporal split, threshold tuned on validation, results reported as an analyst alert-review budget rather than at an arbitrary 0.5 cutoff. |

Full evidence in **[INSIGHTS.md](./INSIGHTS.md)**. Method and the questions it
invites in **[LEARN.md](./LEARN.md)**.

---

## The headline

```python
rule = (newbalanceOrig == 0) & (oldbalanceOrg > 0) \
       & type.isin(["TRANSFER", "CASH_OUT"]) \
       & (amount == oldbalanceOrg)
```

Across all 6,362,620 rows: **precision 0.9999, recall 0.9750 — 8,008 TP, 1 FP,
205 FN.** No model.

PaySim scripts its fraud agents to empty the victim's account, leaving an exact
arithmetic signature that essentially never occurs in legitimate traffic. The
~0.999 scores measure how well a model rediscovers the simulator's script, not how
well it would detect real payment fraud.

---

## Running it

```bash
pip install -r requirements.txt

make data     # generate a small PaySim stand-in (no download)
make all      # profile -> article -> diagnose -> improved -> audit
make test     # 19 tests, including the leakage guards
```

To reproduce the figures in `INSIGHTS.md` exactly, fetch the real dataset first:

```bash
make download   # ~480 MB from the Drive link published with the article
make all
```

Individual stages:

```bash
make profile    # data profile; checks the article's claims
make article    # the published pipeline, reproduced as written
make diagnose   # why its XGBoost result no longer reproduces
make improved   # temporal split, class weighting, PR-AUC, tuned threshold
make audit      # the rule that beats the models
```

The 480 MB CSV is gitignored. Everything except `make download` works without it,
because `data/generate_paysim.py` regenerates the schema, the 1:774 imbalance and
PaySim's fraud script — so the tests exercise the finding rather than trusting a
stored number.

Verified on Python 3.11.15, pandas 3.0.5, scikit-learn 1.9.0, xgboost 3.2.0.

## Layout

```
data/generate_paysim.py    PaySim stand-in, fraud script included
data/download.py           fetch the real 480 MB CSV
src/dataset.py             loading, features, temporal split, the rule
src/profile_data.py        stage 1 - profile and claim checks
src/article_pipeline.py    stage 2 - the published pipeline
src/xgb_diagnosis.py       stage 3 - the base_score regression
src/improved_pipeline.py   stage 4 - corrected pipeline
src/leakage_audit.py       stage 5 - the rule that beats the models
src/run.py                 stage runner
tests/test_fraud.py        19 tests; leakage and split guards
```

## Caveat

This project deliberately does not claim a production-ready fraud model. It
claims the opposite: that the dataset everyone uses for this tutorial cannot
support such a claim. Moving to data with real fraud (IEEE-CIS, Sparkov) is the
next step, and is where entity-level history — velocity, recency, amount against
an account's own trailing distribution — would actually matter. PaySim makes that
impossible: `nameOrig` holds 6,353,307 distinct values across 6,362,620 rows.
