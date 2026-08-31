# Findings

All figures below come from the real PaySim CSV — 6,362,620 rows — fetched with
`make download`. Reproduce with `make profile article diagnose improved audit`.

Verified on Python 3.11.15, pandas 3.0.5, scikit-learn 1.9.0, xgboost 3.2.0.

---

## 1. The tutorial's central claim about the data is false

Directly below a `value_counts()` showing 6,354,407 legitimate against 8,213
fraudulent, the article states the classes are already balanced and no sampling
is needed.

| | count |
|---|---|
| legitimate | 6,354,407 |
| fraudulent | 8,213 |
| **fraud rate** | **0.1291%** |
| **imbalance** | **1:774** |

This is not a detail. It is the entire difficulty of the task, and every
downstream choice the article makes — no class weighting, ROC AUC as the only
metric, a 0.5 decision threshold — follows from getting it wrong.

Two further facts the article never surfaces:

**Fraud is confined to two of five transaction types.**

| type | n | frauds | rate |
|---|---|---|---|
| CASH_OUT | 2,237,500 | 4,116 | 0.184% |
| TRANSFER | 532,909 | 4,097 | 0.769% |
| CASH_IN | 1,399,284 | 0 | — |
| DEBIT | 41,432 | 0 | — |
| PAYMENT | 2,151,495 | 0 | — |

**`nameOrig` is an identifier, not a feature** — 6,353,307 distinct values across
6,362,620 rows. The article drops it, correctly, without saying why.

---

## 2. One of its three results does not reproduce

| Model | Article | Reproduced | |
|---|---|---|---|
| LogisticRegression | 0.8850 | 0.9362 | differs |
| XGBClassifier | 0.9992 | **0.7125** | ✗ |
| RandomForestClassifier | 0.9650098729693373 | 0.9650098729693373 | ✓ exact |

The RandomForest figure matches to sixteen significant figures, which confirms
the data and the split are identical and isolates XGBoost as the anomaly.

### Cause: `base_score`

XGBoost 2.0 changed `base_score` from a fixed 0.5 to an estimate of the class
prior. Here that prior is 0.00129731.

```
auto        used=0.00129731  ROC AUC=0.712453  min_p=3.01e-39  frauds below 1e-4: 904/2435
forced 0.5  used=0.50000000  ROC AUC=0.999174  min_p=3.99e-10  frauds below 1e-4:   7/2435
```

Starting every prediction at a raw margin near −6.6 drives probabilities into the
floating-point floor: the smallest predicted probability is `3.01e-39`, and **904
of 2,435 test frauds are ranked below 1e-4**. The ranking collapses.

The signature of saturation rather than learning is that ROC AUC is
**non-monotonic in boosting rounds** — a model that is genuinely improving does
not shed 0.30 AUC between round 5 and round 50 and then win it back by round 300:

| rounds | 1 | 5 | 10 | 25 | 50 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|---|---|
| ROC AUC | 0.980 | 0.944 | 0.774 | 0.686 | 0.685 | 0.712 | 0.912 | 0.950 |

Either fix restores the published figure:

| | ROC AUC | PR AUC |
|---|---|---|
| default | 0.712453 | 0.315570 |
| `scale_pos_weight=769.8` | 0.998938 | 0.959864 |
| `base_score=0.5` | 0.999174 | 0.964994 |

So the article was right *for the XGBoost of its day*. Copy it today and you get
a materially worse model with no warning. Note also the PR AUC of **0.32** where
ROC AUC reads 0.71 — at a 0.13% base rate ROC AUC is dominated by easy negatives.

---

## 3. The corrected pipeline scores ~1.000 — which is the problem

Temporal split, class weighting, engineered balance residuals, PR AUC, threshold
tuned on validation:

```
train  step <= 281   n=3,820,599   frauds=3,193   rate=0.0836%
val    step <= 335   n=  961,244   frauds=  556   rate=0.0578%
test   step <= 743   n=1,580,777   frauds=4,464   rate=0.2824%
```

| Model | PR AUC | ROC AUC |
|---|---|---|
| RandomForest (balanced) | **0.999998** | 1.000000 |
| XGBClassifier (`scale_pos_weight=1195.6`) | 0.999967 | 1.000000 |
| LogisticRegression (scaled, balanced) | 0.845521 | 0.996073 |
| baseline: rank by `amount / oldbalanceOrg` | 0.0022 | 0.3528 |

At the validation-tuned threshold: **TN 1,576,313 · FP 0 · FN 37 · TP 4,427**,
precision 1.0000, recall 0.9917.

Zero false positives across 1.58 million transactions is not a result to
celebrate. It means a near-deterministic rule exists.

---

## 4. The rule — and why none of this is machine learning

```python
rule = (newbalanceOrig == 0) & (oldbalanceOrg > 0) \
       & type.isin(["TRANSFER", "CASH_OUT"]) \
       & (amount == oldbalanceOrg)
```

Across all 6,362,620 rows:

| | |
|---|---|
| precision | **0.9999** |
| recall | **0.9750** |
| TP / FP / FN | 8,008 / **1** / 205 |

**One false positive in 6.36 million transactions, from three clauses and no
model.** And `P(fraud | amount == oldbalanceOrg) = 0.9991`.

Single-signal rankings:

| signal | ROC AUC | PR AUC |
|---|---|---|
| `amount == oldbalanceOrg` (one boolean) | 0.9881 | 0.9754 |
| the 3-clause rule | 0.9875 | 0.9750 |
| `-errBalOrig` | 0.8897 | 0.0065 |
| `amount` | 0.7899 | 0.0187 |

PaySim scripts its fraud agents to empty the victim's account. That leaves an
exact arithmetic signature which essentially never arises in legitimate traffic,
and any expressive model finds it in its first few splits — which is exactly what
the importances show (`drainedOrig` 0.495, `errBalOrig` 0.104).

### What this means

The ~0.999 figures — the article's and the corrected pipeline's alike — measure
**how well a model rediscovers the simulator's fraud script**, not how well it
would detect real payment fraud. Real fraud does not announce itself with
exact-balance arithmetic.

PaySim is a legitimate exercise in pipeline mechanics at realistic data volume.
Its scores are not evidence that a model is production-ready, and a portfolio
that reports 99.9% on it without this caveat is making a claim it cannot support.

---

## 5. A more honest operating view

Recall at a fixed analyst review budget, which is how a fraud team actually buys
detection:

| review top | alerts | frauds caught | recall | precision |
|---|---|---|---|---|
| 0.01% | 158 | 158 | 0.0354 | 1.0000 |
| 0.05% | 790 | 790 | 0.1770 | 1.0000 |
| 0.10% | 1,580 | 1,580 | 0.3539 | 1.0000 |
| 0.50% | 7,903 | 4,464 | 1.0000 | 0.5648 |
| 1.00% | 15,807 | 4,464 | 1.0000 | 0.2824 |

## 6. Drift worth noticing

The fraud rate moves from 0.0836% in the training window to 0.2824% in the test
window — a **3.4× shift inside one month** of simulated time. Any threshold fixed
on the training window silently changes behaviour by the test window, which is an
argument for calibrating on a rolling basis rather than once.

---

## Recommendation

1. **Do not quote PaySim scores as model quality.** Report the rule baseline
   alongside any model; a model that cannot beat three clauses has not earned its
   complexity.
2. **Pin `base_score` or set `scale_pos_weight` explicitly** on any imbalanced
   XGBoost problem. Relying on defaults across a major version cost 0.29 AUC here
   silently.
3. **Lead with PR AUC** at base rates below ~1%, and never label ROC AUC
   "Accuracy".
4. **Move to a dataset with real fraud** (IEEE-CIS, Sparkov) before drawing any
   conclusion about what a model would catch in production.
