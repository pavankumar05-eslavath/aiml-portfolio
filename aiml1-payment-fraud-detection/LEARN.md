# How this works, and the questions it invites

## Why the metric choice dominates everything here

At a 0.129% base rate:

- **Accuracy** is useless. Always predicting "legitimate" scores 99.871% and
  catches nothing. `tests/test_fraud.py::test_accuracy_is_a_useless_metric_here`
  pins this.
- **ROC AUC** is misleading but not useless. Its x-axis is FPR = FP/(FP+TN), and
  TN is ~6.35M, so thousands of false positives barely move it. This is how the
  article's XGBoost reads 0.71 while its PR AUC is 0.32.
- **PR AUC (average precision)** is the honest choice. Precision is FP/(TP+FP) —
  no TN term — so it cannot be flattered by an ocean of easy negatives. A random
  ranker scores the base rate, 0.0013, not 0.5.

The practical consequence: a 0.5 threshold is arbitrary. What a fraud team can
actually act on is *how much fraud is caught within the number of alerts analysts
can review*, which is why `INSIGHTS.md` reports an alert-budget table.

## Why the split has to be temporal

`step` is an hour counter running 1→743. A random split trains on hour 700 and
tests on hour 300 — the model gets to see the future. Here it barely changes the
score, because the leak in §4 is so strong, but the habit matters: on any problem
where behaviour drifts, a random split on time-ordered data reports a number you
cannot ship.

Two consequences the code handles explicitly:

- **Raw `step` is dropped** (`src/dataset.py::design_matrix`). Under a temporal
  split every test value exceeds every training value, so a tree can only
  extrapolate — it will send all test rows down one branch. Hour-of-day (`step %
  24`) keeps the cyclical part and generalises.
- **The threshold is tuned on a validation window**, never on test. Tuning on test
  is how a portfolio number quietly becomes fiction.

## The feature that matters, and why it is a trap

```
errBalOrig = newbalanceOrig + amount - oldbalanceOrg
```

The sender's books should reconcile: money out equals balance drop. On real
transactional data a non-zero residual is a genuine anomaly signal.

On PaySim it is a leak. The simulator's fraud agent takes the whole balance, so
fraud satisfies `amount == oldbalanceOrg` and `newbalanceOrig == 0` *exactly*.
That is not fraud behaviour — it is simulator behaviour. Which is why
`src/leakage_audit.py` exists, and why the generator in
`data/generate_paysim.py` reproduces the fraud script rather than just the
schema: the tests then exercise the finding instead of trusting a stored number.

## Why the XGBoost bug is worth a whole stage

`XGBClassifier()` scores 0.712 today and 0.999 on the version the article used.
Nothing errors. Nothing warns.

XGBoost 2.0 began estimating `base_score` from the data instead of fixing it at
0.5. At a 0.13% prior the initial raw margin is `logit(0.0013) ≈ -6.6`, and
boosting pushes from there into the floating-point floor — smallest predicted
probability `3.01e-39`, with 904 of 2,435 frauds ranked below 1e-4.

The diagnostic that proves it is saturation rather than genuine learning is the
**non-monotonic AUC-versus-rounds curve** (0.980 → 0.685 → 0.950). Real learning
does not lose 0.3 AUC in the middle. This is a generally useful habit: when a
metric moves non-monotonically in training effort, suspect numerics before
suspecting the data.

## Reading the code

| file | role |
|---|---|
| `src/dataset.py` | loading, features, temporal split, the rule |
| `src/profile_data.py` | stage 1 — profile, and check the article's claims |
| `src/article_pipeline.py` | stage 2 — the published pipeline as written |
| `src/xgb_diagnosis.py` | stage 3 — isolate the `base_score` regression |
| `src/improved_pipeline.py` | stage 4 — the corrected pipeline |
| `src/leakage_audit.py` | stage 5 — find the rule that beats the models |
| `data/generate_paysim.py` | PaySim stand-in, fraud script included |

## Questions this invites

**"Your model gets 0.999 PR AUC. Is it good?"** — No, and that is the finding. A
three-clause rule gets 0.9999 precision at 0.975 recall on the same data, so the
model has not earned its complexity. The score measures recovery of the
simulator's script.

**"Why not just use the rule in production?"** — Because it only works on PaySim.
It encodes `amount == oldbalanceOrg`, an artefact of how the data was generated.
Real attackers do not drain to the cent. The rule's value here is as a *baseline
that exposes the dataset*, not as a candidate model.

**"Why is ROC AUC 1.000 but PR AUC 0.846 for logistic regression?"** — Because
ROC AUC's FPR denominator contains 1.58M true negatives. The linear model ranks
well overall but puts enough legitimate transactions above some frauds to cost
real precision. Precision has no TN term, so it shows the damage.

**"You changed the split and the score barely moved. Was it worth it?"** — Yes,
for two reasons: the split is what makes the drift in §6 visible (0.084% → 0.282%
fraud rate), and a temporal split is the only one that can be trusted if the leak
were ever fixed. Methodology that only matters sometimes still has to be right
every time.

**"What would you do with a real budget?"** — Move off PaySim. Then build
entity-level history: velocity per account, time since last transaction, amount
against that account's own trailing distribution. PaySim makes this impossible —
`nameOrig` has 6,353,307 distinct values in 6,362,620 rows, so there is no
account history to aggregate. That absence is itself a sign the data is
synthetic.

## Running it

```bash
pip install -r requirements.txt
make data       # small stand-in, no download
make all        # every stage
make download   # the real 480 MB CSV; needed to reproduce INSIGHTS.md exactly
make test
```
