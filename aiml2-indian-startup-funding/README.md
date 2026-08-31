# AIML-2 — Indian Startup Funding: a dataset, audited

Audit of the Kaggle dataset
[Indian Startup Funding](https://www.kaggle.com/datasets/sudalairajkumar/indian-startup-funding)
(CC0; collected by [trak.in](https://trak.in)) — 3,044 funding rounds, Jan 2015 → Jan 2020.

The data loads cleanly, parses without error, and reports a null count that looks
unremarkable. Two of its defects are invisible to that check, and both change the
conclusion: the largest amount in the file is in the **wrong currency**, and the **time axis
is unusable after 2017**.

The finding, not the cleaning, is the deliverable.

---

## What this demonstrates

| | |
|---|---|
| **A defect that hides from its own diagnostic** | The escape junk is stored as the 8 *characters* `\\xc2\\xa0` — two backslashes, not a real U+00A0. `grep -P '\xc2\xa0'` returns **zero matches** while 92 cells are affected, and `.strip()` cannot touch them. The obvious regex fix matches one backslash, leaves the other, and the bug survives while appearing fixed. |
| **A currency error worth 10% of the dataset** | `sr_no 61` carries `3,90,00,00,000` in a column headed `Amount in USD`. That is ₹390 crore ≈ **$55M**. Read as USD it is **$3.9B** — making a bike-taxi app the largest venture round in Indian history, ahead of Flipkart's genuine $2.5B. **One cell = $3.85B = 10.1% of the total.** |
| **A trend that runs backwards** | Deal counts fall 993 → 111 from 2016 to 2019, which reads as a funding collapse. Benchmarked against Tracxn and Inc42: 2015–2017 track reality within ~10%, then coverage drops to **14% of deals** by 2019. Indian funding hit a **record high** in 2019. The decline is the scraper, not the market. |
| **Non-random missingness, quantified** | 32% of rounds have no amount, and disclosure runs **56% for Seed vs 100% for Series B** — so any "average deal size" over disclosed rows is biased upward by construction. |
| **Entity resolution that refuses to over-merge** | `Ola`/`Ola Cabs` merge; `Ola Electric` must not. `OYO`/`OyoRooms` merge; `OYOfit` and `FroyoFit` must not. Explicit key maps, not fuzzy matching, with tests pinning **both** directions. |
| **A stated baseline** | `make naive` runs the obvious uncleaned analysis. It is kept in CI because the gap between it and the audited result *is* the measurement. |

Full evidence in **[INSIGHTS.md](./INSIGHTS.md)**. Method, reasoning and the questions it
invites in **[LEARN.md](./LEARN.md)**.

---

## The headline

```
                          as published        corrected
largest round             Rapido  $3.90B      Flipkart  $2.50B
dataset total                     $38.14B               $34.30B
2019 vs 2016 deals        111 vs 993          14% coverage vs 98%
#1 investor               "Undisclosed        Sequoia Capital India
                           Investors" (83)     (122 deals)
```

Three of those four "as published" figures appear in public write-ups of this dataset.

---

## Running it

```bash
pip install -r requirements.txt

make data      # generate the defect-preserving stand-in (no download)
make all       # profile -> naive -> audited -> coverage -> export
make test      # 44 tests, including the coverage and over-merge guards
```

To reproduce the figures in `INSIGHTS.md` exactly, fetch the real dataset first:

```bash
make download  # ~416 KB, CC0, from a GitHub mirror (Kaggle needs auth)
make all
```

Individual stages:

```bash
make profile   # profile the file and locate its defects
make naive     # THE BASELINE: the obvious, uncleaned analysis
make audited   # the corrected analysis, printed next to the baseline
make coverage  # validate against published industry totals
make export    # write data/funding_clean.csv + data/investor_deals.csv
```

The CSV is gitignored. Everything except `make download` works without it, because
`data/generate_sample.py` reproduces the *defects* — literal escape text, Indian digit
grouping, an INR value at `sr_no 61`, the merge/no-merge entity pairs, placeholder
investors, malformed dates — so the tests exercise the findings rather than trusting stored
numbers.

Verified on Python 3.11.15, pandas 3.0.5, matplotlib 3.10.8.

## Layout

```
data/download.py           fetch the real 3,044-row CSV (CC0 mirror)
data/generate_sample.py    defect-preserving stand-in, runs offline
src/dataset.py             all cleaning; single source of truth
src/profile_data.py        stage 1 - locate the defects
src/naive_pipeline.py      stage 2 - the baseline, deliberately uncleaned
src/audited_pipeline.py    stage 3 - corrected analysis, beside the baseline
src/coverage_check.py      stage 4 - validate against published totals
src/run.py                 stage runner
tests/test_funding.py      44 tests; claim, coverage and over-merge guards
```

## Output columns

`data/funding_clean.csv` — one row per round. **Use `amount_usd_adj`, not `amount_usd`**:
the latter is as-published and contains the INR figure. `amount_flag` marks
`confirmed_inr_not_usd` and `review_outlier`.

`data/investor_deals.csv` — one row per investor-round pair, placeholders removed, aliases
merged. Join back on `sr_no`.

## Caveat

**This project does not claim a clean dataset.** It claims a bounded one: good for
2015–2017 cross-sectional questions (geography, sector mix, investor activity, round-size
distribution), unusable for time-series claims through 2020.

Specific limits:

- **One currency error confirmed, five flagged, unknown remainder.** Rapido was verified
  against three outlets. The five `review_outlier` rows are a review queue, not a verdict —
  at least one (Automation Anywhere, $300M) is genuine. Errors below the $250M threshold
  are undetectable without a deal-level external join, because there no ranking looks absurd.
- **`KNOWN_REAL_MEGA` is a hand-curated allowlist.** It is the weakest component here:
  embedded judgement rather than a rule.
- **Entity resolution covers the head, not the tail.** Seven alias pairs were resolved from
  the top of the leaderboard. The top 15 is sound; "number of distinct startups funded"
  is not.
- **`Private Equity` labels 1,361 rows with a $6.0M median**, so it plainly includes
  ordinary VC rounds. Stage-based conclusions are weak, and recovering true stage needs an
  external join.
- **The 76%-in-three-metros figure may be about trak.in, not India.** Coverage analysis
  shows how much is missing per year but not *which* rounds. If the source under-covered
  tier-2 cities, concentration is overstated. This is the largest unresolved threat to the
  surviving findings.

Next step is a deal-level join against a fuller source (Tracxn, Venture Intelligence,
Crunchbase) to identify which rounds are absent and test whether the gap is
small-round-biased or sector-biased. Until that exists, the 2018–2020 rows should not be
used for anything.
