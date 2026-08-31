# LEARN — how this audit works, and why each choice was made

Written to be argued with. If you disagree with a decision here, the code is structured so
you can change it and see what moves.

---

## The shape of the problem

This dataset is 3,044 rows and ten columns. It is small enough to read end to end, which
makes it a good place to learn something uncomfortable: **the defects that matter are not
the ones a null-count finds.**

A standard `df.info()` / `df.isna().sum()` pass on this file reports missing values in five
columns and nothing else. It does not report that the largest number in the file is in the
wrong currency, or that the time axis is unusable after 2017. Both of those change the
conclusion. Neither is visible from inside the file.

That gives the project its structure: a baseline that does the obvious thing, then four
stages that each attack a different *class* of defect.

## Why there is a deliberately-wrong baseline

`src/naive_pipeline.py` reads the CSV with a plain `read_csv` and prints the headline
numbers. It is kept in the repo, and run in CI, because the deliverable is the *gap*.
"Cleaning matters" is a platitude; "cleaning moves the total by 10% and changes which
company is #1" is a measurement.

This is the portfolio's baseline convention applied to a dataset audit: the baseline is not
a dumb model, it is the analysis a competent person produces in ten minutes. If the audit
cannot beat that, the audit is not worth its complexity.

## Class 1 — defects that hide from the obvious check

The escape-text bug is the instructive one.

The published CSV contains cells that *look* like they have a leading non-breaking space.
The natural diagnosis is a U+00A0, and the natural check is:

```bash
grep -cP '\xc2\xa0' data/startup_funding.csv   # -> 0
```

Zero matches. The file looks clean. But `od -c` shows what is actually stored:

```
2607 , \ \ x c 2 \ \ x a 0 1 0 / 7 / 2 0 1 5
```

Two backslashes. The cell holds the eight **characters** `\\xc2\\xa0` — text that was
escaped twice somewhere upstream and written literally. There is no U+00A0 anywhere in the
file, which is why the grep is clean and why `.str.strip()` does nothing.

Then the subtler trap. The obvious fix removes `\xc2` and `\xa0`:

```python
s.str.replace(r"\\x[0-9a-fA-F]{2}", " ", regex=True)
```

The regex matches *one* backslash plus `xc2`. Given two backslashes it consumes the second
and leaves the first. `'\\xc2\\xa0Gurgaon'` becomes `'\ \ Gurgaon'` → `'\ Gurgaon'`. Still
not `'Gurgaon'`. The bug survives in a new form, the city count stays inflated, and
everything still *looks* fixed because the visible `xc2` is gone.

The `\\+` quantifier in `unescape()` is there for that reason, with a `\\+` sweep after it.
`tests/test_funding.py::test_unescape_leaves_no_backslash_debris` asserts on the absence of
debris rather than the presence of the fix, because the half-fix passes any test that only
checks `'xc2' not in value`.

**Generalisable lesson:** when a defect is invisible to your diagnostic, your diagnostic is
part of the problem. Drop to bytes (`od -c`, `repr()`) before theorising.

## Class 2 — defects that parse cleanly and mean something else

`Amount in USD` uses Indian digit grouping: `20,00,00,000` is 20 crore = 200,000,000.

Nothing errors. `int("20,00,00,000".replace(",", ""))` gives the right answer, and even
pandas' `thousands=","` gives the right answer, because it strips separators without
validating grouping. I originally wrote a test asserting `thousands=","` was *wrong*; it
failed, because the claim was false. The test now pins it as **safe**, so nobody "fixes" it
later. Being wrong about the mechanism while right about the risk is worth recording.

The real hazard is human. `3,90,00,00,000` reads as "about 3.9 billion" to an eye trained
on international grouping. It is ₹390 crore. Which brings us to the currency:

**The column header is a claim, not a fact.** `Amount in USD` contains at least one INR
value. No internal consistency check can catch that — ₹390 crore and $390M are both
plausible-looking numbers in a column of plausible-looking numbers. The only way to catch
it is to notice that the *ranking* it produces is absurd (a bike-taxi app out-raising
Flipkart) and go check the round against the press.

So the detection rule is not "validate the format", it is **"be suspicious of your own
top-N"**. Outlier review is not about statistical thresholds here; the value is only 1.5×
Flipkart's, well within a plausible range. It is about domain knowledge: Rapido in 2019 was
not a $3.9B-round company.

`OUTLIER_THRESHOLD = 250_000_000` with a `KNOWN_REAL_MEGA` allowlist encodes exactly that,
and it is the weakest part of this project — a hand-maintained list of "rounds I happen to
know are real". It is honest about being a review queue (`review_outlier`), not a verdict.

## Class 3 — defects that split one thing into two

Entity resolution is where it is easy to overshoot. `Ola` and `Ola Cabs` are the same
company. So the tempting rule is prefix or fuzzy matching:

```python
if key.startswith("ola"): key = "ola"          # merges Ola Electric. Wrong.
if "oyo" in key: key = "oyo"                   # merges OYOfit and FroyoFit. Wrong.
```

`Ola Electric` is a separate company (separate cap table, separate rounds). `OYOfit` is a
different product; `FroyoFit` is an unrelated company that merely contains the substring.
`Paytm Marketplace` (Paytm Mall) is a distinct legal entity from `Paytm`.

So `STARTUP_ALIASES` is an **explicit key → canonical map**. It is more work, it does not
generalise, and it is correct. Fuzzy matching a leaderboard is how you end up reporting
that one company raised money it never raised.

The suite pins this in both directions — `test_split_entities_are_merged` and
`test_near_miss_names_are_not_merged`. The second test is the important one, because a
fuzzy implementation passes the first.

The same logic applies to investors, plus two extra wrinkles:

- **Placeholders are not entities.** `Undisclosed Investors` appears in 83 rows and
  `Undisclosed Investor` in 24. Left in, anonymity ranks as the most active investor in
  Indian venture capital. 410 rows across ~30 spellings are dropped.
- **Delimiters are inconsistent.** Splitting on commas alone leaves
  `Saama Capital and Sequoia Capital` as one fictitious firm. Splitting on `,`, ` and `
  and ` & ` takes Sequoia from 73 deals to 122 — the difference between second and first
  place. The `(?![^(]*\))` guard avoids splitting inside parentheticals like
  `(Alibaba @ 40% equity)`.

## Class 4 — defects in what the file does not contain

The hardest class, and the reason `src/coverage_check.py` exists.

Rounds per year: 936, 993, 687, 310, 111. Every internal check passes. The dates are valid,
the distribution is smooth, nothing is null. And the obvious reading — Indian startup
funding collapsed after 2016 — is exactly backwards. 2019 was a record year.

**Absence leaves no trace in the data.** You cannot detect missing rows by examining
present rows. The only instrument is an external benchmark, so the module hard-codes
published Tracxn and Inc42 aggregates with citations and computes coverage ratios.

Two internal signals *corroborate* it once you know to look: median deal size rises
$1.0M → $12.0M and disclosure rises 59% → 95% across the same window. Both are what you see
when small rounds stop being collected. Neither is sufficient alone — a market genuinely
shifting to late-stage would produce the same pattern. That is precisely why the external
check is load-bearing rather than decorative.

**Generalisable lesson:** a dataset can be internally perfect and still answer your
question wrongly, if the sampling frame changed mid-collection. For any time series from a
scraped source, plot rows-per-period *first* and ask whether the shape is the world or the
scraper.

## Why the tests assert claims

The suite does not check that `clean()` returns a DataFrame. It checks that:

- one mislabelled cell accounts for >9% of the total,
- uncorrected data ranks Rapido above Flipkart,
- 2015–2017 capital lands within 20% of published totals **and 2019 lands below 50%**,
- median deal size in 2019 is >5× 2016,
- `Ola Electric` is not folded into `Ola`.

If a future change makes 2019 look complete, the conclusion in INSIGHTS.md is void and
`test_early_years_match_published_totals_and_late_years_do_not` fails. That is the point:
the tests defend the *argument*, not the implementation.

Two of them are pure regression guards for bugs I hit while writing this:

```python
# amount_flag is a nullable 'string' Series
rounds["amount_flag"] != "review_outlier"   # -> NA on unflagged rows, not True
```

`NA` is falsy in boolean indexing. So `money_subset()` — meant to select the ~2,000 *good*
rows — returned only the 6 flagged ones, the exact inverse of its purpose. The same trap
in `clean()` divided all 2,073 amounts by 71 while looking correct row by row; only the
total was wrong. Both now go through `.eq(...).fillna(False)` / `.ne(...).fillna(True)`, and
`test_adjustment_touches_only_the_flagged_row` plus `test_na_comparison_really_does_yield_na`
pin the behaviour and document the language rule.

The second: `fix_amount` originally returned nullable `Int64` whenever every surviving value
happened to be a whole number, and assigning `amount / 71` into an `Int64` array raises
`cannot safely cast non-equivalent object to int64`. It worked on the real CSV and crashed
on the stand-in — a bug whose existence depended on the data. Hence the explicit
`.astype("Float64")`. Testing against two datasets found it; testing against one would not
have.

## Why there is a defect-preserving stand-in

`*.csv` is gitignored, so CI has no data. `data/generate_sample.py` writes a 458-row file
that reproduces every defect *structurally*: the literal double-backslash text, Indian
grouping, an INR value at `sr_no 61`, city variants, the merge/no-merge entity pairs,
placeholder investors, fused investor lists, malformed dates, ~32% missing amounts and
coverage that decays after 2017.

The tests therefore exercise the findings rather than trusting stored numbers, and the
figures that require the real file are skipped explicitly via `REAL_ONLY`. A stand-in that
reproduced only the *schema* would let every cleaning function rot silently.

## Questions this invites

1. **Is the `KNOWN_REAL_MEGA` allowlist defensible?** It is a hand-curated list. A better
   design might flag every round above a percentile of its own year-and-stage distribution
   and require external confirmation for all of them, accepting more review work for less
   embedded judgement.
2. **Should the INR row be converted or dropped?** I convert at the Aug-2019 rate and keep
   both columns. Dropping loses a real $55M round; converting invents a precision the
   source never had. `amount_flag` exists so you can choose, but the default is a choice.
3. **Is 71 INR/USD the right rate?** It was roughly the Aug-2019 spot rate. Rounds are
   negotiated over months, so any single rate is a fiction; the error is ~2–3%, far below
   the error introduced by the 32% missing amounts.
4. **Does correcting one row and flagging five others create a false sense of completeness?**
   Probably. There may be more currency errors below the $250M threshold, where INR and USD
   values overlap in plausibility and no ranking looks absurd. I have no way to find those
   without a deal-level external join, and the README says so.
5. **Is "76% of rounds in three metros" a fact about India or about trak.in?** Coverage
   analysis shows *how much* is missing per year but not *which* rounds. If the source
   under-covered tier-2 cities, concentration is overstated. This is the largest unresolved
   threat to the surviving findings.
6. **Would this dataset support a model at all?** Predicting round size from sector, city,
   stage and investor identity is feasible on 2015–2017. But with 32% of the target missing
   non-randomly and `Private Equity` covering 45% of rows as a stage label, a model would
   mostly learn the disclosure process. Recovering true stage labels is the prerequisite,
   and that is an external-join problem, not a modelling one.
