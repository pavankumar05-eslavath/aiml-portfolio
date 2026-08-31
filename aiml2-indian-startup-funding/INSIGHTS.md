# INSIGHTS — Indian Startup Funding, audited

Every figure below comes from the real 3,044-row Kaggle CSV and is reproducible with
`make download && make all`. Money figures use `amount_usd_adj` and exclude rounds
flagged `review_outlier`.

---

## 1. The single largest "USD" amount is rupees

Row `sr_no 61`, **Rapido Bike Taxi**, 27 Aug 2019, carries `3,90,00,00,000` in the column
headed **`Amount in USD`**.

That is ₹390 crore ≈ **$55M** — the round actually reported at the time
([Inc42](https://inc42.com/buzz/exclusive-rapido-closes-series-b-funding-round-at-55-mn/),
[Economic Times](https://economictimes.indiatimes.com/small-biz/startups/newsbuzz/rapido-enters-fast-lane-with-390-crore/articleshow/70868579.cms),
[Entrackr](https://entrackr.com/2019/08/rapido-rs-391-cr-westbridge-valuation-rs-1000-cr/)).
Read as USD it becomes **$3.9B**.

| | as published | corrected |
|---|---:|---:|
| Largest round in the dataset | **Rapido, $3.9B** | Flipkart, $2.5B |
| Dataset total | $38.14B | **$34.30B** |

**One cell is $3.85B — 10.1% of the dataset's entire funding total.** It makes a
seed-stage bike-taxi company the largest venture round in Indian history, ahead of
Flipkart's genuine $2.5B SoftBank round.

Five further rounds ≥$250M are flagged `review_outlier` and excluded from money totals.
At least one (Automation Anywhere, $300M Series B, Nov 2018) is genuine; Vogo ($283M) and
GOQii ($450M) look like the same INR mislabelling but I did not confirm them. They are
flagged, not deleted — `amount_flag` lets a downstream user decide.

## 2. Deal counts fall 89% after 2016. The market rose.

This is the finding no amount of internal cleaning reaches — it requires leaving the file.

| year | rounds here | rounds actual | capital here | capital actual | deal coverage | capital coverage |
|---|---:|---:|---:|---:|---:|---:|
| 2015 | 936 | 912 | $8.67B | $7.90B | 103% | 110% |
| 2016 | 993 | 1,017 | $3.83B | $4.30B | 98% | 89% |
| 2017 | 687 | — | $10.16B | $10.40B | — | 98% |
| 2018 | 310 | 743 | $4.22B | $10.60B | **42%** | **40%** |
| 2019 | 111 | 766 | $5.12B | $14.50B | **14%** | **35%** |

Actuals: Tracxn via [TechCrunch (2019)](https://techcrunch.com/2019/12/29/indian-tech-startups-funding-amount-2019/)
and [TechCrunch (2018)](https://techcrunch.com/2019/10/23/india-tech-startups-fundraise);
deal counts from [Inc42 DataLabs](https://inc42.com/datalab/despite-12-7-bn-funding-in-2019-indian-startup-funding-deals-at-a-5-year-low/)
and [Economic Times](https://economictimes.indiatimes.com/small-biz/money/with-4-4-bn-investment-across-1017-deals-in-2016-the-ups-and-downs-of-indias-startup-ecosystem/articleshow/56829457.cms).

2015–2017 track reality within ~10%. From 2018 the trak.in scraper wound down. **Indian
startup funding hit a record high in 2019** — precisely where this dataset shows its
steepest decline.

Two internal signals corroborate that this is selection, not market:

| year | median deal | amount disclosed |
|---|---:|---:|
| 2016 | $1.0M | 59% |
| 2019 | **$12.0M** | **95%** |

Small rounds stopped being recorded, so survivors are larger and better documented.
A rising median deal size alongside a collapsing deal count is the signature of a
shrinking *sample*, not a shrinking market.

**Any trend, CAGR or "funding winter" claim drawn across the full 2015–2020 window is an
artefact of collection.** 2015–2017 is analysable; 2018–2020 is a biased sample of large deals.

## 3. Missingness is not random, so "average deal size" is not meaningful

**32% of rounds (971) have no amount at all.** They are silently dropped from every sum.
Disclosure is strongly stage-dependent:

| stage | rounds | disclosed |
|---|---:|---:|
| Seed / Angel | 1,542 | 56% |
| Private Equity | 1,361 | 79% |
| Debt | 29 | 97% |
| Series A | 24 | 92% |
| Series B | 21 | 100% |

Seed rounds are the ones that hide their size, so any mean over disclosed rows is biased
upward. Report medians by stage, and report the disclosure rate next to them.

## 4. Findings that survive the audit

These hold on 2015–2017, the reliable window, and after entity resolution.

**Geography is extremely concentrated.** Bengaluru + Delhi-NCR + Mumbai = **76% of all
rounds**. Bengaluru alone takes 28% of rounds and $14.68B — more than the next three
cities combined.

**Capital is head-heavy.** Top 10 rounds = 30% of disclosed capital; top 50 = 52%; top
100 = 64%. Median round $1.7M vs mean $15.7M — **the mean is 9.2× the median**. Quote the
median or the distribution, never the mean.

**Top recipients** (entities merged):

| startup | rounds | total |
|---|---:|---:|
| Flipkart | 6 | $4.76B |
| Paytm | 5 | $3.15B |
| Ola | 12 | $2.05B |
| OYO | 9 | $997.0M |
| Udaan | 4 | $870.0M |

**eCommerce takes the most capital** ($9.74B over 340 rounds) while **Consumer Internet
takes the most rounds** (942) for $6.25B — many small cheques versus few large ones.
Median deal is $3.1M in eCommerce against $1.0M in Consumer Internet.

**Top investors** (placeholders removed, aliases merged): Sequoia Capital India 122 deals
across 91 startups, Accel 89, Blume Ventures 61, Kalaari Capital 59, SAIF Partners 56,
Tiger Global 53. **77% of the 2,996 investors appear in exactly one round**, and the median
round names a single investor — the syndicate structure in this data is thin.

## 5. What the cleaning changed

| defect | scale | effect if ignored |
|---|---|---|
| Literal `\\xc2\\xa0` escape text (two backslashes, not U+00A0) | 92 cells | Rows fail to parse and vanish; `\ Gurgaon` ≠ `Gurgaon` |
| INR in a USD column | 1 cell | **10% of the total**; wrong #1 round |
| Split entities (Ola/Ola Cabs, Flipkart/Flipkart.com, OYO/OyoRooms) | 3 pairs | Ola ranks 3rd **and** 4th |
| Placeholder investors | 410 rows | "Undisclosed Investors" ranks **#1 investor** |
| Investor lists fused by ` and ` / ` & ` | — | Sequoia: 73 deals instead of 122 |
| City spelling variants | 107 → 76 | Bangalore and Bengaluru counted separately |
| Round labels | 55 → 16 | Stage analysis fragments |
| Malformed dates | 8 | Rows dropped |

## 6. What would have to be measured to confirm this

- **The currency claim.** I confirmed Rapido against three outlets. The right check is the
  regulatory filing (MCA/RoC) for the Aug 2019 round. The five `review_outlier` rows need
  the same treatment individually — I have not done that.
- **The coverage claim.** I compared against Tracxn and Inc42 aggregates, which are
  themselves proprietary and differ from each other ($12.7B vs $14.5B for 2019). A
  stronger test is a deal-level join against a full source (Tracxn, Venture Intelligence,
  Crunchbase) to identify *which* rounds are missing and confirm the gap is
  small-round-biased rather than sector-biased.
- **The entity resolution.** Seven alias pairs were resolved by hand from the top of the
  leaderboard. The tail is unaudited: 2,276 startup keys and 2,996 investor keys almost
  certainly contain more duplicates that do not change the top 15 but would change any
  count of "distinct startups funded".
- **Whether `Private Equity` means anything.** It covers 1,361 rows with a $6.0M median,
  so it clearly includes ordinary VC rounds. Recovering true stage would need an external
  join; until then, stage-based conclusions here are weak.

## 7. Recommendation

Use this dataset for **2015–2017 cross-sectional questions** — geography, sector mix,
investor activity, round-size distribution. It is genuinely good for those: three
independent aggregates agree with it to within ~10%.

Do not use it for **time-series claims through 2020**, for **totals** without correcting
`sr_no 61`, or for **average deal size** without accounting for stage-dependent
non-disclosure. If the question is "what happened to Indian startup funding after 2017",
this dataset cannot answer it and will confidently give the wrong answer.
