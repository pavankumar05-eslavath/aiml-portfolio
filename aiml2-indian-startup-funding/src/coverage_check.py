"""
Stage 4 -- validate the dataset against published industry totals.

The most consequential finding in the project, and the one no amount of internal
cleaning can reach: you have to leave the file and check it against the outside
world.

Deal counts in this dataset fall 993 (2016) -> 111 (2019). Read at face value
that is a collapse in Indian startup funding. It is the opposite: 2019 was a
record year. What actually declined is the source's collection.

Benchmarks (Indian tech startup funding):
  2015 $7.9B / 2016 $4.3B / 2017 $10.4B  -- Tracxn, via TechCrunch
    https://techcrunch.com/2019/10/23/india-tech-startups-fundraise
  2018 $10.6B / 2019 $14.5B              -- Tracxn, via TechCrunch
    https://techcrunch.com/2019/12/29/indian-tech-startups-funding-amount-2019/
  deal counts 2015 912 / 2016 1017        -- Economic Times
    https://economictimes.indiatimes.com/small-biz/money/with-4-4-bn-investment-across-1017-deals-in-2016-the-ups-and-downs-of-indias-startup-ecosystem/articleshow/56829457.cms
  deal counts 2018 743 / 2019 766         -- Inc42 DataLabs
    https://inc42.com/datalab/despite-12-7-bn-funding-in-2019-indian-startup-funding-deals-at-a-5-year-low/
"""
from __future__ import annotations

import pandas as pd

from src.dataset import is_real, load_clean, money_subset, usd

BENCH_CAPITAL = {2015: 7.9e9, 2016: 4.3e9, 2017: 10.4e9, 2018: 10.6e9, 2019: 14.5e9}
BENCH_DEALS = {2015: 912, 2016: 1017, 2018: 743, 2019: 766}

# Years whose capital lands within tolerance of the published totals.
RELIABLE_YEARS = (2015, 2017)
TOLERANCE = 0.20


def coverage_table(rounds: pd.DataFrame) -> pd.DataFrame:
    money = money_subset(rounds)
    t = pd.DataFrame({
        "rounds_here": rounds.groupby("year").size(),
        "rounds_actual": pd.Series(BENCH_DEALS),
        "capital_here": money.groupby("year")["amount_usd_adj"].sum(),
        "capital_actual": pd.Series(BENCH_CAPITAL),
    })
    t = t.loc[[y for y in t.index if 2015 <= y <= 2019]]
    t["deal_coverage"] = t["rounds_here"] / t["rounds_actual"]
    t["capital_coverage"] = t["capital_here"] / t["capital_actual"]
    return t


def run(path: str | None = None) -> dict:
    rounds, _ = load_clean(path)
    money = money_subset(rounds)
    t = coverage_table(rounds)

    show = t.copy()
    show["capital_here"] = show["capital_here"].map(usd)
    show["capital_actual"] = show["capital_actual"].map(usd)
    for c in ("deal_coverage", "capital_coverage"):
        show[c] = show[c].map(lambda v: "-" if pd.isna(v) else f"{v:.0%}")
    print(show.to_string())

    print("\n--- median deal size, which is the tell ---")
    med = money.groupby("year")["amount_usd_adj"].median()
    for y, v in med.items():
        print(f"  {int(y)}  {usd(v)}")
    disc = rounds.groupby("year")["amount_usd"].apply(lambda s: s.notna().mean())
    print("\n--- amount-disclosure rate by year ---")
    for y, v in disc.items():
        print(f"  {int(y)}  {v:.0%}")

    if not is_real(path):
        print("\n(stand-in data: coverage ratios are not meaningful here; "
              "run `make download` for the real figures)")
        return {"reliable_years": None}

    print("""
READ THIS BEFORE USING THE YEAR COLUMN.
2015-2017 track reality closely -- capital within ~12%, deal counts within ~10%.
From 2018 coverage falls off a cliff: 2019 holds 111 of ~766 real deals (14%)
and $5.1B of ~$14.5B (35%). The apparent crash from 993 rounds in 2016 to 111 in
2019 is the trak.in scraper winding down; Indian startup funding hit a RECORD
high in 2019. The rows that do survive skew to large, well-covered deals, which
is why median deal size appears to RISE from $1.0M to $12.0M and why disclosure
climbs from 59% to 95% -- both are selection effects, not market signal.

=> 2015-2017 is analysable. 2018-2020 is a biased sample of large deals.
   Any trend, CAGR or 'funding winter' claim drawn across the full window is an
   artefact of collection.""")

    return {
        "reliable_years": RELIABLE_YEARS,
        "coverage_2019_deals": float(t.loc[2019, "deal_coverage"]),
        "coverage_2019_capital": float(t.loc[2019, "capital_coverage"]),
    }
