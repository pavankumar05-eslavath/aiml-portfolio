"""
Stage 2 -- THE BASELINE.

The analysis a reader of this dataset gets by doing the obvious thing:
read_csv, groupby, nlargest. No cleaning. This is what the published
walkthroughs of this dataset do, and it is stated here BEFORE the audited
result so the two can be read side by side.

Every number this stage prints is wrong in a way stage 3 quantifies. It is kept
in the repo, and in CI, because the size of the gap is the finding.
"""
from __future__ import annotations

import pandas as pd

from src.dataset import RAW_COLUMNS, resolve_path, usd


def run(path: str | None = None) -> dict:
    # Deliberately naive: no utf-8-sig, no unescaping, no entity resolution.
    # A plain read_csv() names column 0 '\ufeffSr No' because of the BOM.
    df = pd.read_csv(resolve_path(path), dtype="string")
    bom_col = df.columns[0]
    df.columns = RAW_COLUMNS

    print(f"columns as read : first column is {bom_col!r}")
    print("                  (the BOM is why a by-name lookup of 'Sr No' misses)")
    print(f"rows            : {len(df):,}")

    # The naive amount parse: strip commas, to_numeric. This part is actually
    # correct -- Indian grouping means comma removal is right -- but it trusts
    # the column header, which is where the currency error hides.
    amt = pd.to_numeric(df["amount_raw"].str.replace(",", "", regex=False),
                        errors="coerce")
    print(f"amounts parsed  : {amt.notna().sum():,} of {len(df):,} "
          f"({amt.notna().mean():.1%})")
    print(f"                  {amt.isna().sum():,} unparsed, silently dropped from every sum")

    total = amt.sum()
    print(f"\ntotal funding   : {usd(total)}   <-- headline number, and it is wrong")

    print("\nTop 5 startups by total raised (naive):")
    naive_top = (df.assign(amount=amt).groupby("startup", dropna=True)["amount"]
                 .sum().nlargest(5))
    for name, v in naive_top.items():
        print(f"  {name:<28} {usd(v)}")
    print("  ^ Ola and Ola Cabs, Flipkart and Flipkart.com rank as separate companies")

    print("\nTop 5 cities by round count (naive):")
    for name, v in df["city_raw"].value_counts().head(5).items():
        print(f"  {name:<28} {v}")
    print("  ^ Bangalore and Bengaluru are counted as two different cities")

    print("\nTop 5 investors by round count (naive, no splitting):")
    for name, v in df["investors_raw"].value_counts().head(5).items():
        print(f"  {name:<28} {v}")
    print("  ^ 'Undisclosed Investors' is a placeholder, not an investor")

    print("\nRounds per year (naive):")
    year = pd.to_datetime(df["date_raw"], errors="coerce", dayfirst=True,
                          format="mixed").dt.year
    counts = year.value_counts().sort_index()
    for y, v in counts.items():
        print(f"  {int(y)}  {v:>4}")
    print("  ^ reads as a collapse after 2016. Stage 4 shows it is not.")

    largest_idx = amt.idxmax()
    print(f"\nSingle largest round (naive): {df.loc[largest_idx, 'startup']} "
          f"{usd(amt.max())} on {df.loc[largest_idx, 'date_raw']}")
    print("  ^ stage 3 shows this figure is rupees, and it is 10% of the total above")

    return {
        "rows": len(df),
        "total_usd": float(total),
        "amounts_parsed": int(amt.notna().sum()),
        "distinct_cities": int(df["city_raw"].nunique()),
        "largest_round": float(amt.max()),
    }
