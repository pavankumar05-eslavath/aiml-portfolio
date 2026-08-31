"""
Stage 1 -- profile the published file and locate its defects.

Reports what is actually in the CSV, including the two defects that are
invisible to the obvious check.
"""
from __future__ import annotations

import pandas as pd

from src.dataset import RAW_COLUMNS, fix_amount, fix_date, resolve_path, unescape


def run(path: str | None = None) -> dict:
    src = resolve_path(path)
    raw_bytes = src.read_bytes()
    df = pd.read_csv(src, encoding="utf-8-sig", dtype="string")
    df.columns = RAW_COLUMNS

    print(f"shape: {df.shape}")

    print("\n--- missing values per column ---")
    for c in df.columns:
        junk = df[c].str.strip().str.lower().isin(
            ["nan", "n/a", "", "unknown", "undisclosed"]).sum()
        print(f"  {c:<16} null={df[c].isna().sum():>5}  placeholder_text={junk:>5}")

    print("\n--- DEFECT 1: literal escape text, not real bytes ---")
    literal_double = raw_bytes.count(rb"\\xc2")
    literal_single = raw_bytes.count(rb"\xc2") - literal_double
    real_nbsp = raw_bytes.count("\u00a0".encode())
    print(f"  occurrences of the 6 chars  \\xc2  : {literal_single}")
    print(f"  occurrences of the 8 chars \\\\xc2  : {literal_double}")
    print(f"  occurrences of a REAL U+00A0 byte : {real_nbsp}")
    print("  So `grep -P '\\xc2\\xa0'` finds nothing while cells are affected, and")
    print("  .strip() cannot touch them. A regex matching ONE backslash leaves the")
    print("  other behind -- the cell stays distinct and the bug just changes shape.")

    print("\n--- DEFECT 2: Indian digit grouping in 'Amount in USD' ---")
    wide = df["amount_raw"].dropna().loc[
        lambda s: s.str.match(r"^\d{1,2}(,\d{2})+,\d{3}$", na=False)]
    print(f"  values grouped in lakh/crore style (not 3-digit groups): {len(wide)}")
    for v in wide.head(3):
        print(f"    {v:>16} -> {int(str(v).replace(',', '')):,}")
    print("  Removing every comma is the CORRECT parse. The hazard is any 3-digit-group")
    print("  assumption, and the eye: 20,00,00,000 reads as ~20M but is 200M.")

    print("\n--- parse failures before repair ---")
    d_naive = pd.to_datetime(df["date_raw"], format="%d/%m/%Y", errors="coerce")
    print(f"  dates unparseable as %d/%m/%Y : {d_naive.isna().sum()}")
    for v in df.loc[d_naive.isna(), "date_raw"].head(8):
        print(f"    {v!r}")
    print(f"  dates unparseable after repair : {fix_date(unescape(df['date_raw'])).isna().sum()}")

    amt_naive = pd.to_numeric(df["amount_raw"].str.replace(",", "", regex=False),
                              errors="coerce")
    amt_fixed = fix_amount(unescape(df["amount_raw"]))
    print(f"  amounts unparseable, naive    : {amt_naive.isna().sum()}")
    print(f"  amounts unparseable, repaired : {amt_fixed.isna().sum()}"
          f"  (recovered {int(amt_fixed.notna().sum() - amt_naive.notna().sum())})")

    print("\n--- label fragmentation (distinct raw values) ---")
    for col, label in [("stage_raw", "round label"), ("city_raw", "city"),
                       ("industry_raw", "industry"), ("startup", "startup name")]:
        print(f"  {label:<14} {df[col].nunique():>5}")
    print("  e.g. round labels include 'Seed Funding', 'Seed/ Angel Funding',")
    print("       'Seed \\\\nFunding' and 'Seed / Angle Funding' (sic)")

    return {
        "rows": len(df),
        "literal_escape_cells": literal_double + literal_single,
        "real_nbsp_bytes": real_nbsp,
        "indian_grouped_amounts": len(wide),
        "bad_dates_naive": int(d_naive.isna().sum()),
    }
