"""
Stage 3 -- the audited analysis, reported next to the baseline.

Applies the corrections in src/dataset.py, then prints each headline figure
beside the naive one so the cost of skipping the cleaning is explicit.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dataset import (
    INR_PER_USD_AUG_2019,
    load_clean,
    load_raw,
    money_subset,
    usd,
)

OUTPUTS = Path("outputs")


def run(path: str | None = None, save_figures: bool = False) -> dict:
    raw = load_raw(path)
    rounds, inv = load_clean(path)
    money = money_subset(rounds)

    print("--- what the corrections changed ---")
    naive_amt = pd.to_numeric(raw["amount_raw"].str.replace(",", "", regex=False),
                              errors="coerce")
    print(f"  amounts recovered by unescaping      : "
          f"{int(rounds['amount_usd'].notna().sum() - naive_amt.notna().sum())}")
    print(f"  distinct cities   {raw['city_raw'].nunique():>4} -> {rounds['city'].nunique()}")
    print(f"  distinct stages   {raw['stage_raw'].nunique():>4} -> {rounds['stage'].nunique()}")
    print(f"  distinct startups {raw['startup'].nunique():>4} -> {rounds['startup_key'].nunique()}")
    debris = rounds.select_dtypes(include="str").apply(
        lambda c: c.astype("string").str.contains(r"\\", regex=True, na=False).sum()).sum()
    print(f"  cells still holding backslash debris : {debris}")

    print("\n--- the currency error ---")
    flagged = rounds[rounds["amount_flag"].notna()][
        ["sr_no", "date", "startup", "stage", "amount_usd", "amount_usd_adj", "amount_flag"]]
    if not flagged.empty:
        show = flagged.copy()
        show["amount_usd"] = show["amount_usd"].map(usd)
        show["amount_usd_adj"] = show["amount_usd_adj"].map(usd)
        print(show.to_string(index=False))
    published = rounds["amount_usd"].sum()
    corrected = rounds["amount_usd_adj"].sum()
    print(f"\n  total as published : {usd(published)}")
    print(f"  total corrected    : {usd(corrected)}")
    print(f"  one cell moved     : {usd(published - corrected)} "
          f"= {(published - corrected) / published:.1%} of the published total")
    print(f"  (INR 390 crore at ~{INR_PER_USD_AUG_2019:.0f}/USD, Aug 2019)")
    print(f"  money-safe total   : {usd(money['amount_usd_adj'].sum())} "
          "(also drops rounds flagged review_outlier)")

    print("\n--- missingness is not random ---")
    print(f"  rounds with no amount: {rounds['amount_usd'].isna().sum():,} "
          f"({rounds['amount_usd'].isna().mean():.1%})")
    st = rounds.groupby("stage").agg(rounds=("sr_no", "size"),
                                    with_amount=("amount_usd", lambda s: s.notna().sum()))
    st = st[st["rounds"] >= 20].sort_values("rounds", ascending=False)
    st["disclosure"] = (st["with_amount"] / st["rounds"]).map("{:.0%}".format)
    print(st.to_string())
    print("  => any 'average deal size' over disclosed rows is biased upward,")
    print("     because late stages disclose far more often than seed.")

    print("\n--- top startups (entities merged) ---")
    top = (money.groupby("startup_key")
           .agg(name=("startup", "first"), rounds=("sr_no", "size"),
                total=("amount_usd_adj", "sum"))
           .nlargest(10, "total"))
    top["total"] = top["total"].map(usd)
    print(top.reset_index(drop=True).to_string())

    print("\n--- top investors (placeholders dropped, aliases merged) ---")
    top_inv = (inv.groupby("investor_key")
               .agg(name=("investor", "first"), deals=("sr_no", "nunique"),
                    startups=("startup", "nunique"))
               .nlargest(10, "deals"))
    print(top_inv.reset_index(drop=True).to_string())
    solo = inv.groupby("investor_key")["sr_no"].nunique().eq(1).mean()
    print(f"  {solo:.0%} of {inv['investor_key'].nunique():,} investors appear in exactly one round")

    print("\n--- concentration ---")
    metros = ["Bengaluru", "Delhi", "Gurugram", "Noida", "Mumbai"]
    print(f"  Bengaluru + Delhi-NCR + Mumbai = {rounds['city'].isin(metros).mean():.0%} of rounds")
    city = rounds.groupby("city").agg(rounds=("sr_no", "size")).nlargest(8, "rounds")
    city["capital"] = money.groupby("city")["amount_usd_adj"].sum().map(usd)
    print(city.to_string())

    amt = money["amount_usd_adj"].sort_values(ascending=False)
    tot = amt.sum()
    print()
    for n in (10, 50, 100):
        if n <= len(amt):
            print(f"  top {n:>3} rounds = {amt.head(n).sum() / tot:.0%} of disclosed capital")
    print(f"  median {usd(amt.median())} vs mean {usd(amt.mean())} "
          f"-- mean is {amt.mean() / amt.median():.1f}x the median, so quote the median")

    if save_figures:
        _figures(rounds, money)

    return {
        "rows": len(rounds),
        "total_published": float(published),
        "total_corrected": float(corrected),
        "total_money_safe": float(money["amount_usd_adj"].sum()),
        "distinct_startups": int(rounds["startup_key"].nunique()),
        "distinct_investors": int(inv["investor_key"].nunique()),
        "backslash_debris": int(debris),
    }


def _figures(rounds: pd.DataFrame, money: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np

    OUTPUTS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Indian Startup Funding, Jan 2015 - Jan 2020 (audited)",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    q = rounds.groupby(rounds["date"].dt.to_period("Q")).size()
    q.index = q.index.astype(str)
    # Shade the window that matches published industry totals; the fall after it
    # is the source's coverage decaying, not the market contracting.
    ax.bar(q.index, q.values,
           color=["#2b6cb0" if s[:4] <= "2017" else "#a0aec0" for s in q.index])
    ax.set_title("Deal count by quarter\n(grey = under-collected, not a real decline)",
                 fontsize=10)
    ax.set_ylabel("rounds")
    ax.tick_params(axis="x", rotation=90, labelsize=7)

    ax = axes[0, 1]
    c = money.groupby(money["date"].dt.to_period("Q"))["amount_usd_adj"].sum() / 1e9
    c.index = c.index.astype(str)
    ax.bar(c.index, c.values, color="#276749")
    ax.set_title("Disclosed capital by quarter (outliers excluded)")
    ax.set_ylabel("US$ B")
    ax.tick_params(axis="x", rotation=90, labelsize=7)

    ax = axes[1, 0]
    top = rounds["city"].value_counts().head(10).iloc[::-1]
    ax.barh(top.index, top.values, color="#975a16")
    ax.set_title("Top 10 cities by number of rounds")
    ax.set_xlabel("rounds")
    ax.tick_params(labelsize=8)

    ax = axes[1, 1]
    # Log-spaced bins are required: linear bins on a log axis collapse ~95% of
    # rows into a single meaningless block.
    pos = money.loc[money["amount_usd_adj"] > 0, "amount_usd_adj"]
    bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 40)
    ax.hist(pos, bins=bins, color="#822727", edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(pos.median(), color="#1a365d", ls="--", lw=1.6,
               label=f"median {usd(pos.median())}")
    ax.legend(fontsize=8)
    ax.set_title("Round-size distribution (log-log)")
    ax.set_xlabel("US$ per round")
    ax.set_ylabel("rounds (log)")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.tight_layout()
    dest = OUTPUTS / "funding_overview.png"
    fig.savefig(dest, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {dest}")
