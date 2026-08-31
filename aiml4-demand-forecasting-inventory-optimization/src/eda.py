"""Exploratory time-series analysis.

Each figure answers one business question named in its title. Deliberately few:
plotting 600 individual series would be industrious and useless.

Everything here is computed on data **before the test window**. Exploring the
holdout is a soft leak -- every later decision (which features to build, which
segments to cut) would be fitted to it.
"""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")  # must precede pyplot; keeps headless runs from needing a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import Config
from src.data.loader import load_panel, test_window
from src.data.schema import DATE, SERIES_ID, TARGET

sns.set_theme(style="whitegrid", palette="deep")
BLUE, RED, GREEN = "#2b6cb0", "#c53030", "#276749"


def run(cfg: Config) -> dict[str, Any]:
    """Produce the EDA figures and return the findings used in the report."""
    cfg.ensure_dirs()
    figdir = cfg.resolve("paths", "figures_dir")
    panel = load_panel(cfg)
    test_start, _ = test_window(cfg)
    df = panel[panel[DATE] < test_start].copy()

    out: dict[str, Any] = {
        "n_rows": len(df), "n_series": int(df[SERIES_ID].nunique()),
        "n_days": int(df[DATE].nunique()),
        "date_min": str(df[DATE].min().date()), "date_max": str(df[DATE].max().date()),
        "zero_share": float((df[TARGET] == 0).mean()),
        "mean_demand": float(df[TARGET].mean()),
    }

    # ---- Q1: what does total demand look like over time? ----
    daily = df.groupby(DATE)[TARGET].sum()
    weekly = daily.resample("W").sum()
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    axes[0].plot(daily.index, daily.values, lw=0.5, color=BLUE, alpha=0.7)
    axes[0].plot(weekly.index, weekly.values / 7, lw=2, color=RED,
                 label="weekly mean")
    axes[0].set_title("Q1. Is total demand trending, and how noisy is it day to day?")
    axes[0].set_ylabel("units/day (all series)")
    axes[0].legend(fontsize=8)

    yearly = df.assign(year=df[DATE].dt.year).groupby("year")[TARGET].sum()
    axes[1].bar(yearly.index.astype(str), yearly.values, color=GREEN)
    axes[1].set_title("Q1b. Annual totals (2011 and the final year are partial)")
    axes[1].set_ylabel("units")
    fig.tight_layout()
    fig.savefig(figdir / "01_demand_trend.png", dpi=130)
    plt.close(fig)

    growth = float(weekly.iloc[-8:].mean() / weekly.iloc[:8].mean() - 1.0)
    out["weekly_growth_first_to_last_8w"] = growth

    # ---- Q2: seasonality -- which days and months matter? ----
    dow = df.assign(dow=df[DATE].dt.dayofweek).groupby("dow")[TARGET].mean()
    month = df.assign(m=df[DATE].dt.month).groupby("m")[TARGET].mean()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[0].bar(names, dow.values, color=BLUE)
    axes[0].axhline(df[TARGET].mean(), ls="--", color="black", lw=1)
    axes[0].set_title("Q2. Which weekdays sell?")
    axes[0].set_ylabel("mean units/series/day")

    axes[1].bar(range(1, 13), month.reindex(range(1, 13)).values, color=GREEN)
    axes[1].set_xticks(range(1, 13))
    axes[1].set_title("Q2b. Monthly seasonality")

    snap = df.groupby("snap")[TARGET].mean()
    axes[2].bar(["no SNAP", "SNAP day"], snap.reindex([0.0, 1.0]).values, color=RED)
    axes[2].set_title("Q2c. Do SNAP benefit days lift demand?")
    for i, v in enumerate(snap.reindex([0.0, 1.0]).values):
        axes[2].text(i, v, f"{v:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(figdir / "02_seasonality.png", dpi=130)
    plt.close(fig)

    out["dow_ratio_max_min"] = float(dow.max() / dow.min())
    out["dow_peak"] = names[int(dow.idxmax())]
    out["snap_lift_pct"] = float(100.0 * (snap.get(1.0, np.nan) / snap.get(0.0, np.nan) - 1))

    # ---- Q3: how concentrated is demand across SKUs? ----
    per = df.groupby(SERIES_ID)[TARGET].sum().sort_values(ascending=False)
    cum = per.cumsum() / per.sum()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum.values * 100,
                 color=BLUE, lw=2)
    axes[0].axhline(70, ls="--", color=RED, lw=1, label="70% of demand")
    axes[0].set_xlabel("% of series (ranked)")
    axes[0].set_ylabel("cumulative % of demand")
    axes[0].set_title("Q3. Is demand concentrated in a few SKUs?")
    axes[0].legend(fontsize=8)

    cat = df.groupby("cat_id")[TARGET].sum()
    axes[1].bar(cat.index.astype(str), cat.values, color=GREEN)
    axes[1].set_title("Q3b. Demand by category")
    fig.tight_layout()
    fig.savefig(figdir / "03_demand_concentration.png", dpi=130)
    plt.close(fig)

    share_top20 = float(cum.iloc[int(0.2 * len(cum))])
    out["share_of_demand_from_top20pct_series"] = share_top20
    out["demand_by_category"] = {str(k): float(v) for k, v in cat.items()}

    # ---- Q4: intermittency -- how sparse is a typical series? ----
    nz = df.groupby(SERIES_ID)[TARGET].apply(lambda s: float((s > 0).mean()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].hist(nz.values, bins=30, color=BLUE, edgecolor="white")
    axes[0].set_xlabel("share of days with a sale")
    axes[0].set_ylabel("series")
    axes[0].set_title("Q4. How often does a typical SKU actually sell?")
    axes[0].axvline(float(nz.median()), ls="--", color=RED,
                    label=f"median {nz.median():.2f}")
    axes[0].legend(fontsize=8)

    axes[1].hist(df.loc[df[TARGET] > 0, TARGET].clip(upper=30), bins=30,
                 color=GREEN, edgecolor="white")
    axes[1].set_yscale("log")
    axes[1].set_title("Q4b. Size of demand when it does occur (clipped at 30)")
    axes[1].set_xlabel("units on a selling day")
    fig.tight_layout()
    fig.savefig(figdir / "04_intermittency.png", dpi=130)
    plt.close(fig)
    out["median_selling_day_share"] = float(nz.median())

    # ---- Q5: does price move demand? ----
    d = df[df[TARGET] > 0].copy()
    ref = d.groupby(SERIES_ID)["sell_price"].transform("median")
    d["rel_price"] = d["sell_price"] / ref
    bins = [0, 0.85, 0.95, 1.05, 1.15, 10]
    labels = ["<-15%", "-15..-5%", "±5%", "+5..15%", ">+15%"]
    d["price_band"] = pd.cut(d["rel_price"], bins=bins, labels=labels)
    by_price = d.groupby("price_band", observed=True)[TARGET].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].bar(by_price.index.astype(str), by_price.values, color=RED)
    axes[0].set_title("Q5. Demand vs price relative to the SKU's own median")
    axes[0].set_ylabel("mean units on selling days")
    axes[0].tick_params(axis="x", labelsize=8)

    vol = df.groupby(SERIES_ID)[TARGET].agg(["mean", "std"])
    vol = vol[vol["mean"] > 0]
    axes[1].scatter(vol["mean"], vol["std"] / vol["mean"], s=8, alpha=0.5, color=BLUE)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("mean daily demand (log)")
    axes[1].set_ylabel("coefficient of variation")
    axes[1].set_title("Q5b. Low-volume SKUs are far more volatile")
    fig.tight_layout()
    fig.savefig(figdir / "05_price_and_volatility.png", dpi=130)
    plt.close(fig)
    out["demand_by_price_band"] = {str(k): float(v) for k, v in by_price.items()}

    # ---- Q6: example series, one per demand class ----
    from src.evaluation.segmentation import segment_series
    seg = segment_series(cfg, panel, as_of=test_start - pd.Timedelta(days=1))
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, cls in zip(axes, ("Smooth", "Erratic", "Intermittent", "Lumpy"),
                       strict=False):
        pick = seg[seg["demand_class"] == cls].sort_values(
            "total_demand", ascending=False)
        if pick.empty:
            ax.set_visible(False)
            continue
        sid = pick.index[0]
        # Series.last() was removed in pandas 3.0; slice on the index explicitly.
        one = df[df[SERIES_ID] == sid].set_index(DATE)[TARGET].sort_index()
        s = one[one.index > one.index.max() - pd.Timedelta(days=400)]
        ax.plot(s.index, s.values, lw=0.8, color=BLUE)
        ax.set_title(f"Q6. {cls} — {sid} (last 400 days)", fontsize=10, loc="left")
        ax.set_ylabel("units")
    fig.suptitle("Q6. What do the four demand classes actually look like?",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(figdir / "06_demand_classes.png", dpi=130)
    plt.close(fig)

    print(f"rows (pre-test)          : {out['n_rows']:,}")
    print(f"series / days            : {out['n_series']} / {out['n_days']:,}")
    print(f"zero share               : {out['zero_share']:.2%}")
    print(f"mean demand              : {out['mean_demand']:.3f} units/series/day")
    print(f"weekday peak             : {out['dow_peak']} "
          f"(max/min ratio {out['dow_ratio_max_min']:.2f})")
    print(f"SNAP lift                : {out['snap_lift_pct']:+.1f}%")
    print(f"top 20% series demand    : {out['share_of_demand_from_top20pct_series']:.1%}")
    print(f"median selling-day share : {out['median_selling_day_share']:.2f}")
    print(f"growth first->last 8w    : {out['weekly_growth_first_to_last_8w']:+.1%}")
    print(f"demand by price band     : "
          f"{ {k: round(v, 2) for k, v in out['demand_by_price_band'].items()} }")
    print(f"\n[figures] {figdir}")
    return out
