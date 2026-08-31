"""Time-series data quality and leakage audit.

Runs before any modelling and writes reports/data_audit.md.

The central question for a demand panel is not "are there nulls" but **"is a zero
a real zero?"**. A zero can mean three completely different things:

* the product was on shelf and nobody bought it  -> a true zero, must be modelled
* the product did not exist yet in that store    -> not an observation at all
* the product was delisted                        -> the series has ended

Training on all three as if they were the same teaches the model that demand
collapses at the start and end of a product's life, which is an artefact of the
panel's shape rather than anything about demand. The audit separates them.

The leakage section is executable rather than rhetorical: it checks the actual
feature builder against a mutated future and reports whether anything changed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config
from src.data.loader import (
    Fold,
    classify_demand,
    demand_stats,
    load_panel,
    rolling_origin_folds,
    test_window,
)
from src.data.schema import (
    DATA_DICTIONARY,
    DATE,
    PANEL_COLUMNS,
    SERIES_ID,
    TARGET,
    VALID_RANGES,
)

#: A day whose demand exceeds this multiple of the series' non-zero median is
#: flagged for inspection. Not removed: promotional and seasonal peaks are real
#: demand, and clipping them would bias safety stock downwards precisely for the
#: products where stockouts are most expensive.
SPIKE_MULTIPLE = 20.0

#: No sale for this many days at the end of the panel suggests delisting.
DISCONTINUED_DAYS = 60


@dataclass
class AuditResult:
    """Structured findings, so tests can assert on them."""

    n_rows: int
    n_series: int
    n_days: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    schema_ok: bool
    missing_columns: list[str]
    n_duplicate_rows: int
    n_missing_values: dict[str, int]
    n_calendar_gaps: int
    n_negative_demand: int
    n_invalid_values: dict[str, int]
    zero_share: float
    n_series_launched_late: int
    n_series_discontinued: int
    n_spike_days: int
    n_spike_series: int
    demand_class_counts: dict[str, int]
    intermittent_share: float
    coverage: dict[str, Any]
    leakage_checks: dict[str, Any]
    notes: list[str] = field(default_factory=list)


def _missing_values(panel: pd.DataFrame) -> dict[str, int]:
    n = panel.isna().sum()
    return {k: int(v) for k, v in n.items() if v > 0}


def _invalid_values(panel: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    for col, (lo, hi) in VALID_RANGES.items():
        if col in panel.columns:
            s = pd.to_numeric(panel[col], errors="coerce")
            bad = int(((s < lo) | (s > hi)).sum())
            if bad:
                out[col] = bad
    return out


def calendar_gaps(panel: pd.DataFrame) -> int:
    """Count series whose dates are not contiguous between their own start and end.

    Contiguity matters more than completeness here. Series legitimately start at
    different dates (staggered launches); a *hole in the middle* is a different
    thing, and would mean a zero-demand day was dropped rather than recorded.
    """
    g = panel.groupby(SERIES_ID, observed=True)[DATE].agg(["min", "max", "size"])
    expected = (g["max"] - g["min"]).dt.days + 1
    return int((g["size"] != expected).sum())


def zero_taxonomy(panel: pd.DataFrame) -> dict[str, Any]:
    """Split zeros into pre-launch, post-discontinuation and in-life true zeros."""
    panel = panel.sort_values([SERIES_ID, DATE])
    first_sale = panel.loc[panel[TARGET] > 0].groupby(SERIES_ID, observed=True)[DATE].min()
    last_sale = panel.loc[panel[TARGET] > 0].groupby(SERIES_ID, observed=True)[DATE].max()

    d = panel[[SERIES_ID, DATE, TARGET]].merge(
        first_sale.rename("first_sale"), on=SERIES_ID, how="left").merge(
        last_sale.rename("last_sale"), on=SERIES_ID, how="left")

    is_zero = d[TARGET] == 0
    pre = is_zero & d[DATE].lt(d["first_sale"])
    post = is_zero & d[DATE].gt(d["last_sale"])
    true_zero = is_zero & ~pre & ~post

    return {
        "n_zero": int(is_zero.sum()),
        "zero_share": float(is_zero.mean()),
        "n_pre_launch_zero": int(pre.sum()),
        "n_post_discontinuation_zero": int(post.sum()),
        "n_true_zero": int(true_zero.sum()),
        "true_zero_share_of_zeros": float(true_zero.sum() / max(int(is_zero.sum()), 1)),
    }


def lifecycle(panel: pd.DataFrame) -> dict[str, Any]:
    """Series that launch after the panel opens, or stop selling before it closes."""
    panel_start, panel_end = panel[DATE].min(), panel[DATE].max()
    g = panel.groupby(SERIES_ID, observed=True)[DATE].agg(["min", "max"])
    last_sale = panel.loc[panel[TARGET] > 0].groupby(SERIES_ID, observed=True)[DATE].max()

    launched_late = g["min"] > panel_start
    days_since_sale = (panel_end - last_sale).dt.days
    discontinued = days_since_sale > DISCONTINUED_DAYS

    return {
        "n_series": len(g),
        "n_launched_late": int(launched_late.sum()),
        "n_distinct_start_dates": int(g["min"].nunique()),
        "n_discontinued": int(discontinued.sum()),
        "median_history_days": float((g["max"] - g["min"]).dt.days.median()),
        "min_history_days": float((g["max"] - g["min"]).dt.days.min()),
    }


def spikes(panel: pd.DataFrame, multiple: float = SPIKE_MULTIPLE) -> dict[str, int]:
    """Days far above the series' own non-zero median."""
    med = panel.loc[panel[TARGET] > 0].groupby(SERIES_ID, observed=True)[TARGET].median()
    d = panel.merge(med.rename("nz_median"), on=SERIES_ID, how="left")
    flag = d[TARGET] > multiple * d["nz_median"].replace(0, np.nan)
    return {
        "n_spike_days": int(flag.sum()),
        "n_spike_series": int(d.loc[flag, SERIES_ID].nunique()),
    }


def coverage(panel: pd.DataFrame) -> dict[str, Any]:
    """SKU / store / category coverage of the panel."""
    return {
        "n_series": int(panel[SERIES_ID].nunique()),
        "n_skus": int(panel["sku_id"].nunique()),
        "n_stores": int(panel["store_id"].nunique()),
        "stores": sorted(panel["store_id"].unique().tolist()),
        "n_departments": int(panel["dept_id"].nunique()),
        "series_per_category": panel.groupby("cat_id", observed=True)[SERIES_ID]
            .nunique().to_dict(),
        "series_per_department": panel.groupby("dept_id", observed=True)[SERIES_ID]
            .nunique().to_dict(),
        "complete_cross_product": bool(
            panel[SERIES_ID].nunique()
            == panel["sku_id"].nunique() * panel["store_id"].nunique()),
    }


def leakage_probe(cfg: Config, panel: pd.DataFrame, fold: Fold) -> dict[str, Any]:
    """Executable proof that features do not read the future.

    Builds the feature matrix for one fold, then **overwrites every actual after
    the forecast origin with nonsense** and rebuilds. Any feature that changes was
    reading data that would not exist at forecast time.

    This catches the mistake that quietly inflates most published demand-forecast
    results: computing ``rolling_mean_7`` on the target column *after* the origin,
    so the row for day origin+5 contains information from days origin+1..origin+4.
    """
    from src.features.build import build_supervised

    real = build_supervised(cfg, panel, fold, horizons=[1, 7, 28])

    mutated_panel = panel.copy()
    future = mutated_panel[DATE] > fold.origin
    rng = np.random.default_rng(cfg.seed)
    mutated_panel.loc[future, TARGET] = rng.integers(
        500, 1000, size=int(future.sum())).astype("float32")
    mutated = build_supervised(cfg, mutated_panel, fold, horizons=[1, 7, 28])

    feature_cols = [c for c in real.columns
                    if c not in (SERIES_ID, DATE, TARGET, "horizon", "target_date",
                                 "y_true")]
    common = real[[SERIES_ID, "target_date", "horizon"]].merge(
        mutated[[SERIES_ID, "target_date", "horizon"]], how="inner",
        on=[SERIES_ID, "target_date", "horizon"])
    a = real.set_index([SERIES_ID, "target_date", "horizon"]).loc[
        pd.MultiIndex.from_frame(common)]
    b = mutated.set_index([SERIES_ID, "target_date", "horizon"]).loc[
        pd.MultiIndex.from_frame(common)]

    changed: list[str] = []
    for c in feature_cols:
        if c not in a.columns or c not in b.columns:
            continue
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x):
            same = np.allclose(x.fillna(-999).to_numpy(dtype=float),
                               y.fillna(-999).to_numpy(dtype=float))
        else:
            same = bool((x.astype(str) == y.astype(str)).all())
        if not same:
            changed.append(c)

    return {
        "fold": fold.name,
        "origin": str(fold.origin.date()),
        "n_features_checked": len(feature_cols),
        "n_rows_compared": len(a),
        "features_changed_by_future_mutation": changed,
        "leak_free": len(changed) == 0,
    }


def run_audit(cfg: Config) -> AuditResult:
    """Execute the audit and return structured findings."""
    panel = load_panel(cfg)
    missing_cols = [c for c in PANEL_COLUMNS if c not in panel.columns]

    stats = demand_stats(panel)
    stats["demand_class"] = classify_demand(stats)
    class_counts = stats["demand_class"].value_counts().to_dict()
    life = lifecycle(panel)
    zeros = zero_taxonomy(panel)
    sp = spikes(panel)

    folds = rolling_origin_folds(cfg, panel)
    leak = leakage_probe(cfg, panel, folds[-1])

    test_start, test_end = test_window(cfg)
    notes = [
        f"Forecasting grain: SKU x store x day. Target: {TARGET} (units).",
        f"Panel spans {panel[DATE].min().date()} to {panel[DATE].max().date()}; "
        f"the final {(test_end - test_start).days + 1} days "
        f"({test_start.date()}..{test_end.date()}) are held out and never used for "
        "fitting or model selection.",
        "Exogenous price and SNAP values exist for the test window, so the holdout "
        "can be scored with the same feature set as training.",
        "Named holiday events are not in this mirror; holiday indicators are derived "
        "from pandas' US federal holiday calendar and flagged as derived.",
    ]

    return AuditResult(
        n_rows=len(panel),
        n_series=int(panel[SERIES_ID].nunique()),
        n_days=int(panel[DATE].nunique()),
        date_min=panel[DATE].min(),
        date_max=panel[DATE].max(),
        schema_ok=not missing_cols,
        missing_columns=missing_cols,
        n_duplicate_rows=int(panel.duplicated(subset=[SERIES_ID, DATE]).sum()),
        n_missing_values=_missing_values(panel),
        n_calendar_gaps=calendar_gaps(panel),
        n_negative_demand=int((panel[TARGET] < 0).sum()),
        n_invalid_values=_invalid_values(panel),
        zero_share=zeros["zero_share"],
        n_series_launched_late=life["n_launched_late"],
        n_series_discontinued=life["n_discontinued"],
        n_spike_days=sp["n_spike_days"],
        n_spike_series=sp["n_spike_series"],
        demand_class_counts={str(k): int(v) for k, v in class_counts.items()},
        intermittent_share=float(
            sum(v for k, v in class_counts.items() if k in ("Intermittent", "Lumpy"))
            / max(len(stats), 1)),
        coverage={**coverage(panel), **life, "zeros": zeros},
        leakage_checks=leak,
        notes=notes,
    )


def _fmt(d: dict, empty: str = "none") -> str:
    return "\n".join(f"- `{k}`: {v}" for k, v in d.items()) if d else f"_{empty}_"


def render_report(res: AuditResult, cfg: Config) -> str:
    """Render the audit as markdown."""
    L: list[str] = []
    a = L.append
    z = res.coverage["zeros"]

    a("# Time-Series Data Quality & Leakage Audit\n")
    a("Generated by `python -m src.run audit`. Every number is computed, not asserted.\n")

    a("## 1. Grain, target and schema\n")
    a("- **Forecasting grain:** SKU x store x day (`series_id` x `date`)")
    a(f"- **Target:** `{TARGET}` — units sold per series per day")
    a(f"- **Panel:** {res.n_rows:,} rows, {res.n_series:,} series, {res.n_days:,} days")
    a(f"- **Range:** {res.date_min.date()} to {res.date_max.date()}")
    a(f"- **Schema complete:** {res.schema_ok}"
      + (f" (missing {res.missing_columns})" if res.missing_columns else ""))
    a("")
    a("| column | kind | source | description | notes |")
    a("|---|---|---|---|---|")
    for s in DATA_DICTIONARY:
        a(f"| `{s.name}` | {s.kind} | {s.source} | {s.description} | {s.notes or '—'} |")
    a("")

    a("## 2. Structural integrity\n")
    a(f"- duplicate `(series_id, date)` rows: **{res.n_duplicate_rows}**")
    a(f"- series with internal calendar gaps: **{res.n_calendar_gaps}**")
    a(f"- negative demand values: **{res.n_negative_demand}**")
    a(f"- values outside valid ranges: {_fmt(res.n_invalid_values, 'none')}")
    a(f"- missing values: {_fmt(res.n_missing_values, 'none')}")
    a("")
    a("Contiguity is checked *within* each series rather than against the global "
      "calendar. Series legitimately begin on different dates because products "
      "launch at different times; a hole in the middle of a series would be a "
      "different problem, and would mean a genuine zero-demand day had been dropped "
      "rather than recorded.\n")

    a("## 3. Is a zero a real zero?\n")
    a("The most consequential question in the audit. Three different things look "
      "identical in the target column.\n")
    a("| zero type | rows | share of all zeros | treatment |")
    a("|---|---:|---:|---|")
    a(f"| pre-launch (product not yet stocked) | {z['n_pre_launch_zero']:,} | "
      f"{z['n_pre_launch_zero'] / max(z['n_zero'], 1):.2%} | not an observation; "
      "excluded by trimming each series to its first sale |")
    a(f"| post-discontinuation | {z['n_post_discontinuation_zero']:,} | "
      f"{z['n_post_discontinuation_zero'] / max(z['n_zero'], 1):.2%} | series has "
      "ended; flagged, not forecast |")
    a(f"| **true zero (on shelf, no sale)** | **{z['n_true_zero']:,}** | "
      f"**{z['true_zero_share_of_zeros']:.2%}** | real demand signal; must be modelled |")
    a("")
    a(f"Overall zero share: **{res.zero_share:.2%}**. Treating pre-launch and "
      "post-delisting zeros as demand would teach the model that demand collapses at "
      "the start and end of a product's life — an artefact of the panel's shape, not "
      "a property of demand.\n")

    a("## 4. Product lifecycle\n")
    a(f"- series launching after the panel opens: **{res.n_series_launched_late:,}** "
      f"of {res.n_series:,}")
    a(f"- distinct series start dates: **{res.coverage['n_distinct_start_dates']:,}**")
    a(f"- series with no sale in the last {DISCONTINUED_DAYS} days "
      f"(discontinuation candidates): **{res.n_series_discontinued:,}**")
    a(f"- history length: median **{res.coverage['median_history_days']:.0f}** days, "
      f"minimum **{res.coverage['min_history_days']:.0f}**")
    a("")

    a("## 5. Outliers and spikes\n")
    a(f"Days above **{SPIKE_MULTIPLE:.0f}x** the series' own non-zero median: "
      f"**{res.n_spike_days:,}** days across **{res.n_spike_series:,}** series.\n")
    a("These are **not** removed. Promotional and seasonal peaks are real demand, and "
      "clipping them would bias safety stock downwards for exactly the products where "
      "a stockout costs most. They are reported so their influence on error metrics is "
      "visible, and RMSE is read alongside MAE for that reason.\n")

    a("## 6. Intermittency and demand character\n")
    a("Syntetos-Boylan classification on ADI (periods per non-zero period) and CV² "
      f"(squared coefficient of variation of non-zero sizes), cut at ADI={1.32} and "
      f"CV²={0.49}.\n")
    a("| demand class | series | share |")
    a("|---|---:|---:|")
    for k in ("Smooth", "Erratic", "Intermittent", "Lumpy"):
        v = res.demand_class_counts.get(k, 0)
        a(f"| {k} | {v:,} | {v / max(res.n_series, 1):.1%} |")
    a("")
    a(f"**{res.intermittent_share:.1%}** of series are Intermittent or Lumpy. That is "
      "the single most important fact for how this project is evaluated: with most "
      "series mostly zero, **MAPE is undefined or explosive**, so WAPE and MASE are "
      "used instead, and a model that predicts near-zero everywhere can look "
      "deceptively strong on MAE alone.\n")

    a("## 7. Coverage\n")
    a(f"- series: **{res.coverage['n_series']:,}** = "
      f"{res.coverage['n_skus']:,} SKUs x {res.coverage['n_stores']} stores "
      f"(complete cross product: {res.coverage['complete_cross_product']})")
    a(f"- stores: {res.coverage['stores']}")
    a(f"- departments: {res.coverage['n_departments']}")
    a(f"- series per category: {res.coverage['series_per_category']}")
    a("")

    a("## 8. Leakage audit\n")
    a("### Why random splitting would be wrong\n")
    a("A random train/test split places day *t+1* in training and day *t* in test for "
      "the same series. The model then interpolates between neighbouring days it has "
      "already seen, and reports an accuracy no deployed forecaster can reach — at "
      "forecast time the future genuinely does not exist. Every split here is "
      "chronological, and validation uses rolling origins.\n")
    a("### Executable proof that features do not read the future\n")
    lk = res.leakage_checks
    a(f"Features were built for `{lk['fold']}` at origin **{lk['origin']}**, then every "
      "actual after the origin was overwritten with nonsense and the features rebuilt. "
      "Any feature that changes was reading data unavailable at forecast time.\n")
    a(f"- features checked: **{lk['n_features_checked']}**")
    a(f"- rows compared: **{lk['n_rows_compared']:,}**")
    a(f"- features changed by mutating the future: "
      f"**{lk['features_changed_by_future_mutation'] or 'none'}**")
    a(f"- **leak-free: {lk['leak_free']}**\n")
    a("The one place future information legitimately enters is `sell_price`, `snap` and "
      "the calendar **for the target date**, which a retailer sets in advance. That is "
      "an assumption about the business process, not a property of the data, and it is "
      "declared in `configs/config.yaml` under `features.assume_known_future`.\n")

    a("## 9. Notes and assumptions\n")
    for n in res.notes:
        a(f"- {n}")
    a("")
    return "\n".join(L)


def run(cfg: Config) -> AuditResult:
    """Audit entry point: prints a summary and writes the markdown report."""
    res = run_audit(cfg)
    cfg.ensure_dirs()
    dest = cfg.resolve("paths", "reports_dir") / "data_audit.md"
    dest.write_text(render_report(res, cfg), encoding="utf-8")

    z = res.coverage["zeros"]
    print(f"panel                  : {res.n_rows:,} rows, {res.n_series:,} series, "
          f"{res.n_days:,} days ({res.date_min.date()}..{res.date_max.date()})")
    print(f"schema ok              : {res.schema_ok}")
    print(f"duplicates / gaps      : {res.n_duplicate_rows} / {res.n_calendar_gaps}")
    print(f"negative demand        : {res.n_negative_demand}")
    print(f"missing values         : {res.n_missing_values or 'none'}")
    print(f"invalid ranges         : {res.n_invalid_values or 'none'}")
    print(f"zero share             : {res.zero_share:.2%}")
    print(f"  pre-launch zeros     : {z['n_pre_launch_zero']:,}")
    print(f"  post-delist zeros    : {z['n_post_discontinuation_zero']:,}")
    print(f"  TRUE zeros           : {z['n_true_zero']:,} "
          f"({z['true_zero_share_of_zeros']:.1%} of zeros)")
    print(f"launched late / delisted: {res.n_series_launched_late:,} / "
          f"{res.n_series_discontinued:,}")
    print(f"spikes                 : {res.n_spike_days:,} days in "
          f"{res.n_spike_series:,} series")
    print(f"demand classes         : {res.demand_class_counts}")
    print(f"intermittent+lumpy     : {res.intermittent_share:.1%}")
    print(f"LEAKAGE probe          : leak_free={res.leakage_checks['leak_free']} "
          f"({res.leakage_checks['n_features_checked']} features, "
          f"{res.leakage_checks['n_rows_compared']:,} rows)")
    if res.leakage_checks["features_changed_by_future_mutation"]:
        print(f"  CHANGED: {res.leakage_checks['features_changed_by_future_mutation']}")
    print(f"\n[written] {dest}")
    return res
