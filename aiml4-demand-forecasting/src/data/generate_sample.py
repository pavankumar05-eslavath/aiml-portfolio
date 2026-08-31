"""Generate a synthetic stand-in panel with the same schema AND the same quirks.

    python -m src.data.generate_sample

Data is gitignored, so CI has nothing to run on. This writes a panel that
reproduces the properties the pipeline must cope with, so the audit, the
leakage checks and the inventory maths are genuinely exercised offline:

  * ~60% zero demand, i.e. intermittency
  * all four Syntetos-Boylan demand classes present
  * series with different start dates (new product launches)
  * one discontinued series (sales stop part-way through)
  * weekly and annual seasonality, plus a slow trend
  * stepped price changes and SNAP day flags
  * occasional extreme spikes
  * the same date range and test window as the real data, so splits behave

It is NOT a substitute for the real data. Reported metrics come from the real M5
panel; tests asserting real-data figures are marked and skip here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.schema import PANEL_COLUMNS

N_SKUS = 24
STORES = ("CA_1", "TX_1", "WI_1")
STATE_OF = {"CA_1": "CA", "TX_1": "TX", "WI_1": "WI"}
CATS = ("FOODS", "HOUSEHOLD", "HOBBIES")

# Demand-class recipes: (base rate, zero-inflation, spike probability).
# Chosen so the generated panel spans Smooth / Erratic / Intermittent / Lumpy
# rather than only the well-behaved end.
CLASS_RECIPES = {
    "Smooth": (6.0, 0.05, 0.001),
    "Erratic": (5.0, 0.10, 0.030),
    "Intermittent": (0.8, 0.60, 0.002),
    "Lumpy": (1.2, 0.55, 0.040),
}


def build(cfg, seed: int = 42) -> pd.DataFrame:
    """Build the synthetic panel."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2011-01-29")
    end = pd.Timestamp(cfg["split"]["test_end"])
    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)

    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    t = np.arange(n_days)

    # SNAP: first ten days of each month, matching the real ~33% share, and
    # offset per state so the state dimension carries signal.
    dom = dates.day.to_numpy()
    snap_by_state = {
        "CA": ((dom >= 1) & (dom <= 10)).astype(float),
        "TX": ((dom >= 2) & (dom <= 11)).astype(float),
        "WI": ((dom >= 3) & (dom <= 12)).astype(float),
    }

    classes = list(CLASS_RECIPES)
    rows = []
    for i in range(N_SKUS):
        sku = f"{CATS[i % 3]}_{1 + i // 8}_{i:03d}"
        cat = CATS[i % 3]
        dept = f"{cat}_{1 + i % 2}"
        cls = classes[i % len(classes)]
        base, zero_p, spike_p = CLASS_RECIPES[cls]

        # Launch date: a third of SKUs start late, so the panel contains new
        # products whose history is shorter than the window.
        launch = 0 if i % 3 else int(rng.integers(300, 900))
        # One SKU is discontinued part-way through.
        discontinue = n_days - 200 if i == 5 else n_days + 1

        # Stepped price: a few changes over the horizon, like a real price file.
        n_steps = int(rng.integers(1, 5))
        cuts = np.sort(rng.choice(np.arange(60, n_days - 60), size=n_steps, replace=False))
        levels = np.round(rng.uniform(1.5, 9.0) * rng.uniform(0.85, 1.15, n_steps + 1), 2)
        price = np.empty(n_days, dtype=float)
        edges = [0, *cuts.tolist(), n_days]
        for k in range(len(edges) - 1):
            price[edges[k]:edges[k + 1]] = max(levels[k], 0.5)

        for store in STORES:
            state = STATE_OF[store]
            snap = snap_by_state[state]

            weekly = 1.0 + 0.35 * np.sin(2 * np.pi * (dow - 4) / 7)
            annual = 1.0 + 0.20 * np.sin(2 * np.pi * (doy - 80) / 365.25)
            trend = 1.0 + 0.00015 * t
            store_effect = {"CA_1": 1.15, "TX_1": 1.0, "WI_1": 0.9}[store]
            # Price elasticity and a SNAP lift, so the exogenous features matter.
            price_effect = (price / np.median(price)) ** -1.2
            snap_effect = 1.0 + 0.12 * snap

            lam = base * weekly * annual * trend * store_effect * price_effect * snap_effect
            y = rng.poisson(np.clip(lam, 0.01, None)).astype(float)

            y[rng.random(n_days) < zero_p] = 0.0
            spikes = rng.random(n_days) < spike_p
            y[spikes] *= rng.integers(5, 25, size=int(spikes.sum()))

            y[:launch] = 0.0
            if discontinue <= n_days:
                y[discontinue:] = 0.0

            rows.append(pd.DataFrame({
                "series_id": f"{sku}_{store}",
                "date": dates,
                "demand": y.astype("float32"),
                "sku_id": sku,
                "dept_id": dept,
                "cat_id": cat,
                "store_id": store,
                "state_id": state,
                "sell_price": price.astype("float32"),
                "snap": snap.astype("float32"),
            }))

    panel = pd.concat(rows, ignore_index=True)
    # Drop pre-launch rows, mirroring the real mirror which trims each series to
    # begin when the item was first priced in that store.
    first = panel[panel["demand"] > 0].groupby("series_id", observed=True)["date"].min()
    panel = panel.merge(first.rename("first_sale"), on="series_id", how="left")
    panel = panel[panel["date"] >= panel["first_sale"]].drop(columns="first_sale")
    return panel.loc[:, list(PANEL_COLUMNS)].sort_values(
        ["series_id", "date"], ignore_index=True)


def main() -> int:
    cfg = load_config()
    cfg.ensure_dirs()
    panel = build(cfg, seed=cfg.seed)

    dest = cfg.resolve("data", "sample_file")
    panel.to_parquet(dest, index=False)
    # Also write it as the cached panel so `make all` can run immediately.
    panel.to_parquet(cfg.resolve("data", "processed_dir") / "panel.parquet", index=False)

    print(f"wrote {dest} ({len(panel):,} rows, {panel.series_id.nunique()} series, "
          f"{panel.date.nunique()} days)")
    print(f"zero share {(panel.demand == 0).mean():.3f} | "
          f"snap share {panel.snap.mean():.3f} | "
          f"price mean {panel.sell_price.mean():.2f}")
    print("reproduces: intermittency, all four demand classes, staggered launches,")
    print("            a discontinued series, weekly+annual seasonality, price steps,")
    print("            SNAP flags and demand spikes.")
    print("NOTE: synthetic. Reported metrics come from the real M5 panel "
          "(`make download && make panel`).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
