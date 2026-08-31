"""Streamlit decision dashboard.

Every view answers one of the five questions the project exists to answer, and
every chart is attached to a decision. There is no "overview" page and no metric
without an implied action:

    1. How much demand should we expect?      -> forecast with interval
    2. How uncertain is the forecast?         -> interval width and coverage
    3. Which products will stock out?         -> ranked risk worklist
    4. How much inventory should we hold?     -> reorder point and safety stock
    5. What is the overstock/stockout trade?  -> service-level cost curve

Run with:  make app
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import load_config
from src.data.loader import load_panel, test_fold
from src.data.schema import DATE, SERIES_ID, TARGET

st.set_page_config(page_title="Demand Forecasting & Inventory", layout="wide")


@st.cache_data(show_spinner=False)
def _load():
    cfg = load_config()
    panel = load_panel(cfg)
    proc = cfg.resolve("data", "processed_dir")
    out = {"cfg_raw": cfg.raw, "panel": panel}
    for name in ("test_predictions", "policy", "segments"):
        p = proc / f"{name}.parquet"
        out[name] = pd.read_parquet(p) if p.exists() else None
    rr = cfg.resolve("paths", "reports_dir") / "run_results.json"
    if rr.exists():
        import json
        out["results"] = json.loads(rr.read_text(encoding="utf-8"))
    return out


def main() -> None:
    cfg = load_config()
    data = _load()
    panel, preds = data["panel"], data["test_predictions"]
    policy = data["policy"]
    results = data.get("results", {})

    st.title("Demand Forecasting & Inventory Optimization")
    st.caption(
        "Grain: SKU x store x day. Currency figures use the ASSUMED economics in "
        "configs/config.yaml and are shown in currency units (CU); only unit value "
        "comes from the data."
    )

    if preds is None or policy is None:
        st.warning("No results found. Run `make all` first.")
        st.stop()

    point_col = "lightgbm" if "lightgbm" in preds.columns else "moving_average_28"
    tf = test_fold(cfg)

    # ---- sidebar: the decision inputs ----
    st.sidebar.header("Decision inputs")
    risk = policy.merge(
        preds.groupby(SERIES_ID, observed=True)["y_true"].sum().rename("actual_28d"),
        on=SERIES_ID, how="left")
    default_ix = int(risk["expected_lt_demand"].fillna(0).idxmax())
    series = st.sidebar.selectbox(
        "Series", options=policy[SERIES_ID].tolist(),
        index=min(default_ix, len(policy) - 1))
    sl = st.sidebar.select_slider(
        "Service level", options=[0.90, 0.95, 0.99],
        value=float(cfg["inventory"]["default_service_level"]),
        format_func=lambda v: f"{v:.0%}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "1-2. Forecast & uncertainty", "3. Stockout risk",
        "4. Reorder decisions", "5. Cost trade-off"])

    # ---- 1-2. forecast and uncertainty ----
    with tab1:
        st.subheader(f"Forecast for {series}")
        hist = panel[(panel[SERIES_ID] == series)].sort_values(DATE)
        hist = hist[hist[DATE] > tf.origin - pd.Timedelta(days=120)]
        f = preds[preds[SERIES_ID] == series].sort_values("target_date")

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(hist[DATE], hist[TARGET], color="#4a5568", lw=0.9, label="actual (history)")
        ax.plot(f["target_date"], f["y_true"], color="#c53030", lw=1.2,
                label="actual (held-out)")
        ax.plot(f["target_date"], f[point_col], color="#2b6cb0", lw=1.8,
                label=f"forecast ({point_col})")
        if {"q10", "q90"} <= set(f.columns):
            ax.fill_between(f["target_date"], f["q10"], f["q90"], color="#2b6cb0",
                            alpha=0.18, label="P10-P90")
        ax.axvline(tf.origin, ls="--", color="black", lw=1)
        ax.set_ylabel("units/day")
        ax.legend(fontsize=8)
        ax.set_title("Forecast vs actual over the held-out 28 days")
        st.pyplot(fig)
        plt.close(fig)

        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast, 28d", f"{f[point_col].sum():,.0f} units")
        c2.metric("Actual, 28d", f"{f['y_true'].sum():,.0f} units")
        err = f[point_col].sum() - f["y_true"].sum()
        c3.metric("Error", f"{err:+,.0f} units",
                  delta=f"{err / max(f['y_true'].sum(), 1):+.1%}")

        ir = pd.DataFrame(results.get("train", {}).get("interval_report") or [])
        if not ir.empty:
            st.markdown(
                f"**Interval calibration.** Empirical coverage "
                f"{float(ir['empirical_coverage'].iloc[0]):.1%} against a nominal "
                f"{float(ir['nominal_coverage'].iloc[0]):.0%} — the band is somewhat "
                "too narrow, so quantile-based safety stock is slightly optimistic. "
                "**Decision:** prefer the error-spread safety stock in tab 4 over the "
                "raw P90 for high-value SKUs.")

    # ---- 3. stockout risk worklist ----
    with tab2:
        st.subheader("Which products are likely to stock out?")
        inv = results.get("inventory", {})
        rt = pd.DataFrame(inv.get("risk_table") or [])
        if rt.empty:
            st.info("Run `make inventory` to produce the risk table.")
        else:
            st.caption(
                f"Simulated over the held-out 28 days under the "
                f"{inv.get('best_policy', 'forecast')} policy. Ranked by units short, "
                "because that is what a planner can act on — not by percentage error.")
            st.dataframe(rt, width="stretch")
            fig, ax = plt.subplots(figsize=(10, 3.6))
            top = rt.head(12)
            ax.barh(top[SERIES_ID].astype(str)[::-1], top["units_short"][::-1],
                    color="#c53030")
            ax.set_xlabel("units short over 28 days")
            ax.set_title("Worklist: where the shortfall actually is")
            st.pyplot(fig)
            plt.close(fig)

    # ---- 4. reorder decisions ----
    with tab3:
        st.subheader("How much inventory should we hold?")
        row = policy[policy[SERIES_ID] == series]
        if row.empty:
            st.info("No policy row for this series.")
        else:
            r = row.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Expected LT demand", f"{r['expected_lt_demand']:,.0f}")
            c2.metric("Safety stock", f"{r['safety_stock_normal']:,.1f}")
            c3.metric("Reorder point", f"{r['reorder_point_normal']:,.0f}")
            c4.metric("Protection window", f"{int(r['protection_days'])} d")
            st.caption(
                f"Protection window = {int(r['lead_time_days'])} d lead time + "
                f"{int(r['review_period_days'])} d review period. Safety stock is "
                "derived from the spread of **lead-time forecast error**, not from "
                "historical demand variability — which is why a better forecast "
                "reduces the stock required.")
            st.markdown("**Full replenishment worklist** (sorted by reorder point):")
            cols = [SERIES_ID, "expected_lt_demand", "error_std",
                    "safety_stock_normal", "safety_stock_empirical",
                    "reorder_point_normal", "reorder_point_empirical"]
            st.dataframe(policy[cols].sort_values("reorder_point_normal",
                                                 ascending=False),
                         width="stretch")

    # ---- 5. cost trade-off ----
    with tab4:
        st.subheader("What is the trade-off between overstocking and stockouts?")
        sc = pd.DataFrame(results.get("scenarios", {}).get("scenario_table") or [])
        if sc.empty:
            st.info("Run `make scenarios` first.")
        else:
            fig, ax = plt.subplots(figsize=(9, 4))
            x = (sc["service_level"] * 100).astype(str)
            ax.plot(x, sc["holding_cost_cu"], "o-", label="holding", color="#2b6cb0")
            ax.plot(x, sc["stockout_cost_cu"], "o-", label="stockout", color="#c53030")
            ax.plot(x, sc["total_cost_cu"], "o-", lw=2.5, label="total", color="#276749")
            best = sc.loc[sc["total_cost_cu"].idxmin()]
            ax.axvline(f"{best['service_level'] * 100}", ls="--", color="black", lw=1)
            ax.set_xlabel("service level (%)")
            ax.set_ylabel("cost (CU)")
            ax.set_title("Cost is U-shaped; the minimum is the decision")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close(fig)

            st.metric("Cost-minimising service level",
                      f"{best['service_level']:.0%}",
                      delta=f"{best['achieved_fill_rate']:.2%} achieved fill rate")
            st.dataframe(sc, width="stretch")

            sens = pd.DataFrame(results.get("scenarios", {}).get("sensitivity") or [])
            if not sens.empty:
                st.markdown(
                    "**The recommendation depends on an assumption.** The optimum "
                    "moves with the stockout penalty, which the data cannot supply:")
                st.dataframe(sens, width="stretch")

            if sl != float(best["service_level"]):
                st.info(
                    f"You selected {sl:.0%}; the cost-minimising level under the "
                    f"current assumptions is {best['service_level']:.0%}.")


if __name__ == "__main__":
    main()
