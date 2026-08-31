"""Generate the eight analysis notebooks from one source.

The notebooks demonstrate the analysis; they contain no implementation. Every cell
calls into `src/`, which is what stops the notebook version from drifting away from
the code that is tested and served.

    python tools/build_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks"

PRE = "\n".join([
    "import sys, warnings",
    "from pathlib import Path",
    "sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == 'notebooks' "
    "else Path.cwd()))",
    "warnings.filterwarnings('ignore')",
    "",
    "from src.config import load_config",
    "from src.data.loader import load_panel",
    "cfg = load_config()",
    "panel = load_panel(cfg)",
    "print(f'panel: {len(panel):,} rows, {panel.series_id.nunique()} series, "
    "{panel.date.nunique()} days')",
])

NB: dict[str, list[tuple[str, str]]] = {
    "01_data_audit": [
        ("md", "# 01 — Data Audit\n\n"
               "Grain, schema, and the question that matters most for a demand panel: "
               "**is a zero a real zero?**\n\n"
               "A zero can mean the product was on shelf and nobody bought it, or that "
               "it did not exist yet, or that it was delisted. Training on all three as "
               "the same thing teaches the model that demand collapses at the start and "
               "end of a product's life — an artefact of the panel's shape."),
        ("code", PRE),
        ("md", "## Grain and schema"),
        ("code", "from src.data.schema import DATA_DICTIONARY\n"
                 "import pandas as pd\n"
                 "print('grain:', cfg['data']['grain'], '| target:', cfg.target)\n"
                 "pd.DataFrame([{'column': s.name, 'kind': s.kind, 'source': s.source,\n"
                 "               'description': s.description, 'notes': s.notes}\n"
                 "              for s in DATA_DICTIONARY])"),
        ("md", "## Structural integrity and the zero taxonomy"),
        ("code", "from src.data.audit import zero_taxonomy, lifecycle, calendar_gaps, spikes\n"
                 "print('duplicate (series,date):', int(panel.duplicated(['series_id','date']).sum()))\n"
                 "print('series with internal gaps:', calendar_gaps(panel))\n"
                 "print('negative demand:', int((panel.demand < 0).sum()))\n"
                 "zero_taxonomy(panel)"),
        ("code", "lifecycle(panel)"),
        ("code", "spikes(panel)"),
        ("md", "## Intermittency: the fact that shapes every later choice"),
        ("code", "from src.data.loader import demand_stats, classify_demand\n"
                 "st = demand_stats(panel); st['demand_class'] = classify_demand(st)\n"
                 "print(st.demand_class.value_counts())\n"
                 "st[['ADI','CV2','zero_share','mean_demand']].describe().round(3)"),
        ("md", "## The full audit, including the executable leakage probe\n\n"
               "The probe rebuilds features after overwriting every actual past the "
               "forecast origin. Any feature that changes was reading the future."),
        ("code", "from src.data import audit\nres = audit.run(cfg)"),
        ("code", "res.leakage_checks"),
    ],
    "02_time_series_eda": [
        ("md", "# 02 — Exploratory Time-Series Analysis\n\n"
               "Six figures, each answering one business question. All computed on data "
               "**before the test window** — exploring the holdout is a soft leak."),
        ("code", PRE),
        ("code", "from src import eda\nfindings = eda.run(cfg)"),
        ("code", "from IPython.display import Image, display\n"
                 "figdir = cfg.resolve('paths','figures_dir')\n"
                 "for f in ('01_demand_trend','02_seasonality','03_demand_concentration',\n"
                 "          '04_intermittency','05_price_and_volatility','06_demand_classes'):\n"
                 "    display(Image(str(figdir / f'{f}.png')))"),
        ("md", "## What the numbers say"),
        ("code", "{k: v for k, v in findings.items() if not isinstance(v, dict)}"),
        ("md", "Demand is highly concentrated and highly intermittent at the same time: "
               "a small minority of series carry most of the volume, while the median "
               "SKU sells on a minority of days. Those two facts pull forecasting in "
               "opposite directions and are why the project evaluates per segment."),
    ],
    "03_baselines": [
        ("md", "# 03 — Baseline Forecasts\n\n"
               "The bar the ML models must clear. These are not straw men: seasonal "
               "naive exploits the strong weekly cycle, and Croston/SBA are the standard "
               "methods for the sparse series that dominate this panel."),
        ("code", PRE),
        ("code", "from src.forecasting.baselines import BASELINE_DESCRIPTIONS\n"
                 "import pandas as pd\n"
                 "pd.DataFrame(BASELINE_DESCRIPTIONS.items(), columns=['baseline','definition'])"),
        ("md", "## Rolling-origin evaluation of the baselines"),
        ("code", "from src.data.loader import rolling_origin_folds\n"
                 "for f in rolling_origin_folds(cfg, panel):\n"
                 "    print(f'{f.name}: origin {f.origin.date()} -> predict "
                 "{f.start.date()}..{f.end.date()}')"),
        ("code", "from src.evaluation.validate import walk_forward, aggregate_scores, per_fold_table\n"
                 "from src.evaluation.metrics import skill_vs\n"
                 "res = walk_forward(cfg, panel, fit_ml=False)\n"
                 "skill_vs(aggregate_scores(res), 'seasonal_naive').round(4)"),
        ("code", "per_fold_table(res).round(4)"),
        ("md", "## Croston is not a moving average\n\n"
               "A naive implementation collapses to one: mean non-zero size x non-zero "
               "rate equals sum/n, the window mean. Real Croston smooths size and "
               "interval separately and updates only when a sale occurs. SBA then "
               "corrects Croston's known upward bias."),
        ("code", "import numpy as np, pandas as pd\n"
                 "from src.features.build import croston_state\n"
                 "dates = pd.date_range('2020-01-01', periods=90, freq='D')\n"
                 "y = np.where(np.arange(90) % 5 == 0, 10.0, 0.0)\n"
                 "wide = pd.DataFrame([y], index=['S'], columns=dates)\n"
                 "print(croston_state(wide, alpha=0.1))\n"
                 "print('28-day moving average:', y[-28:].mean())"),
        ("md", "## Watch the `zero` baseline's MASE\n\n"
               "Predicting zero everywhere attains the **best MASE** on this data, "
               "because averaging per-row scaled errors rewards predicting nothing on "
               "sparse series. WAPE correctly scores it 1.0. This is why WAPE is the "
               "headline metric here."),
    ],
    "04_feature_engineering": [
        ("md", "# 04 — Feature Engineering\n\n"
               "Every target-derived feature is computed **as of the forecast origin**, "
               "never as of the target date.\n\n"
               "To forecast day `origin+14`, the model may use demand up to `origin`. It "
               "may not use a lag measured from the target date, because that would be "
               "`origin+13` — a day that has not happened yet. Building "
               "`rolling_mean_7` with a panel-wide groupby-shift is the classic way this "
               "goes wrong, and it inflates reported accuracy substantially."),
        ("code", PRE),
        ("code", "from src.data.loader import rolling_origin_folds\n"
                 "from src.features.build import build_supervised, origin_features, feature_columns\n"
                 "fold = rolling_origin_folds(cfg, panel)[-1]\n"
                 "df = build_supervised(cfg, panel, fold)\n"
                 "print(f'{len(df):,} rows, {len(feature_columns(df))} features')\n"
                 "print('lags:', cfg.lags, '| rolling windows:', cfg.rolling_windows)\n"
                 "df.head(3)"),
        ("md", "## Origin-anchored features for one origin"),
        ("code", "f = origin_features(panel[panel.date <= fold.origin], fold.origin, cfg)\n"
                 "f[[c for c in f.columns if c.startswith(('lag_','rolling_mean_'))]].head()"),
        ("md", "## Proof that features cannot see the future\n\n"
               "Overwrite every actual after the origin with nonsense, rebuild, and "
               "compare."),
        ("code", "from src.data.audit import leakage_probe\n"
                 "leakage_probe(cfg, panel, fold)"),
        ("md", "## What is allowed to come from the future\n\n"
               "`sell_price`, `snap` and the calendar for the **target date**. A retailer "
               "sets prices and promotion calendars in advance, so these are genuinely "
               "known at forecast time. That is an assumption about the business process, "
               "declared in `configs/config.yaml` under `features.assume_known_future`."),
        ("code", "cfg['features']"),
    ],
    "05_modeling": [
        ("md", "# 05 — Forecasting Models\n\n"
               "One **global** LightGBM across all series, not one model per SKU: most "
               "series are too sparse to support their own model, new products have no "
               "history at all, and per-series fitting costs scale with the number of "
               "series for no additional insight.\n\n"
               "Objective is Tweedie because demand is a non-negative, zero-inflated "
               "count; squared error treats it as symmetric and unbounded."),
        ("code", PRE),
        ("md", "## Walk-forward validation with refitting per fold\n\n"
               "Runs several minutes: the model is refitted for each fold, because "
               "reusing one model fitted on all training data would let fold 1 be scored "
               "by a model that had seen fold 4."),
        ("code", "from src.evaluation.validate import walk_forward, aggregate_scores, per_fold_table\n"
                 "from src.evaluation.metrics import skill_vs\n"
                 "res = walk_forward(cfg, panel, fit_ml=True, n_origins=20)\n"
                 "skill_vs(aggregate_scores(res), 'seasonal_naive').round(4)"),
        ("code", "per_fold_table(res).round(4)"),
        ("md", "## Feature importance"),
        ("code", "from src.features.build import build_supervised, training_origins\n"
                 "from src.forecasting import ml as ML\n"
                 "from src.data.loader import rolling_origin_folds\n"
                 "fold = rolling_origin_folds(cfg, panel)[-1]\n"
                 "tr = build_supervised(cfg, panel, fold,\n"
                 "                      origins=training_origins(cfg, panel, fold.origin, 20))\n"
                 "m = ML.fit_lightgbm(cfg, tr)\n"
                 "ML.feature_importance(m, top=15)"),
        ("md", "## Classical models on a stratified sample\n\n"
               "ETS and SARIMA are fitted per series, so only on a sample. Their "
               "assumptions (a continuous, roughly Gaussian process) fail on intermittent "
               "series, which is a result worth reporting rather than a bug to hide."),
        ("code", "from src.forecasting.classical import forecast_classical, sample_series\n"
                 "sample = sample_series(cfg, panel)\n"
                 "cl = forecast_classical(cfg, panel, fold, sample)\n"
                 "print('fit failures:', cl.n_failed)\n"
                 "for n in cl.notes: print('note:', n)"),
    ],
    "06_forecast_evaluation": [
        ("md", "# 06 — Forecast Evaluation\n\n"
               "Metrics, horizons, segments, error attribution and uncertainty.\n\n"
               "**MAPE is not reported.** With ~59% of observations zero it is undefined "
               "on most rows. WAPE is the headline; MASE is reported for scale-free "
               "comparison, with the caveat that the `zero` baseline wins on it."),
        ("code", PRE),
        ("code", "from src.run import load_results\nr = load_results(cfg)\nlist(r)"),
        ("md", "## Validation and held-out test"),
        ("code", "import pandas as pd\npd.DataFrame(r['train']['validation_aggregate']).round(4)"),
        ("code", "pd.DataFrame(r['train']['test_table']).round(4)"),
        ("md", "## Accuracy by horizon"),
        ("code", "hb = pd.DataFrame(r['train']['horizon_breakdown'])\n"
                 "hb.pivot(index='horizon', columns='model', values='wape').round(4)"),
        ("md", "## Which model wins in which segment\n\n"
               "The answerable version of \"which model is best\"."),
        ("code", "pd.DataFrame(r['train']['best_per_segment']).round(4)"),
        ("md", "## What drives forecast error"),
        ("code", "pd.DataFrame(r['train']['error_drivers']).round(4)"),
        ("code", "pd.DataFrame(r['train']['promo_errors']).round(4)"),
        ("md", "## Uncertainty: are the intervals honest?\n\n"
               "An unvalidated P90 is an assumption dressed as a measurement."),
        ("code", "pd.DataFrame(r['train']['interval_report']).round(4)"),
        ("code", "pd.DataFrame(r['train']['interval_by_horizon']).round(4).head(28)"),
    ],
    "07_inventory_optimization": [
        ("md", "# 07 — Inventory Optimization\n\n"
               "Safety stock is derived from the spread of **lead-time forecast error**, "
               "not from historical demand variability. That is the entire reason "
               "forecasting has value: a better forecast shrinks the error spread and "
               "therefore the stock required. Sizing from demand variability returns the "
               "same answer no matter how good the forecast is."),
        ("code", PRE),
        ("code", "from src.inventory.policy import PolicyParams, z_for_service_level\n"
                 "p = PolicyParams.from_config(cfg)\n"
                 "print(f'lead time {p.lead_time_days} d + review {p.review_period_days} d "
                 "= {p.protection_days} d of exposure')\n"
                 "print('z(95%) =', round(z_for_service_level(0.95), 4))"),
        ("md", "Under periodic review the exposure is lead time **plus** the review "
               "period: a stockout can occur any time before the *next* order arrives. "
               "Omitting the review period systematically under-protects."),
        ("md", "## Lead-time forecast error and safety stock"),
        ("code", "import pandas as pd\n"
                 "from src.inventory.policy import lead_time_forecast_errors, error_spread, build_policy\n"
                 "proc = cfg.resolve('data','processed_dir')\n"
                 "ew = pd.read_parquet(proc / 'error_windows.parquet')\n"
                 "lt = lead_time_forecast_errors(ew, 'lightgbm', p.protection_days)\n"
                 "sp = error_spread(lt)\n"
                 "print('windows per series (median):', int(sp.n_windows.median()))\n"
                 "sp[['error_std','error_q90','error_q95']].describe().round(3)"),
        ("code", "test_preds = pd.read_parquet(proc / 'test_predictions.parquet')\n"
                 "pol = build_policy(cfg, test_preds, 'lightgbm', service_level=0.95, lt_errors=lt)\n"
                 "pol[['series_id','expected_lt_demand','error_std','safety_stock_normal',\n"
                 "     'safety_stock_empirical','reorder_point_normal']].head(10).round(2)"),
        ("md", "## Simulate: does the policy deliver its service level?\n\n"
               "A reorder point from a formula is a claim. Replaying the held-out demand "
               "day by day is the test. Unmet demand is treated as **lost sales**, not "
               "backorders."),
        ("code", "from src.inventory.simulate import simulate_policy, compare_policies\n"
                 "from src.inventory.policy import baseline_policy\n"
                 "from src.data.loader import test_fold\n"
                 "tf = test_fold(cfg)\n"
                 "base = baseline_policy(cfg, panel[panel.date <= tf.origin], 0.95)\n"
                 "sims = [simulate_policy(panel, base, tf.start, tf.end, p.lead_time_days,\n"
                 "                        p.review_period_days, 'reorder_point_normal',\n"
                 "                        policy_name='baseline_demand_variability'),\n"
                 "        simulate_policy(panel, pol, tf.start, tf.end, p.lead_time_days,\n"
                 "                        p.review_period_days, 'reorder_point_normal',\n"
                 "                        policy_name='forecast_normal_ss')]\n"
                 "compare_policies(sims).round(4)"),
        ("md", "## Cost the policies\n\n"
               "> Only unit value comes from the data. Margin, holding rate, ordering "
               "cost and the stockout penalty are declared assumptions."),
        ("code", "from src.inventory import costs as C\n"
                 "uv = C.unit_values(panel, as_of=tf.origin)\n"
                 "econ = C.Economics.from_config(cfg)\n"
                 "pd.DataFrame([{'policy': s.policy_name,\n"
                 "               **C.cost_summary(C.cost_simulation(s.per_series, uv, econ, days=28))}\n"
                 "              for s in sims]).round(2)"),
    ],
    "08_scenario_analysis": [
        ("md", "# 08 — Scenario Analysis\n\n"
               "Raising the service level buys fewer stockouts with more inventory. The "
               "cost curve is U-shaped and its minimum is the economically reasonable "
               "service level — **under the stated assumptions**, which the data cannot "
               "supply."),
        ("code", PRE),
        ("code", "import pandas as pd\n"
                 "from src.run import load_results\n"
                 "r = load_results(cfg)\n"
                 "sc = pd.DataFrame(r['scenarios']['scenario_table'])\n"
                 "sc.round(4)"),
        ("md", "## The analytical cross-check\n\n"
               "Newsvendor theory says the optimal service level is the critical ratio "
               "`Cu / (Cu + Co)`. Computing both the closed form and the simulated "
               "optimum makes each a check on the other."),
        ("code", "r['scenarios']['newsvendor'], r['scenarios']['cost_optimal_service_level']"),
        ("md", "## Sensitivity: the recommendation moves with the assumptions"),
        ("code", "pd.DataFrame(r['scenarios']['sensitivity']).round(4)"),
        ("md", "## The cost curve"),
        ("code", "import matplotlib.pyplot as plt\n"
                 "fig, ax = plt.subplots(figsize=(9,4))\n"
                 "x = (sc.service_level*100).astype(str)\n"
                 "ax.plot(x, sc.holding_cost_cu, 'o-', label='holding')\n"
                 "ax.plot(x, sc.stockout_cost_cu, 'o-', label='stockout')\n"
                 "ax.plot(x, sc.total_cost_cu, 'o-', lw=2.5, label='total')\n"
                 "ax.set_xlabel('service level (%)'); ax.set_ylabel('cost (CU)')\n"
                 "ax.legend(); ax.set_title('Overstock vs stockout trade-off')\n"
                 "plt.show()"),
        ("md", "## Example decision\n\n"
               "The forecast turned into an order."),
        ("code", "from src.inference.forecaster import DemandForecaster, format_recommendation\n"
                 "from src.data.loader import test_fold\n"
                 "f = DemandForecaster.load(cfg)\n"
                 "seg = pd.read_parquet(cfg.resolve('data','processed_dir')/'segments.parquet')\n"
                 "sid = str(seg.sort_values('total_demand', ascending=False).index[0])\n"
                 "print(format_recommendation(f.recommend(panel, sid, origin=test_fold(cfg).origin)))"),
    ],
}


def build(cells: list[tuple[str, str]]) -> dict:
    out = []
    for kind, body in cells:
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {},
                        "source": body.splitlines(keepends=True)})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                        "outputs": [], "source": body.splitlines(keepends=True)})
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, cells in NB.items():
        path = DEST / f"{name}.ipynb"
        path.write_text(json.dumps(build(cells), indent=1), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
