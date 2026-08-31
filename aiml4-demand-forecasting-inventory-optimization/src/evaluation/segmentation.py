"""Demand segmentation and per-segment model selection.

Two orthogonal dimensions, because they answer different questions:

* **Volume (ABC)** -- how much this series contributes to total demand. Drives how
  much attention and safety stock budget it deserves.
* **Character (Syntetos-Boylan, from ADI and CV²)** -- whether demand is smooth,
  erratic, intermittent or lumpy. Drives *which forecasting method can work at all*.

Crossing them gives the practical grid: a high-volume lumpy SKU is a completely
different operational problem from a low-volume smooth one, even though a single
aggregate WAPE would treat them as interchangeable.

The per-segment comparison is the payoff: it turns "which model is best?" into
"which model is best *for what*", which is the answerable version of the question.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.data.loader import ADI_CUT, CV2_CUT, classify_demand, demand_stats
from src.data.schema import DATE, SERIES_ID

#: Cumulative demand shares defining the ABC volume classes.
ABC_CUTS: tuple[float, float] = (0.70, 0.90)

SEGMENT_ADVICE: dict[str, str] = {
    "Smooth": "Classical exponential smoothing or a global ML model both work. "
              "Demand has a stable level and a clear weekly cycle, so the seasonal "
              "signal is learnable and intervals are near-symmetric.",
    "Erratic": "Global ML model. The level is stable enough to forecast but sizes "
               "vary widely, so the value is in the interval rather than the point "
               "forecast; size safety stock from an upper quantile, not a z-score.",
    "Intermittent": "Croston/SBA or a global ML model with a count-like objective. "
                    "Most days are zero, so a squared-error model biases towards "
                    "the mean and a classical seasonal model has no level to track.",
    "Lumpy": "Hardest class: rare demand AND variable size. Point forecasts are "
             "close to useless; manage these with quantile-based safety stock and "
             "accept a lower service level or a higher inventory cost.",
}


def segment_series(cfg: Config, panel: pd.DataFrame,
                   as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Per-series segmentation computed from history only.

    ``as_of`` restricts the statistics to data available at the forecast origin.
    Segmenting on the full panel including the test window would leak: the segment
    would partly encode how the series behaved during the evaluation period, and
    per-segment results would be flattered.
    """
    hist = panel if as_of is None else panel[panel[DATE] <= as_of]
    stats = demand_stats(hist)
    stats["demand_class"] = classify_demand(stats)

    total = stats["total_demand"].sum()
    ranked = stats.sort_values("total_demand", ascending=False)
    cum = ranked["total_demand"].cumsum() / max(total, 1e-9)
    abc = pd.Series(
        np.where(cum <= ABC_CUTS[0], "A", np.where(cum <= ABC_CUTS[1], "B", "C")),
        index=ranked.index, name="volume_class")
    stats = stats.join(abc)
    stats["segment"] = stats["volume_class"].astype(str) + "-" + stats["demand_class"].astype(str)

    meta = panel.drop_duplicates(SERIES_ID).set_index(SERIES_ID)[
        ["sku_id", "store_id", "cat_id", "dept_id", "state_id"]]
    return stats.join(meta, how="left")


def segment_summary(segments: pd.DataFrame) -> pd.DataFrame:
    """Size and character of each demand class."""
    out = segments.groupby("demand_class", observed=True).agg(
        series=("total_demand", "size"),
        total_demand=("total_demand", "sum"),
        mean_daily_demand=("mean_demand", "mean"),
        median_ADI=("ADI", "median"),
        median_CV2=("CV2", "median"),
        mean_zero_share=("zero_share", "mean"),
    )
    out["demand_share"] = out["total_demand"] / out["total_demand"].sum()
    out["series_share"] = out["series"] / out["series"].sum()
    out["recommended_approach"] = [SEGMENT_ADVICE.get(str(i), "") for i in out.index]
    return out.reset_index()


def abc_xyz_matrix(segments: pd.DataFrame) -> pd.DataFrame:
    """Series counts across the volume x character grid."""
    return pd.crosstab(segments["volume_class"], segments["demand_class"])


def scores_by_segment(
    predictions: pd.DataFrame, segments: pd.DataFrame, model_cols: list[str],
    mase_denom: pd.Series | None = None, by: str = "demand_class",
) -> pd.DataFrame:
    """Accuracy of each model within each segment."""
    from src.evaluation.metrics import score

    d = predictions.merge(segments[[by]], left_on=SERIES_ID, right_index=True,
                          how="left")
    rows = []
    for seg, g in d.groupby(by, observed=True):
        for m in model_cols:
            s = score(g["y_true"], g[m], g[SERIES_ID], mase_denom)
            rows.append({by: seg, "model": m, "n": s.n, "wape": s.wape,
                         "mae": s.mae, "rmse": s.rmse, "mase": s.mase,
                         "bias": s.bias})
    return pd.DataFrame(rows)


def best_model_per_segment(
    seg_scores: pd.DataFrame, by: str = "demand_class", metric: str = "wape",
) -> pd.DataFrame:
    """Winner per segment, with the margin over the best baseline.

    The margin matters: if the ML model wins a segment by 0.5% of WAPE, the
    operationally correct answer is the simpler method.
    """
    baselines = {"naive", "seasonal_naive", "moving_average_7", "moving_average_28",
                 "croston", "sba", "zero"}
    rows = []
    for seg, g in seg_scores.groupby(by, observed=True):
        best = g.loc[g[metric].idxmin()]
        base = g[g["model"].isin(baselines)]
        best_base = base.loc[base[metric].idxmin()] if not base.empty else None
        rows.append({
            by: seg,
            "n": int(best["n"]),
            "best_model": best["model"],
            f"best_{metric}": float(best[metric]),
            "best_baseline": None if best_base is None else best_base["model"],
            f"baseline_{metric}": None if best_base is None else float(best_base[metric]),
            "improvement_pct": None if best_base is None else float(
                100.0 * (best_base[metric] - best[metric]) / best_base[metric]),
            "recommended_approach": SEGMENT_ADVICE.get(str(seg), ""),
        })
    return pd.DataFrame(rows)


def classification_note() -> str:
    return (f"Syntetos-Boylan cut points: ADI={ADI_CUT} (periods per non-zero "
            f"period) and CV²={CV2_CUT} (squared coefficient of variation of "
            "non-zero demand sizes). These are the standard literature values and "
            "are not tuned here.")
