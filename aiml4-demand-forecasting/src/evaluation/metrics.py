"""Forecast accuracy metrics.

## Why MAPE is not reported

MAPE divides by the actual. With ~59% of observations equal to zero, MAPE is
**undefined** for most rows, and on the rows where the actual is 1 a forecast of 2
scores a 100% error while a forecast of 0 scores 100% too -- so it cannot
distinguish over- from under-forecasting on sparse data. Dropping the zero rows to
"fix" it changes the question being asked: it evaluates the model only on days
demand happened to occur, which is not the operational problem.

sMAPE is reported because it was requested, but it has a related pathology on
intermittent data: when actual and forecast are both near zero its denominator
collapses and the metric saturates at 200%. It is included with that caveat, not
as a headline.

## What is reported

* **MAE** -- units of demand, directly interpretable, but not comparable across
  series of different scale.
* **RMSE** -- penalises the large misses that drive stockouts; read next to MAE to
  see how much error is concentrated in spikes.
* **WAPE** -- total absolute error divided by total actual demand. Well defined as
  long as *aggregate* demand is non-zero, which makes it the practical headline for
  sparse panels. Also called weighted MAPE.
* **MASE** -- MAE divided by the in-sample seasonal-naive MAE. Scale-free, so it
  aggregates honestly across series, and 1.0 has a meaning: no better than
  seasonal naive.
* **Bias** -- mean signed error. A forecast can have good MAE and still be
  systematically low, which matters more than accuracy for inventory: persistent
  under-forecasting guarantees stockouts no safety stock was sized for.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

EPS = 1e-9


@dataclass(frozen=True)
class Scores:
    """Accuracy scores for one model on one evaluation set."""

    n: int
    mae: float
    rmse: float
    wape: float
    mase: float
    smape: float
    bias: float
    bias_pct: float
    zero_share_actual: float
    pred_zero_share: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype="float64").ravel()


def mae(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.mean(np.abs(_arr(y) - _arr(f))))


def rmse(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.sqrt(np.mean((_arr(y) - _arr(f)) ** 2)))


def wape(y: np.ndarray, f: np.ndarray) -> float:
    """Total absolute error / total actual. Undefined only if all actuals are zero."""
    y, f = _arr(y), _arr(f)
    denom = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - f)) / denom) if denom > EPS else float("nan")


def smape(y: np.ndarray, f: np.ndarray) -> float:
    """Symmetric MAPE in percent, with the zero-zero case defined as 0 error.

    Without that convention a day where both actual and forecast are zero -- a
    correct prediction -- would be 0/0. Reported with the caveat in the module
    docstring.
    """
    y, f = _arr(y), _arr(f)
    denom = (np.abs(y) + np.abs(f)) / 2.0
    ratio = np.where(denom < EPS, 0.0, np.abs(y - f) / np.maximum(denom, EPS))
    return float(100.0 * np.mean(ratio))


def mase(
    y: np.ndarray, f: np.ndarray, series: pd.Series, denominator: pd.Series,
) -> float:
    """MAE scaled by each series' in-sample seasonal-naive MAE."""
    y, f = _arr(y), _arr(f)
    d = denominator.reindex(pd.Index(series)).to_numpy(dtype="float64")
    ok = np.isfinite(d) & (d > EPS)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs(y[ok] - f[ok]) / d[ok]))


def score(
    y_true, y_pred, series: pd.Series | None = None,
    mase_denominator: pd.Series | None = None,
) -> Scores:
    """Compute the full metric set."""
    y, f = _arr(y_true), _arr(y_pred)
    bias = float(np.mean(f - y))
    total = float(np.sum(y))
    m = (float("nan") if (series is None or mase_denominator is None)
         else mase(y, f, series, mase_denominator))
    return Scores(
        n=int(y.size),
        mae=mae(y, f),
        rmse=rmse(y, f),
        wape=wape(y, f),
        mase=m,
        smape=smape(y, f),
        bias=bias,
        bias_pct=float(bias * y.size / total) if total > EPS else float("nan"),
        zero_share_actual=float(np.mean(y == 0)),
        pred_zero_share=float(np.mean(f < 0.5)),
    )


def score_frame(
    df: pd.DataFrame, pred_col: str, actual_col: str = "y_true",
    by: list[str] | None = None, series_col: str = "series_id",
    mase_denom: pd.Series | None = None,
) -> pd.DataFrame:
    """Score a predictions frame, optionally grouped.

    Grouped scoring is the point of a panel evaluation: a single aggregate number
    hides that a model can be strong on the handful of high-volume series that
    dominate WAPE while being useless on the long tail.
    """
    def one(g: pd.DataFrame) -> pd.Series:
        s = score(g[actual_col], g[pred_col], g[series_col], mase_denom)
        return pd.Series(s.as_dict())

    if not by:
        return one(df).to_frame().T
    return (df.groupby(by, observed=True)[df.columns.tolist()]
            .apply(one).reset_index())


def compare(scores: dict[str, Scores]) -> pd.DataFrame:
    """Model comparison table sorted by WAPE."""
    out = pd.DataFrame({k: v.as_dict() for k, v in scores.items()}).T
    out.index.name = "model"
    return out.sort_values("wape").reset_index()


def skill_vs(table: pd.DataFrame, reference: str, metric: str = "wape") -> pd.DataFrame:
    """Percentage improvement over a reference model.

    Reported because an absolute WAPE is hard to judge, while "12% better than
    seasonal naive" is the claim a reviewer can actually check.
    """
    ref = float(table.loc[table["model"] == reference, metric].iloc[0])
    out = table.copy()
    out[f"skill_vs_{reference}_pct"] = 100.0 * (ref - out[metric]) / ref
    return out
