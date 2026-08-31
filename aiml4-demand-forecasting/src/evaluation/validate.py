"""Rolling-origin (walk-forward) validation.

## Why not a random split

A random train/test split over a panel puts day *t+1* in training and day *t* in
test for the same series. The model then interpolates between days it has already
seen and reports an accuracy no deployed forecaster can reach, because at forecast
time the future does not exist. The error is not subtle: it typically halves
reported WAPE.

## What happens instead

For each fold, the model is **refitted** using only data at or before that fold's
origin, then asked for the next 28 days. Folds are consecutive and
non-overlapping, and every validation window sits entirely before the test window.

Refitting per fold is the expensive but correct choice: reusing one model fitted on
all training data would let fold 1 be scored by a model that had seen fold 4, which
is the same leak in a different costume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.config import Config
from src.data.loader import Fold, rolling_origin_folds, test_fold, train_slice
from src.data.schema import DATE, SERIES_ID, TARGET
from src.evaluation.metrics import Scores, score
from src.forecasting import baselines as B
from src.forecasting import ml as ML


@dataclass
class FoldResult:
    """Predictions and scores for one fold."""

    fold: str
    origin: pd.Timestamp
    predictions: pd.DataFrame
    scores: dict[str, Scores] = field(default_factory=dict)
    n_train_rows: int = 0
    fit_seconds: float = 0.0
    #: Out-of-sample predictions at many origins inside this fold's window, used to
    #: estimate the lead-time forecast-error distribution that sizes safety stock.
    error_windows: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


def _mase_denominator(cfg: Config, history: pd.DataFrame) -> pd.Series:
    return B.mase_denominator(history, SERIES_ID, DATE, TARGET, season=7)


def evaluate_fold(
    cfg: Config, panel: pd.DataFrame, fold: Fold, *,
    fit_ml: bool = True, n_origins: int = 60, quantiles: bool = False,
    models: list[str] | None = None, error_origin_step: int = 1,
    protection_days: int | None = None,
) -> FoldResult:
    """Fit on data up to ``fold.origin`` and score the next ``fold.horizon`` days."""
    import time

    from src.features.build import build_supervised, training_origins

    history = train_slice(cfg, panel, fold)
    denom = _mase_denominator(cfg, history)

    eval_df = build_supervised(cfg, panel, fold)
    preds = eval_df[[SERIES_ID, "origin", "target_date", "horizon", "y_true"]].copy()

    for name in B.all_baselines():
        preds[name] = B.predict(name, eval_df)

    n_train = 0
    elapsed = 0.0
    fitted_ml = None
    if fit_ml:
        origins = training_origins(cfg, panel, fold.origin, n_origins=n_origins)
        train_df = build_supervised(cfg, panel, fold, origins=origins)
        # Guard: nothing at or after the evaluation window may be in training.
        if not train_df.empty:
            assert train_df["target_date"].max() <= fold.origin, (
                "training targets must not reach the evaluation window")
        n_train = len(train_df)

        wanted = models or ["lightgbm", "random_forest"]
        t0 = time.perf_counter()
        if "lightgbm" in wanted:
            m = ML.fit_lightgbm(cfg, train_df)
            fitted_ml = m
            preds["lightgbm"] = m.predict(eval_df)
        if "random_forest" in wanted:
            rf = ML.fit_random_forest(cfg, train_df)
            preds["random_forest"] = rf.predict(eval_df)
        if quantiles:
            qmodels = ML.fit_quantiles(cfg, train_df)
            qp = ML.enforce_monotone_quantiles(
                {q: mm.predict(eval_df) for q, mm in qmodels.items()})
            for q, v in qp.items():
                preds[f"q{int(q * 100)}"] = v
        elapsed = time.perf_counter() - t0

    model_cols = [c for c in preds.columns
                  if c not in (SERIES_ID, "origin", "target_date", "horizon", "y_true")
                  and not c.startswith("q")]
    scores = {c: score(preds["y_true"], preds[c], preds[SERIES_ID], denom)
              for c in model_cols}

    # ---- extra origins for estimating the lead-time error distribution ----
    # Safety stock needs the SPREAD of lead-time forecast error, and one window per
    # fold gives four observations per series -- far too few for a quantile or even a
    # standard deviation. Extra origins are stepped through this fold's own window,
    # reusing the model fitted before the fold started, so every prediction stays
    # out-of-sample. Features at each origin use only actuals up to that origin,
    # which is exactly how a deployed system behaves between refits.
    error_windows = pd.DataFrame()
    if fit_ml and error_origin_step and protection_days:
        extra: list[pd.Timestamp] = []
        o = fold.origin
        while o + pd.Timedelta(days=protection_days) <= fold.end:
            extra.append(o)
            o = o + pd.Timedelta(days=error_origin_step)
        if extra:
            ew = build_supervised(cfg, panel, fold,
                                  horizons=list(range(1, protection_days + 1)),
                                  origins=extra)
            if not ew.empty:
                cols = [SERIES_ID, "origin", "target_date", "horizon", "y_true"]
                error_windows = ew[cols].copy()
                for name in ("moving_average_28", "seasonal_naive"):
                    error_windows[name] = B.predict(name, ew)
                if fitted_ml is not None:
                    error_windows["lightgbm"] = fitted_ml.predict(ew)

    return FoldResult(fold=fold.name, origin=fold.origin, predictions=preds,
                      scores=scores, n_train_rows=n_train, fit_seconds=elapsed,
                      error_windows=error_windows)


def walk_forward(
    cfg: Config, panel: pd.DataFrame, *, fit_ml: bool = True,
    n_origins: int = 60, include_test: bool = False, quantiles: bool = False,
    verbose: bool = True, protection_days: int | None = None,
) -> list[FoldResult]:
    """Run every validation fold, optionally followed by the held-out test fold."""
    folds = rolling_origin_folds(cfg, panel)
    if include_test:
        folds = [*folds, test_fold(cfg)]

    results: list[FoldResult] = []
    for f in folds:
        r = evaluate_fold(cfg, panel, f, fit_ml=fit_ml, n_origins=n_origins,
                          quantiles=quantiles and f.name == "test",
                          protection_days=protection_days)
        results.append(r)
        if verbose:
            best = min(r.scores.items(), key=lambda kv: kv[1].wape)
            print(f"  {f.name:8s} origin {f.origin.date()} "
                  f"predict {f.start.date()}..{f.end.date()} | "
                  f"train rows {r.n_train_rows:,} | best {best[0]} "
                  f"WAPE {best[1].wape:.4f} | {r.fit_seconds:.0f}s")
    return results


def aggregate_scores(results: list[FoldResult]) -> pd.DataFrame:
    """Mean and spread of each model's scores across folds."""
    rows: list[dict[str, Any]] = []
    for r in results:
        for model, s in r.scores.items():
            rows.append({"fold": r.fold, "model": model, **s.as_dict()})
    df = pd.DataFrame(rows)
    agg = df.groupby("model", observed=True).agg(
        folds=("fold", "nunique"),
        mae=("mae", "mean"), rmse=("rmse", "mean"),
        wape=("wape", "mean"), wape_std=("wape", "std"),
        mase=("mase", "mean"), smape=("smape", "mean"),
        bias=("bias", "mean"),
    ).sort_values("wape").reset_index()
    return agg


def per_fold_table(results: list[FoldResult], metric: str = "wape") -> pd.DataFrame:
    """Fold-by-fold values, so a single lucky fold cannot masquerade as a win."""
    rows = {r.fold: {m: getattr(s, metric) for m, s in r.scores.items()}
            for r in results}
    return pd.DataFrame(rows)


def combine_predictions(results: list[FoldResult]) -> pd.DataFrame:
    """Stack predictions from every fold, tagged by fold."""
    frames = []
    for r in results:
        d = r.predictions.copy()
        d.insert(0, "fold", r.fold)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def horizon_breakdown(
    predictions: pd.DataFrame, model_cols: list[str], cfg: Config,
    denom: pd.Series | None = None,
) -> pd.DataFrame:
    """Accuracy by forecast horizon.

    Error should grow with horizon. If it does not, the features are probably
    leaking: a model whose day-28 accuracy matches its day-1 accuracy has
    information it should not have.
    """
    rows = []
    for h in cfg.horizons:
        sub = predictions[predictions["horizon"] == h]
        if sub.empty:
            continue
        for m in model_cols:
            s = score(sub["y_true"], sub[m], sub[SERIES_ID], denom)
            rows.append({"horizon": h, "model": m, **s.as_dict()})
    return pd.DataFrame(rows)
