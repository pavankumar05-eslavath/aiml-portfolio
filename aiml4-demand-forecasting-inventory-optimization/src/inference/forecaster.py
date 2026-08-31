"""Inference pipeline.

Given a series' history, returns the forecast **and the decision it implies**:

    horizon, point forecast, P10/P50/P90, safety stock, reorder point, stockout risk

Feature engineering lives inside this path rather than being reimplemented, which
is what stops a served forecast from drifting away from the evaluated one.

Usage:

    from src.inference.forecaster import DemandForecaster
    f = DemandForecaster.load()
    out = f.forecast(panel, series_id="FOODS_3_090_CA_1")
    rec = f.recommend(panel, series_id="FOODS_3_090_CA_1")
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.data.loader import Fold
from src.data.schema import DATE, SERIES_ID
from src.forecasting.persist import load_bundle
from src.inventory.policy import PolicyParams, z_for_service_level


@dataclass
class Recommendation:
    """A forecast plus the inventory decision derived from it."""

    series_id: str
    origin: pd.Timestamp
    horizon_days: int
    point_forecast: np.ndarray
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray
    expected_lt_demand: float
    safety_stock: float
    reorder_point: float
    stockout_risk: float
    service_level: float
    lead_time_days: int
    review_period_days: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "series_id": self.series_id,
            "horizon": np.arange(1, len(self.point_forecast) + 1),
            "date": pd.date_range(self.origin + pd.Timedelta(days=1),
                                  periods=len(self.point_forecast), freq="D"),
            "forecast": self.point_forecast,
            "p10": self.p10, "p50": self.p50, "p90": self.p90,
        })

    def as_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "origin": str(self.origin.date()),
            "horizon_days": self.horizon_days,
            "point_forecast_total": float(np.sum(self.point_forecast)),
            "expected_lead_time_demand": round(self.expected_lt_demand, 2),
            "safety_stock": round(self.safety_stock, 2),
            "reorder_point": round(self.reorder_point, 2),
            "stockout_risk": round(self.stockout_risk, 4),
            "service_level": self.service_level,
            "lead_time_days": self.lead_time_days,
            "review_period_days": self.review_period_days,
            "caveat": "Inventory figures use the assumed economics in "
                      "configs/config.yaml; they are not measured business results.",
        }


class DemandForecaster:
    """Loads a persisted bundle and produces forecasts and inventory decisions."""

    def __init__(self, bundle: dict[str, Any], cfg: Config) -> None:
        self.bundle = bundle
        self.cfg = cfg
        self.point_model = bundle["point_model"]
        self.quantile_models: dict[float, Any] = bundle.get("quantile_models") or {}
        self.metadata: dict[str, Any] = bundle["metadata"]
        self.policy: pd.DataFrame | None = bundle.get("policy")

    @classmethod
    def load(cls, cfg: Config | None = None, path: str | Path | None = None) -> DemandForecaster:
        cfg = cfg or load_config()
        return cls(load_bundle(cfg, path), cfg)

    # -- core --------------------------------------------------------------
    def _features(self, panel: pd.DataFrame, origin: pd.Timestamp, horizon: int,
                  series: list[str] | None = None) -> pd.DataFrame:
        from src.features.build import build_supervised

        p = panel if series is None else panel[panel[SERIES_ID].isin(series)]
        fold = Fold(name="inference", origin=origin,
                    start=origin + pd.Timedelta(days=1),
                    end=origin + pd.Timedelta(days=horizon))

        # Forecasting past the end of the panel means no actuals and no published
        # price/SNAP for the target dates. Rows are kept (require_actuals=False) and
        # the exogenous values are carried forward from the last observed day.
        # Carrying forward is an assumption and a weaker one than the training-time
        # setup, where price and promo really were known: a live system would read
        # them from the actual price and promo calendar instead.
        needs_future = origin >= p[DATE].max()
        if needs_future:
            last = p.sort_values(DATE).groupby(SERIES_ID, observed=True).tail(1)
            future_dates = pd.date_range(origin + pd.Timedelta(days=1),
                                         periods=horizon, freq="D")
            pad = last.loc[last.index.repeat(horizon)].copy()
            pad[DATE] = np.tile(future_dates, len(last))
            pad["demand"] = np.nan
            p = pd.concat([p, pad], ignore_index=True)

        feats = build_supervised(self.cfg, p, fold,
                                 horizons=list(range(1, horizon + 1)),
                                 require_actuals=False)
        return feats

    def forecast(
        self, panel: pd.DataFrame, series_id: str | None = None,
        origin: pd.Timestamp | None = None, horizon: int | None = None,
    ) -> pd.DataFrame:
        """Point forecast and prediction interval per day."""
        horizon = int(horizon or max(self.cfg.horizons))
        origin = pd.Timestamp(origin) if origin is not None else panel[DATE].max()
        series = [series_id] if series_id else None

        feats = self._features(panel, origin, horizon, series)
        if feats.empty:
            raise ValueError("no features could be built; check the series and origin")

        out = feats[[SERIES_ID, "target_date", "horizon"]].copy()
        out["forecast"] = self.point_model.predict(feats)

        if self.quantile_models:
            from src.forecasting.ml import enforce_monotone_quantiles
            qp = enforce_monotone_quantiles(
                {q: m.predict(feats) for q, m in self.quantile_models.items()})
            for q, v in qp.items():
                out[f"p{int(q * 100)}"] = v
        return out.sort_values([SERIES_ID, "horizon"], ignore_index=True)

    def recommend(
        self, panel: pd.DataFrame, series_id: str,
        origin: pd.Timestamp | None = None, service_level: float | None = None,
    ) -> Recommendation:
        """Forecast one series and derive its inventory decision."""
        params = PolicyParams.from_config(self.cfg, service_level)
        origin = pd.Timestamp(origin) if origin is not None else panel[DATE].max()
        horizon = max(max(self.cfg.horizons), params.protection_days)

        fc = self.forecast(panel, series_id=series_id, origin=origin, horizon=horizon)
        lt = fc[fc["horizon"] <= params.protection_days]
        expected = float(lt["forecast"].sum())

        # Safety stock comes from the stored per-series lead-time error spread when
        # available. Falling back to the predictive interval keeps a new series
        # serviceable, but the fallback is wider and is flagged in the output.
        ss = np.nan
        if self.policy is not None and series_id in set(self.policy[SERIES_ID]):
            row = self.policy.set_index(SERIES_ID).loc[series_id]
            col = ("safety_stock_empirical" if "safety_stock_empirical" in row
                   else "safety_stock_normal")
            ss = float(row[col])
            if float(row.get("service_level", params.service_level)) != params.service_level:
                ss = float(row["error_std"]) * z_for_service_level(params.service_level)
        if not np.isfinite(ss):
            if "p90" in lt.columns:
                ss = float((lt["p90"] - lt["forecast"]).clip(lower=0).pow(2).sum() ** 0.5)
            else:
                ss = float(lt["forecast"].std() * np.sqrt(params.protection_days))

        rop = float(np.ceil(expected + ss))
        # Stockout risk: probability lead-time demand exceeds the reorder point,
        # under a normal approximation to the error distribution. Reported as an
        # approximation because that assumption is questionable for lumpy demand.
        sigma = ss / max(z_for_service_level(params.service_level), 1e-9)
        from scipy import stats
        risk = float(1.0 - stats.norm.cdf((rop - expected) / max(sigma, 1e-9)))

        cols = {c: fc[c].to_numpy() if c in fc.columns else np.full(len(fc), np.nan)
                for c in ("p10", "p50", "p90")}
        return Recommendation(
            series_id=series_id, origin=origin, horizon_days=len(fc),
            point_forecast=fc["forecast"].to_numpy(),
            p10=cols["p10"], p50=cols["p50"], p90=cols["p90"],
            expected_lt_demand=expected, safety_stock=ss, reorder_point=rop,
            stockout_risk=risk, service_level=params.service_level,
            lead_time_days=params.lead_time_days,
            review_period_days=params.review_period_days,
        )

    def recommend_all(
        self, panel: pd.DataFrame, origin: pd.Timestamp | None = None,
        service_level: float | None = None,
    ) -> pd.DataFrame:
        """Reorder recommendations for every series: the replenishment worklist."""
        rows = [self.recommend(panel, s, origin=origin,
                               service_level=service_level).as_dict()
                for s in sorted(panel[SERIES_ID].unique())]
        out = pd.DataFrame(rows)
        return out.sort_values("stockout_risk", ascending=False, ignore_index=True)


def format_recommendation(rec: Recommendation) -> str:
    """Render one recommendation for a human reader."""
    total = float(np.sum(rec.point_forecast))
    return "\n".join([
        "=" * 72,
        f"DEMAND FORECAST & REORDER RECOMMENDATION — {rec.series_id}",
        "=" * 72,
        f"  forecast origin       : {rec.origin.date()}",
        f"  horizon               : {rec.horizon_days} days",
        f"  total forecast demand : {total:,.1f} units",
        f"  lead time + review    : {rec.lead_time_days} + {rec.review_period_days} "
        f"= {rec.lead_time_days + rec.review_period_days} days of exposure",
        f"  expected LT demand    : {rec.expected_lt_demand:,.1f} units",
        f"  safety stock          : {rec.safety_stock:,.1f} units "
        f"(at {rec.service_level:.0%} service level)",
        f"  REORDER POINT         : {rec.reorder_point:,.0f} units",
        f"  stockout risk         : {rec.stockout_risk:.1%} (normal approximation)",
        "",
        "  note: inventory figures use the assumed economics in configs/config.yaml;",
        "        they are not measured business results.",
        "=" * 72,
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default=None, help="series_id to forecast")
    ap.add_argument("--service-level", type=float, default=None)
    ap.add_argument("--output", default=None, help="write the full worklist to CSV")
    args = ap.parse_args()

    from src.data.loader import load_panel

    cfg = load_config()
    panel = load_panel(cfg)
    f = DemandForecaster.load(cfg)

    if args.output:
        wl = f.recommend_all(panel, service_level=args.service_level)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        wl.to_csv(args.output, index=False)
        print(f"wrote {args.output} ({len(wl):,} series)")
        return 0

    series = args.series or sorted(panel[SERIES_ID].unique())[0]
    rec = f.recommend(panel, series, service_level=args.service_level)
    print(format_recommendation(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
