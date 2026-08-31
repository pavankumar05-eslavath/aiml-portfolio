"""Calendar features.

These are the only features that may legitimately be computed for a future date:
a calendar is knowable in advance. Everything derived from the target column must
come from data at or before the forecast origin.

Holiday flags come from pandas' US federal holiday calendar rather than from the
dataset. The public M5 mirror used here does not carry the competition's named
event columns, so holidays are **derived** and labelled as such rather than
presented as a dataset field.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

CALENDAR_FEATURES: tuple[str, ...] = (
    "dow", "day_of_month", "week_of_year", "month", "quarter", "year",
    "is_weekend", "is_month_start", "is_month_end",
    "is_holiday", "days_to_holiday",
    "dow_sin", "dow_cos", "doy_sin", "doy_cos",
)


def holiday_dates(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """US federal holidays in the range, with a margin for distance features."""
    cal = USFederalHolidayCalendar()
    return pd.DatetimeIndex(
        cal.holidays(start=start - pd.Timedelta(days=400),
                     end=end + pd.Timedelta(days=400)))


def calendar_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build calendar features for a set of dates.

    Cyclical encodings are included because day-of-week and day-of-year are
    circular: as integers, December (12) and January (1) look maximally distant,
    which is wrong for a tree only in the sense that it must spend splits
    rediscovering the wrap-around, and wrong for a linear model outright.
    """
    dates = pd.DatetimeIndex(dates)
    hol = holiday_dates(dates.min(), dates.max())

    out = pd.DataFrame({"date": dates})
    out["dow"] = dates.dayofweek.astype("int16")
    out["day_of_month"] = dates.day.astype("int16")
    out["week_of_year"] = dates.isocalendar().week.to_numpy().astype("int16")
    out["month"] = dates.month.astype("int16")
    out["quarter"] = dates.quarter.astype("int16")
    out["year"] = dates.year.astype("int16")
    out["is_weekend"] = (dates.dayofweek >= 5).astype("int8")
    out["is_month_start"] = dates.is_month_start.astype("int8")
    out["is_month_end"] = dates.is_month_end.astype("int8")

    out["is_holiday"] = dates.isin(hol).astype("int8")
    # Signed distance to the nearest holiday: pre-holiday build-up and
    # post-holiday slump are different effects, so the sign is kept.
    hol_i64 = hol.asi8
    d_i64 = dates.asi8
    pos = np.searchsorted(hol_i64, d_i64)
    prev_i = np.clip(pos - 1, 0, len(hol_i64) - 1)
    next_i = np.clip(pos, 0, len(hol_i64) - 1)
    days_prev = (d_i64 - hol_i64[prev_i]) / 86_400_000_000_000
    days_next = (hol_i64[next_i] - d_i64) / 86_400_000_000_000
    signed = np.where(days_next <= days_prev, days_next, -days_prev)
    out["days_to_holiday"] = np.clip(signed, -30, 30).astype("int16")

    doy = dates.dayofyear.to_numpy()
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7).astype("float32")
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7).astype("float32")
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    return out
