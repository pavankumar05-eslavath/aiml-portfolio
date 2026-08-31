"""Schema of the M5 source files and of the modelling panel.

Declared in code rather than inferred, so a change upstream fails loudly instead
of silently altering the model's input space.

Source: MOFC / Kaggle "M5 Forecasting - Accuracy", accessed through a public
long-format mirror. Three files are joined into one panel:

    target   : series_id x date -> demand (units sold)
    static   : series_id -> sku_id, dept_id, cat_id, store_id, state_id
    temporal : series_id x date -> sell_price, snap_CA, snap_TX, snap_WI
"""
from __future__ import annotations

from dataclasses import dataclass

# ---- source columns, as published in the mirror ---------------------------
SRC_SERIES = "item_id"       # series key in the source == SKU x store
SRC_DATE = "timestamp"
SRC_DEMAND = "demand"

TARGET_COLUMNS: tuple[str, ...] = (SRC_SERIES, SRC_DATE, SRC_DEMAND)
STATIC_COLUMNS: tuple[str, ...] = (
    SRC_SERIES, "sku_id", "dept_id", "cat_id", "store_id", "state_id",
)
TEMPORAL_COLUMNS: tuple[str, ...] = (
    SRC_SERIES, SRC_DATE, "snap_CA", "snap_TX", "snap_WI", "sell_price",
)

# ---- panel columns, after the join and rename -----------------------------
SERIES_ID = "series_id"
DATE = "date"
TARGET = "demand"

#: Identity / grain columns.
KEY_COLUMNS: tuple[str, ...] = (SERIES_ID, "sku_id", "store_id", DATE)

#: Series-level attributes, constant through time.
STATIC_FEATURES: tuple[str, ...] = ("sku_id", "dept_id", "cat_id", "store_id", "state_id")

#: Time-varying exogenous inputs that a retailer knows in advance.
KNOWN_FUTURE_FEATURES: tuple[str, ...] = ("sell_price", "snap")

PANEL_COLUMNS: tuple[str, ...] = (
    SERIES_ID, DATE, TARGET, "sku_id", "dept_id", "cat_id", "store_id", "state_id",
    "sell_price", "snap",
)

#: Physically impossible values. Unit sales cannot be negative; price must be > 0.
VALID_RANGES: dict[str, tuple[float, float]] = {
    TARGET: (0, 100_000),
    "sell_price": (0.001, 10_000),
    "snap": (0, 1),
}


@dataclass(frozen=True)
class ColumnSpec:
    """One row of the data dictionary."""

    name: str
    kind: str
    source: str
    description: str
    notes: str = ""


DATA_DICTIONARY: tuple[ColumnSpec, ...] = (
    ColumnSpec(SERIES_ID, "key", "target/static/temporal",
               "SKU x store series identifier.",
               "The forecasting grain. 3,049 SKUs x 10 stores in the full data."),
    ColumnSpec(DATE, "key", "target",
               "Calendar day.", "Daily frequency, contiguous within each series."),
    ColumnSpec(TARGET, "target", "target",
               "Units sold that day.",
               "Integer, non-negative. ~60% of observations are zero, which is why "
               "MAPE is unusable and MASE/WAPE are reported instead."),
    ColumnSpec("sku_id", "categorical", "static",
               "Product identifier, shared across stores."),
    ColumnSpec("dept_id", "categorical", "static", "Department (7 values)."),
    ColumnSpec("cat_id", "categorical", "static",
               "Category: FOODS / HOUSEHOLD / HOBBIES."),
    ColumnSpec("store_id", "categorical", "static", "Store identifier."),
    ColumnSpec("state_id", "categorical", "static",
               "State: CA / TX / WI.", "Determines which SNAP calendar applies."),
    ColumnSpec("sell_price", "numeric", "temporal",
               "Unit selling price for that series that week.",
               "No missing values in the mirror; the mirror trims each series to "
               "begin when the item was first priced in that store."),
    ColumnSpec("snap", "binary", "temporal",
               "1 if that date is a SNAP benefit day in the series' state.",
               "The dataset's only genuine promotion-like signal; active on ~33% "
               "of days. Named holiday events are NOT in this mirror, so holiday "
               "flags are derived from pandas' US federal calendar instead."),
)


def panel_feature_columns() -> list[str]:
    """Columns of the panel that may feed features (excludes target and keys)."""
    return [*STATIC_FEATURES, "sell_price", "snap"]
