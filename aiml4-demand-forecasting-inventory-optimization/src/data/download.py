"""Fetch the M5 source files.

    python -m src.data.download

The repository gitignores data, so the dataset is fetched rather than committed.
Roughly 133 MB across four parquet files.

Source: MOFC / Kaggle "M5 Forecasting - Accuracy" (Walmart unit sales), accessed
through a public long-format mirror that requires no credentials:
https://m5-benchmarks.s3.amazonaws.com/data/
"""
from __future__ import annotations

import sys
import urllib.request

from src.config import Config, load_config
from src.data.schema import STATIC_COLUMNS, TARGET_COLUMNS, TEMPORAL_COLUMNS

EXPECTED_COLUMNS = {
    "train_target": TARGET_COLUMNS,
    "train_static": STATIC_COLUMNS,
    "train_temporal": TEMPORAL_COLUMNS,
    "test_target": TARGET_COLUMNS,
}


def fetch_one(cfg: Config, name: str) -> bool:
    """Download one source file, verifying its columns before keeping it."""
    import pandas as pd

    dest = cfg.raw_file(name)
    if dest.exists():
        print(f"  {dest.name:22s} present ({dest.stat().st_size / 1e6:.1f} MB); skipping")
        return True

    url = f"{cfg['data']['source_base']}/{cfg['data']['files'][name]}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"  fetching {name} <- {url}")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as exc:
        print(f"  download failed for {name}: {exc}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False

    # Verify the schema BEFORE moving into place, so a partial or changed upstream
    # file cannot masquerade as valid data on the next run.
    try:
        cols = tuple(pd.read_parquet(tmp).columns)
    except Exception as exc:
        print(f"  {name} is not readable parquet: {exc}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False

    expected = EXPECTED_COLUMNS[name]
    missing = sorted(set(expected) - set(cols))
    if missing:
        print(f"  {name} is missing columns {missing}; refusing to keep it",
              file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False

    tmp.rename(dest)
    print(f"  wrote {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def main() -> int:
    cfg = load_config()
    cfg.ensure_dirs()
    print("Fetching M5 source files (MOFC/Kaggle M5, public long-format mirror)")
    ok = all(fetch_one(cfg, name) for name in cfg["data"]["files"])
    if not ok:
        print("\nOne or more downloads failed. Fall back to the synthetic panel:\n"
              "  make sample", file=sys.stderr)
        return 1
    print("\nAll source files present. Build the modelling panel with:  make panel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
