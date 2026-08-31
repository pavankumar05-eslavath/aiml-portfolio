"""Fetch the Telco Customer Churn dataset.

    python -m src.data.download

The repository gitignores ``*.csv``, so the dataset is fetched rather than
committed. Every figure in reports/model_report.md comes from this file.

Source: IBM sample dataset, mirrored in
https://github.com/IBM/telco-customer-churn-on-icp4d
Also published on Kaggle as ``blastchar/telco-customer-churn``.
"""
from __future__ import annotations

import sys
import urllib.request

from src.config import load_config
from src.data.schema import EXPECTED_COLUMNS


def main() -> int:
    cfg = load_config()
    dest = cfg.resolve("data", "raw_file")
    url = str(cfg["data"]["source_url"])
    expected = int(cfg["data"]["expected_rows"])

    if dest.exists():
        print(f"{dest.name} already present ({dest.stat().st_size / 1e3:.0f} KB); skipping.")
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            payload = r.read()
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        print("Fall back to `make sample` for the synthetic stand-in.", file=sys.stderr)
        return 1

    text = payload.decode("utf-8-sig")
    header = tuple(text.splitlines()[0].strip().split(","))

    # Compare as SETS, not as an ordered tuple. EXPECTED_COLUMNS is grouped by
    # type (identifier, categoricals, numerics, target) for readability, while the
    # published file interleaves them -- SeniorCitizen and tenure sit among the
    # categoricals. Physical order is irrelevant because load_raw() selects columns
    # by name, so an ordered comparison rejects the correct file.
    missing = sorted(set(EXPECTED_COLUMNS) - set(header))
    extra = sorted(set(header) - set(EXPECTED_COLUMNS))
    if missing or extra:
        print("unexpected header; refusing to write.", file=sys.stderr)
        if missing:
            print(f"  missing columns: {missing}", file=sys.stderr)
        if extra:
            print(f"  unexpected columns: {extra}", file=sys.stderr)
        return 1

    dest.write_bytes(payload)
    rows = len(text.strip().splitlines()) - 1
    print(f"wrote {dest} ({dest.stat().st_size / 1e3:.0f} KB, {rows:,} rows)")
    if rows != expected:
        print(f"WARNING: expected {expected:,} rows, got {rows:,}; "
              "reported figures assume the former.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
