"""
Fetch the real 'Indian Startup Funding' CSV (3,044 rows, ~416 KB).

    python -m data.download

Every figure in INSIGHTS.md comes from this file. It is CC0 (public domain), so
it is mirrored on GitHub and needs no Kaggle credentials -- the Kaggle download
endpoint requires an authenticated session, which CI does not have.

Canonical source: https://www.kaggle.com/datasets/sudalairajkumar/indian-startup-funding
Collected by:     https://trak.in
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/Aditya-Mankar/"
       "Indian-Startup-Funding/master/startup_funding.csv")
DEST = Path("data/startup_funding.csv")
EXPECTED_ROWS = 3044
# The published header, used to verify we fetched the dataset we think we did.
# 'InvestmentnType' and the doubled space in 'City  Location' are the source's
# own typos -- they are load-bearing identifiers, not mistakes to fix here.
EXPECTED_HEADER = (
    "Sr No,Date dd/mm/yyyy,Startup Name,Industry Vertical,SubVertical,"
    "City  Location,Investors Name,InvestmentnType,Amount in USD,Remarks"
)


def main() -> int:
    if DEST.exists():
        print(f"{DEST} already present ({DEST.stat().st_size / 1e3:.0f} KB); skipping.")
        return 0

    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {URL}")
    try:
        with urllib.request.urlopen(URL, timeout=60) as r:
            payload = r.read()
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        print("Fall back to `make data` for the generated stand-in.", file=sys.stderr)
        return 1

    text = payload.decode("utf-8-sig")
    header = text.splitlines()[0].strip()
    if header != EXPECTED_HEADER:
        print(f"unexpected header:\n  got      {header}\n  expected {EXPECTED_HEADER}",
              file=sys.stderr)
        return 1

    DEST.write_bytes(payload)
    rows = len(text.strip().splitlines()) - 1
    print(f"wrote {DEST} ({DEST.stat().st_size / 1e3:.0f} KB, {rows:,} data rows)")
    if rows != EXPECTED_ROWS:
        print(f"WARNING: expected {EXPECTED_ROWS:,} rows, got {rows:,}; "
              "INSIGHTS.md figures assume the former.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
