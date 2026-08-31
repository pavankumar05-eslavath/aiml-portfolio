"""
Fetch the real PaySim CSV (~480 MB) from the Drive link published with the
GeeksforGeeks article. Every figure in INSIGHTS.md comes from this file.

    python -m data.download

Needs `gdown`, because Drive serves a virus-scan interstitial for files this
size rather than the file itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

FILE_ID = "133E0TDrfIjnhwRoGTw9OEozwBXUL38D8"
DEST = Path("data/onlinefraud.csv")
EXPECTED_ROWS = 6_362_620


def main() -> int:
    if DEST.exists():
        print(f"{DEST} already present ({DEST.stat().st_size / 1e6:.0f} MB); skipping.")
        return 0
    try:
        import gdown
    except ImportError:
        print("gdown is required: pip install gdown", file=sys.stderr)
        return 1

    DEST.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=FILE_ID, output=str(DEST), quiet=False)

    if not DEST.exists():
        print("download failed", file=sys.stderr)
        return 1
    print(f"\nwrote {DEST} ({DEST.stat().st_size / 1e6:.0f} MB)")
    print(f"expected {EXPECTED_ROWS:,} data rows; verify with: wc -l {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
