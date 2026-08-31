"""
Stage runner.

    python -m src.run profile     profile the file and locate its defects
    python -m src.run naive       the baseline: the obvious, uncleaned analysis
    python -m src.run audited     the corrected analysis, next to the baseline
    python -m src.run coverage    validate against published industry totals
    python -m src.run export      write the cleaned CSVs
    python -m src.run all         every stage in order

Uses data/startup_funding.csv (the real 3,044-row file) when present, otherwise
the generated stand-in. Override with --data.
"""
from __future__ import annotations

import argparse

from src import audited_pipeline, coverage_check, naive_pipeline, profile_data
from src.dataset import is_real, load_clean, resolve_path

OUT_ROUNDS = "data/funding_clean.csv"
OUT_INVESTORS = "data/investor_deals.csv"


def _export(path: str | None = None) -> dict:
    rounds, inv = load_clean(path)
    rounds.to_csv(OUT_ROUNDS, index=False)
    inv.to_csv(OUT_INVESTORS, index=False)
    print(f"wrote {OUT_ROUNDS}      ({len(rounds):,} rounds)")
    print(f"wrote {OUT_INVESTORS}   ({len(inv):,} investor-round pairs)")
    print("\nUse amount_usd_adj, not amount_usd: the latter is as-published and")
    print("contains an INR figure in a column headed USD. amount_flag marks it.")
    return {"rounds": len(rounds), "investor_rows": len(inv)}


STAGES = {
    "profile": lambda a: profile_data.run(a.data),
    "naive": lambda a: naive_pipeline.run(a.data),
    "audited": lambda a: audited_pipeline.run(a.data, save_figures=a.save_figures),
    "coverage": lambda a: coverage_check.run(a.data),
    "export": lambda a: _export(a.data),
}
ORDER = ["profile", "naive", "audited", "coverage", "export"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=[*ORDER, "all"])
    ap.add_argument("--data", default=None, help="path to a CSV with the published schema")
    ap.add_argument("--save-figures", action="store_true")
    args = ap.parse_args()

    src = resolve_path(args.data)
    if not is_real(args.data):
        print(
            "NOTE: running on the generated stand-in, not the real Kaggle CSV.\n"
            "      It reproduces the DEFECTS so the suite exercises them, but the\n"
            "      figures in INSIGHTS.md come from the real 3,044-row file.\n"
            "      Run `make download` to reproduce them exactly.\n"
        )

    for name in (ORDER if args.stage == "all" else [args.stage]):
        print(f"\n{'=' * 76}\n{name.upper()}   ({src})\n{'=' * 76}")
        STAGES[name](args)


if __name__ == "__main__":
    main()
