"""
Stage runner.

    python -m src.run profile     data profile and claim checks
    python -m src.run article     the published pipeline, reproduced
    python -m src.run diagnose    why its XGBoost number no longer reproduces
    python -m src.run improved    corrected pipeline
    python -m src.run audit       the rule that beats the models
    python -m src.run all         every stage in order

Uses data/onlinefraud.csv (the real 480 MB PaySim CSV) when present, otherwise
the generated stand-in. Override with --data.
"""
from __future__ import annotations

import argparse

from src import article_pipeline, improved_pipeline, leakage_audit, profile_data, xgb_diagnosis
from src.dataset import is_real, resolve_path

STAGES = {
    "profile": lambda a: profile_data.run(a.data),
    "article": lambda a: article_pipeline.run(a.data),
    "diagnose": lambda a: xgb_diagnosis.run(a.data),
    "improved": lambda a: improved_pipeline.run(a.data, save_figures=a.save_figures),
    "audit": lambda a: leakage_audit.run(a.data),
}
ORDER = ["profile", "article", "diagnose", "improved", "audit"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=[*ORDER, "all"])
    ap.add_argument("--data", default=None, help="path to a PaySim-shaped CSV")
    ap.add_argument("--save-figures", action="store_true")
    args = ap.parse_args()

    src = resolve_path(args.data)
    if not is_real(args.data):
        print(
            "NOTE: running on the generated stand-in, not the real PaySim CSV.\n"
            "      Figures in INSIGHTS.md come from the real 6,362,620-row file.\n"
            "      Run `make download` to reproduce them exactly.\n"
        )

    stages = ORDER if args.stage == "all" else [args.stage]
    for name in stages:
        print(f"\n{'=' * 76}\n{name.upper()}   ({src})\n{'=' * 76}")
        STAGES[name](args)


if __name__ == "__main__":
    main()
