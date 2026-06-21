"""CLI entry point: run the outlier pipeline for one or all conditions.

Usage:
    python -m analysis.outlier_pipeline.run                  # all conditions
    python -m analysis.outlier_pipeline.run --condition aigen_perc
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config  # noqa: E402

from .report import run_condition  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run the outlier pipeline.")
    parser.add_argument("--condition", help="Single condition slug, e.g. aigen_perc")
    args = parser.parse_args()

    conditions = [args.condition] if args.condition else config.CONDITIONS

    summaries = []
    for condition in conditions:
        summary_df, participants_df = run_condition(condition)
        summaries.append(summary_df)
        print(f"[{condition}] full={summary_df['full'][0]} "
              f"partial={summary_df['partial'][0]} unusable={summary_df['unusable'][0]} "
              f"total={summary_df['total'][0]}")

    if len(summaries) > 1:
        combined = pd.concat(summaries, ignore_index=True)
        config.COMBINED_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(config.COMBINED_ANALYSIS_DIR / "outlier_report_summary.csv", index=False)


if __name__ == "__main__":
    main()
