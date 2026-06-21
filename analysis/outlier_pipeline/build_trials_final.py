"""Build trials_final.csv per condition: apply short-answer and (for del conditions)
digit-span performance exclusions on top of all_trials.csv, and write exclusion /
digit-span performance reports.
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config  # noqa: E402

from . import digit_span_metrics  # noqa: E402
from .exclusions import MIN_USABLE_SESSIONS, apply_exclusions, participant_table, session_table  # noqa: E402


def _write_digit_span_reports(digit_span_df: pd.DataFrame, outliers_dir: Path) -> pd.DataFrame:
    """Write digit-span performance reports, return the session pass/fail table."""
    ds_dir = outliers_dir / "digit_span"
    ds_dir.mkdir(parents=True, exist_ok=True)

    digit_span_metrics.participant_summary(digit_span_df).to_csv(
        ds_dir / "digit_span_performance.csv", index=False
    )
    digit_span_metrics.try_counts(digit_span_df).to_csv(
        ds_dir / "digit_span_try_counts.csv", index=False
    )
    digit_span_metrics.plot_accuracy_vs_length(
        digit_span_df, ds_dir / "digit_span_accuracy_by_length.png"
    )
    return digit_span_metrics.session_pass_fail(digit_span_df)


def run_condition(condition: str) -> pd.DataFrame:
    """Apply exclusions for one condition slug, write trials_final.csv + reports.

    Returns the filtered trials_final DataFrame.
    """
    paths = config.paths_for(condition)
    is_del = config.mapping_data["CONDITIONS"][condition]["task"] == "delay"

    trials_path = paths.processed_dir / "all_trials.csv"
    if not trials_path.exists():
        print(f"skip {condition}: {trials_path} not found (run aggregate.py first)")
        return pd.DataFrame()
    trials_df = pd.read_csv(trials_path)

    outliers_dir = paths.analysis_dir / "outliers"
    outliers_dir.mkdir(parents=True, exist_ok=True)

    digit_span_pass_fail = None
    if is_del:
        digit_span_path = paths.processed_dir / "all_digit_span.csv"
        if digit_span_path.exists():
            digit_span_df = pd.read_csv(digit_span_path)
            digit_span_pass_fail = _write_digit_span_reports(digit_span_df, outliers_dir)

    sessions = session_table(trials_df, digit_span_pass_fail)
    participants = participant_table(sessions, has_digit_span=digit_span_pass_fail is not None)
    trials_final = apply_exclusions(trials_df, sessions, participants)

    sessions.to_csv(outliers_dir / "exclusion_report_sessions.csv", index=False)
    participants.to_csv(outliers_dir / "exclusion_report_participants.csv", index=False)
    trials_final.to_csv(paths.csv("trials_final"), index=False)

    _print_summary(condition, trials_df, trials_final, sessions, participants)

    return trials_final


def _print_summary(
    condition: str,
    all_trials: pd.DataFrame,
    trials_final: pd.DataFrame,
    sessions: pd.DataFrame,
    participants: pd.DataFrame,
) -> None:
    n_participants = participants["uid"].nunique()
    n_excluded = int(participants["excluded"].sum())
    n_sessions = len(sessions)
    n_usable_sessions = int(sessions["usable"].sum())

    print(f"[{condition}] all_trials: {len(all_trials)} rows, {n_participants} participants "
          f"-> trials_final: {len(trials_final)} rows, {trials_final['uid'].nunique()} participants")
    print(f"[{condition}] participants excluded entirely: {n_excluded}/{n_participants} "
          f"(usable_sessions < {MIN_USABLE_SESSIONS})")
    print(f"[{condition}] sessions usable: {n_usable_sessions}/{n_sessions} "
          f"(full={int(sessions['is_full_session'].sum())}, "
          f"short_answer_dropped={int(sessions['is_short_session'].sum())}, "
          f"digitspan_dropped={int(sessions['is_digitspan_failed'].sum())})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build trials_final.csv with exclusions applied.")
    parser.add_argument("--condition", help="Single condition slug, e.g. aigen_perc")
    args = parser.parse_args()

    conditions = [args.condition] if args.condition else config.CONDITIONS
    for c in conditions:
        run_condition(c)
