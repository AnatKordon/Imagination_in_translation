"""Per-condition outlier report: classify every participant folder, reconstruct
CSVs from data.txt where possible, and summarize completeness."""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config  # noqa: E402

from .structure_check import classify_participant  # noqa: E402
from .session_summary import full_sessions_for  # noqa: E402

EXPECTED_SESSIONS = 5


def _read_uid(files_dir: Path) -> str | None:
    for name in ("participants.csv", "trials.csv", "digit_span.csv"):
        p = files_dir / name
        if p.exists():
            df = pd.read_csv(p)
            if "uid" in df.columns and not df.empty:
                return str(df["uid"].iloc[0])
    return None


def run_condition(condition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify every participant folder for one condition slug (e.g. 'aigen_perc').

    Returns (summary_df, participants_df). Writes both to the condition's
    analysis_dir as outlier_report_summary.csv / outlier_report_participants.csv.
    """
    paths = config.paths_for(condition)
    spec = config.spec_for(condition)

    pdir = paths.participants_dir
    rows = []
    if pdir is not None and pdir.exists():
        for comp_result_dir in sorted(pdir.glob("study_result_*/comp-result_*")):
            result = classify_participant(comp_result_dir, spec.is_del, expect_images=spec.images)
            session_info = {"full_sessions": 0, "total_rows": 0, "max_session": 0}
            uid = None
            trials_csv = result["files_dir"] / "trials.csv"
            if trials_csv.exists():
                uid = _read_uid(result["files_dir"])
                per_uid = full_sessions_for(trials_csv, required_attempts=spec.attempts)
                if uid in per_uid:
                    session_info = per_uid[uid]
                elif per_uid:
                    # fall back to the single uid present if ours wasn't resolved
                    only_uid, info = next(iter(per_uid.items()))
                    uid = uid or only_uid
                    session_info = info
            else:
                uid = _read_uid(result["files_dir"])

            rows.append({
                "condition": condition,
                "study_result": result["study_result"],
                "comp_result": result["comp_result"],
                "uid": uid,
                "status": result["status"],
                "missing": ",".join(result["missing"]),
                "reconstructed": ",".join(sorted(result["reconstructed"])),
                "full_sessions": session_info["full_sessions"],
                "total_rows": session_info["total_rows"],
                "max_session": session_info["max_session"],
                "all_sessions_complete": session_info["full_sessions"] == EXPECTED_SESSIONS,
            })

    participants_df = pd.DataFrame(rows)

    status_counts = (
        participants_df["status"].value_counts().to_dict() if not participants_df.empty else {}
    )
    summary_df = pd.DataFrame([{
        "condition": condition,
        "full": status_counts.get("full", 0),
        "partial": status_counts.get("partial", 0),
        "unusable": status_counts.get("unusable", 0),
        "total": len(participants_df),
    }])

    outliers_dir = paths.analysis_dir / "outliers"
    outliers_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(outliers_dir / "outlier_report_summary.csv", index=False)
    participants_df.to_csv(outliers_dir / "outlier_report_participants.csv", index=False)

    return summary_df, participants_df
