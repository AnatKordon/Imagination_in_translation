# Analysis/aggregate.py
from pathlib import Path
import sys
import pandas as pd

# Make sure we can import config.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config  # now we can reuse your paths

def load_all_participant_csvs():
    logs, infos, feedbacks = [], [], []
    pdir = config.PARTICIPANTS_DIR

    if not pdir.exists():
        raise FileNotFoundError(f"Participants directory not found: {pdir}")

    for participant_folder in sorted(pdir.iterdir()):
        if not participant_folder.is_dir():
            continue

        # log
        for f in participant_folder.glob("participant_*_log.csv"):
            df = pd.read_csv(f)
            logs.append(df)

        # info
        for f in participant_folder.glob("participant_*_info.csv"):
            df = pd.read_csv(f)
            infos.append(df)

        # optional feedback (if you saved it)
        for f in participant_folder.glob("participant_*_feedback.csv"):
            df = pd.read_csv(f)
            feedbacks.append(df)

    all_logs  = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    all_infos = pd.concat(infos, ignore_index=True) if infos else pd.DataFrame()
    all_fb    = pd.concat(feedbacks, ignore_index=True) if feedbacks else pd.DataFrame()

    return all_logs, all_infos, all_fb

def main():
    logs, infos, fb = load_all_participant_csvs()

    # Quick sanity prints
    print("Logs shape:", logs.shape)
    print("Infos shape:", infos.shape)
    print("Feedback shape:", fb.shape)

    # Save combined datasets
    out_dir = config.ANALYSIS_DIR
    if not logs.empty:
        logs.to_csv(out_dir / "all_participants_log.csv", index=False)
    if not infos.empty:
        infos.to_csv(out_dir / "all_participants_info.csv", index=False)
    if not fb.empty:
        fb.to_csv(out_dir / "all_participants_feedback.csv", index=False)

    # Example: simple summary by user
    if not logs.empty and "uid" in logs and "subjective_score" in logs:
        summary = (
            logs.groupby("uid")
                .agg(
                    n_rows=("uid", "size"),
                    n_sessions=("session", "nunique"),
                    n_attempts=("attempt", "nunique"),
                    avg_score=("subjective_score", "mean")
                )
                .reset_index()
        )
        summary.to_csv(out_dir / "summary_by_uid.csv", index=False)
        print("Wrote:", out_dir / "summary_by_uid.csv")

if __name__ == "__main__":
    main()
