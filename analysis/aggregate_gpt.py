# currently this one doesn't work properly
# Analysis/aggregate_gpt.py
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
    gpt_dir = config.PARTICIPANTS_DIR / "gpt-5_descriptions_as_ppt" # These are generations in the same imagination in translation app as participants, only with gpt descriptions differing in verbosity

    if not gpt_dir.exists():
        raise FileNotFoundError(f"Participants directory not found: {gpt_dir}")

    

    # log
    for file in sorted(gpt_dir.iterdir()):
        if file.name.endswith("_log.csv"):
            df = pd.read_csv(file)
            logs.append(df)

    all_logs  = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    # all_infos = pd.concat(infos, ignore_index=True) if infos else pd.DataFrame()
    # all_fb    = pd.concat(feedbacks, ignore_index=True) if feedbacks else pd.DataFrame()

    return all_logs

def main():
    logs = load_all_participant_csvs()

    # Quick sanity prints
    print("Logs shape:", logs.shape)

    # Save combined datasets
    out_dir = config.PROCESSED_DIR
    if not logs.empty:
        logs.to_csv(out_dir / "gpt_log.csv", index=False)
        print("Wrote:", out_dir / "gpt_log.csv")

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
        summary.to_csv(out_dir / "summary_by_uid_gpt.csv", index=False)
        print("Wrote:", out_dir / "summary_by_uid_gpt.csv")

if __name__ == "__main__":
    main()
