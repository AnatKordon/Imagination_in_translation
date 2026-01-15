# Analysis/aggregate.py
from pathlib import Path
import sys
import pandas as pd

# Make sure we can import config.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config  # now we can reuse your paths
pdir = config.PARTICIPANTS_DIR
out_dir = config.PROCESSED_DIR


def _add_source_columns(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Attach JATOS folder identifiers so we can trace provenance."""
    study_result = csv_path.parents[2].name if len(csv_path.parents) >= 3 else ""
    comp_result = csv_path.parents[1].name if len(csv_path.parents) >= 2 else ""
    df = df.copy()
    df["study_result"] = study_result
    df["comp_result"] = comp_result
    return df


def load_all_participant_csvs(pdir: Path):
    """Load all trials/participants CSVs from the nested JATOS export."""
    trials_frames, participant_frames = [], []
   

    if not pdir.exists():
        raise FileNotFoundError(f"Participants directory not found: {pdir}")

    # Each participant lives under study_result_*/comp-result_*/files/
    trial_files = sorted(pdir.glob("**/trials.csv"))
    participant_files = sorted(pdir.glob("**/participants.csv"))

    for f in participant_files:
        df = pd.read_csv(f)
        participant_frames.append(_add_source_columns(df, f))

    for f in trial_files:
        df = pd.read_csv(f)
        trials_frames.append(_add_source_columns(df, f))

    all_trials = pd.concat(trials_frames, ignore_index=True) if trials_frames else pd.DataFrame()
    all_participants = pd.concat(participant_frames, ignore_index=True) if participant_frames else pd.DataFrame()

    return all_trials, all_participants

def main(pdir: Path = config.PARTICIPANTS_DIR, out_dir: Path = config.PROCESSED_DIR):
    trials, participants = load_all_participant_csvs(pdir=pdir)

    # Quick sanity prints
    print("Trials shape:", trials.shape)
    print("Participants shape:", participants.shape)

    # Save combined datasets
    if not trials.empty:
        trials.to_csv(out_dir / "all_trials.csv", index=False)
    if not participants.empty:
        participants.to_csv(out_dir / "all_participants.csv", index=False)

    # Example: simple summary by user
    if not trials.empty and "uid" in trials and "subjective_score" in trials:
        summary = (
            trials.groupby("uid")
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
