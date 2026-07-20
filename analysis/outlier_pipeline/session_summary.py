"""Per-participant session/attempt completeness from a trials.csv."""
from pathlib import Path

import pandas as pd

# Default for the feedback conditions (aigen/nogen). Conditions declare their own
# count via `attempts` in condition_maps.yaml — plain has 1, since it has no
# feedback loop to iterate on. Callers pass config.spec_for(cond).attempts.
REQUIRED_ATTEMPTS = 3


def full_sessions_for(trials_csv: Path, required_attempts: int = REQUIRED_ATTEMPTS) -> dict[str, dict]:
    """Return {uid: {full_sessions, total_rows, max_session}} for a trials.csv.

    A session counts as full when it has `required_attempts` distinct attempts.
    """
    df = pd.read_csv(trials_csv)
    result: dict[str, dict] = {}
    if df.empty or "uid" not in df.columns:
        return result

    for uid, g in df.groupby("uid"):
        attempts_per_session = g.groupby("session")["attempt"].nunique()
        result[uid] = {
            "full_sessions": int((attempts_per_session == required_attempts).sum()),
            "total_rows": int(len(g)),
            "max_session": int(g["session"].max()),
        }
    return result
