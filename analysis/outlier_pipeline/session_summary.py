"""Per-participant session/attempt completeness from a trials.csv."""
from pathlib import Path

import pandas as pd

REQUIRED_ATTEMPTS = 3


def full_sessions_for(trials_csv: Path) -> dict[str, dict]:
    """Return {uid: {full_sessions, total_rows, max_session}} for a trials.csv.

    A session counts as full when it has REQUIRED_ATTEMPTS distinct attempts.
    """
    df = pd.read_csv(trials_csv)
    result: dict[str, dict] = {}
    if df.empty or "uid" not in df.columns:
        return result

    for uid, g in df.groupby("uid"):
        attempts_per_session = g.groupby("session")["attempt"].nunique()
        result[uid] = {
            "full_sessions": int((attempts_per_session == REQUIRED_ATTEMPTS).sum()),
            "total_rows": int(len(g)),
            "max_session": int(g["session"].max()),
        }
    return result
