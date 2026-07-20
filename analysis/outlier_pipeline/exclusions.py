"""Session- and participant-level usability, combining structural completeness
(the condition's expected attempts per session) with quality gates (short answers,
digit-span performance).
"""
import pandas as pd

from .prompt_quality import flag_short_attempts
from .session_summary import REQUIRED_ATTEMPTS

MIN_USABLE_SESSIONS = 3


def session_table(
    trials_df: pd.DataFrame,
    digit_span_pass_fail: pd.DataFrame | None = None,
    ai_flags: pd.DataFrame | None = None,
    required_attempts: int = REQUIRED_ATTEMPTS,
) -> pd.DataFrame:
    """One row per (uid, session): attempts_present, is_full_session, is_short_session,
    is_digitspan_failed, is_ai_session, usable.

    required_attempts: attempts a session needs to be structurally complete, from the
    condition's `attempts` in condition_maps.yaml (3 for aigen/nogen, 1 for plain).

    ai_flags: optional per-trial frame with [uid, session, attempt, ai_suspected]
    (from ai_usage_suspicion consensus). A session is AI-suspected when ANY of its
    attempts is flagged; sessions with no scored row default to not-suspected so that
    scoring gaps never silently exclude anyone.
    """
    df = trials_df.copy()
    df["is_short_attempt"] = flag_short_attempts(df)

    sessions = (
        df.groupby(["uid", "session"])
        .agg(
            attempts_present=("attempt", "nunique"),
            is_short_session=("is_short_attempt", "any"),
        )
        .reset_index()
    )
    sessions["is_full_session"] = sessions["attempts_present"] == required_attempts

    if digit_span_pass_fail is not None:
        merged = sessions.merge(
            digit_span_pass_fail[["uid", "session", "session_ok"]],
            on=["uid", "session"],
            how="left",
        )
        # No digit-span record for that session at all -> can't verify recall -> failed.
        sessions["is_digitspan_failed"] = ~merged["session_ok"].fillna(False)
    else:
        sessions["is_digitspan_failed"] = False

    if ai_flags is not None:
        per_session_ai = (
            ai_flags.assign(ai_suspected=ai_flags["ai_suspected"].fillna(False).astype(bool))
            .groupby(["uid", "session"])["ai_suspected"]
            .any()
            .reset_index()
            .rename(columns={"ai_suspected": "is_ai_session"})
        )
        merged = sessions.merge(per_session_ai, on=["uid", "session"], how="left")
        sessions["is_ai_session"] = merged["is_ai_session"].fillna(False).astype(bool)
    else:
        sessions["is_ai_session"] = False

    sessions["usable"] = (
        sessions["is_full_session"]
        & ~sessions["is_short_session"]
        & ~sessions["is_digitspan_failed"]
        & ~sessions["is_ai_session"]
    )
    return sessions


def participant_table(
    sessions: pd.DataFrame, has_digit_span: bool = False, has_ai: bool = False
) -> pd.DataFrame:
    """Per uid: breakdown of how many existing sessions pass each individual gate,
    plus usable_sessions (passes all gates) and excluded (usable_sessions < MIN_USABLE_SESSIONS)."""
    participants = (
        sessions.groupby("uid")
        .agg(
            full_sessions_structural=("is_full_session", "sum"),
            good_wordcount_sessions=("is_short_session", lambda s: (~s).sum()),
            good_digitspan_sessions=("is_digitspan_failed", lambda s: (~s).sum()),
            good_nonai_sessions=("is_ai_session", lambda s: (~s).sum()),
            usable_sessions=("usable", "sum"),
        )
        .reset_index()
    )
    if not has_digit_span:
        participants["good_digitspan_sessions"] = pd.NA
    if not has_ai:
        participants["good_nonai_sessions"] = pd.NA
    participants["excluded"] = participants["usable_sessions"] < MIN_USABLE_SESSIONS
    return participants


def apply_exclusions(
    trials_df: pd.DataFrame, sessions: pd.DataFrame, participants: pd.DataFrame
) -> pd.DataFrame:
    """Drop rows for non-usable sessions and for fully-excluded participants."""
    excluded_uids = set(participants.loc[participants["excluded"], "uid"])
    usable_pairs = set(
        zip(sessions.loc[sessions["usable"], "uid"], sessions.loc[sessions["usable"], "session"])
    )
    keep_mask = trials_df.apply(
        lambda r: r["uid"] not in excluded_uids and (r["uid"], r["session"]) in usable_pairs,
        axis=1,
    )
    return trials_df[keep_mask].reset_index(drop=True)
