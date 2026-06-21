"""Digit-span memory test metrics, ported from analysis/digit_span/digit_span_analysis.ipynb
so the same exact_match / positional / recall formulas apply to every del condition's
all_digit_span.csv, not just the pilot file the notebook reads.
"""
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MIN_ACCURACY = 0.15
MIN_TRIALS = 15


def _clean_response(x) -> str:
    if pd.isna(x) or x == "":
        return ""
    try:
        f_val = float(x)
        if f_val == int(f_val):
            return str(int(f_val))
        return str(f_val)
    except (ValueError, TypeError):
        return str(x)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["presented_sequence"] = df["presented_sequence"].astype(str)
    df["participant_response"] = df["participant_response"].apply(_clean_response)
    df["sequence_length"] = df["presented_sequence"].apply(len)
    return df


def _positional_acc(row) -> float:
    pres, resp = row["presented_sequence"], row["participant_response"]
    matches = sum(1 for p, r in zip(pres, resp) if p == r)
    return matches / row["sequence_length"]


def _digit_recall(row) -> float:
    pres, resp = Counter(row["presented_sequence"]), Counter(row["participant_response"])
    return sum((pres & resp).values()) / row["sequence_length"]


def add_accuracy_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = _clean(df)
    df["is_exact_match"] = (df["presented_sequence"] == df["participant_response"]).astype(int)
    df["order_acc_pct"] = df.apply(_positional_acc, axis=1)
    df["total_acc_pct"] = df.apply(_digit_recall, axis=1)
    return df


def participant_summary(df: pd.DataFrame) -> pd.DataFrame:
    """uid, avg_exact_match, avg_positional_acc, avg_digit_recall."""
    df = add_accuracy_columns(df)
    return (
        df.groupby("uid")
        .agg(
            avg_exact_match=("is_exact_match", "mean"),
            avg_positional_acc=("order_acc_pct", "mean"),
            avg_digit_recall=("total_acc_pct", "mean"),
        )
        .reset_index()
    )


def try_counts(df: pd.DataFrame) -> pd.DataFrame:
    """uid, session, num_tries."""
    return df.groupby(["uid", "session"]).size().reset_index(name="num_tries")


def session_pass_fail(
    df: pd.DataFrame, min_accuracy: float = MIN_ACCURACY, min_trials: int = MIN_TRIALS
) -> pd.DataFrame:
    """uid, session, exact_match_mean, num_tries, session_ok."""
    df = add_accuracy_columns(df)
    summary = (
        df.groupby(["uid", "session"])
        .agg(exact_match_mean=("is_exact_match", "mean"), num_tries=("is_exact_match", "size"))
        .reset_index()
    )
    summary["session_ok"] = (summary["exact_match_mean"] >= min_accuracy) & (
        summary["num_tries"] >= min_trials
    )
    return summary


def plot_accuracy_vs_length(df: pd.DataFrame, out_path: Path) -> None:
    """Reproduces the notebook's '100% Accuracy vs Sequence Length' line plot."""
    df = add_accuracy_columns(df)
    viz_data = df.groupby(["uid", "sequence_length"])["is_exact_match"].mean().reset_index()
    viz_data["percentage_correct"] = viz_data["is_exact_match"] * 100

    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=viz_data,
        x="sequence_length",
        y="percentage_correct",
        hue="uid",
        marker="o",
        linewidth=2,
        alpha=0.7,
    )
    plt.title("Digit Span Performance: 100% Accuracy vs. Sequence Length", fontsize=15)
    plt.ylabel("Trials with 100% Accuracy (%)")
    plt.xlabel("Number of Digits (Sequence Length)")
    plt.legend(title="Participant ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
