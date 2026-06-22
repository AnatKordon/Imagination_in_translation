"""Run all three AI-usage judges over a condition's trials and combine by 2-of-3 vote.

A prompt is flagged as AI-suspected when at least common.MIN_AGREEMENT of the three
judges score it >= common.THRESHOLD. Output maps back to uid/session/attempt so the
later exclusion wiring can drop the implicated sessions.
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import anthropic  # noqa: E402
from google import genai  # noqa: E402
from openai import OpenAI  # noqa: E402

import config  # noqa: E402

try:
    from . import claude_suspicion, common, gemini_suspicion, gpt_suspicion
except ImportError:  # running as a plain script from inside this folder
    import claude_suspicion
    import common
    import gemini_suspicion
    import gpt_suspicion

# (column prefix, module, client factory) for each judge.
JUDGES = [
    ("gpt", gpt_suspicion, OpenAI),
    ("gemini", gemini_suspicion, genai.Client),
    ("claude", claude_suspicion, anthropic.Anthropic),
]


def _score_unique(
    unique: pd.DataFrame,
    prefix: str,
    module,
    client,
    limit: int | None,
    usage: common.UsageAccumulator,
) -> pd.DataFrame:
    """Score the unique-prompt frame with one judge, returning prefixed score/expl columns."""
    scored = common.score_dataframe(
        unique,
        lambda p: module.score_prompt(client, p, usage=usage),
        score_col=f"{prefix}_score",
        expl_col=f"{prefix}_explanation",
        limit=limit,
    )
    return scored[["prompt", f"{prefix}_score", f"{prefix}_explanation"]]


def run_consensus(condition: str, limit: int | None = None) -> pd.DataFrame:
    """Score every trial prompt for one condition and write per-trial + summary CSVs.

    limit caps how many *unique* prompts are sent to the APIs (cost guard).
    Returns the per-trial DataFrame.
    """
    paths = config.paths_for(condition)
    trials_path = paths.processed_dir / "all_trials.csv"
    if not trials_path.exists():
        raise FileNotFoundError(f"{trials_path} not found (run aggregate.py first)")
    trials = pd.read_csv(trials_path)

    unique = trials[["prompt"]].drop_duplicates().reset_index(drop=True)

    merged = unique
    usage_by_model: dict[str, common.UsageAccumulator] = {}
    for prefix, module, make_client in JUDGES:
        acc = common.UsageAccumulator()
        usage_by_model[module.DEFAULT_MODEL] = acc
        scored = _score_unique(unique, prefix, module, make_client(), limit, acc)
        merged = merged.merge(scored, on="prompt", how="left")

    common.print_usage_report(usage_by_model)

    score_cols = [f"{p}_score" for p, _, _ in JUDGES]
    flags = pd.concat([merged[c] >= common.THRESHOLD for c in score_cols], axis=1)
    merged["n_judges_flagged"] = flags.sum(axis=1)
    merged["ai_suspected"] = merged["n_judges_flagged"] >= common.MIN_AGREEMENT

    keep = ["uid", "session", "attempt", "prompt"]
    keep = [c for c in keep if c in trials.columns]
    per_trial = trials[keep].merge(merged, on="prompt", how="left")

    out_dir = paths.analysis_dir / "outliers" / "ai_usage"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_trial.to_csv(out_dir / "ai_suspicion_scores.csv", index=False)

    _write_summary(condition, per_trial, out_dir)
    return per_trial


def _write_summary(condition: str, per_trial: pd.DataFrame, out_dir: Path) -> None:
    suspected = per_trial[per_trial["ai_suspected"] == True]  # noqa: E712 - NA-safe equality
    score_cols = [f"{p}_score" for p, _, _ in JUDGES]
    fully_scored = per_trial[score_cols].notna().all(axis=1)
    summary = pd.DataFrame([{
        "condition": condition,
        "trials_scored": int(fully_scored.sum()),
        "trials_suspected": int(len(suspected)),
        "sessions_implicated": int(
            suspected[["uid", "session"]].drop_duplicates().shape[0]
        ) if {"uid", "session"}.issubset(suspected.columns) else 0,
    }])
    summary.to_csv(out_dir / "ai_suspicion_summary.csv", index=False)
    print(f"[{condition}] AI-suspected trials: {summary['trials_suspected'][0]}/{summary['trials_scored'][0]} "
          f"(sessions implicated: {summary['sessions_implicated'][0]})")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="3-judge AI-usage consensus for one condition.")
    parser.add_argument("--condition", default="aigen_perc", help="Condition slug, e.g. aigen_perc")
    parser.add_argument("--limit", type=int, default=None, help="Max unique prompts to score (cost guard).")
    args = parser.parse_args()
    run_consensus(args.condition, limit=args.limit)


if __name__ == "__main__":
    main()
