"""Run all three AI-usage judges over a condition's trials and combine by 2-of-3 vote.

A prompt is flagged as AI-suspected when at least common.MIN_AGREEMENT of the three
judges score it >= common.THRESHOLD. Output maps back to uid/session/attempt so the
later exclusion wiring can drop the implicated sessions.

Judge results are checkpointed to _judge_cache.json in the output dir after every
API call, keyed by (model, rubric hash, prompt hash): an interrupted run loses at
most one call, a re-run only pays for prompts not yet scored, and changing the
rubric in common.SYSTEM_PROMPT automatically invalidates old entries.

--report-only rebuilds the summary and per-participant CSVs from an existing
ai_suspicion_scores.csv without any API calls.
"""
import hashlib
import json
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


# Cache entries are invalidated automatically when the rubric changes.
_RUBRIC_HASH = hashlib.sha1(common.SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:8]


def _cache_key(model: str, prompt: str) -> str:
    return f"{model}|{_RUBRIC_HASH}|{hashlib.sha1(prompt.encode('utf-8')).hexdigest()[:16]}"


def _load_judge_cache(out_dir: Path) -> dict:
    path = out_dir / "_judge_cache.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _score_unique(
    unique: pd.DataFrame,
    prefix: str,
    module,
    client,
    limit: int | None,
    usage: common.UsageAccumulator,
    cache: dict,
    out_dir: Path,
) -> pd.DataFrame:
    """Score the unique-prompt frame with one judge, returning prefixed score/expl columns.

    Every fresh API result is checkpointed to _judge_cache.json immediately, so an
    interrupted run can resume without re-paying for prompts already scored.
    """
    cache_path = out_dir / "_judge_cache.json"

    def scorer(prompt: str) -> common.PromptSuspicionResult:
        key = _cache_key(module.DEFAULT_MODEL, prompt)
        if key in cache:
            hit = cache[key]
            return common.PromptSuspicionResult(
                suspicion_score=hit["score"], explanation=hit["explanation"]
            )
        result = module.score_prompt(client, prompt, usage=usage)
        cache[key] = {"score": result.suspicion_score, "explanation": result.explanation}
        cache_path.write_text(json.dumps(cache, indent=0))
        return result

    scored = common.score_dataframe(
        unique,
        scorer,
        score_col=f"{prefix}_score",
        expl_col=f"{prefix}_explanation",
        limit=limit,
        label=prefix,
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

    out_dir = paths.analysis_dir / "outliers" / "ai_usage"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = _load_judge_cache(out_dir)
    if cache:
        print(f"[{condition}] judge cache: {len(cache)} entries loaded (matching ones will be reused)")

    merged = unique
    usage_by_model: dict[str, common.UsageAccumulator] = {}
    for i, (prefix, module, make_client) in enumerate(JUDGES, start=1):
        print(f"[{condition}] judge {i}/{len(JUDGES)}: {prefix} ({module.DEFAULT_MODEL}) "
              f"— scoring {len(unique)} unique prompts...", flush=True)
        acc = common.UsageAccumulator()
        usage_by_model[module.DEFAULT_MODEL] = acc
        scored = _score_unique(unique, prefix, module, make_client(), limit, acc, cache, out_dir)
        print(f"[{condition}] judge {i}/{len(JUDGES)}: {prefix} done "
              f"({acc.calls} API calls, {len(unique) - acc.calls} from cache)", flush=True)
        merged = merged.merge(scored, on="prompt", how="left")

    common.print_usage_report(usage_by_model)

    score_cols = [f"{p}_score" for p, _, _ in JUDGES]
    flags = pd.concat([merged[c] >= common.THRESHOLD for c in score_cols], axis=1)
    merged["n_judges_flagged"] = flags.sum(axis=1)
    merged["ai_suspected"] = merged["n_judges_flagged"] >= common.MIN_AGREEMENT

    keep = ["uid", "session", "attempt", "prompt"]
    keep = [c for c in keep if c in trials.columns]
    per_trial = trials[keep].merge(merged, on="prompt", how="left")

    per_trial.to_csv(out_dir / "ai_suspicion_scores.csv", index=False)

    _write_summary(condition, per_trial, out_dir)
    _write_participant_report(per_trial, out_dir)
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


def _write_participant_report(per_trial: pd.DataFrame, out_dir: Path) -> None:
    """Per-uid AI-suspicion breakdown -> ai_suspicion_by_participant.csv.

    attempts_flagged counts individually flagged attempts; sessions_flagged counts
    sessions with >= 1 flagged attempt (the exclusion granularity: one flagged
    attempt drops the whole session, so attempts_excluded counts every attempt
    belonging to a flagged session).
    """
    if not {"uid", "session"}.issubset(per_trial.columns):
        return
    df = per_trial.copy()
    df["ai_suspected"] = df["ai_suspected"].fillna(False).astype(bool)

    session_flagged = (
        df.groupby(["uid", "session"])["ai_suspected"].any().rename("session_flagged").reset_index()
    )
    df = df.merge(session_flagged, on=["uid", "session"], how="left")

    report = (
        df.groupby("uid")
        .agg(
            attempts_total=("prompt", "size"),
            attempts_flagged=("ai_suspected", "sum"),
            attempts_excluded=("session_flagged", "sum"),
            sessions_total=("session", "nunique"),
        )
        .reset_index()
    )
    per_uid_sessions = session_flagged.groupby("uid")["session_flagged"].sum().rename("sessions_flagged")
    report = report.merge(per_uid_sessions.reset_index(), on="uid", how="left")
    report = report[["uid", "attempts_total", "attempts_flagged", "attempts_excluded",
                     "sessions_total", "sessions_flagged"]]
    report.to_csv(out_dir / "ai_suspicion_by_participant.csv", index=False)


def report_only(condition: str) -> pd.DataFrame:
    """Rebuild summary + per-participant report from the saved scores CSV. No API calls.

    Verdicts are recomputed from the stored per-judge scores, so the output always
    reflects the current THRESHOLD / MIN_AGREEMENT even if they changed since scoring.
    """
    paths = config.paths_for(condition)
    out_dir = paths.analysis_dir / "outliers" / "ai_usage"
    scores_path = out_dir / "ai_suspicion_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"{scores_path} not found (run the scoring pass first)")
    per_trial = pd.read_csv(scores_path)

    score_cols = [f"{p}_score" for p, _, _ in JUDGES]
    flags = pd.concat([per_trial[c] >= common.THRESHOLD for c in score_cols], axis=1)
    per_trial["n_judges_flagged"] = flags.sum(axis=1)
    per_trial["ai_suspected"] = per_trial["n_judges_flagged"] >= common.MIN_AGREEMENT

    per_trial.to_csv(scores_path, index=False)  # refresh verdict columns in place
    _write_summary(condition, per_trial, out_dir)
    _write_participant_report(per_trial, out_dir)
    return per_trial


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="3-judge AI-usage consensus for one condition.")
    parser.add_argument("--condition", default="aigen_perc", help="Condition slug, e.g. aigen_perc")
    parser.add_argument("--limit", type=int, default=None, help="Max unique prompts to score (cost guard).")
    parser.add_argument("--report-only", action="store_true",
                        help="Rebuild summary + participant report from the saved scores CSV (no API calls).")
    args = parser.parse_args()
    if args.report_only:
        report_only(args.condition)
    else:
        run_consensus(args.condition, limit=args.limit)


if __name__ == "__main__":
    main()
