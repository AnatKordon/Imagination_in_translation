"""Offline calibration harness for the AI-usage suspicion rubric.

Scores a content-matched gold set (GPT-5 image descriptions = AI/positive,
real participant descriptions = human/negative) with all three production
judges under a *candidate* system prompt, and reports the metrics that matter:

  - false positives : human prompts that reach THRESHOLD (the old over-flagging bias)
  - false negatives : AI prompts that miss THRESHOLD (the current too-forgiving bias)
  - separation margin: min(AI score) - max(human score), per judge and consensus

To keep cost down and avoid tuning against one fixed set, each iteration draws
a FRESH balanced random sample from the dev pool; a held-out test split is
never touched until final validation.

Usage:
  python eval_prompts.py --variant v2                 # sample 5 AI + 5 human from dev
  python eval_prompts.py --variant v2 --seed 7        # reproducible different sample
  python eval_prompts.py --variant v2 --split test    # held-out final validation (all test rows)
  python eval_prompts.py --variant v2 --split all --misses   # full set + list errors

Results cache per (variant, model, prompt) so re-runs / added variants only
spend on what is new.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from . import claude_suspicion, common, gemini_suspicion, gpt_suspicion
    from . import prompt_variants
except ImportError:
    import claude_suspicion
    import common
    import gemini_suspicion
    import gpt_suspicion
    import prompt_variants

import anthropic  # noqa: E402
from google import genai  # noqa: E402
from openai import OpenAI  # noqa: E402

EVAL_DIR = HERE / "eval_data"
CACHE_PATH = EVAL_DIR / "_score_cache.json"

JUDGES = [
    ("gpt", gpt_suspicion, OpenAI),
    ("gemini", gemini_suspicion, genai.Client),
    ("claude", claude_suspicion, anthropic.Anthropic),
]


def load_gold() -> pd.DataFrame:
    """Labeled gold set with a fixed dev/test split.

    Split is deterministic (every 3rd row within each label -> test) so the
    held-out rows are stable across sessions and never sampled during tuning.
    """
    gpt = pd.read_csv(EVAL_DIR / "gpt_descs.csv")[["prompt"]].assign(label="ai")
    human = pd.read_csv(EVAL_DIR / "human_descs.csv")[["prompt"]].assign(label="human")
    df = pd.concat([gpt, human], ignore_index=True)
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df = df.drop_duplicates(subset="prompt").reset_index(drop=True)
    df["split"] = "dev"
    for label in ("ai", "human"):
        idx = df.index[df["label"] == label].tolist()
        for n, i in enumerate(idx):
            if n % 3 == 2:
                df.loc[i, "split"] = "test"
    return df


def sample_dev(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Fresh balanced random draw from the dev pool only."""
    dev = df[df["split"] == "dev"]
    parts = []
    for label in ("ai", "human"):
        pool = dev[dev["label"] == label]
        parts.append(pool.sample(n=min(per_class, len(pool)), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def _load_cache() -> dict:
    return json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=0))


def _key(variant: str, model: str, prompt: str) -> str:
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{variant}|{model}|{h}"


def score_rows(variant: str, system_prompt: str, rows: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """Score the given rows with all three judges under `system_prompt`."""
    original = common.SYSTEM_PROMPT
    common.SYSTEM_PROMPT = system_prompt  # judges read this at call time
    clients = {name: make() for name, _, make in JUDGES}
    try:
        out = rows.copy()
        for name, module, _ in JUDGES:
            scores = []
            for prompt in out["prompt"]:
                ck = _key(variant, module.DEFAULT_MODEL, prompt)
                if ck not in cache:
                    res = common.retry(lambda p: module.score_prompt(clients[name], p), prompt)
                    cache[ck] = int(res.suspicion_score)
                    _save_cache(cache)
                scores.append(cache[ck])
            out[f"{name}_score"] = scores
    finally:
        common.SYSTEM_PROMPT = original
    return out


def report(variant: str, scored: pd.DataFrame, tag: str) -> dict:
    judge_cols = [f"{n}_score" for n, _, _ in JUDGES]
    flags = pd.concat([scored[c] >= common.THRESHOLD for c in judge_cols], axis=1)
    consensus = flags.sum(axis=1) >= common.MIN_AGREEMENT

    ai = scored[scored["label"] == "ai"]
    hu = scored[scored["label"] == "human"]
    cons_ai = consensus[scored["label"] == "ai"]
    cons_hu = consensus[scored["label"] == "human"]

    print(f"\n===== variant={variant}  set={tag}  (AI n={len(ai)}, human n={len(hu)}; "
          f"THRESHOLD={common.THRESHOLD}, MIN_AGREEMENT={common.MIN_AGREEMENT}) =====")
    for name, _, _ in JUDGES:
        c = f"{name}_score"
        fp = int((hu[c] >= common.THRESHOLD).sum())
        fn = int((ai[c] < common.THRESHOLD).sum())
        margin = (int(ai[c].min()) - int(hu[c].max())) if len(ai) and len(hu) else None
        print(f"  {name:<7} AI[min={ai[c].min():>3} mean={ai[c].mean():5.1f} max={ai[c].max():>3}] "
              f"HU[min={hu[c].min():>3} mean={hu[c].mean():5.1f} max={hu[c].max():>3}] "
              f"FP={fp} FN={fn} margin={margin}")
    cfp, cfn = int(cons_hu.sum()), int((~cons_ai).sum())
    print(f"  CONSENSUS(2of3): caught {int(cons_ai.sum())}/{len(ai)} AI, "
          f"false-flagged {cfp}/{len(hu)} human  ->  FP={cfp} FN={cfn}")
    return {"variant": variant, "set": tag, "n": len(scored), "consensus_FP": cfp, "consensus_FN": cfn}


def show_misses(scored: pd.DataFrame) -> None:
    judge_cols = [f"{n}_score" for n, _, _ in JUDGES]
    consensus = pd.concat([scored[c] >= common.THRESHOLD for c in judge_cols], axis=1).sum(axis=1) >= common.MIN_AGREEMENT
    wrong = ((scored["label"] == "ai") & ~consensus) | ((scored["label"] == "human") & consensus)
    if not wrong.any():
        print("  (no consensus errors on this set)")
        return
    print("  --- consensus errors ---")
    for _, r in scored[wrong].iterrows():
        s = " ".join(f"{n}={r[f'{n}_score']}" for n, _, _ in JUDGES)
        kind = "FN(missed AI)" if r["label"] == "ai" else "FP(human flagged)"
        print(f"   [{r['split']}] {kind} {s} :: {r['prompt'][:90]}...")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"],
                    help="dev = fresh random sample; test = all held-out rows; all = full gold set.")
    ap.add_argument("--per-class", type=int, default=5, help="Per-class size of the dev random sample.")
    ap.add_argument("--seed", type=int, default=0, help="Seed for the dev random sample.")
    ap.add_argument("--misses", action="store_true")
    args = ap.parse_args()

    df = load_gold()
    cache = _load_cache()
    sysprompt = prompt_variants.VARIANTS[args.variant]

    if args.split == "dev":
        rows = sample_dev(df, args.per_class, args.seed)
        tag = f"dev-sample(seed={args.seed},n={len(rows)})"
        print("Sampled prompts:")
        for _, r in rows.iterrows():
            print(f"   {r['label']:<5} {r['prompt'][:80]}...")
    elif args.split == "test":
        rows = df[df["split"] == "test"]
        tag = "HELD-OUT-TEST"
    else:
        rows = df
        tag = "FULL"

    scored = score_rows(args.variant, sysprompt, rows, cache)
    report(args.variant, scored, tag)
    if args.misses:
        show_misses(scored)


if __name__ == "__main__":
    main()
