"""Export the cached calibration scores into human-readable files.

Read-only w.r.t. the APIs: this reads eval_data/_score_cache.json (populated by
eval_prompts.py) plus the two gold CSVs, and writes

  - eval_data/results_scores.csv : one row per (variant, label, prompt) with each
    judge's score, n_judges_flagged, and ai_suspected (2-of-3 >= THRESHOLD).
  - eval_data/results_report.md  : the full baseline vs v1 prompt text, a per-prompt
    score table per variant, and a consensus summary (caught / FP / FN).

Only prompts that were actually scored under a variant appear (cache-driven), so a
variant only lists the rows eval_prompts.py has already spent on.

Usage: python export_results.py
"""
import csv
import hashlib
import json
from pathlib import Path

try:
    from . import common, prompt_variants
except ImportError:
    import common
    import prompt_variants

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE / "eval_data"
CACHE_PATH = EVAL_DIR / "_score_cache.json"
SCORES_CSV = EVAL_DIR / "results_scores.csv"
REPORT_MD = EVAL_DIR / "results_report.md"

# (column label, model id) — must match eval_prompts.JUDGES model ids / cache keys.
JUDGES = [("gpt", "gpt-5.4-mini"), ("gemini", "gemini-2.5-flash"), ("claude", "claude-haiku-4-5")]


def _key(variant: str, model: str, prompt: str) -> str:
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{variant}|{model}|{h}"


def _load_gold():
    """Return [(label, prompt)] in file order (AI rows then human rows)."""
    rows = []
    for fname, label in (("gpt_descs.csv", "ai"), ("human_descs.csv", "human")):
        with open(EVAL_DIR / fname, newline="") as f:
            for r in csv.DictReader(f):
                rows.append((label, r["prompt"].strip()))
    return rows


def _scored_rows(cache: dict, variant: str, gold):
    """Build per-prompt score dicts for the rows present in cache under `variant`."""
    out = []
    for label, prompt in gold:
        scores = {j: cache.get(_key(variant, model, prompt)) for j, model in JUDGES}
        if all(v is None for v in scores.values()):
            continue  # not scored under this variant
        n_flagged = sum(1 for v in scores.values() if v is not None and v >= common.THRESHOLD)
        out.append({
            "variant": variant, "label": label, "prompt": prompt,
            **{f"{j}_score": scores[j] for j, _ in JUDGES},
            "n_judges_flagged": n_flagged,
            "ai_suspected": n_flagged >= common.MIN_AGREEMENT,
        })
    return out


def _write_scores_csv(all_rows):
    cols = ["variant", "label", "gpt_score", "gemini_score", "claude_score",
            "n_judges_flagged", "ai_suspected", "prompt"]
    with open(SCORES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({c: r[c] for c in cols})


def _fmt(v):
    return "-" if v is None else str(v)


def _table(rows):
    lines = ["| label | gpt | gemini | claude | flags | ai_suspected | prompt |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        prompt = r["prompt"].replace("|", "\\|")
        if len(prompt) > 90:
            prompt = prompt[:90] + "…"
        lines.append(
            f"| {r['label']} | {_fmt(r['gpt_score'])} | {_fmt(r['gemini_score'])} | "
            f"{_fmt(r['claude_score'])} | {r['n_judges_flagged']}/3 | "
            f"{'YES' if r['ai_suspected'] else 'no'} | {prompt} |"
        )
    return "\n".join(lines)


def _summary(rows):
    ai = [r for r in rows if r["label"] == "ai"]
    hu = [r for r in rows if r["label"] == "human"]
    caught = sum(1 for r in ai if r["ai_suspected"])
    fp = sum(1 for r in hu if r["ai_suspected"])
    fn = sum(1 for r in ai if not r["ai_suspected"])
    return (f"- AI rows scored: {len(ai)}, caught (consensus): **{caught}/{len(ai)}** "
            f"(false-negatives: {fn})\n"
            f"- Human rows scored: {len(hu)}, false-flagged: **{fp}/{len(hu)}**")


def main():
    cache = json.loads(CACHE_PATH.read_text())
    gold = _load_gold()

    all_rows = []
    per_variant = {}
    for variant in prompt_variants.VARIANTS:
        rows = _scored_rows(cache, variant, gold)
        per_variant[variant] = rows
        all_rows.extend(rows)

    _write_scores_csv(all_rows)

    parts = [
        "# AI-usage suspicion — prompt calibration results",
        "",
        f"Consensus rule: a prompt is `ai_suspected` when at least "
        f"**{common.MIN_AGREEMENT} of 3** judges score it "
        f"**>= {common.THRESHOLD}**. AI rows should be flagged; human rows should not.",
        "",
        "Only prompts already scored (present in the cache) under each variant are listed.",
    ]
    for variant, rows in per_variant.items():
        if not rows:
            continue
        parts += ["", f"## Variant: `{variant}`", "", _summary(rows), "", _table(rows)]

    parts += ["", "---", "", "## Prompt text"]
    for variant in prompt_variants.VARIANTS:
        parts += ["", f"### `{variant}` system prompt", "", "```text",
                  prompt_variants.VARIANTS[variant], "```"]

    REPORT_MD.write_text("\n".join(parts) + "\n")
    print(f"Wrote {SCORES_CSV}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
