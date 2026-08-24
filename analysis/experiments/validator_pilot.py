# Runs the current validator over a small stratified subset and diffs it, tag by tag,
# against the validation already stored in the source csv.
#
# The point is attribution, not accuracy: v1 answered only `in_image`, so every
# rejection looked like a participant error. v2 splits that into three sources. The
# interesting rows are therefore the disagreements - what v1 called a participant
# hallucination and v2 reattributes to the extractor, and any tag v2 newly rejects.
#
# Nothing here writes to Data/. Outputs land under analysis/outputs/experiments/validator/.

from pathlib import Path
import argparse
import json
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in (PROJECT_ROOT, PROJECT_ROOT / "analysis" / "nlp_analysis"):
    if str(extra) not in sys.path:
        sys.path.append(str(extra))

import object_accuracy_detector as validator  # noqa: E402

FULL_DATA_CSV = (
    PROJECT_ROOT
    / "Data/processed_data/Full_experiment/combined/full_data"
    / "trials_final_semantic_tag_image_validation_full_data.csv"
)
CEILING_CSV = (
    PROJECT_ROOT
    / "Data/processed_data/Full_experiment/combined/gpt_ceiling"
    / "gpt_ceiling_semantic_tag_image_validation_verbosity-medium.csv"
)
OUT_DIR = PROJECT_ROOT / "analysis/outputs/experiments/validator"

STRATA = ["generation", "delay"]


def load_subset(n_participant: int, n_ceiling: int, seed: int) -> pd.DataFrame:
    """Spread the participant rows over conditions, feedback arms and images.

    With a handful of rows we cannot fill every cell, so sample one row per stratum
    from a shuffled frame and take the first n - that yields distinct conditions
    before it repeats any, which is what makes a 5-row look worth looking at.
    """
    frames = []

    if n_participant:
        d = pd.read_csv(FULL_DATA_CSV)
        d = d[d["item_evaluations"].notna()].copy()
        d["has_feedback"] = d["feedback_id"].notna()
        shuffled = d.sample(frac=1.0, random_state=seed)
        spread = (
            shuffled.groupby(STRATA + ["has_feedback"], dropna=False)
            .head(1)
            .sample(frac=1.0, random_state=seed)
        )
        picked = spread.head(n_participant)
        if len(picked) < n_participant:  # more rows than strata: top up
            rest = shuffled[~shuffled.index.isin(picked.index)]
            picked = pd.concat([picked, rest.head(n_participant - len(picked))])
        picked = picked.copy()
        picked["source"] = "participant"
        frames.append(picked)

    if n_ceiling:
        c = pd.read_csv(CEILING_CSV)
        c = c[c["item_evaluations"].notna()].copy()
        c["source"] = "gpt_ceiling"
        for column in STRATA:
            c[column] = "ceiling"
        frames.append(c.sample(frac=1.0, random_state=seed).head(n_ceiling))

    return pd.concat(frames, ignore_index=True)


def previous_verdicts(row: pd.Series) -> dict[tuple[str, str], bool]:
    """Map (category, item) -> old in_image, from the stored v1 evaluations."""
    try:
        evaluations = json.loads(row["item_evaluations"])
    except (TypeError, ValueError):
        return {}
    return {(e["category"], e["item"]): bool(e["in_image"]) for e in evaluations}


def run(args) -> None:
    subset = load_subset(args.n_participant, args.n_ceiling, args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    print(f"Subset: {len(subset)} rows | model={args.model} "
          f"| reasoning={args.reasoning_effort} | run={args.tag}\n")

    tag_rows = []
    row_rows = []

    for _, row in subset.iterrows():
        old = previous_verdicts(row)
        started = time.time()
        result = validator.evaluate_row(
            row, model=args.model, reasoning_effort=args.reasoning_effort
        )
        elapsed = time.time() - started

        label = f"{row['source']}/{row.get('generation')}/{row.get('delay')}"
        if result["error"]:
            print(f"[{label}] {row['gt']}  ERROR: {result['error']}")
            continue

        for e in json.loads(result["item_evaluations"]):
            old_in_image = old.get((e["category"], e["item"]))
            tag_rows.append(
                {
                    "source": row["source"],
                    "generation": row.get("generation"),
                    "delay": row.get("delay"),
                    "uid": row.get("uid"),
                    "session": row.get("session"),
                    "attempt": row.get("attempt"),
                    "gt": row["gt"],
                    "description": row["prompt"],
                    "category": e["category"],
                    "item": e["item"],
                    "old_in_image": old_in_image,
                    "new_in_image": e["in_image"],
                    "category_correct": e["category_correct"],
                    "grounded_in_description": e["grounded_in_description"],
                    "error_type": e["error_type"],
                    "confidence": e["confidence"],
                    "note": e["note"],
                }
            )

        row_rows.append(
            {
                "source": row["source"],
                "label": label,
                "gt": row["gt"],
                "secs": round(elapsed, 1),
                "n_extracted": result["n_extracted"],
                "old_n_invalid": row.get("n_invalid_not_in_image"),
                "n_category_misplacement": result["n_category_misplacement"],
                "n_model_hallucination": result["n_model_hallucination"],
                "n_participant_hallucination": result["n_participant_hallucination"],
                "n_scored": result["n_scored"],
                "corrected_rate": result["corrected_hallucination_rate"],
                "n_low_confidence": result["n_low_confidence"],
            }
        )

    tags = pd.DataFrame(tag_rows)
    rows = pd.DataFrame(row_rows)
    if tags.empty:
        print("No successful rows.")
        return

    # v1 judged only the five categories; a tag absent from `old` was never scored.
    known = tags[tags["old_in_image"].notna()].copy()
    known["old_in_image"] = known["old_in_image"].astype(bool)

    slug = f"{args.model}_r-{args.reasoning_effort}{args.tag}_{stamp}"
    tags_path = OUT_DIR / f"pilot_tags_{slug}.csv"
    rows_path = OUT_DIR / f"pilot_rows_{slug}.csv"
    tags.to_csv(tags_path, index=False)
    rows.to_csv(rows_path, index=False)

    print("=" * 100)
    print("PER-ROW: old rejection count vs new attribution")
    print("=" * 100)
    print(rows.to_string(index=False))

    print("\n" + "=" * 100)
    print("FLIP MATRIX  (old in_image -> new error_type), tags scored by both versions")
    print("=" * 100)
    matrix = pd.crosstab(known["old_in_image"], known["error_type"])
    print(matrix.to_string())

    print("\n" + "=" * 100)
    print("DISAGREEMENTS AND ATTRIBUTED ERRORS")
    print("=" * 100)
    interesting = tags[
        (tags["error_type"] != "valid")
        | (tags["old_in_image"] == False)  # noqa: E712 - keeps NaN out
    ]
    for gt_name, block in interesting.groupby("gt", sort=False):
        print(f"\n--- {gt_name} ---")
        print(f"    {str(block.iloc[0]['description'])[:220]!r}")
        for _, t in block.iterrows():
            if t["old_in_image"] is None or pd.isna(t["old_in_image"]):
                change = "new  "
            elif bool(t["old_in_image"]) == bool(t["new_in_image"]):
                change = "same "
            else:
                change = "FLIP "
            verdict = "REJECT" if t["old_in_image"] is False else "accept"
            print(
                f"  {change}[{t['category']:17}] {t['item']!r:38} "
                f"v1={verdict:6} -> {t['error_type']:25} {t['confidence']:6} {t['note']!r}"
            )

    print("\n" + "=" * 100)
    print("TOTALS")
    print("=" * 100)
    n = len(tags)
    print(f"  tags judged                    {n}")
    for error_type in ["valid"] + list(validator.ERROR_TYPES):
        c = int((tags["error_type"] == error_type).sum())
        print(f"  {error_type:30} {c:4}  ({100*c/n:5.1f}%)")
    print(f"  low/medium confidence          {int((tags['confidence']!='high').sum()):4}")
    old_rejects = int((known["old_in_image"] == False).sum())  # noqa: E712
    if old_rejects:
        reattributed = int(
            (
                (known["old_in_image"] == False)  # noqa: E712
                & (known["error_type"].isin(["category_misplacement", "model_hallucination"]))
            ).sum()
        )
        rescued = int(
            ((known["old_in_image"] == False) & (known["error_type"] == "valid")).sum()  # noqa: E712
        )
        kept = int(
            (
                (known["old_in_image"] == False)  # noqa: E712
                & (known["error_type"] == "participant_hallucination")
            ).sum()
        )
        print(f"\n  v1 rejections                  {old_rejects}")
        print(f"    still participant error      {kept}")
        print(f"    reattributed to extractor    {reattributed}")
        print(f"    overturned to valid          {rescued}")
    newly = int(
        ((known["old_in_image"] == True) & (known["error_type"] == "participant_hallucination")).sum()  # noqa: E712
    )
    print(f"  v1 accepted, v2 calls participant error   {newly}")

    validator.print_usage_costs()
    print(f"\n  wrote {tags_path.relative_to(PROJECT_ROOT)}")
    print(f"  wrote {rows_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--n-participant", type=int, default=4)
    parser.add_argument("-c", "--n-ceiling", type=int, default=1)
    parser.add_argument("--model", default=validator.DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reasoning-effort",
        default=validator.REASONING_EFFORT,
        choices=["low", "medium", "high"],
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Suffix for the output filename, e.g. _rep2 for a repeat run.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
