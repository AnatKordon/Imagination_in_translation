# Flattens the validator output to one row per tag.
#
# The validator writes one row per trial with every per-tag verdict packed into
# the `item_evaluations` JSON column, because that keeps the row aligned with the
# trial it came from. That shape is awkward to read or filter by hand, so this
# writes the same information one tag per row.

from pathlib import Path
import argparse
import json
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
import config  # noqa: E402

TRIAL_COLS = ["generation", "delay", "uid", "gt", "session", "attempt", "prompt",
              "subjective_score", "feedback_id"]
OUT = PROJECT_ROOT / "Data/processed_data/Full_experiment/combined/full_data"

# Column order for the readable per-condition files: what the tag is, what the
# judge decided, and why - before the identifiers. Reading left to right should
# answer "is this tag wrong, whose fault, and how sure was the judge" without
# scrolling.
READABLE_COLS = [
    "gt", "tag", "category", "error_type", "confidence", "note",
    "in_image", "category_correct", "grounded_in_description",
    "prompt", "uid", "session", "attempt", "condition", "generation", "delay",
    "subjective_score", "feedback_id",
]


def flatten(slug: str) -> pd.DataFrame:
    path = config.paths_for(slug).processed_dir / "nlp_analysis" / \
        "trials_final_semantic_tag_image_validation.csv"
    d = pd.read_csv(path)
    gen, task = slug.split("_")
    rows = []
    for _, r in d.iterrows():
        if pd.isna(r.get("item_evaluations")):
            continue
        head = {c: r.get(c) for c in TRIAL_COLS if c in d.columns}
        head.setdefault("generation", gen)
        head.setdefault("delay", task)
        head["condition"] = slug
        for e in json.loads(r["item_evaluations"]):
            rows.append({**head,
                         "category": e["category"], "tag": e["item"],
                         "error_type": e["error_type"],
                         "in_image": e["in_image"],
                         "category_correct": e["category_correct"],
                         "grounded_in_description": e["grounded_in_description"],
                         "confidence": e["confidence"], "note": e["note"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conditions", nargs="*", default=list(config.mapping_data["CONDITIONS"]))
    args = ap.parse_args()
    frames = []
    for slug in args.conditions:
        t = flatten(slug)
        frames.append(t)
        # Readable per-condition copy, sorted so the tags needing a human eye
        # come first: errors before valid, least confident first within those.
        readable = t.copy()
        readable["_err"] = (readable["error_type"] != "valid").astype(int)
        readable["_conf"] = readable["confidence"].map({"low": 0, "medium": 1, "high": 2})
        readable = readable.sort_values(
            ["_err", "_conf", "gt", "category", "tag"],
            ascending=[False, True, True, True, True],
        ).drop(columns=["_err", "_conf"])
        cols = [c for c in READABLE_COLS if c in readable.columns]
        readable = readable[cols + [c for c in readable.columns if c not in cols]]
        out_dir = config.paths_for(slug).processed_dir / "nlp_analysis"
        rp = out_dir / "validation_per_tag_readable.csv"
        readable.to_csv(rp, index=False)
        n_err = int((t["error_type"] != "valid").sum())
        print(f"  {slug:11} {len(t):6} tags  ({n_err} non-valid first)  -> {rp.relative_to(PROJECT_ROOT)}")

    all_tags = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "validation_per_tag_full_data.csv"
    all_tags.to_csv(p, index=False)
    print(f"\ncombined: {len(all_tags)} tags -> {p.relative_to(PROJECT_ROOT)}")
    print("\nverdicts:")
    print(all_tags["error_type"].value_counts().to_string())
    print("\nconfidence:")
    print(all_tags["confidence"].value_counts().to_string())
    print(f"\nrows carrying a note: {int((all_tags['note'].fillna('') != '').sum())}")


if __name__ == "__main__":
    main()
