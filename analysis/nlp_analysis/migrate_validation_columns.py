# Renames the validator's output columns to names that mean one thing only.
#
# The original schema called the in_image count `n_validated`. Under validator v1
# that was reasonable: the judge answered exactly one question, so "in the image"
# and "valid" were the same claim. Validator v2 splits them - a tag can be plainly
# visible and still be an extraction artefact (a shape term filed under objects, a
# colour with no entity) - and `n_validated` kept counting those. Anyone reading the
# file later would take "validated" to mean "the participant got this right", which
# it does not.
#
# So the name is retired rather than redefined. Redefining it would leave two
# different numbers behind one name, separable only by checking validator_version -
# the same ambiguity, one level deeper.
#
#   n_validated              -> n_in_image      visible in the ground-truth image
#   n_invalid_not_in_image   -> n_not_in_image  not visible
#   (new) n_valid                               visible AND right category AND said
#
# Idempotent: running twice is a no-op. v1 files get the renames only, since the
# attribution columns v_valid needs are not there to compute from.

from pathlib import Path
import argparse
import ast
import json
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
import config  # noqa: E402

CATEGORIES = ("objects", "stuff", "scene_category", "spatial_relations", "attr_color")

RENAMES = {
    "validated_tags": "in_image_tags",
    "n_validated": "n_in_image",
    "invalid_not_in_image": "not_in_image",
    "n_invalid_not_in_image": "n_not_in_image",
}
for _c in CATEGORIES:
    RENAMES[f"validated_{_c}"] = f"in_image_{_c}"
    RENAMES[f"n_validated_{_c}"] = f"n_in_image_{_c}"
    RENAMES[f"invalid_not_in_image_{_c}"] = f"not_in_image_{_c}"
    RENAMES[f"n_invalid_not_in_image_{_c}"] = f"n_not_in_image_{_c}"


def _num(frame, column):
    return pd.to_numeric(frame[column], errors="coerce") if column in frame else None


def add_valid(frame: pd.DataFrame) -> pd.DataFrame:
    """Add n_valid / valid_<cat> arrays from the per-tag verdicts.

    Derived from item_evaluations rather than by subtraction, so the arrays carry the
    actual tag text and the counts cannot drift from them.
    """
    if "item_evaluations" not in frame.columns:
        return frame
    valid_by_cat, n_valid = [], []
    for cell in frame["item_evaluations"]:
        buckets = {c: [] for c in CATEGORIES}
        total = 0
        if isinstance(cell, str) and cell.strip():
            try:
                evaluations = json.loads(cell)
            except json.JSONDecodeError:
                evaluations = ast.literal_eval(cell)
            for e in evaluations:
                if e.get("error_type") == "valid":
                    buckets.setdefault(e["category"], []).append(e["item"])
                    total += 1
        valid_by_cat.append(buckets)
        n_valid.append(total)
    frame["n_valid"] = n_valid
    frame["valid_tags"] = [json.dumps(b, ensure_ascii=False) for b in valid_by_cat]
    for c in CATEGORIES:
        frame[f"valid_{c}"] = [json.dumps(b[c], ensure_ascii=False) for b in valid_by_cat]
        frame[f"n_valid_{c}"] = [len(b[c]) for b in valid_by_cat]
    return frame


def migrate(path: Path, dry_run: bool = False) -> str:
    frame = pd.read_csv(path)
    todo = {a: b for a, b in RENAMES.items() if a in frame.columns}
    has_attribution = "n_participant_hallucination" in frame.columns
    if not todo and ("valid_tags" in frame.columns or not has_attribution):
        return "already migrated"

    frame = frame.rename(columns=todo)
    note = f"renamed {len(todo)}"
    if has_attribution:
        frame = add_valid(frame)
        # The point of the whole exercise: this identity must hold exactly.
        scored, valid = _num(frame, "n_scored"), _num(frame, "n_valid")
        ppt = _num(frame, "n_participant_hallucination")
        if scored is not None and not (scored == valid + ppt).all():
            bad = int((scored != valid + ppt).sum())
            raise SystemExit(f"{path}: n_scored != n_valid + n_participant_hallucination on {bad} row(s)")
        note += ", added n_valid (identity verified)"
    else:
        note += ", v1 file - no n_valid (no attribution columns)"
    if not dry_run:
        frame.to_csv(path, index=False)
    return note


def targets() -> list[Path]:
    out = []
    for slug in config.mapping_data["CONDITIONS"]:
        out.append(config.paths_for(slug).processed_dir / "nlp_analysis"
                   / "trials_final_semantic_tag_image_validation.csv")
    base = PROJECT_ROOT / "Data/processed_data"
    out += sorted(base.glob("Full_experiment/combined/*/[!_]*semantic_tag_image_validation*.csv"))
    out += sorted(base.glob("wilmas_drawings_2019/*/draw_*/nlp_analysis/drawing_semantic_tag_image_validation*.csv"))
    out += sorted(base.glob("wilmas_drawings_2019/combined/full_data/*.csv"))
    return [p for p in out if p.exists() and "executive_summary" not in p.name]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--paths", nargs="*", type=Path)
    args = ap.parse_args()
    for p in (args.paths or targets()):
        print(f"  {migrate(p, args.dry_run):48} {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
