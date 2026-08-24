# Compares validator pilot runs that judged the same rows under different settings.
#
# The point of the repeat pair (same model, same effort, same prompt) is that it
# measures the judge's own run-to-run variance. Any difference between two
# different settings that is no larger than that floor is not evidence of an
# effect - it is the same noise measured twice. Every contrast below is therefore
# reported next to the floor rather than on its own.

from pathlib import Path
import argparse
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "analysis/outputs/experiments/validator"
KEY = ["gt", "uid", "session", "attempt", "category", "item"]
ERROR_TYPES = ["valid", "category_misplacement", "model_hallucination",
               "participant_hallucination"]


def load(tag: str) -> pd.DataFrame:
    """Newest pilot_tags file whose slug carries this run tag."""
    hits = sorted(OUT_DIR.glob(f"pilot_tags_*{tag}_*.csv"), key=lambda p: p.stat().st_mtime)
    if not hits:
        raise SystemExit(f"No pilot_tags file for tag {tag!r} in {OUT_DIR}")
    d = pd.read_csv(hits[-1])
    for c in ["uid", "session", "attempt"]:
        d[c] = d[c].astype(str)
    d.attrs["path"] = hits[-1].name
    # Duplicate tag text within one row+category would make the join ambiguous.
    return d.drop_duplicates(subset=KEY)


def pair(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a.merge(b, on=KEY, suffixes=("_a", "_b"))


def contrast(name: str, a: pd.DataFrame, b: pd.DataFrame) -> dict:
    m = pair(a, b)
    row = {"contrast": name, "n_tags": len(m)}
    row["error_type_disagree_%"] = round(100 * (m.error_type_a != m.error_type_b).mean(), 2)
    row["in_image_disagree_%"] = round(100 * (m.new_in_image_a != m.new_in_image_b).mean(), 2)
    for field in ["category_correct", "grounded_in_description"]:
        row[f"{field}_disagree_%"] = round(100 * (m[f"{field}_a"] != m[f"{field}_b"]).mean(), 2)
    for et in ERROR_TYPES[1:]:
        n_a = int((m.error_type_a == et).sum())
        n_b = int((m.error_type_b == et).sum())
        both = int(((m.error_type_a == et) & (m.error_type_b == et)).sum())
        union = n_a + n_b - both
        # Jaccard: of every tag either run flagged, how many did both flag?
        row[f"{et}_agree_%"] = round(100 * both / union, 1) if union else None
        row[f"{et}_n"] = f"{n_a}->{n_b}"
    return row


def totals(label: str, d: pd.DataFrame) -> dict:
    scored = (~d.error_type.isin(["category_misplacement", "model_hallucination"])).sum()
    ppt = int((d.error_type == "participant_hallucination").sum())
    out = {"run": label, "file": d.attrs["path"][-28:], "tags": len(d)}
    for et in ERROR_TYPES:
        out[et] = int((d.error_type == et).sum())
    out["corrected_rate_%"] = round(100 * ppt / scored, 2) if scored else None
    out["low_med_conf_%"] = round(100 * (d.confidence != "high").mean(), 1)
    c = d[d.source == "gpt_ceiling"]
    if len(c):
        c_scored = (~c.error_type.isin(["category_misplacement", "model_hallucination"])).sum()
        out["ceiling_ppt_rate_%"] = round(
            100 * (c.error_type == "participant_hallucination").sum() / c_scored, 2
        ) if c_scored else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=["_A", "_B", "_C", "_D"])
    ap.add_argument("--labels", nargs="+",
                    default=["A mini/med", "B mini/high", "C mini/high rep", "D gpt-5.5/med"])
    ap.add_argument("--floor", nargs=2, default=["_B", "_C"],
                    help="The identical-config pair defining the noise floor.")
    args = ap.parse_args()

    data = {t: load(t) for t in args.runs}
    labels = dict(zip(args.runs, args.labels))

    print("=" * 100)
    print("PER-RUN TOTALS")
    print("=" * 100)
    print(pd.DataFrame([totals(labels[t], d) for t, d in data.items()]).to_string(index=False))

    fa, fb = args.floor
    floor = contrast(f"FLOOR  {labels[fa]} vs {labels[fb]}", data[fa], data[fb])
    others = [contrast(f"{labels[a]} vs {labels[b]}", data[a], data[b])
              for a, b in [("_A", "_B"), ("_A", "_D"), ("_B", "_D")]
              if a in data and b in data]

    print("\n" + "=" * 100)
    print("CONTRASTS - read every row against the FLOOR row")
    print("=" * 100)
    cols = ["contrast", "n_tags", "error_type_disagree_%", "in_image_disagree_%",
            "category_correct_disagree_%", "grounded_in_description_disagree_%"]
    print(pd.DataFrame([floor] + others)[cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("PER-ERROR-TYPE REPRODUCIBILITY  (Jaccard: both runs flagged / either flagged)")
    print("=" * 100)
    cols = ["contrast"] + [f"{et}_{s}" for et in ERROR_TYPES[1:] for s in ("agree_%", "n")]
    print(pd.DataFrame([floor] + others)[cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    f = floor["error_type_disagree_%"]
    print(f"  Noise floor ({labels[fa]} vs {labels[fb]}, identical settings): {f}% of tags differ.")
    for o in others:
        d = o["error_type_disagree_%"]
        call = "WITHIN NOISE - no evidence of a real effect" if d <= f else f"exceeds floor by {round(d - f, 2)} pp"
        print(f"  {o['contrast']:34} {d:5.2f}%  -> {call}")


if __name__ == "__main__":
    main()
