# Reorganizes the per-model experiment CSVs (semantic_tags__<model>.csv) into ONE
# wide CSV grouped BY FIELD, so each field's results from all models sit next to
# each other (objects[nano] | objects[mini] | objects[5.5], then stuff[...], ...).
# Goal: eyeball the same prompt across models and spot wrong/missing/hallucinated tags.

from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "analysis" / "outputs" / "experiments" / "semantic_tagging_model"
OUT_PATH = EXPERIMENT_DIR / "comparison_by_field.csv"

FIELDS = ["objects", "stuff", "spatial_relations", "attr_color"]
# Preferred left-to-right model order; anything else is appended alphabetically.
MODEL_ORDER = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.5"]


def model_name(df: pd.DataFrame, path: Path) -> str:
    """Prefer the tagger_model column; fall back to the filename suffix."""
    if "tagger_model" in df.columns and df["tagger_model"].notna().any():
        return str(df["tagger_model"].dropna().iloc[0])
    return path.stem.replace("semantic_tags__", "")


def field_lists(df: pd.DataFrame) -> dict[str, list[list[str]]]:
    """Pull each field as a list-of-lists, parsing the clean JSON `extraction` column."""
    parsed = df["extraction"].apply(json.loads)
    return {f: [p.get(f, []) for p in parsed] for f in FIELDS}


def main() -> None:
    files = sorted(EXPERIMENT_DIR.glob("semantic_tags__*.csv"))
    if not files:
        raise SystemExit(f"No experiment CSVs found in {EXPERIMENT_DIR}")

    frames = {}
    prompts = None
    for path in files:
        df = pd.read_csv(path)
        name = model_name(df, path)
        frames[name] = df
        if prompts is None:
            prompts = df["prompt"].tolist()
        elif df["prompt"].tolist() != prompts:
            # Same seed/sample should guarantee identical order; warn if it ever drifts.
            print(f"  [warn] {name}: prompts differ from the first file's order")

    models = [m for m in MODEL_ORDER if m in frames] + \
             sorted(m for m in frames if m not in MODEL_ORDER)

    per_model_fields = {m: field_lists(frames[m]) for m in models}

    out = pd.DataFrame({"prompt": prompts})
    # Group BY FIELD so all models for a field are adjacent columns.
    for field in FIELDS:
        for m in models:
            col = f"{field} [{m}]"
            out[col] = [", ".join(tags) for tags in per_model_fields[m][field]]

    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out)} rows x {out.shape[1]} cols to {OUT_PATH}")
    print(f"Models (left->right): {models}")


if __name__ == "__main__":
    main()
