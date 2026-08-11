"""
Export combined data across all 9 conditions (3 generation types x 3 delays)
for analysis (e.g. LMM in R).

Produces combined CSVs under
Data/processed_data/Full_experiment/combined/full_data/:
  - trials_final_sim_full_data.csv
  - trials_final_semantic_tag_image_validation_full_data.csv
  - all_digit_span_full_data.csv

Trial-level files keep their original columns plus `generation` and `delay`.
Digit span is only collected in the `del` conditions, so it is combined
across generation types (with a `generation` column) only.
"""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent / "Data" / "processed_data" / "Full_experiment"
OUT_DIR = BASE / "combined" / "full_data"

GENERATIONS = ["aigen", "nogen", "plain"]
DELAYS = ["imm", "perc", "del"]

# ── Wilma's 2019 drawings ─────────────────────────────────────────────────────
# The drawn counterpart of the verbal conditions: transcribed by
# drawings_descriptions.py, tagged by the same tagger and judged by the same image
# validator, so its rows carry the same validation columns as the trial files above.
# `generation` is "drawing" (the response was drawn, not typed) and `delay` reuses the
# same three values, so an analysis can concat this with the trials file and split on
# `generation`.
DRAW_BASE = Path(__file__).resolve().parent.parent / "Data" / "processed_data" / "wilmas_drawings_2019"
DRAW_OUT_DIR = DRAW_BASE / "combined" / "full_data"

# One tree per hint arm; each is exported to its own csv, never merged, because the
# same drawing appears in both and only the arm distinguishes the two rows.
DRAW_ARMS = ["no_hint", "with_hint"]
DRAW_CONDITIONS = {"draw_perc": "perc", "draw_imm": "imm", "draw_del": "del"}
DRAW_VALIDATION_GLOB = "nlp_analysis/drawing_semantic_tag_image_validation_*.csv"

# (relative path within a condition dir, output filename)
FILES = [
    ("trials_final_sim.csv", "trials_final_sim_full_data.csv"),
    (
        "nlp_analysis/trials_final_semantic_tag_image_validation.csv",
        "trials_final_semantic_tag_image_validation_full_data.csv",
    ),
]


def combine(rel_path: str, out_name: str) -> None:
    frames = []
    for gen in GENERATIONS:
        for delay in DELAYS:
            fp = BASE / gen / delay / rel_path
            if not fp.exists():
                print(f"  [MISSING] {fp}")
                continue
            df = pd.read_csv(fp)
            df.insert(0, "delay", delay)
            df.insert(0, "generation", gen)
            frames.append(df)
            print(f"  [OK] {gen}/{delay}: {len(df)} rows")

    if not frames:
        print(f"  No data found for {rel_path}")
        return

    combined = pd.concat(frames, ignore_index=True)
    out_fp = OUT_DIR / out_name
    combined.to_csv(out_fp, index=False)
    print(f"  -> wrote {len(combined)} rows to {out_fp}\n")


def combine_digit_span(out_name: str = "all_digit_span_full_data.csv") -> None:
    # Digit span is only collected in the delayed ("del") conditions.
    frames = []
    for gen in GENERATIONS:
        fp = BASE / gen / "del" / "all_digit_span.csv"
        if not fp.exists():
            print(f"  [MISSING] {fp}")
            continue
        df = pd.read_csv(fp)
        df.insert(0, "generation", gen)
        frames.append(df)
        print(f"  [OK] {gen}/del: {len(df)} rows")

    if not frames:
        print("  No digit span data found")
        return

    combined = pd.concat(frames, ignore_index=True)
    out_fp = OUT_DIR / out_name
    combined.to_csv(out_fp, index=False)
    print(f"  -> wrote {len(combined)} rows to {out_fp}\n")


def combine_drawings() -> None:
    """One csv per hint arm, combining the three drawn conditions.

    Skips the executive-summary files that sit beside the validation csvs, and skips
    an arm entirely if it has not been produced yet.
    """
    for arm in DRAW_ARMS:
        frames = []
        for condition, delay in DRAW_CONDITIONS.items():
            matches = [
                path for path in sorted((DRAW_BASE / arm / condition).glob(DRAW_VALIDATION_GLOB))
                if not path.stem.endswith("_executive_summary")
            ]
            if not matches:
                print(f"  [MISSING] {arm}/{condition}: no validation csv")
                continue
            if len(matches) > 1:
                # Different scope/effort stamps are different runs and must not be
                # averaged together silently.
                print(f"  [WARN] {arm}/{condition}: {len(matches)} validation csvs, "
                      f"using all: {[p.name for p in matches]}")
            for path in matches:
                df = pd.read_csv(path)
                df.insert(0, "delay", delay)
                df.insert(0, "generation", "drawing")
                df.insert(0, "arm", arm)
                frames.append(df)
                print(f"  [OK] {arm}/{condition}: {len(df)} rows ({path.name})")

        if not frames:
            print(f"  No validated drawings found for arm '{arm}'\n")
            continue

        combined = pd.concat(frames, ignore_index=True)
        DRAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_fp = DRAW_OUT_DIR / f"drawing_semantic_tag_image_validation_full_data_{arm}.csv"
        combined.to_csv(out_fp, index=False)
        print(f"  -> wrote {len(combined)} rows to {out_fp}\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for rel_path, out_name in FILES:
        print(f"Combining {rel_path} ...")
        combine(rel_path, out_name)
    print("Combining all_digit_span.csv ...")
    combine_digit_span()
    print("Combining Wilma 2019 drawing validations ...")
    combine_drawings()


if __name__ == "__main__":
    main()
