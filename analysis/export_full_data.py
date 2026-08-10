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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for rel_path, out_name in FILES:
        print(f"Combining {rel_path} ...")
        combine(rel_path, out_name)
    print("Combining all_digit_span.csv ...")
    combine_digit_span()


if __name__ == "__main__":
    main()
