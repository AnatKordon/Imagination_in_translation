from pathlib import Path
import sys
import os
import json
import pandas as pd
from tqdm.auto import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# --------------------
# Setup
# --------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config
from config import PROCESSED_DIR

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_PATH = PROCESSED_DIR / "ppt_w_gpt_w_similarity_trials.csv" # or: "ppt_trials_w_similarity_trials.csv"

# Use a small sample first.
# Set to None only when you intentionally want to run the full dataset.
DRY_RUN_N = 20

# IMPORTANT:
# This controls whether we save a CSV.
# Keep False while measuring cost.
SAVE_OUTPUT = False

MODEL = "gpt-5"  # or "gpt-4o" if you intentionally want the larger model

SYSTEM_PROMPT = """
You are a STRICT semantic tagger for text descriptions of images.
Your job is to STRUCTURE ONLY what the text explicitly states about the imagined image.
Do NOT add likely objects, inferred details, or common-sense fill-ins.

Return ONLY valid JSON with exactly the keys requested. No prose.

Long-text rule: read the entire prompt before producing JSON.

Rules:
- Use lowercase for objects; singular nouns.
- Put multiword attributes as phrases (e.g., "paint chipped", "rusty metal").
- If a category is not present, return an empty list ([]) or null (for single values).
- Keep outputs concise and deduplicated.
"""

USER_PROMPT = """
Extract the following fields from the PROMPT below.

PROMPT: "{PROMPT}"

Return ONLY a JSON object with these keys and types:

{{
  "objects": ["All primary and secondary objects that are explicitly mentioned in description"],

  "attr_color": ["..."],
  "attr_shape": ["..."],
  "attr_size": ["..."],
  "attr_material": ["..."],
  "attr_texture": ["..."],
  "attr_pose": ["..."],
  "attr_action": ["..."],
  "attr_state": ["..."],

  "spatial_relations": ["relation statements like 'on top of', 'underneath', 'next to' or clear frame position words (e.g., 'top right')],

  "world_knowledge": ["mentions for named entities, e.g., 'big apple', 'george clooney'"],

  "scene": ["mentions capturing any scene setting, indoor/outdoor, time of day, weather"],

  "camera_aspects": ["mentions capturing camera angle, shot size, viewpoint, depth of field"],

  "optical_effects": ["mentions capturing any mentioned optical effects"],

  "subjective_detail": ["personal interpretations that appear in text / vibes / aesthetic judgments / speculation (not objective facts)"]
}}

Important constraints:
- Each attribute list should contain attribute phrases found in the prompt (not attached to objects).
- spatial should include only explicit spatial/positional phrases.
- If the prompt is long, scan the entire text before answering.
- If something is unknown, use null or an empty list (do not guess).
- Keep lists deduplicated.
"""


def usage_to_dict(usage):
    """
    Convert OpenAI usage object to a plain dict robustly.
    """
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


def extract_semantics_with_usage(prompt: str):
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(PROMPT=prompt)},
        ],
        text={"format": {"type": "json_object"}},
        # temperature=0.0,
        # top_p=1.0,
        max_output_tokens=5000,
    )

    extraction = json.loads(resp.output_text)
    usage = usage_to_dict(resp.usage)

    return extraction, usage


def print_usage_report(usage_rows, full_n_rows=None):
    usage_df = pd.DataFrame(usage_rows)

    tested_n = len(usage_df)

    total_input = int(usage_df["input_tokens"].sum())
    total_output = int(usage_df["output_tokens"].sum())
    total_tokens = int(usage_df["total_tokens"].sum())

    avg_input = usage_df["input_tokens"].mean()
    avg_output = usage_df["output_tokens"].mean()
    avg_total = usage_df["total_tokens"].mean()

    print("\n" + "=" * 60)
    print("TOKEN USAGE REPORT")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Rows tested: {tested_n}")
    print()
    print("Total tokens in test:")
    print(f"  Input tokens:  {total_input:,}")
    print(f"  Output tokens: {total_output:,}")
    print(f"  Total tokens:  {total_tokens:,}")
    print()
    print("Average per row:")
    print(f"  Input tokens:  {avg_input:,.1f}")
    print(f"  Output tokens: {avg_output:,.1f}")
    print(f"  Total tokens:  {avg_total:,.1f}")

    if full_n_rows is not None and tested_n > 0:
        est_input = avg_input * full_n_rows
        est_output = avg_output * full_n_rows
        est_total = avg_total * full_n_rows

        print()
        print(f"Projected usage for full dataset ({full_n_rows:,} rows):")
        print(f"  Input tokens:  {est_input:,.0f}")
        print(f"  Output tokens: {est_output:,.0f}")
        print(f"  Total tokens:  {est_total:,.0f}")

    print("=" * 60 + "\n")


def main():
    df = pd.read_csv(DATA_PATH).copy()
    full_n_rows = len(df)

    print(f"Full dataset rows: {full_n_rows:,}")

    if DRY_RUN_N is not None:
        df_run = df.head(DRY_RUN_N).copy()
        print(f"DRY RUN: only processing first {len(df_run):,} rows.")
    else:
        df_run = df.copy()
        print("FULL RUN: processing all rows.")

    usage_rows = []
    extractions = []

    for idx, row in tqdm(df_run.iterrows(), total=len(df_run)):
        prompt = str(row["prompt"])

        try:
            extraction, usage = extract_semantics_with_usage(prompt)

            usage_rows.append({
                "idx": idx,
                **usage,
            })

            # Keep in memory only. This does not modify your original CSV.
            extractions.append(extraction)

        except Exception as e:
            print(f"Error at index {idx}: {e}")
            continue

    print_usage_report(usage_rows, full_n_rows=full_n_rows)

    if SAVE_OUTPUT:
        OUT_PATH = PROCESSED_DIR / "nlp_analysis" / "ppt_trials_w_similarity_semantic_tags.csv"
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        out = pd.json_normalize(extractions)
        out.index = df_run.index[:len(out)]

        tagged_df = pd.concat([df_run.iloc[:len(out)].copy(), out], axis=1)
        tagged_df.to_csv(OUT_PATH, index=False)

        print(f"Saved tagged data to {OUT_PATH}")
    else:
        print("SAVE_OUTPUT=False, so no CSV was written.")


if __name__ == "__main__":
    main()