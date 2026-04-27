
from pathlib import Path
import sys
import pandas as pd

# Make sure we can import config.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config  # now we can reuse your paths
import json
import pandas as pd

import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

from config import PROCESSED_DIR, GT_DIR
#measure for each condition seperately
folder_path = Path("/mnt/hdd/anatkorol/Imagination_in_translation/Data/processed_data/comparing_conditions")
condition = "gpt_descs" # "immediate_memory_with_feedback" # change this according to the condition you want to analyze, e.g. "perception_no_feedback", "gpt-5_descriptions_as_ppt", "translation_imagination"
df = pd.read_csv(folder_path / "gpt_descs.csv").copy() # "ppt_w_gpt_semantic_tags.csv").copy()
# df = df[df['uid'] != "gpt-5"]
print(f"Number of rows to process in full df: {len(df)}")
# df = df.head(2) # for testing
print(f"Number of rows to process: {len(df)}")

OUT_PATH = folder_path / "gpt_trials_w_object_validation.csv" # "ppt_trials_w_object_validation.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)  


import json
import base64
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from openai import OpenAI

from config import PROCESSED_DIR
import config

client = OpenAI()


# -----------------------------
# Structured Outputs schema
# -----------------------------
OBJECT_VALIDATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "item_evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "item": {"type": "string"},
                    "is_object": {"type": "boolean"},
                    "in_image": {"type": "boolean"},
                    "valid": {"type": "boolean"},
                },
                "required": ["item", "is_object", "in_image", "valid"],
            },
        },
        "validated_objects": {
            "type": "array",
            "items": {"type": "string"},
        },
        "invalid_not_objects": {
            "type": "array",
            "items": {"type": "string"},
        },
        "invalid_not_in_image": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "item_evaluations",
        "validated_objects",
        "invalid_not_objects",
        "invalid_not_in_image",
    ],
}

SYSTEM_PROMPT = """
You are a strict evaluator of extracted objects from a participant's description of an image.

Your task is to validate each extracted item using:
1. the participant's original description
2. the image itself

Definitions:
- A valid object is a concrete, visually identifiable entity in the image.
- Examples of objects: couch, table, rug, lamp, dog, window.
- Not objects: scene labels or room types (e.g. "living room", "kitchen"), attributes/adjectives
  (e.g. "cozy", "red"), materials (e.g. "wood"), actions, relations, or abstract concepts.

Rules:
- An item is VALID only if:
  A. it is an object/entity, and
  B. it is clearly visible in the image.
- Be conservative.
- Do not guess.
- Use the description only as supporting context, not as proof that the object exists in the image.
- If an item is not really an object, mark it invalid even if the description mentions it.
- Evaluate only the extracted items given to you. Do not add missing objects.

Return only data matching the required schema.
"""

USER_PROMPT = """
Original description:
{prompt}

Extracted objects:
{objects}
"""

def image_to_data_url(image_path: Path) -> str:
    mime = "image/jpeg"
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    elif suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def parse_objects_cell(value):
    """
    Handles cases where the objects column may already be a list,
    a dict-like JSON string, or missing.
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        return value.get("objects", [])

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []

        # First try JSON
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed.get("objects", [])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback for python-literal-like strings
        import ast
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return parsed.get("objects", [])
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    return []

def evaluate_row(row) -> dict:
    try:
        prompt = row["prompt"]
        extracted_objects = parse_objects_cell(row["objects"])

        gt_name = row["gt"]
        gt_path = Path(config.GT_DIR) / gt_name

    except Exception as e:
        return {
            "item_evaluations": None,
            "validated_objects": None,
            "invalid_not_objects": None,
            "n_extracted": None,
            "n_validated": None,
            "n_invalid_not_object": None,
            "n_invalid_not_in_image": None,
            "hallucination_count": None,
            "error": str(e),
        }

    image_data_url = image_to_data_url(gt_path)

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": SYSTEM_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": USER_PROMPT.format(
                            prompt=prompt,
                            objects=json.dumps(extracted_objects, ensure_ascii=False),
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                    },
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "object_validation",
                "schema": OBJECT_VALIDATION_SCHEMA,
                "strict": True,
            }
        },
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=2000,
    )

    parsed = json.loads(response.output_text)

    validated_objects = parsed["validated_objects"]
    invalid_not_objects = parsed["invalid_not_objects"]
    invalid_not_in_image = parsed["invalid_not_in_image"]

    n_extracted = len(extracted_objects) # length count
    n_validated = len(validated_objects)
    n_invalid_not_object = len(invalid_not_objects)
    n_invalid_not_in_image = len(invalid_not_in_image)
    hallucination_count = n_extracted - n_validated

    return {
        "item_evaluations": json.dumps(parsed["item_evaluations"], ensure_ascii=False),
        "validated_objects": json.dumps(validated_objects, ensure_ascii=False),
        "invalid_not_objects": json.dumps(invalid_not_objects, ensure_ascii=False),
        "invalid_not_in_image": json.dumps(invalid_not_in_image, ensure_ascii=False),

        "n_extracted": n_extracted,
        "n_validated": n_validated,
        "n_invalid_not_object": n_invalid_not_object,
        "n_invalid_not_in_image": n_invalid_not_in_image,
        "diff_in_objects": n_extracted - n_validated,

        "error": None,
    }

# -----------------------------
# Run evaluation on every row
# -----------------------------
tqdm.pandas()
eval_results = df.progress_apply(evaluate_row, axis=1)
eval_df = pd.DataFrame(list(eval_results), index=df.index)

final_df = pd.concat([df, eval_df], axis=1)

# -----------------------------
# Save full CSV
# -----------------------------
final_df.to_csv(OUT_PATH, index=False)
print(f"Saved validated data to {OUT_PATH}")

# -----------------------------
# Executive summary - for the whole csv
# -----------------------------
summary_df = final_df.copy()
print(f"\n=== FULL DATA SUMMARY for {condition}===")
# Coerce numeric columns in case some rows failed and produced None / strings
numeric_cols = [
    "n_extracted",
    "n_validated",
    "n_invalid_not_object",
    "n_invalid_not_in_image",
    "diff_in_objects",
]
for col in numeric_cols:
    summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

n_rows = len(summary_df)
n_error_rows = summary_df["error"].notna().sum() if "error" in summary_df.columns else 0
n_success_rows = n_rows - n_error_rows

total_extracted = summary_df["n_extracted"].sum(skipna=True)
total_validated = summary_df["n_validated"].sum(skipna=True)
total_semantic_errors = summary_df["n_invalid_not_object"].sum(skipna=True)
total_hallucinations = summary_df["n_invalid_not_in_image"].sum(skipna=True)
total_invalid_combined = summary_df["diff_in_objects"].sum(skipna=True)
# Safe percentage helper
def pct(num, den):
    return (100 * num / den) if den and den > 0 else 0.0

semantic_error_pct = pct(total_semantic_errors, total_extracted)
hallucination_pct = pct(total_hallucinations, total_extracted)
valid_pct = pct(total_validated, total_extracted)
combined_invalid_pct = pct(total_invalid_combined, total_extracted)

rows_with_hallucination = (summary_df["n_invalid_not_in_image"].fillna(0) > 0).sum()
rows_with_semantic_error = (summary_df["n_invalid_not_object"].fillna(0) > 0).sum()
rows_with_any_invalid = (summary_df["diff_in_objects"].fillna(0) > 0).sum()

avg_extracted_per_row = summary_df["n_extracted"].mean(skipna=True)
avg_validated_per_row = summary_df["n_validated"].mean(skipna=True)
avg_semantic_errors_per_row = summary_df["n_invalid_not_object"].mean(skipna=True)
avg_hallucinations_per_row = summary_df["n_invalid_not_in_image"].mean(skipna=True)

executive_summary = {
    "n_rows": int(n_rows),
    "n_success_rows": int(n_success_rows),
    "n_error_rows": int(n_error_rows),

    "total_extracted_objects": int(total_extracted),
    "total_validated_objects": int(total_validated),

    "total_semantic_errors": int(total_semantic_errors),
    "semantic_error_pct_of_all_extracted_objects": round(semantic_error_pct, 2),

    "total_hallucinations": int(total_hallucinations),
    "hallucination_pct_of_all_extracted_objects": round(hallucination_pct, 2),

    "total_invalid_combined": int(total_invalid_combined),
    "combined_invalid_pct_of_all_extracted_objects": round(combined_invalid_pct, 2),

    "valid_object_pct_of_all_extracted_objects": round(valid_pct, 2),

    "rows_with_at_least_one_hallucination": int(rows_with_hallucination),
    "rows_with_at_least_one_semantic_error": int(rows_with_semantic_error),
    "rows_with_at_least_one_invalid_item": int(rows_with_any_invalid),

    "avg_extracted_objects_per_row": round(avg_extracted_per_row, 3) if pd.notna(avg_extracted_per_row) else None,
    "avg_validated_objects_per_row": round(avg_validated_per_row, 3) if pd.notna(avg_validated_per_row) else None,
    "avg_semantic_errors_per_row": round(avg_semantic_errors_per_row, 3) if pd.notna(avg_semantic_errors_per_row) else None,
    "avg_hallucinations_per_row": round(avg_hallucinations_per_row, 3) if pd.notna(avg_hallucinations_per_row) else None,
}

print("\n=== EXECUTIVE SUMMARY ===")
for k, v in executive_summary.items():
    print(f"{k}: {v}")

# Save summary as a one-row CSV
summary_out_path = OUT_PATH.with_name(OUT_PATH.stem + "_executive_summary.csv")
pd.DataFrame([executive_summary]).to_csv(summary_out_path, index=False)
print(f"Saved executive summary to {summary_out_path}")