
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
#measure for each condition seperately (set via config.CURRENT_CONDITION)
folder_path = PROCESSED_DIR / "nlp_analysis"
df = pd.read_csv(folder_path / "semantic_tags.csv").copy()
df = df[df['uid'] != "gpt-5"]
print(f"Number of rows to process in full df: {len(df)}")
# df = df.head(2) # for testing
print(f"Number of rows to process: {len(df)}")

OUT_PATH = folder_path / "object_validation.csv"
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
                    "is_object": {
                        "type": "boolean",
                        "description": "Semantic test judged from the item text alone (ignore the image): is it a concrete, countable, bounded, visually depictable object per the definition? False for stuff, scene/room labels, colors/attributes/materials, actions, relations, abstractions.",
                    },
                    "in_image": {
                        "type": "boolean",
                        "description": "Visual test judged from the image alone: is that object clearly, actually visible? Only meaningful when is_object is true. Be strict on presence and set false when uncertain. If is_object is false, set in_image to false.",
                    },
                    "valid": {
                        "type": "boolean",
                        "description": "True iff is_object AND in_image.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One short sentence justifying is_object and in_image for this item.",
                    },
                },
                "required": ["item", "is_object", "in_image", "valid", "reason"],
            },
        },
        "validated_objects": {
            "type": "array",
            "description": "Items with valid == true (is_object AND in_image).",
            "items": {"type": "string"},
        },
        "invalid_not_objects": {
            "type": "array",
            "description": "Items with is_object == false (tagger/semantic errors). Checked first; such items never appear in invalid_not_in_image.",
            "items": {"type": "string"},
        },
        "invalid_not_in_image": {
            "type": "array",
            "description": "Items with is_object == true but in_image == false (participant hallucinations / false memories).",
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
You are a STRICT evaluator of objects that were extracted from a participant's description of an image.

For EACH extracted item you must decide two INDEPENDENT things, then route the item:
1. is_object — is the item really an object? (a semantic test)
2. in_image  — is that object actually visible in the image? (a visual test)

You are given the participant's original description and the ground-truth image. Evaluate ONLY the
items provided. Do NOT add, infer, or invent objects that are not in the given list.

--------------------------------------------------------------------------------
OBJECT DEFINITION (must match the semantic tagger that produced these items)
--------------------------------------------------------------------------------
An OBJECT is a concrete, countable, bounded, visually depictable entity: living beings, people,
animals, plants, artifacts, furniture, vehicles, decorations, and structural / room features.

INCLUDES:
- Room and architectural features when named, e.g. wall, floor, ceiling, rug, carpet, door,
  window, staircase, counter, shelf, cabinet, fireplace.
- Explicitly named object parts, e.g. "table leg", "door handle".
- Do NOT split ordinary compound object names (e.g. "coffee table", "ceiling fan") — they are one object.

NOT objects (these make is_object = false):
- Stuff / amorphous substances and atmosphere: sky, water, sand, snow, smoke, fog, shadow, light,
  clouds, rain, dust, steam, grass, dirt, mud.
- Scene / room labels: living room, bedroom, kitchen, office, bathroom, beach, forest, city,
  street, indoors, outdoors, "scene", "setting".
- Bare attributes: colors, sizes, shapes, materials, textures, styles (red, large, round, wooden, modern).
- Actions, spatial relations, and abstract concepts (happiness, beauty, loneliness).

NOTE: the items you receive were already filtered by the tagger to be objects, so is_object = false
should be RARE and reserved for genuine tagger mistakes.

--------------------------------------------------------------------------------
THE TWO TESTS
--------------------------------------------------------------------------------
is_object — judge from the ITEM TEXT ALONE, independent of the image. Apply the definition above.

in_image — judge from the IMAGE ALONE. Is the object clearly, actually visible?
- The description is only context; it is NOT proof that the object is in the image.
- Be lenient on wording: a synonym or near-synonym of a clearly visible entity counts as present.
- Be STRICT on presence: do NOT credit objects that are merely implied, occluded, plausible, or
  "probably there". If you are uncertain whether the object is really visible, set in_image = false.
- in_image is only meaningful when is_object is true. If is_object is false, set in_image = false.

valid = is_object AND in_image.

--------------------------------------------------------------------------------
ROUTING (every item goes into EXACTLY ONE array — check is_object FIRST)
--------------------------------------------------------------------------------
1. is_object = false                    -> invalid_not_objects   (a tagger / semantic error)
2. is_object = true  and in_image=false -> invalid_not_in_image  (a participant HALLUCINATION / false memory)
3. is_object = true  and in_image=true  -> validated_objects

An item that fails the object test goes ONLY to invalid_not_objects; never also to invalid_not_in_image.

INVARIANT: len(validated_objects) + len(invalid_not_objects) + len(invalid_not_in_image) must equal
the number of extracted items, and no item may appear in more than one array.

Return only data matching the required schema, including a short `reason` per item.
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