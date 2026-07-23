
# Validates the objects extracted by semantic_tagging.py against the ground-truth image.
# For each row the LLM returns one JSON object (stable schema, low ambiguity) and the code
# expands it into adjacent dataframe columns, exactly like semantic_tagging.py.

from pathlib import Path
import sys
import base64
import json
import os
import argparse
from collections import defaultdict

import pandas as pd
from tqdm.auto import tqdm

# Make sure we can import config.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config  # now we can reuse your paths

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ── Input / output ────────────────────────────────────────────────────────────
# Input is the per-condition tag table written by semantic_tagging.py; output lands
# beside it under <condition>/nlp_analysis/.
TAGS_FILENAME = "trials_final_semantic_tags.csv"
OUT_FILENAME = "trials_final_object_validation.csv"

# Rows whose uid is a synthetic/model run rather than a participant.
EXCLUDE_UIDS = {"gpt-5"}

# ── Default model for the full run ────────────────────────────────────────────
DEFAULT_MODEL = "gpt-4.1"  # vision judge; must accept image input.

# ── Resumable full run ────────────────────────────────────────────────────────
# The full run is resumable: rows already validated in OUT_PATH are skipped, so
# re-running continues where a previous run stopped instead of re-judging (and
# re-billing) everything. Rows are matched on KEY_COLS (a unique key per trial).
# Rows that previously errored are NOT treated as done and get retried. Progress
# is written every CHECKPOINT_EVERY rows, so an interruption (crash / Ctrl-C)
# keeps the work done so far. Set MAX_NEW_ROWS to an int to validate only that
# many new rows this run (then resume later); leave None to do all remaining rows.
KEY_COLS = ["uid", "session", "attempt"]
CHECKPOINT_EVERY = 25
MAX_NEW_ROWS = None  # limit new rows for testing

# ── Token usage & cost tracking ───────────────────────────────────────────────
# Exact token counts come from resp.usage (free, from the API). The dollar figure
# is an ESTIMATE = tokens x your account's price (USD per 1M tokens). Image tokens
# are already folded into input_tokens by the API.
USAGE = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "calls": 0})

PRICING = {  # USD per 1,000,000 tokens
    "gpt-4o-mini":  {"input": 0.15, "output": 0.60},
    "gpt-4o":       {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1":      {"input": 2.00, "output": 8.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4":      {"input": 2.50, "output": 15.00},
    "gpt-5.5":      {"input": 5.00, "output": 30.00},
}

PRINT_USAGE_PER_PROMPT = False  # per-row token breakdown (noisy on a full run)


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

# Columns the LLM step adds to each row (also the shape of a failed row).
RESULT_COLS = [
    "item_evaluations",
    "validated_objects",
    "invalid_not_objects",
    "invalid_not_in_image",
    "n_extracted",
    "n_validated",
    "n_invalid_not_object",
    "n_invalid_not_in_image",
    "diff_in_objects",
    "error",
]

NUMERIC_COLS = [
    "n_extracted",
    "n_validated",
    "n_invalid_not_object",
    "n_invalid_not_in_image",
    "diff_in_objects",
]


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
    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        return value.get("objects", [])

    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return []

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


def _accumulate_usage(model, *, input_tokens=0, output_tokens=0, reasoning_tokens=0) -> None:
    acc = USAGE[model]
    acc["input_tokens"] += input_tokens or 0
    acc["output_tokens"] += output_tokens or 0
    acc["reasoning_tokens"] += reasoning_tokens or 0
    acc["calls"] += 1


def _track_usage(resp, model, prompt) -> None:
    u = getattr(resp, "usage", None)
    if u is None:
        return
    out_details = getattr(u, "output_tokens_details", None)
    input_tokens = getattr(u, "input_tokens", 0) or 0
    output_tokens = getattr(u, "output_tokens", 0) or 0
    total_tokens = getattr(u, "total_tokens", input_tokens + output_tokens) or 0
    reasoning_tokens = getattr(out_details, "reasoning_tokens", 0) if out_details else 0
    _accumulate_usage(model, input_tokens=input_tokens, output_tokens=output_tokens,
                      reasoning_tokens=reasoning_tokens)
    if PRINT_USAGE_PER_PROMPT:
        preview = " ".join(str(prompt).split())[:90]
        print(f"  [{model}] in={input_tokens:,} out={output_tokens:,} total={total_tokens:,} "
              f"reasoning={reasoning_tokens:,}\n      «{preview}…»")


def print_usage_costs() -> None:
    """Print accumulated tokens per model and a dollar estimate where PRICING is set."""
    print("\n=== Token usage & estimated cost per model ===")
    for model, acc in USAGE.items():
        line = (f"{model}: {acc['calls']} calls | "
                f"in={acc['input_tokens']:,} out={acc['output_tokens']:,} "
                f"(reasoning={acc['reasoning_tokens']:,})")
        price = PRICING.get(model) or {}
        if price.get("input") is not None and price.get("output") is not None:
            cost = (acc["input_tokens"] / 1e6 * price["input"]
                    + acc["output_tokens"] / 1e6 * price["output"])
            line += f" | est. ${cost:.4f}"
        else:
            line += "  | est. $? (set PRICING for this model)"
        print(line)


def _failed(msg: str) -> dict:
    """A result record for a row we could not judge (kept so it can be retried)."""
    rec = {c: None for c in RESULT_COLS}
    rec["error"] = msg
    return rec


def evaluate_row(row, model: str = DEFAULT_MODEL) -> dict:
    try:
        prompt = row["prompt"]
        extracted_objects = parse_objects_cell(row["objects"])

        gt_name = row["gt"]
        gt_path = Path(config.GT_DIR) / gt_name
        image_data_url = image_to_data_url(gt_path)
    except Exception as e:
        return _failed(str(e))

    try:
        response = client.responses.create(
            model=model,
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
        _track_usage(response, model, prompt)
        parsed = json.loads(response.output_text)
    except Exception as e:
        print(f"  [warn] validation failed ({model}): {e}")
        return _failed(str(e))

    validated_objects = parsed["validated_objects"]
    invalid_not_objects = parsed["invalid_not_objects"]
    invalid_not_in_image = parsed["invalid_not_in_image"]

    n_extracted = len(extracted_objects)  # length count
    n_validated = len(validated_objects)
    n_invalid_not_object = len(invalid_not_objects)
    n_invalid_not_in_image = len(invalid_not_in_image)

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


def _row_keys(frame: pd.DataFrame, key_cols) -> pd.Series:
    """Stable per-row key (as a tuple of strings) for resume matching."""
    return frame[key_cols].astype(str).apply(tuple, axis=1)


def validate_dataframe_resumable(frame: pd.DataFrame, model: str, out_path: Path,
                                 key_cols=KEY_COLS, checkpoint_every: int = CHECKPOINT_EVERY,
                                 max_new_rows: int | None = None) -> pd.DataFrame:
    """Validate frame's extracted objects, skipping rows already done in out_path.

    Resumes from a previous (possibly interrupted) run: any row whose key_cols
    already appear in out_path WITHOUT an error is left untouched, so it is never
    re-judged or re-billed. Previously-errored rows are dropped and retried. The
    output is re-written every `checkpoint_every` newly-judged rows so an
    interruption keeps completed work.
    """
    if out_path.exists():
        prev_df = pd.read_csv(out_path)
        if "error" in prev_df.columns:
            ok = prev_df["error"].isna()
            n_retry = int((~ok).sum())
            if n_retry:
                print(f"Retrying {n_retry} previously-failed row(s)")
            done_df = prev_df[ok].copy()
        else:
            done_df = prev_df
        done_keys = set(_row_keys(done_df, key_cols)) if len(done_df) else set()
        print(f"Resuming: {len(done_df)} rows already validated in {out_path.name}")
    else:
        done_df = None
        done_keys = set()

    todo = frame[~_row_keys(frame, key_cols).isin(done_keys)]
    if max_new_rows is not None:
        todo = todo.head(max_new_rows)
    print(f"Rows to validate this run: {len(todo)} (of {len(frame)} total)")
    if len(todo) == 0:
        print("Nothing to do — all rows already validated.")
        return done_df if done_df is not None else frame.iloc[0:0]

    def _save(new_rows):
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([done_df, new_df], ignore_index=True) if done_df is not None else new_df
        combined.to_csv(out_path, index=False)
        return combined

    new_rows = []
    since_ckpt = 0
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc=model):
        rec = row.to_dict()
        rec.update(evaluate_row(row, model=model))
        rec["judge_model"] = model
        new_rows.append(rec)
        since_ckpt += 1
        if since_ckpt >= checkpoint_every:
            _save(new_rows)
            since_ckpt = 0

    return _save(new_rows)


# -----------------------------
# Executive summary
# -----------------------------
def _pct(num, den):
    """Safe percentage helper."""
    return (100 * num / den) if den and den > 0 else 0.0


def executive_summary(final_df: pd.DataFrame, condition: str) -> dict:
    """Aggregate counts/rates over one condition's validated table."""
    summary_df = final_df.copy()
    # Coerce numeric columns in case some rows failed and produced None / strings
    for col in NUMERIC_COLS:
        summary_df[col] = pd.to_numeric(summary_df.get(col), errors="coerce")

    n_rows = len(summary_df)
    n_error_rows = int(summary_df["error"].notna().sum()) if "error" in summary_df.columns else 0
    n_success_rows = n_rows - n_error_rows

    total_extracted = summary_df["n_extracted"].sum(skipna=True)
    total_validated = summary_df["n_validated"].sum(skipna=True)
    total_semantic_errors = summary_df["n_invalid_not_object"].sum(skipna=True)
    total_hallucinations = summary_df["n_invalid_not_in_image"].sum(skipna=True)
    total_invalid_combined = summary_df["diff_in_objects"].sum(skipna=True)

    avg_extracted_per_row = summary_df["n_extracted"].mean(skipna=True)
    avg_validated_per_row = summary_df["n_validated"].mean(skipna=True)
    avg_semantic_errors_per_row = summary_df["n_invalid_not_object"].mean(skipna=True)
    avg_hallucinations_per_row = summary_df["n_invalid_not_in_image"].mean(skipna=True)

    return {
        "condition": condition,
        "generation": config.mapping_data["CONDITIONS"].get(condition, {}).get("generation"),
        "task": config.mapping_data["CONDITIONS"].get(condition, {}).get("task"),

        "n_rows": int(n_rows),
        "n_success_rows": int(n_success_rows),
        "n_error_rows": int(n_error_rows),

        "total_extracted_objects": int(total_extracted),
        "total_validated_objects": int(total_validated),

        "total_semantic_errors": int(total_semantic_errors),
        "semantic_error_pct_of_all_extracted_objects": round(_pct(total_semantic_errors, total_extracted), 2),

        "total_hallucinations": int(total_hallucinations),
        "hallucination_pct_of_all_extracted_objects": round(_pct(total_hallucinations, total_extracted), 2),

        "total_invalid_combined": int(total_invalid_combined),
        "combined_invalid_pct_of_all_extracted_objects": round(_pct(total_invalid_combined, total_extracted), 2),

        "valid_object_pct_of_all_extracted_objects": round(_pct(total_validated, total_extracted), 2),

        "rows_with_at_least_one_hallucination": int((summary_df["n_invalid_not_in_image"].fillna(0) > 0).sum()),
        "rows_with_at_least_one_semantic_error": int((summary_df["n_invalid_not_object"].fillna(0) > 0).sum()),
        "rows_with_at_least_one_invalid_item": int((summary_df["diff_in_objects"].fillna(0) > 0).sum()),

        "avg_extracted_objects_per_row": round(avg_extracted_per_row, 3) if pd.notna(avg_extracted_per_row) else None,
        "avg_validated_objects_per_row": round(avg_validated_per_row, 3) if pd.notna(avg_validated_per_row) else None,
        "avg_semantic_errors_per_row": round(avg_semantic_errors_per_row, 3) if pd.notna(avg_semantic_errors_per_row) else None,
        "avg_hallucinations_per_row": round(avg_hallucinations_per_row, 3) if pd.notna(avg_hallucinations_per_row) else None,
    }


# ── CLI: pick which full-experiment condition(s) to validate ─────────────────
# Each condition's tags are read from <condition>/nlp_analysis/trials_final_semantic_tags.csv
# and the validation is written beside it. Pass one or more condition slugs, or
# 'all' for every condition in config.CONDITIONS. With no flag it falls back to
# config.CONDITION (the CURRENT_CONDITION in condition_maps.yaml).
#   python object_accuracy_detector.py --condition aigen_perc
#   python object_accuracy_detector.py --condition aigen_perc aigen_imm aigen_del
#   python object_accuracy_detector.py --condition all
def _parse_args():
    ap = argparse.ArgumentParser(
        description="Validate tagged objects against the ground-truth image, per condition."
    )
    ap.add_argument(
        "--condition", "-c", nargs="+", default=None, metavar="SLUG",
        help="Condition slug(s) (e.g. aigen_perc) or 'all' for every condition. "
             "Default: config.CONDITION.",
    )
    ap.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Judge model (default: {DEFAULT_MODEL}).")
    ap.add_argument("--max-new-rows", "-n", type=int, default=MAX_NEW_ROWS,
                    help="Validate at most this many new rows per condition (for testing).")
    return ap.parse_args()


def _resolve_conditions(arg) -> list[str]:
    """Turn the --condition value into a validated list of condition slugs."""
    if not arg:
        return [config.CONDITION]
    if len(arg) == 1 and arg[0].lower() == "all":
        return list(config.CONDITIONS)
    unknown = [c for c in arg if c not in config.CONDITIONS]
    if unknown:
        raise SystemExit(
            f"Unknown condition(s): {unknown}. "
            f"Valid slugs: {list(config.CONDITIONS)} or 'all'."
        )
    return list(arg)


def run_full_for_condition(condition: str, model: str = DEFAULT_MODEL,
                           max_new_rows: int | None = MAX_NEW_ROWS) -> dict | None:
    """Resumably validate one condition's tagged objects, writing results beside them."""
    nlp_dir = config.paths_for(condition).processed_dir / "nlp_analysis"
    in_path = nlp_dir / TAGS_FILENAME
    if not in_path.exists():
        print(f"skip {condition}: {TAGS_FILENAME} not found ({in_path}) — run semantic_tagging.py first")
        return None

    cond_df = pd.read_csv(in_path)
    cond_df = cond_df[~cond_df["uid"].isin(EXCLUDE_UIDS)].copy()

    out_path = nlp_dir / OUT_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {condition}: {len(cond_df)} rows -> {out_path} ===")

    validate_dataframe_resumable(
        cond_df, model, out_path,
        checkpoint_every=CHECKPOINT_EVERY, max_new_rows=max_new_rows,
    )
    print(f"Saved validated data to {out_path}")

    # Summarize whatever is on disk (including rows validated in earlier runs).
    final_df = pd.read_csv(out_path)
    summary = executive_summary(final_df, condition)
    print(f"\n=== EXECUTIVE SUMMARY — {condition} ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    summary_path = out_path.with_name(out_path.stem + "_executive_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"Saved executive summary to {summary_path}")
    return summary


if __name__ == "__main__":
    args = _parse_args()
    conditions = _resolve_conditions(args.condition)
    print(f"Full run over {len(conditions)} condition(s): {conditions}")

    summaries = [s for s in (run_full_for_condition(c, args.model, args.max_new_rows)
                             for c in conditions) if s is not None]
    print_usage_costs()

    # One combined table of all conditions run, for side-by-side comparison.
    if len(summaries) > 1:
        combined_path = config.COMBINED_PROCESSED_DIR / "nlp_analysis" / "object_validation_executive_summary.csv"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(combined_path, index=False)
        print(f"\nSaved combined summary for {len(summaries)} condition(s) to {combined_path}")
