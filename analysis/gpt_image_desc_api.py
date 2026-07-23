# GPT descriptions of the ground-truth images — the gpt-5.5_desc baseline.
#
# What this is for: participants write a prompt describing a GT image (from perception
# or from memory). This script asks a GPT model to do the same thing while LOOKING at
# the image, i.e. with no memory load and no time pressure. That gives an upper bound
# on how good a description of these images can be, to compare against the human
# prompts (verbosity / semantic-tag counts / object accuracy).
#
# One description per GT image (N_PER_IMAGE=1), verbosity medium by default.
#
# Outputs (both under OUT_DIR, the designated gpt-5.5_desc folder):
#   gpt-5.5_desc_descriptions_verbosity-<v>.csv   uid/gt/session/attempt/prompt/... —
#       same column scheme as trials_final.csv, so it drops into existing analyses.
#   gpt-5.5_desc_semantic_tags_verbosity-<v>.csv  the same rows + semantic tags, same
#       scheme as each condition's nlp_analysis/trials_final_semantic_tags.csv, so
#       notebooks (e.g. cross_gen_semantic_counts.ipynb) can load it as a reference line.
#
# Usage:
#   python analysis/gpt_image_desc_api.py                       # describe + tag, verbosity medium
#   python analysis/gpt_image_desc_api.py --verbosity high
#   python analysis/gpt_image_desc_api.py --n-per-image 3        # repeats -> attempt 1..3
#   python analysis/gpt_image_desc_api.py --skip-tagging
#   python analysis/gpt_image_desc_api.py --force                # re-describe (re-bills)

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import base64
import json
import os
from typing import List

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

import config

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Allowed image extensions
IMG_EXTS = {".png", ".jpg", ".jpeg"}

# Description model. gpt-5.5 matches the tagger's DEFAULT_MODEL in semantic_tagging.py.
DESC_MODEL = "gpt-5.5"
# Valid gpt-5.x efforts: none, low, medium, high, xhigh. Describing an image well is
# the whole point of this baseline, so we do not skimp here.
REASONING_EFFORT = "high"
# "low" | "medium" | "high" — how long the description is. Medium is the default because
# it is the closest match to the length of a participant prompt.
DEFAULT_VERBOSITY = "medium"
N_PER_IMAGE = 1  # descriptions per image; >1 fills attempt 1..N for variability checks

# Designated gpt-5.5_desc folder, beside the per-condition processed data.
OUT_DIR = config.COMBINED_PROCESSED_DIR / "gpt-5.5_desc"

# Same framing the participants work under: the text is a prompt for a diffusion model.
INSTRUCTIONS = (
    "You are a visual assistant that specializes in describing images for use as "
    "prompts for diffusion models. Be precise and concrete: include composition and "
    "camera angle, subjects, lighting, color palette, materials and textures, style, "
    "mood, and any salient small details. Describe only what is visible; do not "
    "speculate about what is outside the frame or invent details. English only. "
    "Output must contain only ASCII letters, digits, spaces, "
    "and these punctuation marks: . , ! ? : ; ' \" - ( ). "
    "Do not use any other characters (no emojis, curly quotes, en dashes or em dashes, "
    "slashes, brackets, ellipses, bullets). Do not add newlines."
)

USER_TEXT = "Please, describe the picture as precisely as possible in English only."


def encode_image_to_base64(image_path: Path) -> str:
    """Convert an image to a base64 data URI string."""
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    suffix = image_path.suffix.lower().replace(".", "")
    if suffix == "jpg":
        suffix = "jpeg"  # the data URI media type is image/jpeg, not image/jpg
    return f"data:image/{suffix};base64,{encoded}"


def generate_diffusion_prompt(image_path: Path, verbosity: str = DEFAULT_VERBOSITY,
                              model: str = DESC_MODEL):
    """One description of one image. Returns (response, text)."""
    img_uri = encode_image_to_base64(image_path)

    response = client.responses.create(
        model=model,
        # System/developer guidance goes here, not as an assistant message.
        instructions=INSTRUCTIONS,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": USER_TEXT},
                    {"type": "input_image", "image_url": img_uri},
                ],
            }
        ],
        # Responses API uses max_output_tokens (not max_tokens); it must cover the
        # internal reasoning as well as the visible answer.
        max_output_tokens=10000,
        reasoning={"effort": REASONING_EFFORT},
        text={"verbosity": verbosity},
        store=False,
    )
    return response, response.output_text


def describe_all_images(gt_dir: Path, verbosity: str = DEFAULT_VERBOSITY,
                        model: str = DESC_MODEL, n_per_image: int = N_PER_IMAGE,
                        skip: set | None = None) -> List[dict]:
    """Describe every image in gt_dir, n_per_image times each.

    Rows follow the participant column scheme (uid / gt / session / attempt / prompt)
    so these rows can be concatenated with, or plotted beside, the trial tables.
    `skip` holds (gt, attempt) pairs already described in a previous run.
    """
    skip = skip or set()
    images = [p for p in sorted(gt_dir.glob("*")) if p.suffix.lower() in IMG_EXTS]
    # session = the GT image index (1..5), matching aggregate_gpt_desc_to_csv.py.
    gt_to_session = {p.name: i + 1 for i, p in enumerate(images)}

    rows = []
    for image_path in images:
        for attempt in range(1, n_per_image + 1):
            if (image_path.name, attempt) in skip:
                print(f"skip {image_path.name} (attempt {attempt}): already described")
                continue
            print(f"Describing {image_path.name} (attempt {attempt})...")
            try:
                full_response, description = generate_diffusion_prompt(
                    image_path, verbosity=verbosity, model=model
                )
            except Exception as e:
                print(f"Failed on {image_path.name}: {e}")
                continue
            if not description or not description.strip():
                print(f"Failed on {image_path.name}: empty output "
                      f"(status={getattr(full_response, 'status', '?')})")
                continue
            print(f"  {description[:120]}...")
            usage = getattr(full_response, "usage", None)
            rows.append({
                "uid": model,
                "gt": image_path.name,
                "session": gt_to_session[image_path.name],
                "attempt": attempt,
                "prompt": description.strip(),
                "gen": pd.NA,          # no image is generated from these
                "subjective_score": pd.NA,
                "verbosity": verbosity,
                "desc_model": model,
                "reasoning_effort": REASONING_EFFORT,
                "input_tokens": getattr(usage, "input_tokens", pd.NA) if usage else pd.NA,
                "output_tokens": getattr(usage, "output_tokens", pd.NA) if usage else pd.NA,
                "full_response": full_response.model_dump_json(),
            })
    return rows


def tag_descriptions(desc_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Semantic-tag the gpt-5.5_desc descriptions with the SAME tagger the trials use.

    Reuses semantic_tagging.extract_semantics (same system prompt, schema and model),
    which is what makes the counts comparable to the per-condition tag tables.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "analysis" / "nlp_analysis"))
    import semantic_tagging as st

    rows = []
    for _, row in desc_df.iterrows():
        print(f"Tagging {row['gt']} (attempt {row['attempt']})...")
        tags = st.extract_semantics(row["prompt"], model=st.DEFAULT_MODEL)
        rec = row.to_dict()
        rec.pop("full_response", None)  # keep the tag table readable
        rec["extraction"] = json.dumps(tags, ensure_ascii=False)
        rec["tagger_model"] = st.DEFAULT_MODEL
        rec.update(tags)  # objects / stuff / scene_category / spatial_relations / ...
        rows.append(rec)

    tagged = pd.DataFrame(rows)
    tagged.to_csv(out_path, index=False)
    st.print_usage_costs()
    return tagged


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Generate gpt-5.5_desc descriptions of the GT images and semantic-tag them."
    )
    ap.add_argument("--verbosity", default=DEFAULT_VERBOSITY,
                    choices=["low", "medium", "high"],
                    help="Description length (default: medium).")
    ap.add_argument("--model", default=DESC_MODEL, help=f"Describing model (default: {DESC_MODEL}).")
    ap.add_argument("--n-per-image", type=int, default=N_PER_IMAGE,
                    help="Descriptions per image; >1 fills attempt 1..N (default: 1).")
    ap.add_argument("--skip-tagging", action="store_true",
                    help="Only write the descriptions CSV, do not call the semantic tagger.")
    ap.add_argument("--force", action="store_true",
                    help="Re-describe images already in the output CSV (re-bills them).")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    desc_path = OUT_DIR / f"gpt-5.5_desc_descriptions_verbosity-{args.verbosity}.csv"
    tags_path = OUT_DIR / f"gpt-5.5_desc_semantic_tags_verbosity-{args.verbosity}.csv"

    # Resume: keep whatever was already described (5 images, but calls are not free).
    done_df = pd.read_csv(desc_path) if (desc_path.exists() and not args.force) else None
    skip = set() if done_df is None else set(
        zip(done_df["gt"].astype(str), done_df["attempt"].astype(int))
    )

    new_rows = describe_all_images(
        config.GT_DIR, verbosity=args.verbosity, model=args.model,
        n_per_image=args.n_per_image, skip=skip,
    )
    new_df = pd.DataFrame(new_rows)
    desc_df = (pd.concat([done_df, new_df], ignore_index=True)
               if done_df is not None and not new_df.empty else
               (new_df if done_df is None else done_df))
    if desc_df.empty:
        print("No descriptions produced — nothing written.")
        return
    desc_df = desc_df.sort_values(["session", "attempt"]).reset_index(drop=True)
    desc_df.to_csv(desc_path, index=False)
    print(f"\nWrote {len(desc_df)} descriptions to {desc_path}")

    for r in desc_df.itertuples():
        print(f"\n{r.gt} (attempt {r.attempt})\n{r.prompt}")

    if args.skip_tagging:
        print("\n--skip-tagging: not tagging. Re-run without the flag to produce "
              f"{tags_path.name}.")
        return

    tagged = tag_descriptions(desc_df, tags_path)
    print(f"\nWrote {len(tagged)} tagged rows to {tags_path}")


if __name__ == "__main__":
    main()
