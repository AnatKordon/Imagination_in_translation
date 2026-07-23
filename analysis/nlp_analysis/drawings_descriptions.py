# Verbal transcriptions of Wilma's 2019 drawings — the drawn-recall counterpart of our
# verbal-recall conditions.
#
# WHY THIS EXISTS
# In our experiment participants convey a remembered (or perceived) image in WORDS.
# In Wilma's 2019 study participants conveyed the same SUN scenes in a DRAWING. To put
# the two on one axis we need the drawings in text form, then run them through the SAME
# semantic tagger (semantic_tagging.py) the verbal prompts go through.
#
# WHAT MAKES IT A FAIR COMPARISON (the whole design of this file)
#   1. Same tagger, same schema, same model. This script only produces the `prompt`
#      column; every count downstream comes from semantic_tagging.extract_semantics,
#      unchanged. Nothing about the drawing route gets its own tagging rules.
#   2. Transcription, not interpretation. The tagger's core rule for verbal prompts is
#      "extract ONLY what the participant explicitly states". The drawing analogue is
#      "report ONLY what the participant explicitly DREW". So the describer is barred
#      from completing the scene from schema knowledge (a bed implies a bedroom implies
#      a lamp), from inferring occluded or implied content, and from adding mood,
#      lighting, atmosphere or style — all things a vision model volunteers by default
#      and which would silently inflate the drawings' tag counts.
#   3. No target leakage. The filename encodes the target scene (e.g. high_lighthouse).
#      It is NEVER shown to the describer, which sees the pixels only. Otherwise the
#      describer could confabulate target-consistent content that the participant never
#      drew, and object-accuracy against the GT would be inflated.
#   4. Written text on the drawing is split out. Wilma's participants often annotate
#      ("blue sky", an arrow labelled "Door", "white bed w/ dark brown accents"). That is
#      VERBAL content produced by the participant — including it silently would mix the
#      two media in the drawn condition only. So the model returns two texts:
#        prompt              -> depicted content only, labels ignored
#        prompt_with_labels  -> depicted content + what the labels assert
#      `prompt` is the default (drawing vs. words is then a clean medium contrast);
#      `prompt_with_labels` lets you re-run the tagger as a robustness check with
#      --text-variant labels. Raw label strings are kept in `written_text` either way.
#   5. Blank / illegible sheets are flagged (`is_blank`), not hallucinated into content.
#
# DATA CAVEAT (found while writing this — matters for any analysis of this folder)
# perception_drawings/ contains 1051 files: 444 with the `c` prefix and 607 with the `w`
# prefix. All 607 `w` files are byte-identical copies of files in
# immediate_memory_drawings/. Treating the folder as "perception" therefore mislabels
# 58% of it as perception when it is immediate memory. This script filters each
# condition by its filename prefix (c / w / none), so only the 444 real perception
# drawings are used. See FILENAME_PREFIX below.
#
# Usage:
#   python analysis/nlp_analysis/drawings_descriptions.py --condition draw_del
#   python analysis/nlp_analysis/drawings_descriptions.py --condition all
#   python analysis/nlp_analysis/drawings_descriptions.py -c all --skip-tagging
#   python analysis/nlp_analysis/drawings_descriptions.py -c draw_perc --all-scenes
#   python analysis/nlp_analysis/drawings_descriptions.py -c all --text-variant labels
#   python analysis/nlp_analysis/drawings_descriptions.py -c draw_del --limit 5   # smoke test

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import base64
import json
import os
import re

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm.auto import tqdm

import config

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Paths ─────────────────────────────────────────────────────────────────────
DRAWINGS_ROOT = PROJECT_ROOT / "Data" / "other_datasets" / "wilmas_drawings_2019"
OUT_ROOT = PROJECT_ROOT / "Data" / "processed_data" / "wilmas_drawings_2019"

# ── Conditions ────────────────────────────────────────────────────────────────
# Slugs mirror config.CONDITIONS (aigen_perc / aigen_imm / aigen_del) so the drawn
# conditions line up one-to-one with the verbal ones in cross-condition plots.
# `prefix` is the filename marker that identifies a drawing as genuinely belonging to
# this condition — see the DATA CAVEAT above; "" means "no letter prefix".
CONDITIONS = {
    "draw_perc": {"folder": "perception_drawings",       "prefix": "c",  "task": "perception"},
    "draw_imm":  {"folder": "immediate_memory_drawings", "prefix": "w",  "task": "immediate_memory"},
    "draw_del":  {"folder": "delayed_memory_drawings",   "prefix": "",   "task": "delayed_memory"},
}

# ── Scene -> ground truth ─────────────────────────────────────────────────────
# Keyed by (scene token, memorability half) as they appear in the filename, because the
# same scene appears in both a high- and a low-memorability version and only one of the
# two is our GT. Values are the filenames in config.GT_DIR.
SCENE_MAP = {
    ("bedroom",        "low"):  "bedroom_l.jpg",
    ("conferenceroom", "high"): "conference_room_h.jpg",
    ("lighthouse",     "high"): "lighthouse_h.jpg",
    ("livingroom",     "high"): "living_room_h.jpg",
    ("playground",     "high"): "playground_h.jpg",
}

# c14_10_low_street.jpg / w10_11_high_amusementpark.jpg / 10_16_xxxx.jpg
FNAME_RE = re.compile(
    r"^(?P<prefix>[a-z]*)(?P<subject>\d+)_(?P<trial>\d+)_(?P<sun>high|low)_(?P<scene>[a-z]+)$"
)

# ── Model settings ────────────────────────────────────────────────────────────
# Same describing model / effort as the GT baseline in analysis/gpt_image_desc_api.py,
# so "GPT looking at the GT image" and "GPT looking at a drawing of it" are produced by
# the same reader and differ only in what they are looking at.
DESC_MODEL = "gpt-5.5"
REASONING_EFFORT = "high"
# Participant prompts run ~55 words at the median (aigen_del), which is what "medium"
# lands near. Raising this makes the describer pad, which inflates drawing tag counts
# relative to the verbal conditions — change it only as a deliberate sensitivity check.
DEFAULT_VERBOSITY = "medium"

CHECKPOINT_EVERY = 20

# ── Prompt ────────────────────────────────────────────────────────────────────
INSTRUCTIONS = (
    "You are transcribing a hand-made drawing into words. The drawing was made by a "
    "study participant who was asked to reproduce a photograph, either while looking at "
    "it or from memory. Your transcription will be compared against text written by "
    "other participants who described the same photographs in words instead of drawing "
    "them, so it must report the drawing's content and nothing more.\n\n"

    "Report only what is actually drawn on the page. Every thing you mention must "
    "correspond to marks you can point to. Do not complete the scene from what such a "
    "scene usually contains: if a bed is drawn, do not add a lamp, a window or a floor "
    "because bedrooms have them. Do not infer objects that are implied, occluded, cut "
    "off at the edge, or merely suggested. Do not guess what an unclear shape was meant "
    "to be; if a shape is genuinely unidentifiable, either say it is an unidentifiable "
    "shape or leave it out. Better to omit a doubtful thing than to invent one.\n\n"

    "Do not add mood, atmosphere, ambience, lighting, weather, time of day, style, "
    "genre, or any evaluation of the drawing, the drawer, or the medium. Do not mention "
    "that this is a drawing, a sketch, pencil, crayon, lines, strokes, shading, paper, "
    "or the page. Mention a color only where color is actually applied on the page, and "
    "name it plainly. Mention size, shape, material or condition only where the drawing "
    "itself shows it.\n\n"

    "Mention each drawn thing once. Say where things are and how they relate spatially "
    "when the drawing makes that clear. Write flowing prose in the voice of a person "
    "saying what is in the picture, not a list, not a report, and no headings. Let the "
    "length follow the drawing: a sparse drawing gets a short text, a dense one a longer "
    "text. Never pad to fill space.\n\n"

    "Handle words written on the page separately. Participants sometimes label parts of "
    "the drawing or write notes on it. Copy every such piece of writing verbatim into "
    "written_text. Then produce two texts. In `description`, report only what is drawn "
    "and ignore the writing completely: do not let a label tell you what a shape is, and "
    "do not report a thing that exists on the page only as a word. In "
    "`description_with_labels`, report the same drawn content and additionally state what "
    "the writing asserts, phrased as ordinary description. If nothing is written on the "
    "page, leave written_text empty and make the two texts identical.\n\n"

    "If the page is blank, or has nothing beyond stray marks with no identifiable "
    "content, set is_blank to true and leave both texts empty.\n\n"

    "English only. Output must contain only ASCII letters, digits, spaces, and these "
    "punctuation marks: . , ! ? : ; ' \" - ( ). Do not use any other characters (no "
    "emojis, curly quotes, en dashes or em dashes, slashes, brackets, ellipses, "
    "bullets). Do not add newlines."
)

USER_TEXT = "Transcribe this drawing."

DRAWING_SCHEMA = {
    "type": "json_schema",
    "name": "drawing_transcription",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "is_blank": {"type": "boolean"},
            "written_text": {"type": "array", "items": {"type": "string"}},
            "description": {"type": "string"},
            "description_with_labels": {"type": "string"},
        },
        "required": ["is_blank", "written_text", "description", "description_with_labels"],
    },
}


# ── Discovery ─────────────────────────────────────────────────────────────────
def parse_filename(path: Path) -> dict | None:
    """Pull subject / trial / memorability half / scene out of a drawing filename.

    Returns None for names that do not follow the pattern (e.g. the 17 `..._xxxx.jpg`
    files in the delayed set, whose scene was never recorded).
    """
    m = FNAME_RE.match(path.stem.lower())
    if m is None:
        return None
    return m.groupdict()


def collect_drawings(condition: str, all_scenes: bool = False) -> list[dict]:
    """List the drawings that belong to `condition`, in a participant-row shape.

    Only files carrying this condition's filename prefix are kept, which is what
    excludes the immediate-memory copies sitting inside perception_drawings/.
    By default only the five scenes we have a GT for are returned.
    """
    spec = CONDITIONS[condition]
    folder = DRAWINGS_ROOT / spec["folder"]
    if not folder.exists():
        raise SystemExit(f"Missing drawings folder: {folder}")

    # session = the GT image index, assigned the same way as in gpt_image_desc_api.py
    # (alphabetical order of config.GT_DIR), so `session` means the same thing here.
    gt_images = sorted(p.name for p in config.GT_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    gt_to_session = {name: i + 1 for i, name in enumerate(gt_images)}

    rows, skipped_prefix, unparsed = [], 0, 0
    for path in sorted(folder.glob("*.jpg")):
        parsed = parse_filename(path)
        if parsed is None:
            unparsed += 1
            continue
        if parsed["prefix"] != spec["prefix"]:
            skipped_prefix += 1
            continue

        gt = SCENE_MAP.get((parsed["scene"], parsed["sun"]))
        if gt is None and not all_scenes:
            continue

        rows.append({
            "uid": f"{spec['prefix']}{parsed['subject']}",
            "gt": gt if gt is not None else pd.NA,
            "session": gt_to_session.get(gt, pd.NA) if gt is not None else pd.NA,
            "attempt": 1,
            "condition": condition,
            "task": spec["task"],
            "scene": parsed["scene"],
            "sun_half": parsed["sun"],
            "trial_index": int(parsed["trial"]),
            "original_name": path.name,
            "path": str(path),
        })

    # A participant drew each scene once, but guard the key anyway: semantic_tagging
    # resumes on (uid, session, attempt), so a repeat must not collide.
    seen: dict[tuple, int] = {}
    for r in rows:
        key = (r["uid"], str(r["session"]))
        seen[key] = seen.get(key, 0) + 1
        r["attempt"] = seen[key]

    print(f"{condition}: {len(rows)} drawings "
          f"({skipped_prefix} skipped as not-this-condition, {unparsed} unparsable filenames)")
    return rows


# ── Describing ────────────────────────────────────────────────────────────────
def encode_image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    suffix = image_path.suffix.lower().lstrip(".")
    if suffix == "jpg":
        suffix = "jpeg"
    return f"data:image/{suffix};base64,{encoded}"


def transcribe_drawing(image_path: Path, verbosity: str = DEFAULT_VERBOSITY,
                       model: str = DESC_MODEL):
    """One transcription of one drawing. Returns (response, parsed dict).

    The filename is deliberately not part of the request: the model sees pixels only,
    so it cannot be steered by the target scene name.
    """
    response = client.responses.create(
        model=model,
        instructions=INSTRUCTIONS,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": USER_TEXT},
                {"type": "input_image", "image_url": encode_image_to_base64(image_path)},
            ],
        }],
        max_output_tokens=10000,
        reasoning={"effort": REASONING_EFFORT},
        text={"format": DRAWING_SCHEMA, "verbosity": verbosity},
        store=False,
    )
    return response, json.loads(response.output_text)


def describe_condition(condition: str, out_path: Path, *, verbosity: str, model: str,
                       all_scenes: bool, limit: int | None) -> pd.DataFrame:
    """Transcribe one condition's drawings, resuming from whatever out_path holds."""
    records = collect_drawings(condition, all_scenes=all_scenes)

    done_df = pd.read_csv(out_path) if out_path.exists() else None
    done = set(done_df["original_name"].astype(str)) if done_df is not None else set()
    if done:
        print(f"Resuming: {len(done)} already transcribed in {out_path.name}")

    todo = [r for r in records if r["original_name"] not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"To transcribe this run: {len(todo)} (of {len(records)})")
    if not todo:
        return done_df if done_df is not None else pd.DataFrame()

    def _save(new_rows) -> pd.DataFrame:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([done_df, new_df], ignore_index=True) if done_df is not None else new_df
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        return combined

    new_rows, since_ckpt = [], 0
    for r in tqdm(todo, desc=f"{condition} ({model})"):
        try:
            response, parsed = transcribe_drawing(Path(r["path"]), verbosity=verbosity, model=model)
        except Exception as e:
            print(f"  [warn] failed on {r['original_name']}: {e}")
            continue

        usage = getattr(response, "usage", None)
        rec = dict(r)
        rec.update({
            # `prompt` is the tagger's input column, and it is the labels-excluded text:
            # the drawn-medium counterpart of a participant's typed description.
            "prompt": parsed["description"].strip(),
            "prompt_with_labels": parsed["description_with_labels"].strip(),
            "written_text": json.dumps(parsed["written_text"], ensure_ascii=False),
            "has_written_text": bool(parsed["written_text"]),
            "is_blank": parsed["is_blank"],
            "gen": pd.NA,               # no image is generated from these
            "subjective_score": pd.NA,  # Wilma's task had no similarity rating
            "verbosity": verbosity,
            "desc_model": model,
            "reasoning_effort": REASONING_EFFORT,
            "input_tokens": getattr(usage, "input_tokens", pd.NA) if usage else pd.NA,
            "output_tokens": getattr(usage, "output_tokens", pd.NA) if usage else pd.NA,
        })
        new_rows.append(rec)
        since_ckpt += 1
        if since_ckpt >= CHECKPOINT_EVERY:
            _save(new_rows)
            since_ckpt = 0

    return _save(new_rows)


# ── Tagging ───────────────────────────────────────────────────────────────────
def tag_descriptions(desc_df: pd.DataFrame, out_path: Path, text_variant: str) -> pd.DataFrame:
    """Semantic-tag the transcriptions with the SAME tagger the verbal prompts use.

    `text_variant` picks which text is tagged: "depicted" (the default, drawn content
    only) or "labels" (drawn content plus what the participant's writing asserts).
    Reusing semantic_tagging.extract_semantics unchanged is what makes these counts
    comparable to each condition's trials_final_semantic_tags.csv.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "analysis" / "nlp_analysis"))
    import semantic_tagging as st

    source_col = "prompt" if text_variant == "depicted" else "prompt_with_labels"

    done_df = pd.read_csv(out_path) if out_path.exists() else None
    done = set(done_df["original_name"].astype(str)) if done_df is not None else set()
    todo = desc_df[~desc_df["original_name"].astype(str).isin(done)]
    print(f"Tagging {len(todo)} rows from `{source_col}` (of {len(desc_df)}) -> {out_path.name}")

    rows = []
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc=f"tag ({st.DEFAULT_MODEL})"):
        text = row[source_col]
        # A blank sheet has no content to tag; empty tags are the honest zero here, and
        # spending a call on an empty string would only invite the tagger to invent.
        tags = dict(st._EMPTY) if not isinstance(text, str) or not text.strip() \
            else st.extract_semantics(text, model=st.DEFAULT_MODEL)
        rec = row.to_dict()
        rec["tagged_text_variant"] = text_variant
        rec["extraction"] = json.dumps(tags, ensure_ascii=False)
        rec["tagger_model"] = st.DEFAULT_MODEL
        rec.update(tags)  # objects / stuff / scene_category / spatial_relations / ...
        rows.append(rec)

    tagged = pd.concat([done_df, pd.DataFrame(rows)], ignore_index=True) if done_df is not None \
        else pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tagged.to_csv(out_path, index=False)
    st.print_usage_costs()
    return tagged


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args():
    ap = argparse.ArgumentParser(
        description="Transcribe Wilma's 2019 drawings into text and semantic-tag them, "
                    "so drawn recall can be compared with our verbal recall conditions."
    )
    ap.add_argument("--condition", "-c", nargs="+", default=["all"], metavar="SLUG",
                    help=f"Condition slug(s) {list(CONDITIONS)} or 'all' (default: all).")
    ap.add_argument("--verbosity", default=DEFAULT_VERBOSITY, choices=["low", "medium", "high"],
                    help="Transcription length (default: medium, closest to participant prompts).")
    ap.add_argument("--model", default=DESC_MODEL, help=f"Describing model (default: {DESC_MODEL}).")
    ap.add_argument("--text-variant", default="depicted", choices=["depicted", "labels"],
                    help="Which text to tag: drawn content only (default) or content plus "
                         "what the participant's written labels assert.")
    ap.add_argument("--all-scenes", action="store_true",
                    help="Transcribe every scene, not just the five we have a GT for.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Transcribe at most N new drawings per condition (smoke test).")
    ap.add_argument("--skip-tagging", action="store_true",
                    help="Only write the transcriptions CSV, do not call the semantic tagger.")
    return ap.parse_args()


def _resolve_conditions(arg) -> list[str]:
    if len(arg) == 1 and arg[0].lower() == "all":
        return list(CONDITIONS)
    unknown = [c for c in arg if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"Unknown condition(s): {unknown}. Valid: {list(CONDITIONS)} or 'all'.")
    return list(arg)


def main() -> None:
    args = _parse_args()
    scope = "all_scenes" if args.all_scenes else "gt_scenes"

    for condition in _resolve_conditions(args.condition):
        desc_path = OUT_ROOT / condition / f"drawing_descriptions_{scope}.csv"
        tags_path = (OUT_ROOT / condition / "nlp_analysis" /
                     f"drawing_semantic_tags_{scope}_{args.text_variant}.csv")

        print(f"\n=== {condition} ===")
        desc_df = describe_condition(
            condition, desc_path, verbosity=args.verbosity, model=args.model,
            all_scenes=args.all_scenes, limit=args.limit,
        )
        if desc_df.empty:
            print("No transcriptions produced — nothing written.")
            continue
        print(f"Wrote {len(desc_df)} transcriptions to {desc_path}")

        if args.skip_tagging:
            print(f"--skip-tagging: re-run without the flag to produce {tags_path.name}.")
            continue
        tagged = tag_descriptions(desc_df, tags_path, args.text_variant)
        print(f"Wrote {len(tagged)} tagged rows to {tags_path}")


if __name__ == "__main__":
    main()
