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
#      "report ONLY what the participant explicitly PUT ON THE PAGE", drawn or written
#      (see 4). So the describer is barred from completing the scene from schema
#      knowledge and from inferring occluded or implied content.
#      Note what this does NOT bar. The tagger has attr_color and adjectives categories,
#      and adjectives explicitly covers "brightness terms, subjective descriptions,
#      evaluative descriptions". If the describer were forbidden to mention colour or
#      atmosphere at all, those two categories would be structurally unreachable for
#      drawings — a guaranteed zero against participants who typed "blue sky" or "dimly
#      lit bedroom". And atmosphere often IS on the page: heavy shading, rain, a sun or
#      moon, a dark wash, or a written note. So the rule is REPORT IT WHEN THE PAGE SHOWS
#      OR STATES IT, and never otherwise. What stays banned is the vision model's habit of
#      editorializing ("a serene, peaceful scene") over any sketch, and any remark about
#      the drawing as a drawing (skill, medium, strokes, paper).
#   3. No target leakage, in EITHER direction. The filename encodes the target scene
#      (e.g. high_lighthouse) and is NEVER shown to the describer, which sees pixels only.
#      Just as important, the INSTRUCTIONS below must never name a target scene or its
#      furniture: an example like "if it shows a lighthouse, say a lighthouse" would prime
#      the describer toward the very scenes we score against. Every example in the prompt
#      is therefore drawn from outside the 30 SUN categories in this dataset (bicycle,
#      teapot, ladder). Keep it that way when editing the prompt.
#   4. Writing on the page ADDS to the drawing. Per the paper's Methods, participants
#      "were provided a black ballpoint pen and colored pencils and were instructed to
#      optionally color or label aspects of the image" — labelling and colouring were
#      PERMITTED AND OPTIONAL, not required, and not a mandated substitute for drawing.
#      (The Image Drawing / perception group was told only to "draw this picture", with no
#      mention of labelling at all.) So writing is genuine participant output and dropping
#      it would discard content they chose to convey, but its presence varies BETWEEN
#      PARTICIPANTS and between conditions rather than being part of the task everyone
#      performed. `has_written_text` is kept per row so label use can be a covariate, or
#      an analysis can be re-run on label-free drawings only. The paper reports the
#      parallel figure for colour: 56.3% of delayed-recall drawings contained any colour,
#      so colour tags carry the same optionality confound. The drawing stays PRIMARY: the
#      describer settles what the marks show first, then lets legible writing name what it
#      could not identify, add detail, or report something undrawn. One merged text comes
#      back in `prompt`; the raw label strings are kept in `written_text` for auditing.
#      Writing the model cannot confidently read is dropped rather than guessed, because a
#      misread label becomes an object that was never in the scene (see the OCR note in
#      transcribe_drawing).
#   5. Blank / illegible sheets are flagged (`is_blank`), not hallucinated into content.
#
# DATA CAVEAT — the perception folder is contaminated UPSTREAM, not locally
# perception_drawings/ holds 1051 files: 444 with the `c` prefix and 607 with the `w`
# prefix. Our copy is byte-identical to the published dataset (Bainbridge, Hall & Baker
# 2019, "Image Drawings", doi:10.7910/DVN/1VRU1T), so the mix-up is in the release:
#   - that dataset documents its own filenames as c[subnum]_[imnum]_[memorability]_[scene]
#     and states "the c indicates these were participants in the control experiments";
#   - the `w` prefix is documented only in the immediate-recall release
#     (doi:10.7910/DVN/FVMI5W): "the w indicates these were participants in the working
#     memory (immediate recall) experiments";
#   - all 607 `w` files here are byte-identical to files in immediate_memory_drawings/,
#     and none of them is unique to the perception release.
# So they are immediate-recall drawings misfiled into the perception dataset. Taking the
# folder at face value would label 58% of "perception" as something it is not AND put the
# same drawing in both arms of the comparison. This script therefore selects each
# condition by filename prefix (c / w / none), leaving 444 genuine perception drawings.
# Consequence for stats: perception is 16 subjects vs. 30 for immediate recall. That N
# asymmetry is real and must be modelled, not repaired by re-admitting the duplicates.
#
# HOW MANY DRAWINGS EXIST FOR OUR 5 GT IMAGES (143) — AND WHY IT IS NOT 30 PER IMAGE
# The intuition "N participants, so N drawings per image" does not hold: NO participant
# ever saw all the images. The paper's design (Methods) is 30 scene categories x 2
# memorability versions = 60 stimuli, and each participant was shown 30 of them, one per
# category, with the high/low assignment counterbalanced across participants:
#   "participants viewed 30 images, one per category, half of which were high memorable
#    images and the other half low memorable images. Which category images were high
#    memorable or low memorable was counterbalanced across participants so that EACH
#    IMAGE WAS SEEN BY 15 PARTICIPANTS"
# So 30 immediate-recall participants x 30 drawings each = 900 drawings, but only 15 per
# stimulus. Our data matches exactly: 56 of the 60 stimuli have exactly 15 immediate
# drawings, and no participant drew both versions of a category.
#
# Per condition, for our five GT photographs:
#     low_bedroom          perc  6  imm 15  del  6   = 27
#     high_conferenceroom        7      15       6   = 28
#     high_lighthouse            7      15       6   = 28
#     high_livingroom            7      15      10   = 32
#     high_playground            6      15       7   = 28
#                                                     --- 143
#   - immediate is at its designed ceiling of 15 per image.
#   - delayed is free recall, so it is whatever people happened to remember; the paper
#     notes "no images ... were recalled by >12 participants", consistent with our 2-11.
#   - perception is SHORT. The paper specifies 24 participants, "twelve participants per
#     image", each drawing 30 images -> 720 drawings, 12 per stimulus. This release holds
#     only 444 c-files from 16 subjects (ids 1-6, 8-16, 18; missing 7, 17, 19-24), giving
#     5-9 per stimulus. So roughly a third of the perception drawings are absent from the
#     published data, on top of the duplicate problem described above. Treat perception's
#     N as the binding constraint in any power calculation.
#
# Taking the other memorability version (e.g. bedroom_h) would not raise N: that is a
# different photograph and cannot join a comparison scored against our GT. Use
# --all-scenes only for analyses that do not compare against our GT.
#
# Usage:
#   python analysis/nlp_analysis/drawings_descriptions.py --condition draw_del
#   python analysis/nlp_analysis/drawings_descriptions.py --condition all
#   python analysis/nlp_analysis/drawings_descriptions.py -c all --skip-tagging
#   python analysis/nlp_analysis/drawings_descriptions.py -c draw_perc --all-scenes
#   python analysis/nlp_analysis/drawings_descriptions.py -c draw_del --reasoning-effort low
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
# Reading a rough pencil sketch is a genuinely hard perceptual call (see --reasoning-effort
# to change it). Effort lands in the output filename, so runs at different efforts never
# resume into each other or get silently mixed in one CSV.
REASONING_EFFORT = "high"
# Participant prompts run ~55 words at the median (aigen_del), which is what "medium"
# lands near. Raising this makes the describer pad, which inflates drawing tag counts
# relative to the verbal conditions — change it only as a deliberate sensitivity check.
DEFAULT_VERBOSITY = "medium"

CHECKPOINT_EVERY = 20

# ── Prompt ────────────────────────────────────────────────────────────────────
INSTRUCTIONS = (
    "YOUR TASK\n"
    "You are given one hand-made drawing. Write a description of the scene it shows. "
    "The description is the only record of this drawing that survives: a later stage "
    "reads your text alone and extracts from it every object, material, colour, spatial "
    "relation and descriptor it mentions. Anything you leave out is lost, and anything "
    "you add is counted as though the drawing contained it. So report everything on the "
    "page, and nothing else.\n\n"

    "These drawings are being compared against descriptions that other people typed of "
    "the same photographs. Write the kind of text one of those people would have typed: "
    "plain prose naming what is there.\n\n"

    "READ THE DRAWING\n"
    "Name things the way an ordinary person would, at the level of detail people "
    "naturally use: if you can see it is a bicycle, say a bicycle; if it is a teapot, say "
    "a teapot. Do not retreat into geometry. 'A tall tapered form with a wide dark cap' "
    "is a failure when the marks plainly show a recognizable thing.\n\n"

    "Before naming something, ask whether you would recognize it from the marks alone, "
    "with no writing on the page. If yes, name it. If two readings are about equally "
    "plausible, do not pick one: describe the shape briefly instead, for instance three "
    "round shapes in a row along the bottom. Keep such fallback wording short.\n\n"

    "Recognize things whole rather than taking them apart. A stick figure is a person, "
    "not a round head above an oval body, and not an oval-headed figure. Do not describe "
    "how a figure is rendered. Mention a part only when it is a feature someone would "
    "remark on, such as a ladder with a broken rung.\n\n"

    "Report what a mark DEPICTS, never the mark itself. A wavy line along the edge of "
    "water is the water's edge, or ripples; it is not 'a wavy line'. Colour laid down "
    "inside an outline is simply that thing's colour, so write a blue teapot, not 'blue "
    "shading'. An outline is merely how a thing was drawn and is never worth mentioning. "
    "The words line, outline, stroke, mark, squiggle, scribble and shading must not "
    "appear in your description as things in the scene. The single exception is the brief "
    "fallback wording for a shape you genuinely cannot identify.\n\n"

    "BE COMPLETE\n"
    "Go over the whole page and account for everything on it, including small or "
    "peripheral items. Do not summarize, do not skip things for being minor, and do not "
    "stop early because the main subject is covered. Completeness is not padding: "
    "mention each thing once, in as few words as it takes, and never inflate the text "
    "with repetition or elaboration.\n\n"

    "Report colour wherever colour appears, naming the coloured thing: a red bicycle, "
    "green leaves, a yellow patch across the top. Colour was optional for these "
    "participants, so many pages are plain line work; on those, say nothing about "
    "colour rather than inventing any.\n\n"

    "Report lighting, weather, time of day or mood ONLY where the page actually shows or "
    "states it: heavy dark shading, rain, a sun or a moon, snow, a dark wash across a "
    "room, or a written note such as dimly lit. When the page shows such a thing, say so "
    "plainly, exactly as you would report an object. What you must not do is supply "
    "atmosphere the page does not support: never call a scene serene, peaceful, cozy, "
    "eerie or dramatic because the subject matter suggests it.\n\n"

    "WRITING ON THE PAGE\n"
    "These participants were given a pen and coloured pencils and told they could "
    "optionally colour or label parts of what they drew. Labelling was their choice, so "
    "many pages carry no writing at all, and such a page is not deficient. Where a "
    "participant did write something, it names something they meant to convey as present "
    "in the scene. Sometimes a line or arrow connects the word to the part it names, and "
    "sometimes the word simply sits next to it or floats near the edge; either way it "
    "belongs to whatever it is nearest or points to.\n\n"

    "Fold any writing you can read into the description as ordinary content, whether it "
    "names something you could not identify, adds a detail such as a colour or a "
    "condition, or reports something never drawn. Where a thing is both drawn and "
    "written, report it ONCE.\n\n"

    "Read a word only if you are confident of it. If it is unclear, or you would be "
    "guessing between two plausible readings, leave it out and fall back on what the "
    "marks show. A dropped label costs far less than an invented object: a misread word "
    "becomes a thing that was never in the scene. Never let the surrounding picture talk "
    "you into a reading of a half-legible word.\n\n"

    "Copy the writing you are confident of, verbatim, into written_text: one entry per "
    "label or note, multi-word labels kept intact. Never write ABOUT the writing in your "
    "description: no 'the writing says', no 'a label reads', no mention of words, "
    "labels, notes or handwriting. The description should read as though the participant "
    "simply described the scene.\n\n"

    "DO NOT INVENT\n"
    "Report only what the participant put on the page, drawn or written. Do not complete "
    "the scene from what such a scene usually contains: if a bicycle is drawn, do not add "
    "a road because bicycles are ridden on roads. Do not add things that are merely "
    "implied, occluded or suggested. Where the page is ambiguous, prefer leaving a thing "
    "out to inventing it. Naming what is recognizably there is not inventing; adding what "
    "ought to be there is.\n\n"

    "Two kinds of marks are part of the paperwork rather than the scene, and must never "
    "be reported as things in it. The first is a pointer: a line or arrow drawn purely to "
    "connect a written word to the part it names. Ignore it, and never describe it as a "
    "line crossing the picture. The second is the sheet's own border: many drawings sit "
    "inside a hand-drawn rectangle marking the edge of the picture. That rectangle is not "
    "a wall, a window or a frame; ignore it and describe only what is inside it.\n\n"

    "Say nothing about the drawing AS a drawing: no remarks on skill, effort, style, "
    "medium, pencil, crayon, strokes, shading technique, paper or the page, and no "
    "mention that this is a drawing or a sketch at all.\n\n"

    "HOW TO WRITE IT\n"
    "Say where things are and how they relate spatially wherever the page makes that "
    "clear. Write flowing prose in the voice of a person saying what is in the picture: "
    "not a list, not a report, no headings. Let the length follow the content, so a "
    "sparse page gets a short text and a crowded one a longer text.\n\n"

    "If the page is blank, or holds nothing beyond stray marks with no identifiable "
    "content and no legible writing, set is_blank to true and leave the description "
    "empty.\n\n"

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
        },
        "required": ["is_blank", "written_text", "description"],
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
                       model: str = DESC_MODEL, effort: str = REASONING_EFFORT):
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
        # Generous cap: it must cover the internal reasoning AND the answer. At high
        # effort a dense drawing can spend most of the budget thinking, and when nothing
        # is left the API returns an empty output_text that json.loads reports as
        # "Expecting value: line 1 column 1". Observed at 10000; 25000 clears it.
        max_output_tokens=25000,
        reasoning={"effort": effort},
        text={"format": DRAWING_SCHEMA, "verbosity": verbosity},
        store=False,
    )
    return response, json.loads(response.output_text)


def describe_condition(condition: str, out_path: Path, *, verbosity: str, model: str,
                       effort: str, all_scenes: bool, limit: int | None) -> pd.DataFrame:
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

    new_rows, since_ckpt, failed = [], 0, []
    for r in tqdm(todo, desc=f"{condition} ({model})"):
        try:
            response, parsed = transcribe_drawing(Path(r["path"]), verbosity=verbosity,
                                                  model=model, effort=effort)
        except Exception as e:
            # A failure drops an observation, so never let it pass silently. The row stays
            # out of the CSV, which means the next run retries it (see the resume logic).
            print(f"  [warn] failed on {r['original_name']}: {e}")
            failed.append(r["original_name"])
            continue

        usage = getattr(response, "usage", None)
        rec = dict(r)
        rec.update({
            # The participant's whole response in one text: what the marks show, with
            # legible writing folded in. `prompt` is the tagger's input column and matches
            # the trials_final scheme, so these rows drop into the existing analyses.
            "prompt": parsed["description"].strip(),
            "written_text": json.dumps(parsed["written_text"], ensure_ascii=False),
            "has_written_text": bool(parsed["written_text"]),
            "is_blank": parsed["is_blank"],
            "gen": pd.NA,               # no image is generated from these
            "subjective_score": pd.NA,  # Wilma's task had no similarity rating
            "verbosity": verbosity,
            "desc_model": model,
            "reasoning_effort": effort,
            "input_tokens": getattr(usage, "input_tokens", pd.NA) if usage else pd.NA,
            "output_tokens": getattr(usage, "output_tokens", pd.NA) if usage else pd.NA,
        })
        new_rows.append(rec)
        since_ckpt += 1
        if since_ckpt >= CHECKPOINT_EVERY:
            _save(new_rows)
            since_ckpt = 0

    if failed:
        print(f"  !! {len(failed)} drawing(s) produced no output and are MISSING from "
              f"{out_path.name}; re-run the same command to retry them: {failed}")
    return _save(new_rows)


# ── Tagging ───────────────────────────────────────────────────────────────────
def tag_descriptions(desc_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Semantic-tag the transcriptions with the SAME tagger the verbal prompts use.

    Reusing semantic_tagging.extract_semantics unchanged is what makes these counts
    comparable to each condition's trials_final_semantic_tags.csv.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "analysis" / "nlp_analysis"))
    import semantic_tagging as st

    done_df = pd.read_csv(out_path) if out_path.exists() else None
    done = set(done_df["original_name"].astype(str)) if done_df is not None else set()
    todo = desc_df[~desc_df["original_name"].astype(str).isin(done)]
    print(f"Tagging {len(todo)} rows (of {len(desc_df)}) -> {out_path.name}")

    rows = []
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc=f"tag ({st.DEFAULT_MODEL})"):
        text = row["prompt"]
        # A blank sheet has no content to tag; empty tags are the honest zero here, and
        # spending a call on an empty string would only invite the tagger to invent.
        tags = dict(st._EMPTY) if not isinstance(text, str) or not text.strip() \
            else st.extract_semantics(text, model=st.DEFAULT_MODEL)
        rec = row.to_dict()
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
    ap.add_argument("--reasoning-effort", default=REASONING_EFFORT,
                    choices=["none", "low", "medium", "high", "xhigh"],
                    help=f"How hard the describer thinks before writing (default: {REASONING_EFFORT}). "
                         "Appears in the output filename, so efforts never mix in one CSV.")
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
        desc_path = OUT_ROOT / condition / f"drawing_descriptions_{scope}_{args.reasoning_effort}.csv"
        tags_path = (OUT_ROOT / condition / "nlp_analysis" /
                     f"drawing_semantic_tags_{scope}_{args.reasoning_effort}.csv")

        print(f"\n=== {condition} ===")
        desc_df = describe_condition(
            condition, desc_path, verbosity=args.verbosity, model=args.model,
            effort=args.reasoning_effort, all_scenes=args.all_scenes, limit=args.limit,
        )
        if desc_df.empty:
            print("No transcriptions produced — nothing written.")
            continue
        print(f"Wrote {len(desc_df)} transcriptions to {desc_path}")

        if args.skip_tagging:
            print(f"--skip-tagging: re-run without the flag to produce {tags_path.name}.")
            continue
        tagged = tag_descriptions(desc_df, tags_path)
        print(f"Wrote {len(tagged)} tagged rows to {tags_path}")


if __name__ == "__main__":
    main()
