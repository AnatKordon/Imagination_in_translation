# extracting semantic tags based on the DSG framework using open ai api
""" 
This will include 4 calls:
1. extract tuples from user description 
2. extract questions from tupples 
3. extract dependencies from tupples 
4. only in the final prompt provide the image and not the description, and answer the questions in 2 based on it
"""

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
import re
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

from config import PROCESSED_DIR
df = pd.read_csv(PROCESSED_DIR / "ppt_w_gpt_w_similarity_trials.csv").copy()
df = df.head(2) # for testing 
# save new df:
OUT_PATH = PROCESSED_DIR / "nlp_analysis" / "ppt_w_gpt_dsg_tagging.csv"

# demo before improving prompts:

TUPLE_EXTRACTION_PROMPT = """
You are generating Davidsonian Scene Graph (DSG) tuples from a text description.
Task: given input prompts, describe each scene with skill-specific tuples.
Only include semantics explicitly stated in the text.
Do NOT add inferred objects/attributes/relations.
Do not generate same tuples again. 
Output format: id | tuple

Use ONLY these tuple skills (keep tuples atomic):
ENTITY-WHOLE: presence of an object/entity
ENTITY-PART: presence of a named part of an entity (e.g., “wheel of a bicycle”)
ATTRIBUTE-COLOR / SIZE / SHAPE / TEXTURE / MATERIAL / STYLE / STATE / TYPE
ATTRIBUTE-COUNT: exact count of an entity
ATTRIBUTE-TEXT-RENDERING: exact text written on an object/surface
RELATION-SPATIAL: left/right/above/below/inside/next-to/on-top-of/etc.
RELATION-ACTION: subject is doing an action (e.g., “man riding bicycle”)
RELATION-SCALE: comparative size/scale between two entities (e.g., “A larger than B”)

Tuple format constraints (pick the simplest that fits):
- ENTITY-WHOLE: entity(<object>)
- ENTITY-PART: part(<part>, of=<object>)
- ATTRIBUTE-*: attr(<object_or_part>, <attr_type>=<value>)
- COUNT: count(<object>)=<integer>
- TEXT-RENDERING: text(<object_or_surface>)="<exact text>"
- RELATION-SPATIAL: spatial(<objectA>, <relation>, <objectB>)
- RELATION-ACTION: action(<subject>, <verb_phrase>)
- RELATION-SCALE: scale(<objectA>, <comp>, <objectB>) where comp in (larger, smaller, same_size)

Input prompt:
{TEXT_PROMPT}

"""

#they talk about negation handling but i don't think it makes sense according to paper:
# where does the id come from?

# 2. 
QUESTION_EXTRACTION_PROMPT = """
Task: given input prompts and skill-specific tuples, re-write tuple each in natural language question.
Output format: id | question

Rules:
- One tuple -> one atomic yes/no question.
- Keep questions short and visually answerable.
- For COUNT use: "Are there exactly N <objects>?"
- For TEXT-RENDERING use: "Is the text '<exact text>' written on <surface>?"
- Do not add new details.

Input prompt:
{TEXT_PROMPT}

Tuples:
{ID2TUPLES}

"""

# 3. ***this part isn't needed unless i also perform the 4th step***
# TUPLE_DEPENDENCY_EXTRACTION = """
# Task: given input prompts and tuples, describe the parent tuples of each tuple.
# Output format: id | dependencies (comma separated)

# Rules:
# - A tuple’s parents are the minimal tuples that must be true for the tuple to be valid to ask about.
# - Typical dependencies:
#   - attr(X, ...) depends on entity(X) or part(X, ...)
#   - spatial(A, ..., B) depends on entity(A) and entity(B)
#   - action(S, ...) depends on entity(S)
#   - count(X)=N depends on entity(X)
#   - text(surface)="..." depends on entity(surface)

# Input prompt:
# {TEXT_PROMPT}

# Tuples:
# {ID2TUPLES}
# """



# function for extracting the 3 types of information and printing them out for now


MODEL_NAME = "gpt-4o-mini"  # replace with your open LLM name if different

def call_llm_text(prompt: str, model: str = MODEL_NAME) -> str:
    """
    Simple text-only call using Responses API.
    """
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        temperature=0,  # deterministic-ish for extraction
    )
    return resp.output_text  # easiest accessor for final text output


def parse_id_pipe_lines(text: str) -> dict:
    """
    Parses lines formatted like:  t1 | entity(cat)
    Returns dict: { "t1": "entity(cat)", ... }
    Ignores empty lines and non-matching lines.
    """
    out = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Accept "t1 | ..." with flexible spacing
        m = re.match(r"^(\S+)\s*\|\s*(.+?)\s*$", line)
        if not m:
            continue
        _id, val = m.group(1), m.group(2)
        out[_id] = val
    return out


def format_id2dict_for_prompt(d: dict) -> str:
    """
    Formats dict back to `id | value` lines to feed into next prompt.
    """
    return "\n".join([f"{k} | {v}" for k, v in d.items()])



for idx, row in df.iterrows():
    text_prompt = row["prompt"]  

    # 1) tuples
    prompt1 = TUPLE_EXTRACTION_PROMPT.format(TEXT_PROMPT=text_prompt)
    tuples_raw = call_llm_text(prompt1)
    id2tuples = parse_id_pipe_lines(tuples_raw)

    # 2) questions (uses tuples)
    prompt2 = QUESTION_EXTRACTION_PROMPT.format(
        TEXT_PROMPT=text_prompt,
        ID2TUPLES=format_id2dict_for_prompt(id2tuples),
    )
    questions_raw = call_llm_text(prompt2)
    id2questions = parse_id_pipe_lines(questions_raw)

    # # 3) dependencies (uses tuples)
    # prompt3 = TUPLE_DEPENDENCY_EXTRACTION.format(
    #     TEXT_PROMPT=text_prompt,
    #     ID2TUPLES=format_id2dict_for_prompt(id2tuples),
    # )
    # deps_raw = call_llm_text(prompt3)
    # id2dependency = parse_id_pipe_lines(deps_raw)

    print("\n" + "="*80)
    print(f"ROW {idx}")
    print("-"*80)
    print("TUPLES (id2tuples):")
    print(json.dumps(id2tuples, indent=2, ensure_ascii=False))
    print("\nQUESTIONS (id2questions):")
    print(json.dumps(id2questions, indent=2, ensure_ascii=False))
    # print("\nDEPENDENCIES (id2dependency):")
    # print(json.dumps(id2dependency, indent=2, ensure_ascii=False))


"""
DRAFTS NOT USED
THE REAL TUPLES NEEDED FOR 1:
###
USER:
DESCRIPTION:
{DESCRIPTION_TEXT}

SEMANTIC PARTS (extracted lists; may be empty):
objects={OBJECTS}
attr_color={ATTR_COLOR}
attr_shape={ATTR_SHAPE}
attr_size={ATTR_SIZE}
attr_material={ATTR_MATERIAL}
attr_texture={ATTR_TEXTURE}
attr_pose={ATTR_POSE}
attr_action={ATTR_ACTION}
attr_state={ATTR_STATE}
spatial_relations={SPATIAL_RELATIONS}
world_knowledge={WORLD_KNOWLEDGE}
scene={SCENE}
camera_aspects={CAMERA_ASPECTS}
optical_effects={OPTICAL_EFFECTS}
subjective_detail={SUBJECTIVE_DETAIL}

Now output the DSG tuples."""