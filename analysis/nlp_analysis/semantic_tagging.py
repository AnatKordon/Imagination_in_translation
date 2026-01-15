# this is a simplified demo (not the DSG) which I planned - aimed to tet the abilities of a gpt-40mini to answer freely regarding a description and it's attributes.

#LLM returns one JSON object per row (stable schema, low ambiguity).
#code expands it into separate dataframe columns (adjacent columns).

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

from config import PROCESSED_DIR
# df = pd.read_csv(PROCESSED_DIR / "ppt_w_gpt_w_similarity_trials.csv").copy()
# df = df 
# # save new df:
# OUT_PATH = PROCESSED_DIR / "nlp_analysis" / "ppt_w_gpt_semantic_tags.csv"

#perception data
df = pd.read_csv("/mnt/hdd/anatkorol/Imagination_in_translation/Data/processed_data/10122025_pilot_2/ppt_w_gpt_trials.csv").copy()
OUT_PATH = Path("/mnt/hdd/anatkorol/Imagination_in_translation/Data/processed_data/10122025_pilot_2/nlp_analysis") / "ppt_w_gpt_semantic_tags.csv"

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



def extract_semantics(prompt: str) -> dict:
  SYSTEM = SYSTEM_PROMPT

  resp = client.responses.create(
      model="gpt-4o",
      input=[
         {"role":"system","content":SYSTEM},
         {"role": "user", "content": USER_PROMPT.format(PROMPT=prompt)},
        ],
      text={"format": {"type": "json_object"}},
      temperature=0.0,  
      top_p=1.0,
      max_output_tokens=5000,
  )
  return json.loads(resp.output_text)


from tqdm.auto import tqdm
tqdm.pandas()
df["extraction"] = df["prompt"].progress_apply(extract_semantics)

out = pd.json_normalize(df["extraction"])
tagged_df = pd.concat([df, out], axis=1) # does this
#save
tagged_df.to_csv(OUT_PATH, index=False)