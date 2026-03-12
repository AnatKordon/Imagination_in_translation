# generating descriptions for the drawings (for later comparing them with our descriptions)

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
import os
import csv
import base64
from pathlib import Path
from openai import OpenAI
import config
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

from config import WILMA_IMAGES


# --------------------------------------------------
# 1. Define the mapping between drawing filenames and GT
# --------------------------------------------------

SCENE_MAP = {
    "low_bedroom": "bedroom_l.jpg",
    "high_conference_room": "conference_room_h.jpg",
    "high_lighthouse": "lighthouse_l.jpg",
    "high_living_room": "living_room_h.jpg",
    "low_tower": "tower_l.jpg",
}

IMAGE_DIR = Path(config.WILMA_IMAGES)
OUTPUT_CSV = Path("/mnt/hdd/anatkorol/Imagination_in_translation/Data/processed_data/wilmas_drawings_2019/perception_drawing_descriptions.csv")


# --------------------------------------------------
# 2. Find relevant images
# --------------------------------------------------

records = []

for file in IMAGE_DIR.glob("*.jpg"):
    fname = file.name.lower()

    for key, gt in SCENE_MAP.items():
        if key in fname:
            records.append({
                "original_name": file.name,
                "gt": gt,
                "path": str(file)
            })


print(f"Found {len(records)} relevant drawings")


# --------------------------------------------------
# 3. Save initial CSV
# --------------------------------------------------

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["original_name", "gt", "description"])
    writer.writeheader()
    for r in records:
        writer.writerow({
            "original_name": r["original_name"],
            "gt": r["gt"],
            "description": ""
        })


# --------------------------------------------------
# 4. Helper to encode image
# --------------------------------------------------

def encode_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")


# --------------------------------------------------
# 5. GPT prompt
# --------------------------------------------------

PROMPT = (
"You are a visual assistant that describes images in the way a human would naturally "
"describe what they see in a drawing. Focus on the objects and scene content.\n\n"

"Write a natural description of the scene and the objects in it. "
"Mention the main objects, their positions, and any notable details. "
"Do not list items or number them. Do not sound like a report. "
"Write as if a person is casually describing what they see.\n\n"

"Do not comment on drawing quality or artistic technique. "
"Only describe what appears in the scene.\n\n"

"Output must contain only ASCII letters, digits, spaces, "
"and these punctuation marks: . , ! ? : ; ' \" - ( ). "
"Do not use any other characters. "
"Do not use bullet points, lists, or newlines."
)


# --------------------------------------------------
# 6. Run GPT descriptions
# --------------------------------------------------

for r in records:

    print("Processing:", r["original_name"])

    base64_image = encode_image(r["path"])

    response = client.chat.completions.create(
        model="gpt-4o", # can be changed
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500
    )

    description = response.choices[0].message.content
    r["description"] = description
    print("Description:", description)

# --------------------------------------------------
# 7. Save final CSV
# --------------------------------------------------

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["original_name", "gt", "description"]
    )
    writer.writeheader()

    for r in records:
        writer.writerow({
            "original_name": r["original_name"],
            "gt": r["gt"],
            "description": r["description"]
        })


print("Finished. Results saved to:", OUTPUT_CSV)