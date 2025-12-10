#this generates images based on prompts and filename template from a csv (for example gpt_trials.csv)
#previously it did this (i changed it on 10.12.2025): generating image using gpt-image-1 for all our pilot prompts in retrospect to check how well it does, for later including all analyses on it.

from pathlib import Path
import sys
# Make sure we can import config.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import re
import math
from tqdm import tqdm
import base64
from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

from config import PROCESSED_DIR, PANELS_DIR # uses your project-level config.py
OUT_PATH = Path("/mnt/hdd/anatkorol/Imagination_in_translation/Data/participants_data/pilot-2_10122025_gpt-image-1/jatos_results_files_20251210075934/study_result_gpt-5/comp_result_gpt-5/files")
DATA = OUT_PATH / "gpt_trials.csv" # participants and gpt descriptions
 # path for all the generated images
# #DATA = PROCESSED_DIR / "participants_log_with_gpt_pilot_08092025.csv" # participants and gpt descriptions
# # GPT_IMAGES_DATA = PROCESSED_DIR / "participants_log_with_gpt_pilot_08092025_gpt-image-1_generation.csv" # this is how to save it
# #looping through all prompts to generate an image for each using gpt-image-1 and saving them for further analysis
# # modifying the df to include the paths to the generated images

# df = pd.read_csv(DATA).copy()
# #building the folders for saving the images the same way as originally with the users

# def build_gpt_image_filename(uid: str, session: int, attempt: int, img_index: int) -> str:
#     """
#     like this: 00aeccd632c742d48a9ffe94da201493_session01_attempt01_img01_gpt-image.png
#     """
#     return (
#         f"{uid}_session{session:02d}_"
#         f"attempt{attempt:02d}_"
#         f"img{img_index:02d}_gpt-image.png"
#     )


# def build_gpt_image_path(row) -> Path:
#     """
#     Data/<PARTICIPANTS_DIR>/<uid>/gen_images/session_01/<filename>
#     """
#     uid        = str(row["uid"])
#     session    = int(row["session"])
#     attempt    = int(row["attempt"])
#     img_index  = int(row["img_index"])

#     filename   = build_gpt_image_filename(uid, session, attempt, img_index)

#     return (
#         GPT_IMAGES
#         / uid
#         / "gen_images"
#         / f"session_{session:02d}"
#         / filename
#     )


# a new generation function for single image generation and saving it to a given path
def generate_gpt_image_from_prompt(prompt: str,
                                   out_path: Path,
                                   size: str = "1024x1024")-> Path:
    """
    Generate a single image with gpt-image-1 and save to out_path.
    Returns out_path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resp = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=size,       # 1024x1024, 1024x1536, 1536x1024 for gpt-image-1
        n=1,
    )

    b64 = resp.data[0].b64_json
    img_bytes = base64.b64decode(b64)

    with open(out_path, "wb") as f:
        f.write(img_bytes)
    print(f"✅ Generated: {out_path.name}")
    
    #inspecting full response:
    revised_prompt = getattr(resp.data[0], "revised_prompt", None)
    print(revised_prompt)

    return out_path, revised_prompt


def regenerate_images_with_gpt(
   csv_path: Path,
    out_dir: Path,
    overwrite: bool = False,
    save_every: int = 10,
) -> pd.DataFrame:
    """
    Read CSV, generate images for each prompt, save with filename from 'gen' column.
    """
    # Work on a copy so we don't mutate the original df in-place
    df = pd.read_csv(csv_path).copy()

    # Make sure columns exist
    if "revised_prompt" not in df.columns:
        df["revised_prompt"] = pd.NA

    # enumerate gives a clean 1..N counter for save_every logic
    for n, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df)), start=1):
        filename = str(row["gen"])
        out_path = Path(out_dir) / filename

        if out_path.exists() and not overwrite:
            # File already exists: just record the filename, leave revised_prompt as-is/NA
            print(f"⏭️  Skipping (exists): {filename}")
            continue

        prompt = str(row["prompt"])
        try:
            _, revised_prompt = generate_gpt_image_from_prompt(prompt, out_path)
            df.at[idx, "revised_prompt"] = revised_prompt
        except Exception as e:
            print(f"❌ Error generating {filename}: {e}")
            continue

        # Periodic save
        if n % save_every == 0:
            df.to_csv(csv_path, index=False)
            print(f"💾 Progress saved after {n} images")

    # Final save
    df.to_csv(csv_path, index=False)
    print(f"✅ Final save complete → {csv_path}")

    return df

def main():
    print(f"📂 Input CSV: {DATA}")
    print(f"📁 Output directory: {OUT_PATH}")

    if not DATA.exists():
        print(f"❌ CSV file not found: {DATA}")
        return

    regenerate_images_with_gpt(DATA, OUT_PATH, overwrite=False, save_every=10)


if __name__ == "__main__":
    main()