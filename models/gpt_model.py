import base64
from openai import OpenAI
import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from config import  params, GPT_IMAGES

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

#THIS FUNCTION CURRENTLY WORKS RETROSPECTIVELY, NOT FOR USE IN APP CODE - SHOULD BE MODIFIED


GEN_DIR = GPT_IMAGES
# when we get back to it: add fallback for mujltiple generations~!
#  for now i am just printing revised prompts as in generation mode (not edit or responses) - it's supposed to be null
def send_gpt_request(prompt, attempt, session_num, user_id, out_path: Path = GEN_DIR):
    img_data = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        n=1,
        quality="medium",
        size="1024x1024" # I changed from default 1024x1024
    )
    
    #construct folders if they don't exist - not sure it should be inside this function...
    out_path.mkdir(parents=True, exist_ok=True)
    user_id_dir = out_path / user_id
    user_id_dir.mkdir(parents=True, exist_ok=True)
    gen_images = user_id_dir / "gen_images"
    gen_images.mkdir(parents=True, exist_ok=True)
    session_dir = gen_images / f"session_{session_num:02d}"
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving generated images to: {session_dir}"
          )
    local_paths: List[Path] = []
    revised_prompts: List[Optional[str]] = []

    for i, item in enumerate(img_data.data, start=1):
        # Save image
        img_b64 = getattr(item, "b64_json", None)
        if not img_b64:
            # If you ever switch to URL returns, handle that here.
            raise ValueError("No base64 image found in response; set response_format='b64_json'.")

        image_bytes = base64.b64decode(img_b64)
        filename = f"{user_id}_session{session_num:02d}_attempt{attempt:02d}_img{i:02d}_gpt-image.png"
        path = session_dir / filename
        path.write_bytes(image_bytes)
        local_paths.append(path)

        # Log revised prompt if present (usually only for edit/variation calls)
        rp = getattr(item, "revised_prompt", None)
        #  for now i am just printing revised prompts as in generation mode (not edit or responses) - it's supposed to be null
        print(f"revised prompt is: {rp}")
        revised_prompts.append(rp)

    # For generate, Data/participants_data/pilot_08092025_SD/6d3f4debdb254a30a2964dc9cad815df/gen_images/session_04this will almost always be [None, None]
    # Keep your local params log if you like:
    params["revised_prompt"] = revised_prompts

    return local_paths, revised_prompts

