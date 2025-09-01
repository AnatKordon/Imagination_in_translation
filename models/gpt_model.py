import base64
from openai import OpenAI
import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from config import GEN_DIR, params

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

#  for now i am just printing revised prompts as in generation mode (not edit or responses) - it's supposed to be null
def send_gpt_request(prompt, iteration, session_num, user_id):
    img_data = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        n=2,
        size="1024x1024" # I changed from default 1024x1024
    )
    
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    local_paths: List[Path] = []
    revised_prompts: List[Optional[str]] = []

    for i, item in enumerate(image_data.data, start=1):
        # Save image
        img_b64 = getattr(item, "b64_json", None)
        if not img_b64:
            # If you ever switch to URL returns, handle that here.
            raise ValueError("No base64 image found in response; set response_format='b64_json'.")

        image_bytes = base64.b64decode(img_b64)
        filename = f"{user_id}_session{session_num:02d}_iter{iteration:02d}_{i:02d}_open_ai.png"
        path = GEN_DIR / filename
        path.write_bytes(image_bytes)
        local_paths.append(path)

        # Log revised prompt if present (usually only for edit/variation calls)
        rp = getattr(item, "revised_prompt", None)
        #  for now i am just printing revised prompts as in generation mode (not edit or responses) - it's supposed to be null
        print(f"revised prompt is: {rp}")
        revised_prompts.append(rp)

    # For generate, this will almost always be [None, None]
    # Keep your local params log if you like:
    params["revised_prompt"] = revised_prompts

    return local_paths, revised_prompts

def here():
    print(gpt_model)
    #upload it