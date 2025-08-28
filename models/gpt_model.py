import base64
from openai import OpenAI
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from config import GEN_DIR
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def send_gpt_request(prompt, iteration, session_num, user_id):
    img = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        n=4,
        size="1024x1024" # I changed from default 1024x1024
    )
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    local_paths: List[Path] = []

    for idx, data in enumerate(img.data, start=1):
        image_bytes = base64.b64decode(data.b64_json)
        filename = f"{user_id}_session{session_num:02d}_iter{iteration:02d}_{idx:02d}_open_ai.png"
        path = GEN_DIR / filename
        path.write_bytes(image_bytes)
        local_paths.append(path)
    return local_paths

def here():
    print(gpt_model)
    #upload it