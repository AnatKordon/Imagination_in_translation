import sys
from pathlib import Path
# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import os
import base64
from typing import List
from openai import OpenAI
import pandas as pd
import config
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Allowed image extensions
IMG_EXTS = {".png", ".jpg", ".jpeg"}

def encode_image_to_base64(image_path: Path) -> str:
    """Convert an image to a base64 data URI string."""
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    suffix = image_path.suffix.lower().replace('.', '')
    return f"data:image/{suffix};base64,{encoded}" #jpg/jpeg/png

def generate_diffusion_prompt(image_path: Path) -> str:
    img_uri = encode_image_to_base64(image_path)

    response = client.responses.create(
        model="gpt-5",  # your GPT-5 reasoning model id
        # Put your “system/dev” guidance here (not as an assistant message)
        instructions=(
            'You are a visual assistant that specializes in describing images for use as prompts for diffusion models'
            "Output must contain only ASCII letters, digits, spaces, "
            "and these punctuation marks: . , ! ? : ; ' \" - ( ). "
            "Do not use any other characters (no emojis, curly quotes, en dashes/em dashes, "
            "slashes, brackets, ellipses, bullets). Do not add newlines. "
            # "You are a visual assistant that specializes in describing images for "
            # "use as prompts for diffusion models. Be precise and concrete; include "
            # "composition/camera angle, subjects, lighting, color palette, materials/"
            # "textures, style, mood, and any salient small details. English only."
        ),
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Please, describe the picture as precisely as possible in English only."
            
                    },
                    {
                        "type": "input_image",
                        "image_url": img_uri
                    }
                ]
            }
        ],
        # Responses API uses max_output_tokens (not max_tokens)
        # max_output_tokens=800,
        # Reasoning control for GPT-5 models
        reasoning={"effort": "high"},   # can change 
        text={ "verbosity": "low" }, 
        # # Optional hygiene:
        store=False,
        # metadata={"task": "diffusion_prompt"}
    )
    # print(f"Raw response: {response}")
    print(response.output_text)
    
    # SDK convenience accessor; falls back to structured path if needed
    return response, response.output_text

def describe_all_images(gt_dir: Path) -> List[dict]:
    """Iterate over all images in GT_DIR and describe each."""
    all_descriptions = []
    for image_path in sorted(gt_dir.glob("*")):
        if image_path.suffix.lower() not in IMG_EXTS:
            continue
        print(f"Describing {image_path.name}...")
        try:
            full_response, description = generate_diffusion_prompt(image_path)
            all_descriptions.append({
                "image": image_path.name,
                "description": description,
                "full_response": full_response.model_dump_json()

            })
        except Exception as e:
            print(f"Failed on {image_path.name}: {e}")
    return all_descriptions

if __name__ == "__main__":
    image_dir = config.GT_DIR

    all_descriptions = describe_all_images(image_dir)

    # Optional: print all results
    for r in all_descriptions:
        print(f"\n🖼️ {r['image']}\n📜 {r['description']}")
   
    pd.DataFrame(all_descriptions).to_csv("gpt-5_image_descriptions_verbosity-low.csv", index=False)