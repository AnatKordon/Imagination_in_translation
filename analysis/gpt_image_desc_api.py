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
    """Send an image to the GPT model and receive a descriptive prompt."""
    image_data_uri = encode_image_to_base64(image_path)
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a visual assistant that specializes in describing images "
                "for use as prompts for diffusion models."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please, describe the picture as precisely as possible in English only."},
                {"type": "image_url", "image_url": {"url": image_data_uri}}
            ]
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-5", 
        messages=messages,
        temperature=0.0, # lowest temperature - deterministic
        max_tokens=10000
    )
    return response.choices[0].message.content.strip()

def describe_all_images(gt_dir: Path) -> List[dict]:
    """Iterate over all images in GT_DIR and describe each."""
    all_descriptions = []
    for image_path in sorted(gt_dir.glob("*")):
        if image_path.suffix.lower() not in IMG_EXTS:
            continue
        print(f"Describing {image_path.name}...")
        try:
            description = generate_diffusion_prompt(image_path)
            all_descriptions.append({
                "image": image_path.name,
                "description": description
            })
        except Exception as e:
            print(f"Failed on {image_path.name}: {e}")
    return all_descriptions

if __name__ == "__main__":
    image_dir = config.GT_DIR
    results = describe_all_images(image_dir)

    # Optional: print all results
    for r in results:
        print(f"\n🖼️ {r['image']}\n📜 {r['description']}")

    pd.DataFrame(results).to_csv("image_descriptions.csv", index=False)