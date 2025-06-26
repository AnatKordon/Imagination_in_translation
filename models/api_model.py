import os
import requests
import time
import uuid
import json
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import base64


# Load Stability API key
load_dotenv()
api_key = os.getenv("STABILITY_API_KEY")
api_host = "https://api.stability.ai"
engine_id = "stable-diffusion-v1-5"

def generate_image_and_log(prompt, step_number, user_id, ground_truth_id):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    payload = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 7.5,
        "height": 512,
        "width": 512,
        "samples": 1,
        "steps": 30
    }

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers=headers,
        json=payload,
    )

    if not response.ok:
        raise Exception(f"API error {response.status_code}: {response.text}")

    result = response.json()

    image_base64 = result["artifacts"][0]["base64"]
    seed = result["artifacts"][0]["seed"]
    sampler = result["artifacts"][0].get("finish_reason", "unknown")

    # Save image
    image_id = str(uuid.uuid4())
    image_data = base64.b64decode(image_base64)
    image_path = f"images/{user_id}_{image_id}_step{step_number}.png"

    with open(image_path, "wb") as f:
        f.write(image_data)

    # Log metadata
    log_entry = {
        "image_id": image_id,
        "step_number": step_number,
        "prompt_text": prompt,
        "user_id": user_id,
        "timestamp": int(time.time()),
        "seed": seed,
        "sampler": sampler,
        "cfg_scale": payload["cfg_scale"],
        "num_inference_steps": payload["steps"],
        "image_path": image_path,
        "ground_truth_id": ground_truth_id,
        # Add CLIP similarity and embeddings later
    }

    os.makedirs("logs", exist_ok=True)
    with open("logs/generation_log.jsonl", "a") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")

    return image_path, log_entry
