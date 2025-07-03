from typing import Optional, Dict, Any
import os
import requests
from dotenv import load_dotenv
from PIL import Image
from shutil import copyfile

# ------- Load API key from .env -------
load_dotenv()
STABILITY_KEY = os.getenv("STABILITY_API_KEY")

# ------- Function to send generation request -------
def send_generation_request(
    host: str,
    params: Dict[str, Any],
    user_id: str,
    iteration: int,
    session_num: int,
    true_image_path: Optional[str] = None
) -> str:
    """
    Generates an image using Stability AI API and saves it locally with a flat filename structure.

    Parameters:
    - host: API endpoint URL.
    - params: Generation settings (prompt, aspect_ratio, seed, etc.).
    - user_id: Unique ID for the user.
    - iteration: Generation iteration number.
    - session_num: Session number for tracking.
    - true_image_path: Optional path to original reference image.

    Returns:
    - Path to the saved generated image.
    """

    # ------- Prepare headers for the request -------
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }
    # ------- Prepare payload (text-to-image) -------
    files = {
        "prompt": (None, params["prompt"]),
        "image": open("data/wilma_ground_truth/bridge_l.jpg", "rb"), #style guide, from api documentation - to generate images similar to our dataset, "rb" - read mode, binary mode
        "fidelity": (None, "0.2"), #setting lower fidelity to style image to allow variablity and adjstment to all images in dataset
        "aspect_ratio": (None, params["aspect_ratio"]),
        "output_format": (None, params["output_format"]),
        "model": (None, params["model"]),
        "seed": (None, str(params["seed"]))
        #"style_preset": (None, "photographic") # try "analog-film", "photographic"
    }
    # ------- Handle optional image and mask files -------
    if "image" in params:
        files["image"] = open(params["image"], "rb")
    if "mask" in params:
        files["mask"] = open(params["mask"], "rb")

    # ------- Send request -------
    print(f"Sending REST request to {host}...")
    response = requests.post(host, headers=headers, files=files)

    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")
    
    # ------- Handle API response and content filtering -------
    output_image = response.content
    returned_seed = response.headers.get("seed")
    finish_reason = response.headers.get("finish-reason")

    if finish_reason == 'CONTENT_FILTERED':
        raise Warning("NSFW content filtered.")

    # ------- Save generated image with flat filename structure -------
    # Make sure Logs/gen_images/ directory exists
    output_dir = os.path.join("Logs", "gen_images")
    os.makedirs(output_dir, exist_ok=True)

    # Save image using flat structure
    filename = f"{user_id}_session{session_num}_iter{iteration}.png"
    image_path = os.path.join(output_dir, filename)


    with open(image_path, "wb") as f:
        f.write(output_image)

    return image_path

