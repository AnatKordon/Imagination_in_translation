from typing import Optional, Dict, Any
import os
from config import GEN_DIR  # Importing configuration settings
import requests
from dotenv import load_dotenv
from PIL import Image
from shutil import copyfile
from pathlib import Path
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
    img_index: Optional[int] = None,
    on_image_saved = None  # for google drive upload - to save the idx
) -> tuple[Path, str]:
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
        # I removed the style image
        # "image": open(config.STYLE_IMAGE, "rb"), #style guide, from api documentation - to generate images similar to our dataset, "rb" - read mode, binary mode
        # "fidelity": (None, "0.2"), #setting lower fidelity to style image to allow variablity and adjstment to all images in dataset
        "aspect_ratio": (None, params["aspect_ratio"]),
        "output_format": (None, params["output_format"]),
        "model": (None, params["model"]),
        "seed": (None, str(params["seed"])) # not sure...
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

    GEN_DIR.mkdir(parents=True, exist_ok=True)

    # Canonical filename: uid_sessionXX_attemptYY_imgZZ[_seedSSS].png
    parts = [f"{user_id}_session{session_num:02d}", f"attempt{iteration:02d}"] # this is structure of image name
    if img_index is not None:
        parts.append(f"img{img_index:02d}") # include index in structure
    if returned_seed:
        parts.append(f"seed{returned_seed}")
    filename   = "_".join(parts) + ".png"
    image_path = GEN_DIR / filename

    with open(image_path, "wb") as f:
        f.write(output_image)

    if on_image_saved is not None:
        try:
            on_image_saved(image_path, img_index, returned_seed)
        except Exception as cb_err:
            print(f"⚠️ on_image_saved failed: {cb_err}")

    return image_path, returned_seed

