import os
import requests
from dotenv import load_dotenv
from PIL import Image
from shutil import copyfile

# ------- Load API key from .env -------
load_dotenv()
STABILITY_KEY = os.getenv("STABILITY_API_KEY")

# ------- Function to send generation request -------
def send_generation_request(host, params, user_id, iteration, session_num, true_image_path=None):
    """
    Generates an image using Stability AI API and handles image + metadata logging.

    - Sends a POST request to the Stability AI endpoint with the given prompt and parameters.
    - Saves the generated image locally under a user-specific folder, named by iteration.
    - Optionally saves a true/reference image on the first iteration (if provided).
    - Extracts key metadata (prompt, seed, model, etc.) and logs it into a per-user JSON file.
    - Supports future flexibility for image-to-image workflows by managing optional 'image' and 'mask' fields.

    Parameters:
    - host: URL of the Stability API endpoint.
    - params: Dictionary of generation settings (prompt, aspect_ratio, seed, etc.).
    - user_id: Unique identifier to organize images and logs per user.
    - iteration: Index of current generation to name and track image progress.
    - true_image_path: Path to original image provided by user.

    Returns:
    - Path to the saved generated image of the current iteration.
    """

    # ------- Prepare headers for the request -------
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }
    
    # ------- Handle optional image and mask files -------
    # This setup supports both text-to-image and image-based tasks (we will only use text to image)
    # In our project, we're using only text-to-image, so 'image' and 'mask' are usually not provided.
    # We keep this section for compatibility with the Stability AI example and future flexibility.
    files = {
        "prompt": (None, params["prompt"]),
        "aspect_ratio": (None, params["aspect_ratio"]),
        "output_format": (None, params["output_format"]),
        "model": (None, params["model"]),
        "seed": (None, str(params["seed"]))
    }

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
    # Make sure images/ directory exists
    os.makedirs("images", exist_ok=True)

    # Save image using flat structure
    filename = f"{user_id}_session{session_num}_iter{iteration}.png"
    image_path = os.path.join("images", filename)

    with open(image_path, "wb") as f:
        f.write(output_image)

    return image_path



# === Main code ===
params = {
    "prompt": "a playground with ",
    "aspect_ratio": "1:1",
    "seed": 1,
    "output_format": "png",
    "model": "sd3.5-large-turbo"
}

send_generation_request(
    host="https://api.stability.ai/v2beta/stable-image/generate/sd3",
    params=params,
    user_id="124",
    iteration=1,
    session_num=1,
)



