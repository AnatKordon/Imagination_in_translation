import os
import requests
import json
from dotenv import load_dotenv
from PIL import Image
from shutil import copyfile

# ------- Load API key from .env -------
load_dotenv()
STABILITY_KEY = os.getenv("STABILITY_API_KEY")

# ------- Function to send generation request -------
def send_generation_request(host, params, user_id, iteration, true_image_path=None):
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

    # Optional image or mask support (for future inpainting)
    if "image" in params:
        files["image"] = open(params["image"], "rb")
    if "mask" in params:
        files["mask"] = open(params["mask"], "rb")


    # ------- Send request -------
    print(f"Sending REST request to {host}...")
    response = requests.post(host, headers=headers, files=files)

    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")
    
    # ------- Get generated image bytes -------
    # The response content is the generated image in binary format.
    output_image = response.content
    returned_seed = response.headers.get("seed")
    finish_reason = response.headers.get("finish-reason")

    if finish_reason == 'CONTENT_FILTERED':
        raise Warning("NSFW content filtered.")

    # Paths and filenames for saving the image
    user_folder = f"images/user_{user_id}"
    os.makedirs(user_folder, exist_ok=True)
    gen_filename = f"gen_{iteration}.png"
    gen_path = os.path.join(user_folder, gen_filename)

    # ------- Save generated image -------
    with open(gen_path, "wb") as f:
        f.write(output_image)

    # Save true image if this is first time and provided
    if iteration == 1 and true_image_path:
        true_dest = os.path.join(user_folder, "true_image.png")
        if not os.path.exists(true_dest):
            from shutil import copyfile
            copyfile(true_image_path, true_dest)

    # ------- Save log to JSON and update user log -------
    log_entry = {
        "iteration": iteration,
        "prompt": params["prompt"],
        "seed": returned_seed,
        "aspect_ratio": params["aspect_ratio"],
        "output_format": params["output_format"],
        "model": params["model"],
        "image_path": gen_path,
    }

    log_path = f"logs/user_{user_id}.json"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)  # ensure logs folder exists
    update_user_log(log_path, log_entry)

    return gen_path

    
# ------- Function to update user log -------
def update_user_log(log_path, log_entry):
    """
    Appends a single JSON entry to the user's log file.

    Ensures each generation attempt (with metadata) is stored as a separate
    line in a JSONL file for easy tracking and later analysis.
    """
    # Load existing log if exists
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_data = json.load(f)
    else:
        log_data = []

    # Append new entry
    log_data.append(log_entry)

    # Save updated log
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)



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
    iteration=1
)



