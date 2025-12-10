import os
from pathlib import Path

# Directories (MIGHT NEED TO BE CHANGED)
ROOT = Path(__file__).resolve().parent  # root of the project, where this file is located
GT_DIR = ROOT / "GT_images" / "wilma_ground_truth" / "chosen" # folder with target images
LOG_DIR = ROOT / "logs" / "users_data" # folder with CSV log files per user/session
# GEN_DIR = ROOT / "logs"/"gen_images"  # folder where generated images are saved
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1bbDtQ7WrDTyaoMTJfIlgix7QUG3is78U?usp=drive_link"
STYLE_IMAGE = GT_DIR / "sample_image" / "bridge_l.jpg"  # Path to the style image
IMG_H = 260  # The height of images is limited to 260 px so the user doesn't need to scroll
MAX_LENGTH = 10000  # Maximum length of the prompt text
N_OUT = 1  # Number of images to generate per prompt (1 or 4)
MAX_SESSIONS = 8             # total sessions per participant
REQUIRED_ATTEMPTS = 3         # exactly 3 attempts per session
PROLIFIC_URL = "https://app.prolific.com/submissions/complete?cc=C1OJX362"  # Prolific completion URL

#analysis
ANALYSIS_DIR = ROOT / "analysis"  
PARTICIPANTS_DIR = ROOT / "Data" / "participants_data" / "pilot_08092025_gpt-image-1" #"pilot_08092025_SD" # original pilot data with sd3.5 generations
PROCESSED_DIR = ROOT / "Data" / "processed_data" / "08092025_pilot"
PANELS_DIR = ANALYSIS_DIR / "pilot_gpt-image-1" / "panels" # This is changed according to the model used

GPT_IMAGES = ROOT / "Data" / "participants_data" / "pilot_08092025_gpt-image-1"  # folder with gpt-image-1 generations for the pilot participants data



## for error handling
websites = [".com", ".net", ".org", ".edu", ".gov", ".io", ".co", ".uk", ".de", ".fr", ".jp", ".ru","https", "http", "www."]
KNOWN_ERRORS = {
    "required_field": "some-field: is required",
    "content_moderation": "Your request was flagged by our content moderation system",
    "payload_too_large": "payloads cannot be larger than 10MiB",
    "language_not_supported": "English is the only supported language",
    "rate_limit": "You have exceeded the rate limit",
    "server_error": "An unexpected server error has occurred",
    "Invalid_Api" :"authorization: invalid or missing header value"
}


## Params

params = {
    "prompt": "",
    #  "image": str(GT_DIR /"sample_image"/"bridge_l.jpg"),  # Path to the style image
    "aspect_ratio": "1:1", 
    "output_format": "png",
    # "model": "sd3.5-large-turbo", # , "sd3.5-large", "sd3.5-large-turbo"
    "revised_prompt": None  # this is for openai api - to assure that no revision of prompt is done
}

API_CALL = "stability_ai"  # ["open_ai", "stability_ai"]