import os
from pathlib import Path

# Directories (MIGHT NEED TO BE CHANGED)
ROOT = Path(__file__).resolve().parent  # root of the project, where this file is located
GT_DIR = ROOT.parent / "GT_images" / "wilma_ground_truth" # folder with target images
LOG_DIR = ROOT.parent / "logs" / "users_data" # folder with CSV log files per user/session
GEN_DIR = ROOT.parent / "logs"/"gen_images"  # folder where generated images are saved
STYLE_IMAGE = GT_DIR / "sample_image" / "bridge_l.jpg"  # Path to the style image
MAX_ATTEMPTS = 5 # Attempts to improve the description are limited to 5
IMG_H = 260  # The height of images is limited to 260 px so the user doesn't need to scroll
MAX_LENGTH = 10000  # Maximum length of the prompt text
##  Params for image generation


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
    "prompt": "a playground with ",
    "image": str(GT_DIR /"sample_image"/"bridge_l.jpg"),  # Path to the style image
    "aspect_ratio": "1:1", 
    "output_format": "png",
    "model": "sd3.5-large-turbo"
}