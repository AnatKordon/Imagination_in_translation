import os
from pathlib import Path

# Directories (MIGHT NEED TO BE CHANGED)
ROOT = Path(__file__).resolve().parent  # root of the project, where this file is located
GT_DIR = ROOT.parent / "GT_images" / "wilma_ground_truth" # folder with target images
LOG_DIR = ROOT.parent / "logs" / "users_data" # folder with CSV log files per user/session
GEN_DIR = ROOT.parent / "gen_images"  # folder with images for the UI (e.g. fallback image)

MAX_ATTEMPTS = 5 # Attempts to improve the description are limited to 5
IMG_H = 260  # The height of images is limited to 260 px so the user doesn't need to scroll
MAX_LENGTH = 10000  # Maximum length of the prompt text
##  Params for image generation


## for erroe handling
websites = [".com", ".net", ".org", ".edu", ".gov", ".io", ".co", ".uk", ".de", ".fr", ".jp", ".ru","https", "http", "www."]

params = {
    "prompt": "a playground with ",
    "aspect_ratio": "1:1",
    "seed": 1,
    "output_format": "png",
    "model": "sd3.5-large-turbo"
}