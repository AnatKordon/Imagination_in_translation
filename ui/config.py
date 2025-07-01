import os
from pathlib import Path

# Directories (MIGHT NEED TO BE CHANGED)
ROOT = Path(__file__).resolve().parent
GT_DIR = ROOT.parent / "data" / "wilma_ground_truth" # folder with target images
GEN_DIR = ROOT / "generated" # folder for generated images
LOG_DIR = ROOT / "data" / "logs" # folder with CSV log files per user/session
FALLBACK = ROOT / "mona_lisa_2.jpg" # TO BE REMOVED: dummy placeholder picture that is used instead of a generated one

MAX_ATTEMPTS = 5 # Attempts to improve the description are limited to 5
IMG_H = 260  # The height of images is limited to 260 px so the user doesn't need to scroll

