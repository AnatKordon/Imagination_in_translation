import os
from pathlib import Path
from dataclasses import dataclass
import yaml
import pandas as pd

# Directories - old data, currently experiment runs in jatos (MIGHT NEED TO BE CHANGED)
ROOT = Path(__file__).resolve().parent  # root of the project, where this file is located
GT_DIR = ROOT / "GT_images" / "wilma_ground_truth" / "CHOSEN_IMAGES" # folder with target images
LOG_DIR = ROOT / "logs" / "users_data" # folder with CSV log files per user/session
# GEN_DIR = ROOT / "logs"/"gen_images"  # folder where generated images are saved
# DRIVE_FOLDER = "https://drive.google.com/drive/folders/1bbDtQ7WrDTyaoMTJfIlgix7QUG3is78U?usp=drive_link"
# STYLE_IMAGE = GT_DIR / "sample_image" / "bridge_l.jpg"  # Path to the style image
# IMG_H = 260  # The height of images is limited to 260 px so the user doesn't need to scroll
MAX_LENGTH = 10000  # Maximum length of the prompt text
N_OUT = 1  # Number of images to generate per prompt (1 or 4)
MAX_SESSIONS = 5            # total sessions per participant
REQUIRED_ATTEMPTS = 3         # exactly 3 attempts per session
# PROLIFIC_URL = "https://app.prolific.com/submissions/complete?cc=C1OJX362"  # Prolific completion URL

# =====================================================================
# Analysis paths.
#
# condition_maps.yaml is the declarative source (which conditions exist,
# their generation/task, canonical filenames). This module is the logic
# layer that turns it into paths + helpers. A condition slug is
# "<gen>_<task>" (e.g. aigen_perc); its folder path is the slug split on
# "_". The JATOS export folder is auto-discovered by glob.
# =====================================================================
YAML_PATH = ROOT / "condition_maps.yaml"
with open(YAML_PATH, "r") as f:
    mapping_data = yaml.safe_load(f)

DATASET = mapping_data["DATASET"]
FILES = mapping_data["FILENAMES"]
CONDITIONS = list(mapping_data["CONDITIONS"].keys())          # ordered 9 slugs
LEGACY = mapping_data.get("LEGACY", {})

GROUPS_BY_GEN = {
    g: [c for c in CONDITIONS if mapping_data["CONDITIONS"][c]["generation"] == g]
    for g in ("aigen", "nogen", "plain")
}
GROUPS_BY_TASK = {
    t: [c for c in CONDITIONS if mapping_data["CONDITIONS"][c]["task"] == t]
    for t in ("perception", "immediate", "delay")
}


@dataclass
class Paths:
    """Resolved paths for a single condition across the three trees."""
    condition: str
    experiment_dir: Path
    participants_dir: Path     # auto-globbed jatos_results_files_* (None until data lands)
    processed_dir: Path
    analysis_dir: Path

    def csv(self, key: str) -> Path:
        """Path to a canonical CSV by FILENAMES key, e.g. .csv('trials_final')."""
        return self.processed_dir / FILES[key]


def paths_for(condition: str) -> Paths:
    """Build all paths for any Full_experiment condition slug ('<gen>_<task>')."""
    gen, task = condition.split("_")          # "aigen_perc" -> "aigen", "perc"
    rel = Path(DATASET) / gen / task          # nested: <gen>/<task>
    exp = ROOT / "Data" / "participants_data" / rel
    jatos = next(exp.glob("jatos_results_files_*"), None)   # None until data lands
    return Paths(
        condition=condition,
        experiment_dir=exp,
        participants_dir=jatos,
        processed_dir=ROOT / "Data" / "processed_data" / rel,
        analysis_dir=ROOT / "analysis" / "outputs" / rel,  # outputs/ separates results from code
    )


def _legacy_paths(condition: str) -> Paths:
    """Build paths for a frozen pilot from its explicit *_sub fields."""
    lc = LEGACY[condition]
    exp = ROOT / "Data" / "participants_data" / lc["exp_sub"]
    return Paths(
        condition=condition,
        experiment_dir=exp,
        participants_dir=exp / lc["jatos_sub"],
        processed_dir=ROOT / "Data" / "processed_data" / lc["processed_sub"],
        analysis_dir=ROOT / "analysis" / lc["analysis_sub"],
    )


# Cross-condition output areas (sit beside aigen/nogen/plain, under DATASET).
COMBINED_PROCESSED_DIR = ROOT / "Data" / "processed_data" / DATASET / "combined"
COMBINED_ANALYSIS_DIR = ROOT / "analysis" / "outputs" / DATASET / "combined"


def load(conditions, sim: bool = False) -> "pd.DataFrame":
    """Concatenate the analysis table for ANY list of condition slugs into one
    tidy DataFrame, tagged with condition/generation/task columns. sim=True loads
    the similarity version (trials_final_sim). Conditions whose CSV is missing are
    skipped with a printed notice."""
    key = "trials_final_sim" if sim else "trials_final"
    frames = []
    for c in conditions:
        p = paths_for(c).csv(key)
        if not p.exists():
            print(f"skip {c}: {p.name} not found ({p})")
            continue
        df = pd.read_csv(p)
        cm = mapping_data["CONDITIONS"][c]
        df["condition"], df["generation"], df["task"] = c, cm["generation"], cm["task"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --- Back-compat single-condition globals (every existing script keeps working) ---
CONDITION = mapping_data["CURRENT_CONDITION"]
_p = _legacy_paths(CONDITION) if CONDITION in LEGACY else paths_for(CONDITION)
EXPERIMENT_DIR = _p.experiment_dir
PARTICIPANTS_DIR = _p.participants_dir
PROCESSED_DIR = _p.processed_dir
ANALYSIS_DIR = _p.analysis_dir
# For legacy conditions the analysis CSV name is explicit; otherwise the
# no-similarity canonical table is the default.
CSV_PATH = (PROCESSED_DIR / LEGACY[CONDITION]["df"]) if CONDITION in LEGACY else _p.csv("trials_final")

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

#API_CALL = "stability_ai"  # ["open_ai", "stability_ai"]