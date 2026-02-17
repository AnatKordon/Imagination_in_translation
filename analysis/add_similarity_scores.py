
# this code adds similarity scores according to the functions in similarity folder
from pathlib import Path
import sys
import pandas as pd

# Make sure we can import config.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config as config
from similarity.CLIP_similarity import get_clip_visual_embedding, get_clip_text_embedding, cosine_similarity
from similarity.vgg_similarity import compute_similarity_score, VGGEmbedder
from visualize_per_ppt import path_from_row_jatos
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights
from similarity.LPIPS_similarity import compute_lpips_score

#change the paths acccording to task
CSV_PATH = config.PROCESSED_DIR /"ppt_w_gpt_trials_corrected.csv" # should use corrected gt if nesseceary, this is the original good file with participant data: "participants_log_with_gpt_pilot_08092025.csv"  # Path to my CSV for analysis
OUTPUT_CSV = config.PROCESSED_DIR / "ppt_w_gpt_w_similarity_trials.csv"  # Path to save the output CSV with similarity scores

USE_CORRECTED = True  # <-- set this based on whether gt column was corrected "corrected: True/False"

# ---- Process CSV ----
df = pd.read_csv(CSV_PATH).copy()

# handling missing gt values:
import numpy as np
import pandas as pd

USE_CORRECTED = True  # <-- set this based on your "corrected: True/False"

def pick_gt_name(row: pd.Series, use_corrected: bool) -> str | None:
    """Return the filename/title to use for GT, or None if missing."""
    if use_corrected:
        val = row.get("gt_corrected")
        # if gt_corrected missing, you said: "ignore nan" => no similarity
        if pd.isna(val) or str(val).strip() == "":
            return None
        return str(val)
    else:
        val = row.get("gt")
        if pd.isna(val) or str(val).strip() == "":
            return None
        return str(val)

#vgg setup
weights = VGG16_Weights.IMAGENET1K_V1
vgg_imagenet = models.vgg16(weights=weights)

vgg_fc7_embedder = VGGEmbedder(model=vgg_imagenet, layer='Classifier_4') # this is fc7, else: 'Layer_30'

if __name__ == "__main__":
    # CLIP

    clip_distances = []
    clip_vis_text_similarities = []
    clip_text_embeddings = []
    lpips_distances = []
    lpips_scaled_similarities = []

    vgg_fc7_distances = []
    vgg_fc7_similarities = []
    vgg_fc7_scaled_similarities = []

    token_nums = []
    for idx, row in df.iterrows():
        # path for gt and gen per row
        gt_name = pick_gt_name(row, USE_CORRECTED)
        gen_path = path_from_row_jatos(row) # in the function it knows to use the 'gen' row and turn it into a full path (Previously I used function: path_from_row that worked but failed for shuffled data as it can't be reconstructed)
        #extract prompt earlier
        prompt_clip_embed, token_num = get_clip_text_embedding(row['prompt']) # Note that for prompts longer than 77 words it is truncated
        token_nums.append(token_num)

        #handle missing GT case: if no GT, we can't compute similarity, so we can either skip or assign a default value (e.g., NaN)
        if gt_name is None:
            # no GT => no similarity
            clip_distances.append(pd.NA)
            vgg_fc7_distances.append(pd.NA)
            clip_vis_text_similarities.append(pd.NA)  # up to you if you still want prompt alignment
            continue

        gt_path = config.GT_DIR / gt_name

        
        
        gt__clip_embed = get_clip_visual_embedding(gt_path)
        gen__clip_embed = get_clip_visual_embedding(gen_path)
        
        
        # Compute cosine similarity for images
        _, _, clip_cosine_distance = compute_similarity_score(gt__clip_embed, gen__clip_embed)
        clip_distances.append(clip_cosine_distance)

        #compute visual_text alighnment using clip
        clip_visual_text_alignment = cosine_similarity(gen__clip_embed, prompt_clip_embed)
        clip_vis_text_similarities.append(clip_visual_text_alignment) # note that this is similarity, not distance


        # # LPIPS (based on vgg)
        # lpips_dist_value = compute_lpips_score(gt_path, gen_path)
        # lpips_distances.append(lpips_dist_value)

        # #add CLIP textual embedding for each prompt
        # clip_text_embedding, real_token_num = get_clip_text_embedding(row['prompt'])
        # #clip_text_embeddings.append(clip_text_embedding)
        # token_nums.append(real_token_num)
        embedding_gt = vgg_fc7_embedder.get_embedding(img_path=str(gt_path))
        embedding_gen = vgg_fc7_embedder.get_embedding(img_path=str(gen_path))

        # Compute similarity score
        _, _, vgg_fc7_distance = compute_similarity_score(embedding1=embedding_gt, embedding2=embedding_gen)

        # add logging for distance score (not the simialrity we show the user - UserID, SessionID, Iteration, cosine_distance)
        vgg_fc7_distances.append(vgg_fc7_distance)

    #append to csv
    df["clip_cosine_distance"] = clip_distances
    # df["clip_scaled_similarity"] = clip_scaled_similarities

    # df["lpips_distance"] = lpips_distances

    df["vgg_fc7_distance"] = vgg_fc7_distances
    # df["vgg_fc7_similarity"] = vgg_fc7_similarities
    # df["vgg_fc7_scaled_similarity"] = vgg_fc7_scaled_similarities

    df["clip_vis_text_similarity"] = clip_vis_text_similarities
    df["token_num"] = token_nums

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved results to {OUTPUT_CSV}")
