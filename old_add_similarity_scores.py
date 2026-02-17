import pandas as pd
from pathlib import Path
from config import  GT_DIR, GEN_DIR, LOG_DIR
from similarity.CLIP_similarity import get_clip_visual_embedding, get_clip_text_embedding
from similarity.vgg_similarity import compute_similarity_score, VGGEmbedder
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights
from similarity.LPIPS_similarity import compute_lpips_score

CSV_PATH = Path(LOG_DIR) / "improving_descriptions_step_by_step.csv"  # Path to my CSV for analysis
OUTPUT_CSV = Path(LOG_DIR) / "improving_descriptions_step_by_step_with_clip_lpips.csv"

# ---- Process CSV ----
df = pd.read_csv(CSV_PATH)


#vgg setup
weights = VGG16_Weights.IMAGENET1K_V1
vgg_imagenet = models.vgg16(weights=weights)

vgg_last_conv_embedder = VGGEmbedder(model=vgg_imagenet, layer='Classifier_4') # this is fc7, else: 'Layer_30'

if __name__ == "__main__":
    # CLIP
    clip_distances = []
    clip_scaled_similarities = []

    lpips_distances = []
    lpips_scaled_similarities = []

    vgg_last_conv_distances = []
    vgg_last_conv_similarities = []

    clip_text_embeddings = []
    token_nums = []
    for idx, row in df.iterrows():
        gt_path = GT_DIR / row['gt']
        gen_path = Path(row['gen'])
        gt_embed = get_clip_visual_embedding(gt_path)
        gen_embed = get_clip_visual_embedding(gen_path)
        # Compute cosine similarity using PyTorch
        clip_scaled_sim, clip_cosine_distance = compute_similarity_score(gt_embed, gen_embed)
        clip_distances.append(clip_cosine_distance)
        clip_scaled_similarities.append(clip_scaled_sim)

        # LPIPS (based on vgg)
        lpips_dist_value = compute_lpips_score(gt_path, gen_path)
        lpips_distances.append(lpips_dist_value)

        #add CLIP textual embedding for each prompt
        clip_text_embedding, real_token_num = get_clip_text_embedding(row['prompt'])
        #clip_text_embeddings.append(clip_text_embedding)
        token_nums.append(real_token_num)
          
        embedding_gt = vgg_last_conv_embedder.get_embedding(img_path=str(gt_path))
        embedding_gen = vgg_last_conv_embedder.get_embedding(img_path=str(gen_path))

        # Compute similarity score
        vgg_last_conv_similarity, vgg_last_conv_distance = compute_similarity_score(embedding1=embedding_gt, embedding2=embedding_gen)

        # add logging for distance score (not the simialrity we show the user - UserID, SessionID, Iteration, cosine_distance)
        vgg_last_conv_distances.append(vgg_last_conv_distance)
        vgg_last_conv_similarities.append(vgg_last_conv_similarity)

    #append to csv
    df["clip_cosine_distance"] = clip_distances
    df["clip_scaled_similarity"] = clip_scaled_similarities

    df["lpips_distance"] = lpips_distances

    df["vgg_last_conv_distance"] = vgg_last_conv_distances
    df["vgg_last_conv_similarity"] = vgg_last_conv_similarities

    # df['clip_text_embedding'] = clip_text_embeddings # I'm not using the raw embedding in the csv because it will blow up
    df["real_token_num"] = token_nums

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved results to {OUTPUT_CSV}")
