# compute_clip_similarity.py
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from torch.nn.functional import cosine_similarity
from scipy.spatial.distance import cosine

import sys
from pathlib import Path

# Add the project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))


# ---- CONFIG ----
from ui.config import GT_DIR, GEN_DIR, LOG_DIR
CSV_PATH = Path(LOG_DIR / "improving_descriptions_step_by_step.csv")  # Path to my CSV for analysis
OUTPUT_CSV = Path(LOG_DIR / "improving_descriptions_step_by_step_with_clip.csv")

# ---- Load CLIP ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_clip_embedding(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)                                                                                                                                                                                                                                                                                                                                                                                
    return features[0]  # (512,)

# ---- Process CSV ----
df = pd.read_csv(CSV_PATH)


def compute_cosine_similarity_metrics(embedding1, embedding2):
    """
    Computes cosine similarity and cosine distance (and optionally a scaled similarity score).

    Args:
        embedding1: np.ndarray or torch.Tensor
        embedding2: np.ndarray or torch.Tensor

    Returns:
        scaled_similarity (0–100), cosine_distance (float)
    """
    if isinstance(embedding1, torch.Tensor):
        embedding1 = embedding1.detach().cpu().numpy()
    if isinstance(embedding2, torch.Tensor):
        embedding2 = embedding2.detach().cpu().numpy()

    cosine_dist = cosine(embedding1, embedding2)  # 0 = identical
    cosine_sim = 1 - cosine_dist                  # [-1, 1]
    scaled_sim = int(((cosine_sim + 1) / 2) * 100)  # map to [0, 100]

    return scaled_sim, cosine_dist


clip_distances = []
for idx, row in df.iterrows():
    gt_path = GT_DIR / row['gt']
    gen_path = GEN_DIR / row['gen']
    try:
        gt_embed = get_clip_embedding(gt_path)
        gen_embed = get_clip_embedding(gen_path)
        # Compute cosine similarity using PyTorch
        clip_scaled_sim, clip_cosine_distance = compute_cosine_similarity_metrics(gt_embed, gen_embed)
        cos_sim_using_torch = cosine_similarity(gt_embed, gen_embed, dim=0).item()
        clip_distance = 1 - cos_sim_using_torch
    except Exception as e:
        print(f"Error with row {idx}: {e}")
        clip_distance = None
    clip_distances.append(clip_distances)

print(f'torch nn clip distance: {clip_distance}')
# df["clip_cosine_distance_using_scipy"] = clip_cosine_distance
# df["clip_scaled_similarity"] = clip_scaled_sim
df["clip_cosine_distance_using_torch"] = clip_distances

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved results to {OUTPUT_CSV}")
