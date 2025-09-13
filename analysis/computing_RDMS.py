# Analysis/aggregate.py
from pathlib import Path
import sys
import pandas as pd

# Make sure we can import config.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config  # now we can reuse your paths

# computing RDM correlation between gt images (using VGG) and prompt text (using SGPT)
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms, models
from torchvision.models import VGG16_Weights
from PIL import Image
from similarity.vgg_similarity import VGGEmbedder, _to_numpy, compute_similarity_score
from similarity.SGPT_embedder import SGPTEmbedder
# -----------------------------
# Utility
# -----------------------------

weights = VGG16_Weights.IMAGENET1K_V1
vgg_imagenet = models.vgg16(weights=weights)


def cosine_sim_matrix(emb: np.ndarray) -> np.ndarray:
    """emb shape [N, D], assumed L2-normalized. Returns NxN cosine similarity."""
    return emb @ emb.T

def cosine_rdm(emb: np.ndarray) -> np.ndarray:
    """Return cosine *distance* matrix (NxN), i.e., 1 - cosine_sim."""
    # emb need not be normalized; pdist with 'cosine' handles it.
    # But we’ll L2 normalize explicitly for stability.
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    D = squareform(pdist(emb, metric="cosine"))  # NxN, zeros on diag
    return D

def rdm_upper_tri_vector(rdm: np.ndarray) -> np.ndarray:
    n = rdm.shape[0]
    iu = np.triu_indices(n, k=1)
    return rdm[iu]


# -----------------------------
# RDM pipeline
# -----------------------------
def build_rdm_alignment(
    df: pd.DataFrame,
    gt_root: Path,
    vgg_layer: str = "Classifier_4",
    sgpt_model: str = "Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit",
    max_len: int = 512, # okay as max token len for my dataset is 219
    batch_size: int = 8, 
    correlation_method: str = "spearman",
    drop_duplicates: bool = False # I don't have any by this point
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    For each (uid, attempt):
      - Build VGG RDM over that user's GT set.
      - Build SGPT RDM over that user's prompts (same GT order).
      - Compute Spearman correlation between upper triangles.

    Returns:
      df_rdm_results: uid, attempt, n_items, rdm_corr_spearman
      rdm_vgg_dict: {(uid, attempt): RDM ndarray}
      rdm_sgpt_dict: {(uid, attempt): RDM ndarray}
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 0) Optional: ensure one row per (uid, attempt, gt). If multiple, average the text per (uid,attempt,gt).
    if drop_duplicates:
        # If duplicates exist, keep first; or you can groupby and join texts—here we keep first for alignment.
        dup_cols = ["uid", "attempt", "gt"]
        if df.duplicated(subset=dup_cols).any():
            df = (df.sort_values(dup_cols)
                    .drop_duplicates(subset=dup_cols, keep="first")
                    .reset_index(drop=True))

    # Initialize embedder for desired layer
    embedder = VGGEmbedder(model=vgg_imagenet, layer='Classifier_4') #can also try "Layer_30" which is last conv layer
    # 1) Precompute/cache VGG embeddings for all GT images present
    vgg = VGGEmbedder(model=vgg_imagenet, layer=vgg_layer)
    unique_gts = df["gt"].dropna().unique().tolist()
    gt2vgg: Dict[str, np.ndarray] = {}
    for gt in unique_gts:
        p = Path(gt_root) / gt
        gt2vgg[gt] = vgg.get_embedding(str(p))

    # 2) Prepare SGPT embedder
    sgpt = SGPTEmbedder(model_name=sgpt_model, device=device, max_length=max_len, batch_size=batch_size)

    # 3) Iterate per (uid, attempt)
    results = []
    rdm_vgg_dict = {}
    rdm_sgpt_dict = {}

    for (uid, attempt), sub in df.groupby(["uid", "attempt"], sort=True):
        # Order by GT to ensure consistent pairing between image/text RDMs
        sub = sub.dropna(subset=["gt", "prompt"]).copy()
        if sub.empty:
            continue
        sub = sub.sort_values("gt")

        # Image embeddings (VGG) in this order
        img_embs = np.stack([gt2vgg[g] for g in sub["gt"].tolist()], axis=0)  # [N, D_img]

        # Text embeddings (SGPT) in the same GT order
        txt_embs = sgpt.encode(sub["prompt"].tolist())  # [N, D_txt], already L2-normalized

        # Build cosine distance RDMs
        rdm_vgg = cosine_rdm(img_embs)    # NxN
        rdm_sgpt = cosine_rdm(txt_embs)   # NxN
        print(f"shapes: {rdm_vgg.shape}, {rdm_sgpt.shape}")
        # Store RDMs if needed later
        rdm_vgg_dict[(uid, attempt)] = rdm_vgg
        rdm_sgpt_dict[(uid, attempt)] = rdm_sgpt

        # Compare upper triangles (Spearman)
        if rdm_vgg.shape[0] >= 3:
            v1 = rdm_upper_tri_vector(rdm_vgg)
            v2 = rdm_upper_tri_vector(rdm_sgpt)
            if correlation_method == "spearman":
                r, p = spearmanr(v1, v2)
            elif correlation_method == "pearson":
                r, p = pearsonr(v1, v2)

        else:
            # Not enough items to form a stable RDM
            r, p = np.nan, np.nan

        results.append({
            "uid": uid,
            "attempt": attempt,
            "n_items": int(rdm_vgg.shape[0]),
            f"rdm_corr_{correlation_method}": r,
            "rdm_corr_p": p,
        })

    df_rdm_results = pd.DataFrame(results).sort_values(["uid", "attempt"]).reset_index(drop=True)
    return df_rdm_results, rdm_vgg_dict, rdm_sgpt_dict

if __name__ == "__main__":
    # Example usage
    CSV_PATH = config.PROCESSED_DIR / 'participants_log_with_gpt_with_distances_and_alignment_and_len_pilot_08092025_.csv'
    
    OUTPUT_CSV = "/mnt/hdd/anatkorol/Imagination_in_translation/Data/processed_data/08092025_pilot/rdm_alignment_results.csv" # it is a smaller csv

    df = pd.read_csv(CSV_PATH)
    df_rdm, rdm_vgg_dict, rdm_sgpt_dict = build_rdm_alignment(df, gt_root=config.GT_DIR)

    print(df_rdm.head())
    df_rdm.to_csv(OUTPUT_CSV, index=False)
    print(f"RDM alignment results saved to {OUTPUT_CSV}")