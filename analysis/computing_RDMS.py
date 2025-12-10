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
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr, spearmanr
# -----------------------------
# Utility
# -----------------------------

weights = VGG16_Weights.IMAGENET1K_V1
vgg_imagenet = models.vgg16(weights=weights)


# def cosine_sim_matrix(emb: np.ndarray) -> np.ndarray:
#     """emb shape [N, D], assumed L2-normalized. Returns NxN cosine similarity."""
#     return emb @ emb.T

def cosine_rdm(emb: np.ndarray) -> np.ndarray:
    """Return cosine *distance* matrix (NxN), i.e., 1 - cosine_sim."""
    # emb need not be normalized; pdist with 'cosine' handles it.
    # But we’ll L2 normalize explicitly for stability.
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    D = squareform(pdist(emb, metric="cosine"))  # NxN, zeros on diag
    return D


def upper_tri(x: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(x, k=1)
    return x[iu]

def save_rdm_csv(rdm: np.ndarray, labels: List[str], path: Path):
    df = pd.DataFrame(rdm, index=labels, columns=labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, encoding="utf-8")

# -----------------------------
# RDM pipeline
# -----------------------------
def build_rdm_alignment(
    df: pd.DataFrame,
    gt_root: Path,
    out_dir: Path,
    attempts=(1,2,3),
    vgg_layer: str = "Classifier_4",
    sgpt_model: str = "Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit",
    sgpt_pooling="mean",  # can be "mean", "weighted_end", "weighted_start"
    max_len: int = 2047, # from model card token limit is 2048, so 2047 should be safe
    batch_size: int = 8, 
    correlation_method: str = "spearman",
    align="intersection",                 # 'intersection' -> same GT order across attempts per uid
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

    out_dir.mkdir(parents=True, exist_ok=True)
    
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
    for uid, df_uid in df.groupby("uid"):
        # decide GT order per attempt
        if align == "intersection":
            sets = []
            for a in attempts:
                s = set(df_uid.loc[df_uid["attempt"]==a, "gt"].dropna().unique())
                sets.append(s)
            common = set.intersection(*sets) if len(sets)>0 else set()
            if len(common) < 3:
                # not enough items to form a decent RDM across all attempts
                continue
            gt_order_by_attempt = {a: sorted(common) for a in attempts}
        else:
            gt_order_by_attempt = {a: sorted(df_uid.loc[df_uid["attempt"]==a, "gt"].dropna().unique())
                                   for a in attempts}

        # store RDMs to plot per uid
        rdms_vgg: Dict[int, np.ndarray] = {}
        rdms_txt: Dict[int, np.ndarray] = {}
        labels_by_attempt: Dict[int, List[str]] = {}

        # per attempt
        for a in attempts:
            order = gt_order_by_attempt[a]
            if len(order) < 3:  # need at least 3 items for a stable RDM
                continue

            sub = (df_uid[df_uid["attempt"]==a]
                    .dropna(subset=["gt","prompt"])
                    .sort_values("gt")
                    .drop_duplicates(subset=["gt"], keep="first"))
            sub = sub[sub["gt"].isin(order)]
            # enforce exact order
            sub = sub.set_index("gt").loc[order].reset_index()

            gts = sub["gt"].tolist()
            prompts = sub["prompt"].fillna("").tolist()
            if len(gts) < 3: continue

            img_embs = np.stack([gt2vgg[g] for g in gts], axis=0)   # [N,D_img]
            txt_embs = sgpt.encode(prompts)                         # [N,D_txt]

            rdm_v = cosine_rdm(img_embs)
            rdm_t = cosine_rdm(txt_embs)

            # # save RDM CSVs with labels
            # save_rdm_csv(rdm_v, gts, out_dir / f"uid_{uid}_attempt_{a}_VGG_RDM.csv")
            # save_rdm_csv(rdm_t, gts, out_dir / f"uid_{uid}_attempt_{a}_SGPT_RDM.csv")

            # store for plotting & correlation
            rdms_vgg[a] = rdm_v
            rdms_txt[a] = rdm_t
            labels_by_attempt[a] = gts

            # Spearman correlation of upper triangles
            v1, v2 = upper_tri(rdm_v), upper_tri(rdm_t)
            r, p = (np.nan, np.nan) if len(v1) < 3 else spearmanr(v1, v2)
            results.append({"uid": uid, "attempt": a, "n_items": len(gts),
                            "rdm_corr_spearman": r, "rdm_corr_p": p})

        # draw a panel per uid (2 rows × 3 attempts)
        if len(rdms_vgg) > 0:
            ncols = len(attempts)
            fig, axes = plt.subplots(2, ncols, figsize=(4*ncols, 8), constrained_layout=True)
            for j, a in enumerate(attempts):
                r_v = rdms_vgg.get(a); r_t = rdms_txt.get(a)
                labs = labels_by_attempt.get(a, [])
                for ax in [axes[0, j], axes[1, j]]:
                    ax.axis("off")  # default off; enable if we have data

                if r_v is not None:
                    sns.heatmap(r_v, ax=axes[0, j], square=True, vmin=0, vmax=2, cmap="magma",
                                cbar=(j==ncols-1), xticklabels=labs, yticklabels=labs)
                    axes[0, j].tick_params(axis="x", rotation=90)
                    axes[0, j].tick_params(axis="y", rotation=0)
                    axes[0, j].set_title(f"VGG — Attempt {a}")
                    axes[0, j].axis("on")

                if r_t is not None:
                    sns.heatmap(r_t, ax=axes[1, j], square=True, vmin=0, vmax=2, cmap="magma",
                                cbar=(j==ncols-1), xticklabels=labs, yticklabels=labs)
                    axes[1, j].tick_params(axis="x", rotation=90)
                    axes[1, j].tick_params(axis="y", rotation=0)
                    axes[1, j].set_title(f"SGPT ({sgpt_pooling}) — Attempt {a}")
                    axes[1, j].axis("on")

            fig.suptitle(f"RDMs — uid {uid}", y=1.02, fontsize=14)
            fig_path = out_dir / f"uid_{uid}_RDMs_panel.png"
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)

    df_rdm = pd.DataFrame(results).sort_values(["uid","attempt"]).reset_index(drop=True)
    # also write the tidy results table next to the images
    # df_rdm.to_csv(out_dir / "rdm_alignment_per_uid_attempt_spearman_002.csv", index=False)
    return df_rdm

if __name__ == "__main__":
    # Example usage
    CSV_PATH = config.PROCESSED_DIR / 'participants_log_with_gpt_pilot_08092025_gpt-image-1_generation_with_distances_and_alignment_pilot_08092025.csv'
    OUT_DIR = config.ANALYSIS_DIR / "pilot_gpt-image-1" / "rdms"
    OUTPUT_CSV = "/mnt/hdd/anatkorol/Imagination_in_translation/Data/processed_data/08092025_pilot/rdm_alignment_results_gpt-image-1-generation.csv" # it is a smaller csv

    df = pd.read_csv(CSV_PATH)
    df_rdm = build_rdm_alignment(df, gt_root=config.GT_DIR, out_dir=OUT_DIR)

    print(df_rdm.head())
    df_rdm.to_csv(OUTPUT_CSV, index=False)
    print(f"RDM alignment results saved to {OUTPUT_CSV}")