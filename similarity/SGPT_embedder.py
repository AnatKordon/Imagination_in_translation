from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel


# -----------------------------
# SGPT text embedder (batched, normalized)
# -----------------------------
class SGPTEmbedder:
    def __init__(self,
                 model_name: str = "Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit",
                 device: str = None,
                 max_length: int = 2047, # from model card, base model max length is 2048 
                 batch_size: int = 8):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size

    def encode(self, texts: List[str]) -> np.ndarray:
        """Weighted-mean pooled embeddings, L2-normalized, shape [N, H]."""
        embs = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                toks = self.tokenizer(batch, padding=True, truncation=True,
                                      max_length=self.max_length, return_tensors="pt").to(self.device)
                out = self.model(**toks, output_hidden_states=True, return_dict=True)
                last_hidden = out.last_hidden_state  # [bs, seq, hid]

                # Weighted mean pooling (your recipe)
                bs, seq_len, hid = last_hidden.shape #batch size, sequence length, hidden size
                weights = torch.arange(1, seq_len+1, device=last_hidden.device).float().view(1, seq_len, 1)
                weights = weights.expand(bs, seq_len, hid)
                mask = toks["attention_mask"].unsqueeze(-1).expand(bs, seq_len, hid).float()

                sum_emb = torch.sum(last_hidden * mask * weights, dim=1)
                sum_w = torch.sum(mask * weights, dim=1)
                emb = sum_emb / (sum_w + 1e-8)  # [bs, hid]

                # L2 normalize
                emb = emb / (emb.norm(p=2, dim=1, keepdim=True) + 1e-8)
                embs.append(emb.cpu().numpy())

        return np.vstack(embs)
