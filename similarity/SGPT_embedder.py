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
                 batch_size: int = 8,
                 pooling: str = "mean"):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.batch = batch_size
        assert pooling in {"mean", "weighted_end", "weighted_start"}
        self.pooling = pooling

    def encode(self, texts: List[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch):
                batch = texts[i:i+self.batch]
                toks = self.tok(batch, padding=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt").to(self.device)
                hid = self.model(**toks, output_hidden_states=True, return_dict=True).last_hidden_state  # [bs, L, H]
                bs, sequence_len, H = hid.shape #batch size, sequence length, hidden size
                mask = toks["attention_mask"].unsqueeze(-1).float()  # [bs,L,1]

                # weights on every token in the sequence - Adva's suggestion is weighted mean (last) - but i'm going with flat mean for now because in my descriptions the main things isn't at the end
                if self.pooling == "weighted_end":
                    weights = torch.arange(1, sequence_len+1, device=hid.device).float().view(1,sequence_len,1).expand(bs,sequence_len,H)
                elif self.pooling == "weighted_start":
                    weights = torch.arange(sequence_len, 0, -1, device=hid.device).float().view(1,sequence_len,1).expand(bs,sequence_len,H)
                else:  # "mean"
                    weights = torch.ones_like(hid)

                num = (hid * mask * weights).sum(dim=1)                   # [bs,H]
                den = (mask * weights).sum(dim=1) + 1e-8                  # [bs,1]
                emb = num / den
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                out.append(emb.cpu().numpy())
        return np.vstack(out)
    

