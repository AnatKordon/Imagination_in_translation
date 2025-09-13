# compute_clip_similarity.py
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

from scipy.spatial.distance import cosine


# ---- Load CLIP ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_visual_embedding(image_path: Path, clip_processor=clip_processor) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)                                                                                                                                                                                                                                                                                                                                                                                
    return features[0]  # (512,)


def get_clip_text_embedding(text: str, clip_tokenizer=clip_tokenizer, clip_model=clip_model) -> torch.Tensor:
    real_token_num = len(clip_tokenizer.tokenize(text))
    tokens = clip_tokenizer(text, truncation=True, max_length=77, padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_text_features(**tokens)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features[0], real_token_num



def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    return float(torch.matmul(emb1, emb2))

def compute_clip_similarity(image_path: Path, text: str) -> float:
    image_emb = get_clip_visual_embedding(image_path)
    text_emb, _ = get_clip_text_embedding(text)
    similarity = cosine_similarity(image_emb, text_emb)
    return similarity

## now i will call it on my dataframe
if __name__ == "__main__":
    df = pd.read_csv("data/processed/similarity_data.csv")
    df["clip_similarity"] = df.apply(lambda row: compute_clip_similarity(Path(row["image_path"]), row["description"]), axis=1)
    df.to_csv("data/processed/similarity_data.csv", index=False)
    print("CLIP similarity computation completed and saved to data/processed/similarity_data.csv")