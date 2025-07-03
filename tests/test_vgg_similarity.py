import sys
from pathlib import Path

# Adding project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from similarity.vgg_similarity import compute_similarity_score

def test_identical_similarity():
    img1 = torch.rand(3, 224, 224)
    img2 = img1.clone()
    _, cosine_distance = compute_similarity_score(img1, img2)
    print(cosine_distance)
    assert cosine_distance < 1e-6, f"Expected distance ~0, got {cosine_distance}"

