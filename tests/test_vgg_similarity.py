from pathlib import Path
import pytest
import torch
from torchvision import models
from torchvision.models import VGG16_Weights

import numpy as np

#import my class and simoilarity function
from similarity.vgg_similarity import compute_similarity_score, VGGEmbedder

#Initalize model and embedder for use in all test functions
weights = VGG16_Weights.IMAGENET1K_V1
vgg_imagenet = models.vgg16(weights=weights)

# Initialize embedder
embedder = VGGEmbedder(model=vgg_imagenet, layer='Classifier_4')

def test_identical_similarity():
    # testing two identical image
    img_path1 = Path(r'GT_images\wilma_ground_truth\badlands_h.jpg')
    img_path2 = Path(r'GT_images\wilma_ground_truth\badlands_h.jpg') #identical
    # Get embeddings for both
    embedding1 = embedder.get_embedding(img_path=str(img_path1))
    embedding2 = embedder.get_embedding(img_path=str(img_path2))

    # Compute similarity
    _ , cosine_distance = compute_similarity_score(embedding1=embedding1, embedding2=embedding2)

    # Assert distance is 0 for identical images
    assert cosine_distance == 0

def test_opposite_similarity():
    #compute_similarity_score accepts a 1D numpy array
    v1 = np.random.rand(1000)
    # Create its exact opposite by negating
    v2 = -v1
    _, cosine_distance = compute_similarity_score(v1, v2)
    assert cosine_distance > 1.999 

