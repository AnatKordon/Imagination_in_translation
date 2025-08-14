# citing:
# Shoham, A., Grosbard, I. D., Patashnik, O., Cohen-Or, D., & Yovel, G. (2024). Using deep neural networks to disentangle visual and semantic information in human perception and memory. Nature Human Behaviour, 8(4), 702-717.:
# Simonyan, K. & Zisserman, A. Very Deep Convolutional Networks for Large-Scale Image Recognition. 3rd International Conference on Learning Representations, ICLR 2015 - Conference Track Proceedings (2014) doi:10.48550/arxiv.1409.1556.
# Deng, J. et al. ImageNet: A large-scale hierarchical image database. in 2009 IEEE Conference on Computer Vision and Pattern Recognition 248–255 (IEEE, 2009). doi:10.1109/CVPR.2009.5206848.
import os
from pathlib import Path
import numpy as np
from typing import List
from PIL import Image
import torch
from torchvision import transforms, models
from torchvision.models import vgg16, VGG16_Weights
from scipy.spatial.distance import cosine

class VGGEmbedder:
    """
    Extracts an embedding from a specified VGG layer for a single image,
    and computes similarity scores between embeddings.
    """

    def __init__(self, model: torch.nn.Module, layer: str):
        """
        Initializes the embedder.

        Args:
            model: Preloaded VGG16 model.
            layer: The layer name to extract (after experimenting we chose 'Classifier_4', fully connected 7).
        """
        self.model = model
        self.model.eval() #set on eval to avoid chang
        self.layer = layer
        self.embedding = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #creates a device object - use gpuif available
        self.model.to(self.device)  # move model to gpu if exists, else cpu

        # Register hook for the chosen layer - all model's inner representations without modifying it
        for idx, layer_module in enumerate(model.features):
            if f'Layer_{idx}' == layer:
                layer_module.register_forward_hook(self.get_embeddings_by_layer(layer))
        for idx, layer_module in enumerate(model.classifier):
            if f'Classifier_{idx}' == layer:
                layer_module.register_forward_hook(self.get_embeddings_by_layer(layer))

        # Define  transforms to match images to what model is used to
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

    def get_embeddings_by_layer(self, layer_name: str):
        """
        Returns a hook function to save the layer output as an embedding.
        """
        def hook(model, input, output):
            self.embedding = output.detach().flatten().cpu().numpy()
        return hook

    def preprocess_image(self, img_path: str) -> torch.Tensor:
        """
        Loads and preprocesses an image for the model.

        Args:
            img_path: Path to image file.

        Returns:
            Preprocessed image tensor.
        """
        img = Image.open(img_path).convert('RGB')
        img_tensor = torch.unsqueeze(self.transforms(img), 0).to(self.device)
        return img_tensor

    def get_embedding(self, img_path: str) -> np.ndarray:
        """
        Returns the embedding of the image for the specified layer.

        Args:
            img_path: Path to image file.

        Returns:
            Embedding as a numpy array.
        """
        self.embedding = None  # clear previous
        img_path = Path(img_path)
        prep_img = self.preprocess_image(img_path)
        with torch.no_grad():
            _ = self.model(prep_img) #no need for the output
        return self.embedding  #returns embeddings of the chosen layer

# computing the similarity score - for the user and for the logging
def compute_similarity_score(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Computes cosine similarity score scaled to [0, 100].

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Similarity score between 0 and 100 (higher = more similar).
        """
        cosine_distance = cosine(embedding1, embedding2) #dissimilarity value - lower is higher similarity, used for logging ranges [0, 2]
        similarity = 1 - cosine_distance  # ranges [-1,+1]
        scaled_similarity = int(((similarity + 1) / 2) * 100)  # maps to [0,100] for visibility, rounding for user friendliness
        return scaled_similarity, cosine_distance  # maybe we should only log similarity
def get_vgg_embedder():
    """
    Initializes and returns a VGGEmbedder instance with preloaded VGG16 model.
    """
    # Load the pre-trained VGG16 model with ImageNet weights
    weights = VGG16_Weights.IMAGENET1K_V1
    vgg_imagenet = vgg16(weights=weights)
    return VGGEmbedder(model=vgg_imagenet, layer='Classifier_4')

if __name__ == "__main__":
    # Load VGG model
    weights = VGG16_Weights.IMAGENET1K_V1
    vgg_imagenet = models.vgg16(weights=weights)

    # Initialize embedder for desired layer
    embedder = VGGEmbedder(model=vgg_imagenet, layer='Classifier_4') #can also try "Layer_30" which is last conv layer
    img_gt = Path(r'data\wilma_ground_truth\kitchen_h.jpg')  # ground truth path - to be changed, for path change slashes to forward slashes
    img_gen = Path(r'data/wilma_ground_truth/airport_terminal_h.jpg')  # generated image path - to be changed
    # Get embeddings for two images
    embedding_gt = embedder.get_embedding(img_path=str(img_gt))
    embedding_gen = embedder.get_embedding(img_path=str(img_gen))
    # Compute similarity score
    ObjeciveSimilarityScore, cosine_distance = compute_similarity_score(embedding1=embedding_gt, embedding2=embedding_gen)
    #present user with ObjeciveSimilarityScore, log cosine_distance for analyses
    print(f"Similarity score: {int(ObjeciveSimilarityScore)}")
    print(cosine_distance)
    # add logging for distance score (not the simialrity we show the user - UserID, SessionID, Iteration, cosine_distance)

