from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# Load model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Ensure model is on the right device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
