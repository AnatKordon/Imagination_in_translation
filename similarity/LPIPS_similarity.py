import lpips
import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

# loading
lpips_model = lpips.LPIPS(net='vgg')  # 'alex', 'vgg', 'squeeze'

def compute_lpips_score(gt_path: Path, gen_path: Path, lpips_model=lpips_model):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # LPIPS expects same size
        transforms.ToTensor(),
    ])

    img1 = transform(Image.open(gt_path).convert("RGB")).unsqueeze(0)
    img2 = transform(Image.open(gen_path).convert("RGB")).unsqueeze(0)

    # LPIPS expects tensors in [-1, 1]
    img1 = (img1 - 0.5) * 2
    img2 = (img2 - 0.5) * 2

    with torch.no_grad():
        dist_value = lpips_model(img1, img2).item()
    # invert and scale: 0 = identical → 100 similarity
    
    return dist_value

