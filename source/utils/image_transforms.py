from typing import Tuple
from PIL import Image
import torch
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def make_transform(resize: int | Tuple[int, int] = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize, resize), antialias=True) if isinstance(resize, int) else transforms.Resize(resize, antialias=True),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])