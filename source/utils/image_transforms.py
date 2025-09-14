from typing import Tuple
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def make_transform(resize: int | Tuple[int, int] = 512) -> transforms.Compose:
    """ if resize is int, smaller edge is resized to that value; if tuple, resize to that exact size and squash/stretch """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(resize),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])