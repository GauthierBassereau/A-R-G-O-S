from typing import Optional
from dataclasses import dataclass
import torch

@dataclass
class ImageDecoderTransposeConfig:
    feature_dim: int = 1024
    observation_shape: tuple = (3, 224, 224)
    depth: int = 64
    kernel_size: int = 5
    stride: int = 3
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None