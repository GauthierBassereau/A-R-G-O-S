import logging
from dataclasses import dataclass
from typing import Optional
import torch

# ---------- Configs for Pipelines ----------
@dataclass
class TrainerConfig:
    device: str = "mps"
    learning_rate: float = 1e-3
    batch_size: int = 2
    checkpoint_dir: str = "checkpoints"
    checkpoint_frequency: int = 1000
    image_log_frequency: int = 100

# ---------- Configs for Models ----------
@dataclass
class ImageDecoderTransposeConfig:
    feature_dim: int = 1024
    observation_shape: tuple = (3, 512, 512)
    depth: int = 64
    kernel_size: int = 5
    stride: int = 3
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None
    
# ---------- Configs for DatasetStream ----------
@dataclass
class HFStreamConfig:
    hf_dataset: str = "laion/laion-coco"
    split: str = "train"
    streaming: bool = True
    batch_size: int = 64 # Overwritten by TrainerConfig
    image_size: int = 512
    yield_partial_final: bool = False  # yield the last small batch
    max_concurrency: int = 32
    per_host_limit: int = 8
    total_timeout_sec: float = 2.0
    connect_timeout_sec: float = 2.0
    read_timeout_sec: float = 2.0
    retries: int = 1
    user_agent: str = "hf-image-loader/1.0"
    ssl: bool = False # Don't care about man in the middle attack
    ttl_dns_cache: int = 300
    url_key: str = "URL"
    text_key: str = "TEXT"

# ---------- Config Logging ----------
def config_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S"
    )
