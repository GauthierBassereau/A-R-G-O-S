from typing import Optional
from dataclasses import dataclass
import torch
import logging

# ---------- Logging ----------

def config_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S"
    )

# ---------- Configs for Pipelines ----------



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
class HTTPConfig:
    connect_timeout: float = 0.6
    read_timeout: float = 1.2
    total_retries: int = 1
    user_agent: str = "Mozilla/5.0 (LAION-stream/0.1)"
    pool_size: int = 128  # same as max_workers by default

@dataclass
class StreamConfig:
    max_workers: int = 128 # Number of parallel fetches
    max_in_flight: int = 128 # Memory usage ~ max_in_flight * image_size_in_bytes
    shuffle_buffer: int = 512
    log_interval_s: float = 2.0
    url_col_name: str = "URL"
    transform_size: int = 512
    batch_size: int = 2
    seed: Optional[int] = None # Random seed only used for buffer pop / shuffling (doesn't affect global)