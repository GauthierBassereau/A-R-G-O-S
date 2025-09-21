import logging
from dataclasses import dataclass
from typing import Optional
import torch

# ---------- Configs for Pipelines ----------
@dataclass
class TrainerConfig:
    device: str = "cuda"
    learning_rate: float = 3e-4
    batch_size: int = 10
    gradient_accumulation_steps: int = 3
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
    
@dataclass
class WorldModelFMConfig:
    input_dim: int = 1024
    latent_dim: int = 1024
    depth: int = 12
    num_heads: int = 16
    mlp_ratio: float = 4.0
    time_embed_dim: int = 256
    time_cond_dim: int = 1024
    attention_dropout: float = 0.0
    cross_attention_dropout: float = 0.0
    mlp_dropout: float = 0.0
    text_context_dim: int = 1024
    history_context_dim: int = 1024
    use_history_rope: bool = True
    history_rope_base: float = 10000.0
    condition_use_gate: bool = True
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
    max_concurrency: int = 64
    per_host_limit: int = 64
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
