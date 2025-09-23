import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple
import torch

# ---------- Configs for Pipelines ----------
@dataclass
class TrainDecoderConfig:
    device: str = "cuda:1"
    learning_rate: float = 3e-4
    batch_size: int = 10
    gradient_accumulation_steps: int = 3
    checkpoint_frequency: int = 1000
    image_log_frequency: int = 100


@dataclass
class PretrainWorldModelConfig:
    device: str = "cuda:1"
    learning_rate: float = 1e-3
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    text_dropout_prob: float = 0.5
    checkpoint_frequency: int = 1000
    log_frequency: int = 1
    adam_betas: Optional[Tuple[float, float]] = (0.9, 0.999)
    weight_decay: float = 0.0
    cfg_log_scale: float = 4.0
    checkpoint_dir: str = "checkpoints/world_model"
    decoder_checkpoint_path: Optional[str] = "checkpoints/decoder_step_2000.pt"
    decoder_device: Optional[str] = "cuda:0"
    image_log_frequency: int = 1000
    rectified_flow_sample_steps: int = 50
    rectified_flow_logit_normal_sampling_t: bool = True
    rectified_flow_predict_velocity: bool = True

# ---------- Configs for Models ----------
@dataclass
class ImageDecoderTransposeConfig:
    feature_dim: int = 1024
    observation_shape: tuple = (3, 512, 512)
    depth: int = 64
    kernel_size: int = 5
    stride: int = 3
    
@dataclass
class WorldModelFMConfig:
    input_dim: int = 1024
    latent_dim: int = 1024
    depth: int = 12
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    time_embed_dim: int = 256
    time_cond_dim: int = 1024
    attention_dropout: float = 0.0
    cross_attention_dropout: float = 0.0
    mlp_dropout: float = 0.0
    text_context_dim: int = 2048
    history_context_dim: int = 1024
    use_history_rope: bool = True
    history_rope_base: float = 1000.0
    condition_use_gate: bool = True
    
# ---------- Configs for DatasetStream ----------
@dataclass
class HFStreamConfig:
    hf_dataset: str = "laion/laion-coco"
    split: str = "train"
    streaming: bool = True
    batch_size: int = 2 # Overwritten by TrainerConfig
    image_size: int = 512
    yield_partial_final: bool = False  # yield the last small batch
    max_concurrency: int = 16
    per_host_limit: int = 16
    total_timeout_sec: float = 10.0
    connect_timeout_sec: float = 10.0
    read_timeout_sec: float = 10.0
    retries: int = 1
    user_agent: str = "hf-image-loader/1.0"
    ssl: bool = False # Don't care about man in the middle attack
    ttl_dns_cache: int = 300
    url_key: str = "URL"
    text_key: str = "TEXT"
    encode_images: bool = True
    encode_texts: bool = True
    image_encoder_device: Optional[str] = "cuda:0"
    text_encoder_device: Optional[str] = "cuda:0"
    image_encoder_text_head: bool = False
    image_encoder_normalize: bool = True
    text_encoder_normalize: bool = True
    transform: Optional[Callable[[Any], torch.Tensor]] = None

# ---------- Config Logging ----------
def config_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S"
    )
