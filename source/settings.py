from typing import Optional, Literal
from pydantic import BaseModel


class WorldEncoderDinov3Config(BaseModel):
    head_ckpt: Optional[str] = None
    backbone_ckpt: Optional[str] = None
    repo: str = "facebookresearch/dinov3"
    hub_entry: str = "dinov3_vitl16_dinotxt_tet1280d20h24l"
    source: Literal["github", "local"] = "github"
    device: Optional[str] = "cuda"
    dtype: Optional[str] = None

class ActionEncoderMLPConfig(BaseModel):
    model_name: str = "MLP"
    checkpoint_path: str = ""
    input_dim: int = 10
    hidden_dim: int = 256
    output_dim: int = 512
    num_layers: int = 3

class FmBackboneDiTConfig(BaseModel):
    model_name: str = "DiT-B/2"
    checkpoint_path: str = ""
    depth: int = 12
    hidden_dim: int = 768
    patch_size: int = 2
    num_heads: int = 12

class WorldModelFM_cfg(BaseModel):
    model_name: str = "WorldModelFM"
    device_map: str = "mps"
    dtype: str = "float16"
    world_encoder_cfg: WorldEncoderDinov3Config | None = None
    action_encoder_cfg: ActionEncoderMLPConfig | None = None
    fm_backbone: FmBackboneDiTConfig | None = None
    num_timesteps: int = 1000
    cfg_scale: float = 3.0

class WorldDecoder_DINOv3_cfg(BaseModel):
    model_type: str = "Decoder_DINOv3_vits16"
    checkpoint_path: str = ""
    input_dim: int = 512
    output_dim: int = 3
