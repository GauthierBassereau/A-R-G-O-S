from pydantic import BaseModel

class WorldEncoder_DINOv3_cfg(BaseModel):
    model_type: str = "DINOv3_vits16"
    checkpoint_path: str = ""
    output_dim: int = 512

class TextEncoder_DINOv3_cfg(BaseModel):
    model_name: str = "t5-small"
    pretrained: bool = True
    output_dim: int = 512

class ActionEncoder_MLP_cfg(BaseModel):
    model_name: str = "MLP"
    checkpoint_path: str = ""
    input_dim: int = 10
    hidden_dim: int = 256
    output_dim: int = 512
    num_layers: int = 3

class FmBackbone_DiT_cfg(BaseModel):
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
    world_encoder_cfg: WorldEncoder_DINOv3_cfg
    text_encoder_cfg: TextEncoder_t5_cfg
    action_encoder_cfg: ActionEncoder_MLP_cfg
    fm_backbone: FmBackbone_DiT_cfg
    num_timesteps: int = 1000
    cfg_scale: float = 3.0

class WorldDecoder_DINOv3_cfg(BaseModel):
    model_type: str = "Decoder_DINOv3_vits16"
    checkpoint_path: str = ""
    input_dim: int = 512
    output_dim: int = 3