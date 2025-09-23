from dataclasses import fields, is_dataclass
from typing import Any, Dict, Tuple
from torchvision import transforms
import torch

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

def tensor_to_image(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a batch-normalized tensor into a single image numpy array."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    if tensor.ndim != 4:
        raise ValueError("tensor must be a 4D batch in BCHW format")
    vis = (tensor * std + mean).clamp(0, 1)
    return vis[0].detach().cpu().permute(1, 2, 0).numpy()

def _serialize_config_value(value: Any) -> Any:
    """Convert config values into basic python types for logging."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_serialize_config_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_config_value(v) for k, v in value.items()}
    if is_dataclass(value):
        return dataclass_to_dict(value)
    return str(value)


def dataclass_to_dict(config: Any) -> Dict[str, Any]:
    """Serialize a dataclass instance to primitives for logging or config dumps."""
    if not is_dataclass(config):
        raise TypeError("config must be a dataclass instance")
    serialized: Dict[str, Any] = {}
    for field in fields(config):
        serialized[field.name] = _serialize_config_value(getattr(config, field.name))
    return serialized


def collect_configs(**configs: Any) -> Dict[str, Any]:
    """Gather multiple configs into a nested mapping suitable for logging."""
    collected: Dict[str, Any] = {}
    for name, config in configs.items():
        collected[name] = dataclass_to_dict(config) if is_dataclass(config) else _serialize_config_value(config)
    return collected