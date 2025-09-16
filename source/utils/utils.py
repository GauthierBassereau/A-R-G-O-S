from dataclasses import fields, is_dataclass
from typing import Any, Dict
import torch


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