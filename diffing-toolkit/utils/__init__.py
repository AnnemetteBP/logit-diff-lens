"""
Utility functions and helpers shared across the project.
"""

from .configs import (
    ModelConfig,
    DatasetConfig,
    get_model_configurations,
    get_dataset_configurations,
)
from .model import load_model, load_model_from_config, get_ft_model_id

__all__ = [
    "ModelConfig",
    "DatasetConfig",
    "get_model_configurations",
    "get_dataset_configurations",
    "load_model",
    "load_model_from_config",
    "get_ft_model_id",
]


def get_layer_indices(*args, **kwargs):
    from .activations import get_layer_indices as _get_layer_indices

    return _get_layer_indices(*args, **kwargs)


__all__.append("get_layer_indices")
