"""Bilinear layer training on modular arithmetic."""

from .core import (
    BilinearModularModel,
    ModularArithmeticDataset,
    TrainingConfig,
    generate_dataset,
    load_checkpoint,
    save_checkpoint,
    train,
    validate,
)

__version__ = "0.1.0"
__all__ = [
    "ModularArithmeticDataset",
    "generate_dataset",
    "BilinearModularModel",
    "TrainingConfig",
    "load_checkpoint",
    "save_checkpoint",
    "train",
    "validate",
]
