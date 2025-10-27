"""Core functionality for bilinear modular arithmetic training."""

from .dataset import ModularArithmeticDataset, generate_dataset
from .train import BilinearModularModel, TrainingConfig, load_checkpoint, save_checkpoint, train, validate

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
