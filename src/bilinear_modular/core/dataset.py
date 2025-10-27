"""Dataset generation and loading for modular arithmetic."""

import json
from pathlib import Path

import arguably
import torch as th
from loguru import logger


class ModularArithmeticDataset:
    """Efficient reader for modular arithmetic datasets.

    Stores data in memory as PyTorch tensors for fast access without the overhead
    of PyTorch DataLoader machinery.

    Args:
        mod_basis: The modulus for arithmetic operations (e.g., 113)
        data_dir: Directory containing the dataset files (default: data/{mod_basis})
        train_split: Fraction of data to use for training (default: 0.8)
        one_hot: Whether to return one-hot encoded vectors (default: True)
        seed: Random seed for splitting (default: 42)
        batch_size: Batch size for iteration (default: 128)
    """

    def __init__(
        self,
        mod_basis: int,
        data_dir: Path | None = None,
        train_split: float = 0.8,
        one_hot: bool = True,
        seed: int = 42,
        batch_size: int = 128,
    ):
        self.mod_basis = mod_basis
        self.data_dir = data_dir or Path(f"data/{mod_basis}")
        self.train_split = train_split
        self.one_hot = one_hot
        self.seed = seed
        self.batch_size = batch_size

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Dataset metadata not found at {metadata_path}. "
                f"Generate the dataset first with: generate_dataset({mod_basis})"
            )

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # Load raw data as torch tensors
        data_path = self.data_dir / "samples.pt"
        data = th.load(data_path, weights_only=True)
        self.a_values = data["a"]
        self.b_values = data["b"]
        self.c_values = data["c"]

        # Create train/val split
        generator = th.Generator().manual_seed(seed)
        n_samples = len(self.a_values)
        indices = th.randperm(n_samples, generator=generator)

        split_idx = int(n_samples * train_split)
        self.train_indices = indices[:split_idx]
        self.val_indices = indices[split_idx:]

        # Iterator state
        self._current_epoch_indices = None
        self._current_idx = 0
        self._is_training = True

    def __len__(self) -> int:
        """Total number of samples."""
        return len(self.a_values)

    def _to_one_hot(self, values: th.Tensor) -> th.Tensor:
        """Convert integer values to one-hot encoding."""
        return th.nn.functional.one_hot(values.long(), num_classes=self.mod_basis).float()

    def get_batch(self, indices: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Get a batch of samples by indices.

        Args:
            indices: Tensor of indices to retrieve

        Returns:
            Tuple of (inputs, targets) where:
            - inputs: (batch_size, 2 * mod_basis) if one_hot else (batch_size, 2)
            - targets: (batch_size, mod_basis) if one_hot else (batch_size,)
        """
        a_batch = self.a_values[indices]
        b_batch = self.b_values[indices]
        c_batch = self.c_values[indices]

        if self.one_hot:
            # Concatenate one-hot encoded a and b
            a_one_hot = self._to_one_hot(a_batch)
            b_one_hot = self._to_one_hot(b_batch)
            inputs = th.cat([a_one_hot, b_one_hot], dim=1)
            targets = self._to_one_hot(c_batch)
        else:
            # Return raw integers
            inputs = th.stack([a_batch, b_batch], dim=1)
            targets = c_batch

        return inputs, targets

    def get_train_batch(self, batch_size: int | None = None) -> tuple[th.Tensor, th.Tensor]:
        """Get a random batch from training set.

        Args:
            batch_size: Number of samples in the batch (uses default if None)

        Returns:
            Tuple of (inputs, targets)
        """
        if batch_size is None:
            batch_size = self.batch_size
        indices = self.train_indices[th.randint(0, len(self.train_indices), (batch_size,))]
        return self.get_batch(indices)

    def get_val_batch(self, batch_size: int | None = None) -> tuple[th.Tensor, th.Tensor]:
        """Get a random batch from validation set.

        Args:
            batch_size: Number of samples in the batch (uses default if None)

        Returns:
            Tuple of (inputs, targets)
        """
        if batch_size is None:
            batch_size = self.batch_size
        indices = self.val_indices[th.randint(0, len(self.val_indices), (batch_size,))]
        return self.get_batch(indices)

    def get_all_train(self) -> tuple[th.Tensor, th.Tensor]:
        """Get all training samples.

        Returns:
            Tuple of (inputs, targets)
        """
        return self.get_batch(self.train_indices)

    def get_all_val(self) -> tuple[th.Tensor, th.Tensor]:
        """Get all validation samples.

        Returns:
            Tuple of (inputs, targets)
        """
        return self.get_batch(self.val_indices)

    @property
    def train_size(self) -> int:
        """Number of training samples."""
        return len(self.train_indices)

    @property
    def val_size(self) -> int:
        """Number of validation samples."""
        return len(self.val_indices)

    def train(self) -> "ModularArithmeticDataset":
        """Set dataset to training mode for iteration."""
        self._is_training = True
        return self

    def eval(self) -> "ModularArithmeticDataset":
        """Set dataset to evaluation mode for iteration."""
        self._is_training = False
        return self

    def __iter__(self) -> "ModularArithmeticDataset":
        """Initialize iterator for epoch."""
        # Shuffle training indices at the start of each epoch
        if self._is_training:
            generator = th.Generator().manual_seed(self.seed + th.randint(0, 10000, (1,)).item())
            perm = th.randperm(len(self.train_indices), generator=generator)
            self._current_epoch_indices = self.train_indices[perm]
        else:
            self._current_epoch_indices = self.val_indices

        self._current_idx = 0
        return self

    def __next__(self) -> tuple[th.Tensor, th.Tensor]:
        """Get next batch in iteration."""
        if self._current_epoch_indices is None:
            raise RuntimeError("Iterator not initialized. Call iter(dataset) first.")

        if self._current_idx >= len(self._current_epoch_indices):
            raise StopIteration

        # Get batch indices
        end_idx = min(self._current_idx + self.batch_size, len(self._current_epoch_indices))
        batch_indices = self._current_epoch_indices[self._current_idx : end_idx]
        self._current_idx = end_idx

        return self.get_batch(batch_indices)


@arguably.command
def generate_dataset(
    mod_basis: int = 113,
    *,
    output_dir: str | None = None,
) -> ModularArithmeticDataset:
    """Generate a complete modular arithmetic dataset.

    Creates all possible combinations of (a, b) where both are in [0, mod_basis),
    and computes c = (a + b) % mod_basis for each pair using efficient torch operations.

    Args:
        mod_basis: The modulus for arithmetic operations (default: 113)
        output_dir: Directory to save the dataset (default: data/{mod_basis})

    Returns:
        ModularArithmeticDataset instance for the generated data

    Example:
        >>> # Generate dataset for mod 113
        >>> dataset = generate_dataset(113)
        >>> # Or from command line:
        >>> # python -m bilinear_modular.core.dataset generate-dataset 113
    """
    # Set output directory
    if output_dir is None:
        output_dir = f"data/{mod_basis}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating modular arithmetic dataset for mod {mod_basis}...")

    # Generate all combinations efficiently using torch
    a_range = th.arange(mod_basis, dtype=th.long)
    b_range = th.arange(mod_basis, dtype=th.long)

    # Create meshgrid for all combinations
    a_values, b_values = th.meshgrid(a_range, b_range, indexing="ij")
    a_values = a_values.flatten()
    b_values = b_values.flatten()

    # Compute modular sum
    c_values = (a_values + b_values) % mod_basis

    # Save to disk as .pt file
    samples_path = output_path / "samples.pt"
    th.save(
        {
            "a": a_values,
            "b": b_values,
            "c": c_values,
        },
        samples_path,
    )

    # Save metadata
    metadata = {
        "mod_basis": mod_basis,
        "n_samples": len(a_values),
        "input_dim": 2 * mod_basis,  # One-hot encoded a and b
        "output_dim": mod_basis,  # One-hot encoded c
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.success(f"Generated {len(a_values)} samples")
    logger.success(f"Saved to {output_path}")
    logger.info(f"  - samples.pt: {samples_path.stat().st_size / 1024:.1f} KB")
    logger.info(f"  - metadata.json: {metadata_path.stat().st_size / 1024:.1f} KB")

    # Return dataset instance
    return ModularArithmeticDataset(mod_basis=mod_basis, data_dir=output_path)


if __name__ == "__main__":
    arguably.run()
