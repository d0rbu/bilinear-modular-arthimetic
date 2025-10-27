"""Dataset generation and loading for modular arithmetic."""

import json
from pathlib import Path
from typing import NamedTuple

import arguably
import numpy as np
import torch


class ModularArithmeticSample(NamedTuple):
    """A single sample from the modular arithmetic dataset."""

    a: int
    b: int
    c: int  # (a + b) % mod_basis


class ModularArithmeticDataset:
    """Efficient reader for modular arithmetic datasets.

    Stores data in memory as numpy arrays for fast access without the overhead
    of PyTorch DataLoader machinery.

    Args:
        mod_basis: The modulus for arithmetic operations (e.g., 113)
        data_dir: Directory containing the dataset files (default: data/{mod_basis})
        train_split: Fraction of data to use for training (default: 0.8)
        one_hot: Whether to return one-hot encoded vectors (default: True)
        seed: Random seed for splitting (default: 42)
    """

    def __init__(
        self,
        mod_basis: int,
        data_dir: Path | None = None,
        train_split: float = 0.8,
        one_hot: bool = True,
        seed: int = 42,
    ):
        self.mod_basis = mod_basis
        self.data_dir = data_dir or Path(f"data/{mod_basis}")
        self.train_split = train_split
        self.one_hot = one_hot
        self.seed = seed

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Dataset metadata not found at {metadata_path}. "
                f"Generate the dataset first with: generate_dataset({mod_basis})"
            )

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # Load raw data
        data_path = self.data_dir / "samples.npz"
        data = np.load(data_path)
        self.a_values = data["a"]
        self.b_values = data["b"]
        self.c_values = data["c"]

        # Create train/val split
        rng = np.random.RandomState(seed)
        n_samples = len(self.a_values)
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        split_idx = int(n_samples * train_split)
        self.train_indices = indices[:split_idx]
        self.val_indices = indices[split_idx:]

    def __len__(self) -> int:
        """Total number of samples."""
        return len(self.a_values)

    def _to_one_hot(self, values: np.ndarray) -> np.ndarray:
        """Convert integer values to one-hot encoding."""
        one_hot_matrix = np.zeros((len(values), self.mod_basis), dtype=np.float32)
        one_hot_matrix[np.arange(len(values)), values] = 1
        return one_hot_matrix

    def get_batch(self, indices: np.ndarray, as_torch: bool = False) -> tuple:
        """Get a batch of samples by indices.

        Args:
            indices: Array of indices to retrieve
            as_torch: Whether to return torch tensors (default: False)

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
            inputs = np.concatenate([a_one_hot, b_one_hot], axis=1)
            targets = self._to_one_hot(c_batch)
        else:
            # Return raw integers
            inputs = np.stack([a_batch, b_batch], axis=1)
            targets = c_batch

        if as_torch:
            inputs = torch.from_numpy(inputs)
            targets = torch.from_numpy(targets)

        return inputs, targets

    def get_train_batch(self, batch_size: int, as_torch: bool = False) -> tuple:
        """Get a random batch from training set.

        Args:
            batch_size: Number of samples in the batch
            as_torch: Whether to return torch tensors (default: False)

        Returns:
            Tuple of (inputs, targets)
        """
        indices = np.random.choice(self.train_indices, size=batch_size, replace=False)
        return self.get_batch(indices, as_torch=as_torch)

    def get_val_batch(self, batch_size: int, as_torch: bool = False) -> tuple:
        """Get a random batch from validation set.

        Args:
            batch_size: Number of samples in the batch
            as_torch: Whether to return torch tensors (default: False)

        Returns:
            Tuple of (inputs, targets)
        """
        indices = np.random.choice(self.val_indices, size=batch_size, replace=False)
        return self.get_batch(indices, as_torch=as_torch)

    def get_all_train(self, as_torch: bool = False) -> tuple:
        """Get all training samples.

        Args:
            as_torch: Whether to return torch tensors (default: False)

        Returns:
            Tuple of (inputs, targets)
        """
        return self.get_batch(self.train_indices, as_torch=as_torch)

    def get_all_val(self, as_torch: bool = False) -> tuple:
        """Get all validation samples.

        Args:
            as_torch: Whether to return torch tensors (default: False)

        Returns:
            Tuple of (inputs, targets)
        """
        return self.get_batch(self.val_indices, as_torch=as_torch)

    @property
    def train_size(self) -> int:
        """Number of training samples."""
        return len(self.train_indices)

    @property
    def val_size(self) -> int:
        """Number of validation samples."""
        return len(self.val_indices)


@arguably.command
def generate_dataset(
    mod_basis: int = 113,
    *,
    output_dir: str | None = None,
) -> ModularArithmeticDataset:
    """Generate a complete modular arithmetic dataset.

    Creates all possible combinations of (a, b) where both are in [0, mod_basis),
    and computes c = (a + b) % mod_basis for each pair.

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

    print(f"Generating modular arithmetic dataset for mod {mod_basis}...")

    # Generate all combinations
    a_values = []
    b_values = []
    c_values = []

    for a in range(mod_basis):
        for b in range(mod_basis):
            c = (a + b) % mod_basis
            a_values.append(a)
            b_values.append(b)
            c_values.append(c)

    # Convert to numpy arrays
    a_values = np.array(a_values, dtype=np.int32)
    b_values = np.array(b_values, dtype=np.int32)
    c_values = np.array(c_values, dtype=np.int32)

    # Save to disk
    samples_path = output_path / "samples.npz"
    np.savez(samples_path, a=a_values, b=b_values, c=c_values)

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

    print(f"✓ Generated {len(a_values)} samples")
    print(f"✓ Saved to {output_path}")
    print(f"  - samples.npz: {samples_path.stat().st_size / 1024:.1f} KB")
    print(f"  - metadata.json: {metadata_path.stat().st_size / 1024:.1f} KB")

    # Return dataset instance
    return ModularArithmeticDataset(mod_basis=mod_basis, data_dir=output_path)


if __name__ == "__main__":
    arguably.run()
