"""Tests for dataset generation and loading."""

import shutil
from pathlib import Path

import pytest
import torch as th

from bilinear_modular.core import ModularArithmeticDataset, generate_dataset


@pytest.fixture
def cleanup_test_data():
    """Clean up test data directories after tests."""
    yield
    # Clean up any test directories
    test_dirs = [Path("data/7"), Path("data/11")]
    for test_dir in test_dirs:
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_generate_dataset_small(cleanup_test_data):
    """Test dataset generation with small mod basis."""
    mod_basis = 7
    dataset = generate_dataset(mod_basis)

    # Check metadata
    assert dataset.mod_basis == mod_basis
    assert dataset.metadata["mod_basis"] == mod_basis
    assert dataset.metadata["n_samples"] == mod_basis * mod_basis
    assert dataset.metadata["input_dim"] == 2 * mod_basis
    assert dataset.metadata["output_dim"] == mod_basis

    # Check data shape
    assert len(dataset) == mod_basis * mod_basis
    assert dataset.train_size + dataset.val_size == len(dataset)

    # Verify files exist
    data_dir = Path(f"data/{mod_basis}")
    assert (data_dir / "samples.pt").exists()
    assert (data_dir / "metadata.json").exists()


def test_dataset_correctness(cleanup_test_data):
    """Test that dataset contains correct modular arithmetic results."""
    mod_basis = 11
    dataset = generate_dataset(mod_basis)

    # Check a few specific examples
    for i in range(len(dataset)):
        a = dataset.a_values[i].item()
        b = dataset.b_values[i].item()
        c = dataset.c_values[i].item()

        # Verify modular arithmetic
        assert c == (a + b) % mod_basis
        assert 0 <= a < mod_basis
        assert 0 <= b < mod_basis
        assert 0 <= c < mod_basis


def test_one_hot_encoding(cleanup_test_data):
    """Test one-hot encoding of samples."""
    mod_basis = 7
    dataset = generate_dataset(mod_basis)

    # Get a batch with one-hot encoding
    inputs, targets = dataset.get_batch(th.tensor([0, 1, 2]))

    # Check shapes
    assert inputs.shape == (3, 2 * mod_basis)
    assert targets.shape == (3, mod_basis)

    # Check one-hot properties
    assert th.all(inputs.sum(dim=1) == 2)  # Two 1s per sample (one for a, one for b)
    assert th.all(targets.sum(dim=1) == 1)  # One 1 per target


def test_raw_integer_mode(cleanup_test_data):
    """Test dataset with raw integer values (no one-hot)."""
    mod_basis = 7
    generate_dataset(mod_basis)  # Generate first
    dataset = ModularArithmeticDataset(mod_basis, one_hot=False)

    # Get a batch without one-hot encoding
    inputs, targets = dataset.get_batch(th.tensor([0, 1, 2]))

    # Check shapes
    assert inputs.shape == (3, 2)  # Just a and b
    assert targets.shape == (3,)  # Just c

    # Check values are integers
    assert inputs.dtype == th.long
    assert targets.dtype == th.long


def test_train_val_split(cleanup_test_data):
    """Test train/validation split."""
    mod_basis = 11
    dataset = generate_dataset(mod_basis)

    # Check split proportions
    total_size = len(dataset)
    expected_train = int(total_size * dataset.train_split)
    assert dataset.train_size == expected_train
    assert dataset.val_size == total_size - expected_train

    # Check no overlap
    train_set = set(dataset.train_indices.tolist())
    val_set = set(dataset.val_indices.tolist())
    assert len(train_set & val_set) == 0
    assert len(train_set | val_set) == total_size


def test_batch_retrieval(cleanup_test_data):
    """Test batch retrieval functions."""
    mod_basis = 7
    dataset = generate_dataset(mod_basis)

    # Test train batch
    batch_size = 10
    inputs, targets = dataset.get_train_batch(batch_size)
    assert inputs.shape[0] == batch_size
    assert targets.shape[0] == batch_size

    # Test val batch
    inputs, targets = dataset.get_val_batch(batch_size)
    assert inputs.shape[0] == batch_size
    assert targets.shape[0] == batch_size

    # Test get all
    inputs, targets = dataset.get_all_train()
    assert inputs.shape[0] == dataset.train_size

    inputs, targets = dataset.get_all_val()
    assert inputs.shape[0] == dataset.val_size


def test_torch_tensors(cleanup_test_data):
    """Test that all returned values are torch tensors."""
    mod_basis = 7
    dataset = generate_dataset(mod_basis)

    # Get batch
    inputs, targets = dataset.get_batch(th.tensor([0, 1, 2]))

    assert isinstance(inputs, th.Tensor)
    assert isinstance(targets, th.Tensor)
    assert inputs.shape == (3, 2 * mod_basis)
    assert targets.shape == (3, mod_basis)


def test_dataset_reload(cleanup_test_data):
    """Test that dataset can be reloaded from disk."""
    mod_basis = 7

    # Generate dataset
    dataset1 = generate_dataset(mod_basis)

    # Load dataset from disk
    dataset2 = ModularArithmeticDataset(mod_basis)

    # Check they're equivalent
    assert dataset1.mod_basis == dataset2.mod_basis
    assert len(dataset1) == len(dataset2)
    assert th.equal(dataset1.a_values, dataset2.a_values)
    assert th.equal(dataset1.b_values, dataset2.b_values)
    assert th.equal(dataset1.c_values, dataset2.c_values)


def test_missing_dataset_error():
    """Test that loading non-existent dataset raises error."""
    with pytest.raises(FileNotFoundError):
        ModularArithmeticDataset(mod_basis=9999)


def test_iterator_protocol(cleanup_test_data):
    """Test __iter__ and __next__ methods for training loops."""
    mod_basis = 7
    dataset = generate_dataset(mod_basis)
    dataset.batch_size = 5

    # Test training iteration
    dataset.train()
    batches = []
    for batch in dataset:
        batches.append(batch)
        if len(batches) >= 3:  # Just test a few batches
            break

    assert len(batches) == 3
    for inputs, targets in batches:
        assert isinstance(inputs, th.Tensor)
        assert isinstance(targets, th.Tensor)
        assert inputs.shape[0] <= 5  # Batch size
        assert targets.shape[0] <= 5

    # Test validation iteration
    dataset.eval()
    val_batches = []
    for batch in dataset:
        val_batches.append(batch)
        if len(val_batches) >= 2:
            break

    assert len(val_batches) == 2
