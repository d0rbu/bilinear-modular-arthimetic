"""Tests for visualization module."""

# Import after adding parent to path
import sys
import tempfile
from pathlib import Path

import pytest
import torch as th

sys.path.insert(0, str(Path(__file__).parent.parent))

from viz.interaction_matrices import (
    compute_interaction_matrix,
    compute_top_eigenvectors,
    extract_bilinear_weights,
    load_checkpoint,
)


def test_compute_interaction_matrix():
    """Test interaction matrix computation."""
    # Create simple bilinear weights
    d_out, d_in_0, d_in_1 = 3, 4, 5
    bilinear_weights = th.randn(d_out, d_in_0, d_in_1)

    # Create output vector
    output_vec = th.zeros(d_out)
    output_vec[0] = 1.0

    # Compute interaction matrix
    interaction = compute_interaction_matrix(bilinear_weights, output_vec)

    # Check shape
    assert interaction.shape == (d_in_0, d_in_1)

    # Check that it equals the first slice when using one-hot
    assert th.allclose(interaction, bilinear_weights[0])


def test_compute_interaction_matrix_weighted():
    """Test interaction matrix with weighted output vector."""
    d_out, d_in_0, d_in_1 = 3, 4, 5
    bilinear_weights = th.randn(d_out, d_in_0, d_in_1)

    # Create weighted output vector
    output_vec = th.tensor([0.5, 0.3, 0.2])

    # Compute interaction matrix
    interaction = compute_interaction_matrix(bilinear_weights, output_vec)

    # Check shape
    assert interaction.shape == (d_in_0, d_in_1)

    # Check that it equals the weighted sum manually
    expected = 0.5 * bilinear_weights[0] + 0.3 * bilinear_weights[1] + 0.2 * bilinear_weights[2]
    assert th.allclose(interaction, expected)


def test_compute_top_eigenvectors():
    """Test eigenvector computation."""
    # Create symmetric matrix for real eigenvalues
    n = 10
    matrix = th.randn(n, n)
    matrix = (matrix + matrix.T) / 2  # Make symmetric

    k = 3
    eigenvalues, eigenvectors = compute_top_eigenvectors(matrix, k=k)

    # Check shapes
    assert eigenvalues.shape == (k,)
    assert eigenvectors.shape == (n, k)

    # Check that eigenvalues are sorted by absolute value
    abs_vals = th.abs(eigenvalues)
    assert th.all(abs_vals[:-1] >= abs_vals[1:])


def test_load_checkpoint_with_model_state_dict():
    """Test loading checkpoint with model_state_dict key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"

        # Create test checkpoint
        bilinear_weights = th.randn(5, 10, 10)
        checkpoint = {
            "model_state_dict": {"bilinear.weight": bilinear_weights},
            "epoch": 100,
        }
        th.save(checkpoint, checkpoint_path)

        # Load and extract
        loaded = load_checkpoint(checkpoint_path)
        weights = extract_bilinear_weights(loaded)

        assert th.allclose(weights, bilinear_weights)


def test_load_checkpoint_direct_state_dict():
    """Test loading checkpoint with weights directly in state dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"

        # Create test checkpoint (direct state dict)
        bilinear_weights = th.randn(5, 10, 10)
        checkpoint = {"bilinear_layer.weight": bilinear_weights}
        th.save(checkpoint, checkpoint_path)

        # Load and extract
        loaded = load_checkpoint(checkpoint_path)
        weights = extract_bilinear_weights(loaded)

        assert th.allclose(weights, bilinear_weights)


def test_extract_bilinear_weights_fallback():
    """Test fallback weight extraction based on shape."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"

        # Create checkpoint without 'bilinear' in key name
        weights_3d = th.randn(5, 10, 10)
        checkpoint = {
            "model_state_dict": {
                "layer.weight": weights_3d,
                "other.weight": th.randn(5, 10),  # 2D, should be ignored
            }
        }
        th.save(checkpoint, checkpoint_path)

        # Load and extract
        loaded = load_checkpoint(checkpoint_path)
        weights = extract_bilinear_weights(loaded)

        # Should find the 3D tensor
        assert th.allclose(weights, weights_3d)


def test_extract_bilinear_weights_error():
    """Test error handling when no bilinear weights found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"

        # Create checkpoint with no 3D weights
        checkpoint = {
            "model_state_dict": {
                "layer1.weight": th.randn(5, 10),
                "layer2.weight": th.randn(10, 5),
            }
        }
        th.save(checkpoint, checkpoint_path)

        # Should raise error
        loaded = load_checkpoint(checkpoint_path)
        with pytest.raises(ValueError, match="Could not find bilinear weights"):
            extract_bilinear_weights(loaded)
