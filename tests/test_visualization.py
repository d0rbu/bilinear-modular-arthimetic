"""Tests for visualization module."""

import tempfile
from pathlib import Path

import torch as th

from bilinear_modular.viz.interaction_matrices import plot_interaction_matrix, plot_singular_vectors


def test_plot_interaction_matrix():
    """Test interaction matrix plotting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_matrix.png"

        # Create simple interaction matrix
        matrix = th.randn(10, 10)

        # Plot it
        plot_interaction_matrix(matrix, output_idx=0, save_path=save_path)

        # Check that file was created
        assert save_path.exists()


def test_plot_singular_vectors():
    """Test singular vector plotting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_svd.png"

        # Create SVD components
        m, n = 10, 8
        U = th.randn(m, n)  # noqa: N806
        S = th.rand(n)  # noqa: N806
        Vh = th.randn(n, n)  # noqa: N806

        # Plot them
        plot_singular_vectors(U, S, Vh, output_idx=0, save_path=save_path, num_components=3)

        # Check that file was created
        assert save_path.exists()


def test_interaction_matrix_computation():
    """Test interaction matrix computation using einsum."""
    # Create simple bilinear weights
    d_out, d_in_0, d_in_1 = 3, 4, 5
    bilinear_weights = th.randn(d_out, d_in_0, d_in_1)

    # Create output vector
    output_vec = th.zeros(d_out)
    output_vec[0] = 1.0

    # Compute interaction matrix using einsum
    interaction = th.einsum("ijk,i->jk", bilinear_weights, output_vec)

    # Check shape
    assert interaction.shape == (d_in_0, d_in_1)

    # Check that it equals the first slice when using one-hot
    assert th.allclose(interaction, bilinear_weights[0])


def test_interaction_matrix_weighted():
    """Test interaction matrix with weighted output vector."""
    d_out, d_in_0, d_in_1 = 3, 4, 5
    bilinear_weights = th.randn(d_out, d_in_0, d_in_1)

    # Create weighted output vector
    output_vec = th.tensor([0.5, 0.3, 0.2])

    # Compute interaction matrix using einsum
    interaction = th.einsum("ijk,i->jk", bilinear_weights, output_vec)

    # Check shape
    assert interaction.shape == (d_in_0, d_in_1)

    # Check that it equals the weighted sum manually
    expected = 0.5 * bilinear_weights[0] + 0.3 * bilinear_weights[1] + 0.2 * bilinear_weights[2]
    assert th.allclose(interaction, expected)


def test_svd_computation():
    """Test SVD computation on interaction matrices."""
    # Create a random matrix
    m, n = 10, 8
    matrix = th.randn(m, n)

    # Compute SVD
    U, S, Vh = th.linalg.svd(matrix, full_matrices=False)  # noqa: N806

    # Check shapes
    assert U.shape == (m, min(m, n))
    assert S.shape == (min(m, n),)
    assert Vh.shape == (min(m, n), n)

    # Check that singular values are sorted
    assert th.all(S[:-1] >= S[1:])

    # Check reconstruction (approximately)
    reconstructed = U @ th.diag(S) @ Vh
    assert th.allclose(reconstructed, matrix, atol=1e-5)
