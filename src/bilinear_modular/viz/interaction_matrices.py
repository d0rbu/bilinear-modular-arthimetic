"""Visualize interaction matrices from bilinear layers.

This module provides functionality to:
1. Load checkpoints from trained bilinear models
2. Extract and visualize interaction matrices for specific outputs
3. Compute and visualize top eigenvector components
"""

from pathlib import Path

import arguably
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from loguru import logger


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load a model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary containing model state and metadata
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = th.load(checkpoint_path, map_location="cpu", weights_only=False)
    logger.info("Checkpoint loaded successfully")
    return checkpoint


def extract_bilinear_weights(checkpoint: dict) -> th.Tensor:
    """Extract bilinear layer weights from checkpoint.

    Args:
        checkpoint: Loaded checkpoint dictionary

    Returns:
        Bilinear weights tensor of shape (d_out, d_in_0, d_in_1)
    """
    # Look for bilinear weights in the checkpoint
    # The exact key depends on how the model was saved
    # Common patterns: 'model.weight', 'bilinear.weight', etc.

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Find bilinear weight key
    bilinear_keys = [k for k in state_dict if "bilinear" in k.lower() and "weight" in k]

    if not bilinear_keys:
        # Fallback: look for any weight tensor with 3 dimensions
        weight_keys = [k for k in state_dict if "weight" in k]
        for key in weight_keys:
            tensor = state_dict[key]
            if tensor.ndim == 3:
                logger.warning(f"Using {key} as bilinear weights (guessing based on shape)")
                return tensor
        raise ValueError(f"Could not find bilinear weights in checkpoint. Available keys: {list(state_dict.keys())}")

    key = bilinear_keys[0]
    logger.info(f"Found bilinear weights at key: {key}")
    return state_dict[key]


def compute_interaction_matrix(bilinear_weights: th.Tensor, output_vec: th.Tensor) -> th.Tensor:
    """Compute interaction matrix for a specific output vector.

    The interaction matrix is computed as a weighted sum:
        M = sum_i bilinear_weights[i] * output_vec[i]

    This gives us a (d_in_0, d_in_1) matrix showing how the two inputs
    interact to produce the weighted output.

    Args:
        bilinear_weights: Tensor of shape (d_out, d_in_0, d_in_1)
        output_vec: Output vector of shape (d_out,) for weighting

    Returns:
        Interaction matrix of shape (d_in_0, d_in_1)
    """
    # Einsum notation: i is d_out, j is d_in_0, k is d_in_1
    # We want: M[j,k] = sum_i W[i,j,k] * v[i]
    interaction = th.einsum("ijk,i->jk", bilinear_weights, output_vec)
    return interaction


def compute_top_eigenvectors(matrix: th.Tensor, k: int = 5) -> tuple[th.Tensor, th.Tensor]:
    """Compute top k eigenvectors and eigenvalues of a matrix.

    Args:
        matrix: Square matrix to decompose
        k: Number of top eigenvectors to return

    Returns:
        Tuple of (eigenvalues, eigenvectors) for top k components
    """
    # Convert to numpy for eigendecomposition
    matrix_np = matrix.detach().cpu().numpy()

    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(matrix_np)

    # Sort by absolute value of eigenvalues
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx[:k]]
    eigenvectors = eigenvectors[:, idx[:k]]

    return th.from_numpy(eigenvalues), th.from_numpy(eigenvectors)


def plot_interaction_matrix(
    matrix: th.Tensor,
    output_idx: int,
    save_path: Path,
    title: str | None = None,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot and save an interaction matrix heatmap.

    Args:
        matrix: Interaction matrix to plot (d_in_0, d_in_1)
        output_idx: Output class index this matrix corresponds to
        save_path: Path to save the figure
        title: Optional custom title
        cmap: Colormap to use
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
    """
    matrix_np = matrix.detach().cpu().numpy()

    # Auto-scale if not provided
    if vmin is None:
        vmin = -np.abs(matrix_np).max()
    if vmax is None:
        vmax = np.abs(matrix_np).max()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix_np, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    if title is None:
        title = f"Interaction Matrix for Output Class {output_idx}"
    ax.set_title(title, fontsize=14, pad=20)

    ax.set_xlabel("Input 1 (b)", fontsize=12)
    ax.set_ylabel("Input 0 (a)", fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Weight", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved interaction matrix plot to {save_path}")


def plot_eigenvector_components(
    eigenvectors: th.Tensor,
    eigenvalues: th.Tensor,
    output_idx: int,
    save_path: Path,
    title: str | None = None,
):
    """Plot top eigenvector components.

    Args:
        eigenvectors: Matrix of eigenvectors (d_in, k)
        eigenvalues: Vector of eigenvalues (k,)
        output_idx: Output class index
        save_path: Path to save the figure
        title: Optional custom title
    """
    eigenvectors_np = eigenvectors.detach().cpu().numpy()
    eigenvalues_np = eigenvalues.detach().cpu().numpy()

    k = eigenvectors_np.shape[1]

    fig, axes = plt.subplots(k, 1, figsize=(12, 3 * k))
    if k == 1:
        axes = [axes]

    for i, (ax, eigval) in enumerate(zip(axes, eigenvalues_np, strict=True)):
        eigvec = eigenvectors_np[:, i]

        ax.plot(eigvec.real, label="Real", linewidth=2, alpha=0.8)
        ax.plot(eigvec.imag, label="Imaginary", linewidth=2, alpha=0.8)

        ax.set_xlabel("Component Index", fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.set_title(
            f"Eigenvector {i + 1} (λ = {eigval.real:.3f} + {eigval.imag:.3f}i)",
            fontsize=12,
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    if title is None:
        title = f"Top {k} Eigenvector Components for Output Class {output_idx}"
    fig.suptitle(title, fontsize=14, y=1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved eigenvector components plot to {save_path}")


@arguably.command
def visualize(
    checkpoint_path: str,
    *,
    output_indices: list[int] | None = None,
    mod_basis: int = 113,
    num_eigenvectors: int = 5,
    output_dir: str = "fig",
):
    """Visualize interaction matrices and eigenvector components from a trained bilinear model.

    Args:
        checkpoint_path: Path to the model checkpoint file
        output_indices: List of output indices to visualize (default: [0, 1, mod_basis-1])
        mod_basis: Modular basis (P) used in training
        num_eigenvectors: Number of top eigenvectors to compute and plot
        output_dir: Directory to save figures
    """
    # Setup
    checkpoint_path_obj = Path(checkpoint_path)
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(exist_ok=True, parents=True)

    # Default output indices if not provided
    if output_indices is None:
        output_indices = [0, 1, mod_basis - 1]

    logger.info(f"Visualizing interaction matrices for output indices: {output_indices}")

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path_obj)
    bilinear_weights = extract_bilinear_weights(checkpoint)

    d_out, d_in_0, d_in_1 = bilinear_weights.shape
    logger.info(f"Bilinear weights shape: (d_out={d_out}, d_in_0={d_in_0}, d_in_1={d_in_1})")

    # Process each output index
    for out_idx in output_indices:
        if out_idx >= d_out or out_idx < 0:
            logger.warning(f"Output index {out_idx} out of range [0, {d_out}). Skipping.")
            continue

        logger.info(f"Processing output index {out_idx}")

        # Create output vector (one-hot)
        output_vec = th.zeros(d_out)
        output_vec[out_idx] = 1.0

        # Compute interaction matrix
        interaction = compute_interaction_matrix(bilinear_weights, output_vec)

        # Plot interaction matrix
        matrix_path = output_dir_obj / f"interaction_matrix_output_{out_idx}.png"
        plot_interaction_matrix(interaction, out_idx, matrix_path)

        # Compute and plot eigenvector components
        eigenvalues, eigenvectors = compute_top_eigenvectors(interaction, k=num_eigenvectors)

        eigvec_path = output_dir_obj / f"eigenvectors_output_{out_idx}.png"
        plot_eigenvector_components(eigenvectors, eigenvalues, out_idx, eigvec_path)

    logger.success(f"Visualization complete! Figures saved to {output_dir_obj}")


if __name__ == "__main__":
    arguably.run()
