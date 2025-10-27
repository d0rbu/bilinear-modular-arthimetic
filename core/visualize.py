"""Visualization utilities for bilinear modular arithmetic analysis."""

from pathlib import Path

import arguably
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from loguru import logger


def plot_interaction_matrix(
    interaction_matrix: th.Tensor | np.ndarray,
    title: str = "Interaction Matrix",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot a single interaction matrix.

    Args:
        interaction_matrix: 2D tensor/array of shape (input_dim, input_dim)
        title: Title for the plot
        save_path: Optional path to save the figure
        show: Whether to show the plot
    """
    if isinstance(interaction_matrix, th.Tensor):
        interaction_matrix = interaction_matrix.cpu().numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(interaction_matrix, cmap="RdBu_r", aspect="auto")
    plt.colorbar(label="Weight")
    plt.title(title)
    plt.xlabel("Input B")
    plt.ylabel("Input A")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved interaction matrix plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_interaction_matrices(
    interaction_matrices: th.Tensor,
    num_to_plot: int = 9,
    save_dir: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot multiple interaction matrices in a grid.

    Args:
        interaction_matrices: Tensor of shape (hidden_dim, input_dim, input_dim)
        num_to_plot: Number of matrices to plot (will be arranged in a grid)
        save_dir: Optional directory to save individual plots
        show: Whether to show the plot
    """
    if isinstance(interaction_matrices, th.Tensor):
        interaction_matrices = interaction_matrices.cpu().numpy()

    hidden_dim = interaction_matrices.shape[0]
    num_to_plot = min(num_to_plot, hidden_dim)

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_to_plot)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for idx in range(num_to_plot):
        ax = axes[idx]
        im = ax.imshow(interaction_matrices[idx], cmap="RdBu_r", aspect="auto")
        ax.set_title(f"Hidden Unit {idx}")
        ax.set_xlabel("Input B")
        ax.set_ylabel("Input A")
        plt.colorbar(im, ax=ax)

    # Hide unused subplots
    for idx in range(num_to_plot, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "interaction_matrices_grid.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved interaction matrices grid to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def compute_and_plot_eigenvectors(
    interaction_matrix: th.Tensor | np.ndarray,
    num_components: int = 5,
    title: str = "Top Eigenvector Components",
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigendecomposition and plot top eigenvector components.

    Args:
        interaction_matrix: 2D tensor/array of shape (input_dim, input_dim)
        num_components: Number of top eigenvectors to analyze
        title: Title for the plot
        save_path: Optional path to save the figure
        show: Whether to show the plot

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    if isinstance(interaction_matrix, th.Tensor):
        interaction_matrix = interaction_matrix.cpu().numpy()

    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(interaction_matrix)

    # Sort by magnitude of eigenvalues
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Plot top eigenvector components
    fig, axes = plt.subplots(num_components, 1, figsize=(12, 3 * num_components))

    if num_components == 1:
        axes = [axes]

    for i in range(num_components):
        ax = axes[i]
        ax.plot(eigenvectors[:, i])
        ax.set_title(f"Eigenvector {i + 1} (λ = {eigenvalues[i]:.4f})")
        ax.set_xlabel("Component Index")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved eigenvector components plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return eigenvalues, eigenvectors


def plot_eigenvalue_spectrum(
    interaction_matrices: th.Tensor,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot the eigenvalue spectrum for all interaction matrices.

    Args:
        interaction_matrices: Tensor of shape (hidden_dim, input_dim, input_dim)
        save_path: Optional path to save the figure
        show: Whether to show the plot
    """
    if isinstance(interaction_matrices, th.Tensor):
        interaction_matrices = interaction_matrices.cpu().numpy()

    hidden_dim = interaction_matrices.shape[0]

    plt.figure(figsize=(12, 6))

    for idx in range(hidden_dim):
        eigenvalues = np.linalg.eigvalsh(interaction_matrices[idx])
        plt.plot(eigenvalues, alpha=0.3, linewidth=0.5)

    plt.title("Eigenvalue Spectrum Across All Hidden Units")
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved eigenvalue spectrum plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


@arguably.command
def visualize_checkpoint(
    checkpoint_path: str,
    output_dir: str = "visualizations",
    num_matrices: int = 9,
    num_eigenvectors: int = 5,
) -> None:
    """Visualize interaction matrices and eigenvectors from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        output_dir: Directory to save visualizations (default: "visualizations")
        num_matrices: Number of interaction matrices to plot (default: 9)
        num_eigenvectors: Number of top eigenvectors to analyze (default: 5)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint
    checkpoint = th.load(checkpoint_path, weights_only=False)
    model_state = checkpoint["model_state_dict"]

    # Extract bilinear weights
    # Shape: (hidden_dim, input_dim, input_dim)
    bilinear_weight = model_state["bilinear.weight"]

    logger.info(f"Bilinear weight shape: {bilinear_weight.shape}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot interaction matrices grid
    logger.info(f"Plotting {num_matrices} interaction matrices...")
    plot_interaction_matrices(
        bilinear_weight,
        num_to_plot=num_matrices,
        save_dir=output_path,
        show=False,
    )

    # Plot eigenvalue spectrum
    logger.info("Plotting eigenvalue spectrum...")
    plot_eigenvalue_spectrum(
        bilinear_weight,
        save_path=output_path / "eigenvalue_spectrum.png",
        show=False,
    )

    # Analyze top eigenvectors for first few matrices
    num_to_analyze = min(3, bilinear_weight.shape[0])
    for idx in range(num_to_analyze):
        logger.info(f"Analyzing eigenvectors for hidden unit {idx}...")
        compute_and_plot_eigenvectors(
            bilinear_weight[idx],
            num_components=num_eigenvectors,
            title=f"Top Eigenvector Components - Hidden Unit {idx}",
            save_path=output_path / f"eigenvectors_unit_{idx}.png",
            show=False,
        )

    logger.info(f"All visualizations saved to {output_path}")


if __name__ == "__main__":
    arguably.run()
