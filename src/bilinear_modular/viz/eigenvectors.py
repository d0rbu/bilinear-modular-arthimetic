"""
Visualization of eigenvectors from eigendecomposition.

This module creates visualizations of eigenvectors and their components
for understanding the learned representations in the bilinear layer.
"""

import json
from pathlib import Path

import arguably
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from loguru import logger

# Use non-interactive backend for server environments
matplotlib.use("Agg")


def load_svd_results(eigendecomp_dir: Path) -> dict:
    """
    Load SVD results from eigendecomposition.

    Args:
        eigendecomp_dir: Directory containing eigendecomposition results

    Returns:
        Dictionary with U, S, Vh matrices and shape metadata
    """
    result_file = eigendecomp_dir / "svd_results.pt"
    if not result_file.exists():
        raise FileNotFoundError(f"SVD results not found: {result_file}")

    logger.info(f"Loading SVD results from {result_file}")
    results = th.load(result_file, map_location="cpu", weights_only=False)
    return results


def load_metadata(eigendecomp_dir: Path) -> dict:
    """Load metadata from eigendecomposition."""
    metadata_file = eigendecomp_dir / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file) as f:
        metadata = json.load(f)
    return metadata


def plot_eigenvector_components(
    eigenvector: th.Tensor,
    eigenvalue: float,
    idx: int,
    title: str | None = None,
    mod_basis: int = 113,
) -> plt.Figure:
    """
    Plot the components of a single eigenvector.

    Args:
        eigenvector: Eigenvector of shape (d_in,)
        eigenvalue: Corresponding eigenvalue
        idx: Index of this eigenvector
        title: Optional custom title
        mod_basis: Modular arithmetic basis

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    components = eigenvector.numpy()
    x = np.arange(len(components))

    # Plot 1: Bar chart of eigenvector components
    ax1.bar(x, components, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.axhline(y=0, color="k", linestyle="-", linewidth=0.8)
    ax1.set_xlabel("Component Index")
    ax1.set_ylabel("Component Value")
    ax1.set_title(f"Eigenvector {idx} Components\nEigenvalue: {eigenvalue:.4f}")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Polar/circular representation for modular arithmetic
    # This helps visualize periodicities
    angles = 2 * np.pi * x / mod_basis
    ax2 = plt.subplot(122, projection="polar")
    ax2.plot(angles, np.abs(components), "o-", alpha=0.7, markersize=3)
    ax2.set_title("Polar View (|components|)")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_top_eigenvectors_heatmap(
    eigenvectors: th.Tensor,
    eigenvalues: th.Tensor,
    n_top: int = 10,
    title: str | None = None,
) -> plt.Figure:
    """
    Plot a heatmap of the top eigenvectors.

    Args:
        eigenvectors: Matrix of eigenvectors, shape (d_in, n_eigenvectors)
        eigenvalues: Vector of eigenvalues, shape (n_eigenvectors,)
        n_top: Number of top eigenvectors to display
        title: Optional custom title

    Returns:
        Matplotlib figure
    """
    n_show = min(n_top, eigenvectors.shape[1])
    eigenvectors_np = eigenvectors[:, :n_show].numpy().T  # Shape: (n_show, d_in)
    eigenvalues_np = eigenvalues[:n_show].numpy()

    fig, ax = plt.subplots(figsize=(12, max(6, n_show * 0.5)))

    # Create heatmap
    im = ax.imshow(eigenvectors_np, aspect="auto", cmap="RdBu_r", interpolation="nearest")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Component Value", rotation=270, labelpad=20)

    # Set ticks and labels
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([f"#{i} (λ={eigenvalues_np[i]:.3f})" for i in range(n_show)])
    ax.set_xlabel("Component Index")
    ax.set_ylabel("Eigenvector")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax.set_title("Top Eigenvectors Heatmap")

    plt.tight_layout()
    return fig


def plot_eigenvalue_spectrum(
    eigenvalues: th.Tensor,
    title: str | None = None,
) -> plt.Figure:
    """
    Plot the eigenvalue spectrum.

    Args:
        eigenvalues: Vector of eigenvalues
        title: Optional custom title

    Returns:
        Matplotlib figure
    """
    eigenvalues_np = eigenvalues.numpy()
    abs_eigenvalues = np.abs(eigenvalues_np)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Eigenvalues (with sign)
    x = np.arange(len(eigenvalues_np))
    colors = ["red" if ev < 0 else "blue" for ev in eigenvalues_np]
    ax1.bar(x, eigenvalues_np, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.axhline(y=0, color="k", linestyle="-", linewidth=0.8)
    ax1.set_xlabel("Eigenvector Index")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_title("Eigenvalue Spectrum (Signed)")
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Eigenvalue magnitudes (log scale)
    ax2.semilogy(x, abs_eigenvalues, "o-", alpha=0.7, markersize=5)
    ax2.set_xlabel("Eigenvector Index (sorted by magnitude)")
    ax2.set_ylabel("Eigenvalue Magnitude (log scale)")
    ax2.set_title("Eigenvalue Magnitude Spectrum")
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_eigenvector_2d_structure(
    eigenvectors: th.Tensor,
    eigenvalues: th.Tensor,
    mod_basis: int = 113,
    n_top: int = 5,
    title: str | None = None,
) -> plt.Figure:
    """
    Plot 2D structure of top eigenvectors to reveal patterns.

    For modular arithmetic, we expect to see periodic/Fourier-like patterns.

    Args:
        eigenvectors: Matrix of eigenvectors, shape (d_in, n_eigenvectors)
        eigenvalues: Vector of eigenvalues
        mod_basis: Modular arithmetic basis
        n_top: Number of top eigenvectors to display
        title: Optional custom title

    Returns:
        Matplotlib figure
    """
    n_show = min(n_top, eigenvectors.shape[1])

    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
    if n_show == 1:
        axes = [axes]

    for i in range(n_show):
        ax = axes[i]
        eigenvector = eigenvectors[:, i].numpy()
        eigenvalue = eigenvalues[i].item()

        # Create a 2D representation based on modular structure
        # Reshape into a grid if possible
        d_in = len(eigenvector)

        # For mod_basis, inputs are often one-hot encoded pairs
        # If d_in = 2 * mod_basis, we can visualize as two separate vectors
        if d_in == 2 * mod_basis:
            # Split into two parts
            part1 = eigenvector[:mod_basis]
            part2 = eigenvector[mod_basis:]

            # Create a 2D plot showing both parts
            x = np.arange(mod_basis)
            ax.plot(x, part1, "o-", label="Input 1", alpha=0.7, markersize=3)
            ax.plot(x, part2, "s-", label="Input 2", alpha=0.7, markersize=3)
            ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
            ax.set_xlabel("Position (mod basis)")
            ax.set_ylabel("Component Value")
            ax.legend()
        else:
            # Just plot the components
            x = np.arange(d_in)
            ax.plot(x, eigenvector, "o-", alpha=0.7, markersize=3)
            ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
            ax.set_xlabel("Component Index")
            ax.set_ylabel("Component Value")

        ax.set_title(f"Eigenvec {i}\nλ={eigenvalue:.4f}")
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


@arguably.command
def visualize_eigenvectors(
    eigendecomp_dir: str,
    n_top: int = 5,
    fig_dir: str = "fig/eigenvectors",
):
    """
    Visualize eigenvectors from SVD eigendecomposition results.

    Args:
        eigendecomp_dir: Directory containing SVD results (svd_results.pt and metadata.json)
        n_top: Number of top singular vectors to visualize
        fig_dir: Directory to save figures
    """
    logger.info("Starting SVD eigenvector visualization")
    logger.info(f"Eigendecomp dir: {eigendecomp_dir}")
    logger.info(f"Top-{n_top} components")

    eigendecomp_path = Path(eigendecomp_dir)
    if not eigendecomp_path.exists():
        raise FileNotFoundError(f"Eigendecomposition directory not found: {eigendecomp_path}")

    # Load SVD results and metadata
    results = load_svd_results(eigendecomp_path)
    metadata = load_metadata(eigendecomp_path)

    logger.info(f"Loaded metadata: {metadata['bilinear_shape']}")
    logger.info(f"Top-k components in results: {metadata['top_k']}")

    # Extract SVD components
    U = results["U"]  # Output directions (d_out, top_k)  # noqa: N806
    S = results["S"]  # Singular values (top_k,)  # noqa: N806
    Vh = results["Vh"]  # Input interaction matrices (top_k, d_in_0 * d_in_1)  # noqa: N806
    shape = results["shape"]

    d_out, d_in_0, d_in_1 = shape["d_out"], shape["d_in_0"], shape["d_in_1"]

    logger.info(f"Bilinear shape: ({d_out}, {d_in_0}, {d_in_1})")
    logger.info(f"Singular values: {S.tolist()}")

    # Create output directory
    fig_path = Path(fig_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    # Limit n_top to available components
    n_top = min(n_top, S.shape[0])

    # 1. Plot singular value spectrum
    fig = plot_eigenvalue_spectrum(
        S,
        title="Singular Value Spectrum",
    )
    fig.savefig(fig_path / "singular_value_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved singular value spectrum")

    # 2. Visualize input interaction matrices (right singular vectors Vh)
    # These represent the most important input interaction patterns
    for i in range(n_top):
        singular_value = S[i].item()
        input_vector = Vh[i, :]  # (d_in_0 * d_in_1,)

        # Reshape back to interaction matrix
        interaction_matrix = input_vector.reshape(d_in_0, d_in_1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot as heatmap
        im = axes[0].imshow(interaction_matrix.numpy(), aspect="auto", cmap="RdBu_r")
        axes[0].set_xlabel("Input 2")
        axes[0].set_ylabel("Input 1")
        axes[0].set_title(f"Component {i}: Interaction Matrix\nSingular value: {singular_value:.4f}")
        plt.colorbar(im, ax=axes[0])

        # Plot flattened vector
        axes[1].plot(input_vector.numpy(), "o-", alpha=0.7, markersize=2)
        axes[1].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        axes[1].set_xlabel("Flattened Index")
        axes[1].set_ylabel("Component Value")
        axes[1].set_title("Flattened View")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(fig_path / f"input_component_{i}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.debug("Saved input component visualizations")

    # 3. Plot heatmap of top input components (Vh)
    n_show = min(n_top, Vh.shape[0])
    fig, ax = plt.subplots(figsize=(12, max(6, n_show * 0.5)))

    im = ax.imshow(Vh[:n_show, :].numpy(), aspect="auto", cmap="RdBu_r", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Component Value", rotation=270, labelpad=20)

    ax.set_yticks(range(n_show))
    ax.set_yticklabels([f"#{i} (σ={S[i].item():.3f})" for i in range(n_show)])
    ax.set_xlabel("Flattened Input Index")
    ax.set_ylabel("Input Component")
    ax.set_title(f"Top {n_show} Input Components (Right Singular Vectors)")

    plt.tight_layout()
    fig.savefig(fig_path / "input_components_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved input components heatmap")

    # 4. Visualize output directions (left singular vectors U)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot as heatmap
    im = axes[0].imshow(U.numpy(), aspect="auto", cmap="RdBu_r")
    axes[0].set_xlabel("Component Index")
    axes[0].set_ylabel("Output Dimension")
    axes[0].set_title("Output Directions (Left Singular Vectors)")
    plt.colorbar(im, ax=axes[0])

    # Plot specific output components
    for i in range(min(5, U.shape[1])):
        axes[1].plot(U[:, i].numpy(), "o-", label=f"Comp {i} (σ={S[i].item():.3f})", alpha=0.7, markersize=3)
    axes[1].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    axes[1].set_xlabel("Output Dimension")
    axes[1].set_ylabel("Component Value")
    axes[1].set_title("Top Output Components")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_path / "output_directions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved output directions")

    logger.info(f"SVD visualization complete! Figures saved to {fig_path}")


if __name__ == "__main__":
    arguably.run()
