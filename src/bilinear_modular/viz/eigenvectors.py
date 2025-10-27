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


def load_eigendecomp_results(eigendecomp_dir: Path, output_idx: int) -> dict:
    """
    Load eigendecomposition results for a specific output direction.

    Args:
        eigendecomp_dir: Directory containing eigendecomposition results
        output_idx: Index of the output direction

    Returns:
        Dictionary with eigenvalues, eigenvectors, and metadata
    """
    result_file = eigendecomp_dir / f"output_{output_idx}.pt"
    if not result_file.exists():
        raise FileNotFoundError(f"Eigendecomposition results not found: {result_file}")

    logger.info(f"Loading eigendecomposition results from {result_file}")
    results = th.load(result_file, map_location="cpu", weights_only=False)
    return results


def load_summary(eigendecomp_dir: Path) -> dict:
    """Load summary metadata from eigendecomposition."""
    summary_file = eigendecomp_dir / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    with open(summary_file) as f:
        summary = json.load(f)
    return summary


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
    mod_basis: int = 113,
    output_indices: list[int] | None = None,
    n_top: int = 5,
    fig_dir: str = "fig/eigenvectors",
):
    """
    Visualize eigenvectors from eigendecomposition results.

    Args:
        eigendecomp_dir: Directory containing eigendecomposition results
        mod_basis: The modular arithmetic basis (default: 113)
        output_indices: List of output indices to visualize. If None, visualizes all available.
        n_top: Number of top eigenvectors to visualize per output
        fig_dir: Directory to save figures
    """
    logger.info("Starting eigenvector visualization")
    logger.info(f"Eigendecomp dir: {eigendecomp_dir}")
    logger.info(f"Mod basis: {mod_basis}, Top-{n_top} eigenvectors")

    eigendecomp_path = Path(eigendecomp_dir) / f"mod_{mod_basis}"
    if not eigendecomp_path.exists():
        raise FileNotFoundError(f"Eigendecomposition directory not found: {eigendecomp_path}")

    # Load summary
    summary = load_summary(eigendecomp_path)
    logger.info(f"Loaded summary: {summary['bilinear_shape']}")

    # Determine which outputs to visualize
    if output_indices is None:
        output_indices = summary["output_indices"]

    # Create output directory
    fig_path = Path(fig_dir) / f"mod_{mod_basis}"
    fig_path.mkdir(parents=True, exist_ok=True)

    # Visualize each output direction
    for output_idx in output_indices:
        logger.info(f"Visualizing output direction {output_idx}")

        try:
            results = load_eigendecomp_results(eigendecomp_path, output_idx)
        except FileNotFoundError:
            logger.warning(f"No results found for output {output_idx}, skipping")
            continue

        eigenvectors = results["eigenvectors"]  # Shape: (d_in, n_eigenvectors)
        eigenvalues = results["eigenvalues"]

        output_fig_dir = fig_path / f"output_{output_idx}"
        output_fig_dir.mkdir(exist_ok=True)

        # 1. Plot eigenvalue spectrum
        fig = plot_eigenvalue_spectrum(
            eigenvalues,
            title=f"Output {output_idx}: Eigenvalue Spectrum",
        )
        fig.savefig(output_fig_dir / "eigenvalue_spectrum.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Saved eigenvalue spectrum")

        # 2. Plot heatmap of top eigenvectors
        fig = plot_top_eigenvectors_heatmap(
            eigenvectors,
            eigenvalues,
            n_top=n_top,
            title=f"Output {output_idx}: Top {n_top} Eigenvectors",
        )
        fig.savefig(output_fig_dir / "eigenvectors_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Saved eigenvectors heatmap")

        # 3. Plot individual eigenvector components
        for i in range(min(n_top, eigenvectors.shape[1])):
            eigenvector = eigenvectors[:, i]
            eigenvalue = eigenvalues[i].item()

            fig = plot_eigenvector_components(
                eigenvector,
                eigenvalue,
                i,
                title=f"Output {output_idx}: Eigenvector {i}",
                mod_basis=mod_basis,
            )
            fig.savefig(output_fig_dir / f"eigenvector_{i}_components.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        logger.debug("Saved individual eigenvector plots")

        # 4. Plot 2D structure
        fig = plot_eigenvector_2d_structure(
            eigenvectors,
            eigenvalues,
            mod_basis=mod_basis,
            n_top=n_top,
            title=f"Output {output_idx}: Eigenvector Structure",
        )
        fig.savefig(output_fig_dir / "eigenvector_structure.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Saved eigenvector structure plot")

        logger.info(f"✓ Completed visualizations for output {output_idx}")

    logger.info(f"Eigenvector visualization complete! Figures saved to {fig_path}")


if __name__ == "__main__":
    arguably.run()

