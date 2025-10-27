"""Visualize interaction matrices from bilinear layers.

This module provides functionality to:
1. Load checkpoints from trained bilinear models
2. Extract and visualize interaction matrices for specific outputs
3. Compute and visualize top singular vector components
"""

from pathlib import Path

import arguably
import matplotlib.pyplot as plt
import torch as th
from loguru import logger

from ..core.train import BilinearModularModel


def _remove_compiled_prefix(state_dict: dict) -> dict:
    """Remove '_orig_mod.' prefix from state dict keys if present.

    This handles models that were saved after being compiled with th.compile().
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_key = key[len("_orig_mod.") :]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


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
    matrix_cpu = matrix.detach().cpu()

    # Auto-scale if not provided
    if vmin is None:
        vmin = -matrix_cpu.abs().max().item()
    if vmax is None:
        vmax = matrix_cpu.abs().max().item()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix_cpu, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

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


def plot_singular_vectors(  # noqa: N802
    U: th.Tensor,  # noqa: N803
    S: th.Tensor,  # noqa: N803
    Vh: th.Tensor,  # noqa: N803
    output_idx: int,
    save_path: Path,
    num_components: int = 5,
    title: str | None = None,
):
    """Plot top singular vector components.

    Args:
        U: Left singular vectors of shape (m, m)
        S: Singular values of shape (min(m, n),)
        Vh: Right singular vectors of shape (n, n)
        output_idx: Output class index
        save_path: Path to save the figure
        num_components: Number of top components to plot
        title: Optional custom title
    """
    U_cpu = U.detach().cpu()  # noqa: N806
    S_cpu = S.detach().cpu()  # noqa: N806
    Vh_cpu = Vh.detach().cpu()  # noqa: N806

    k = min(num_components, len(S_cpu))

    fig, axes = plt.subplots(2, k, figsize=(4 * k, 8))
    if k == 1:
        axes = axes.reshape(2, 1)

    for i in range(k):
        # Plot left singular vector (U)
        ax_u = axes[0, i]
        u_vec = U_cpu[:, i]
        ax_u.bar(range(len(u_vec)), u_vec)
        ax_u.set_title(f"U[{i}] (σ = {S_cpu[i]:.3f})")
        ax_u.set_xlabel("Component Index")
        ax_u.set_ylabel("Value")
        ax_u.grid(True, alpha=0.3)

        # Plot right singular vector (Vh)
        ax_v = axes[1, i]
        v_vec = Vh_cpu[i, :]
        ax_v.bar(range(len(v_vec)), v_vec)
        ax_v.set_title(f"V[{i}]")
        ax_v.set_xlabel("Component Index")
        ax_v.set_ylabel("Value")
        ax_v.grid(True, alpha=0.3)

    if title is None:
        title = f"Top {k} Singular Vectors for Output Class {output_idx}"
    fig.suptitle(title, fontsize=14, y=1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved singular vector plot to {save_path}")


@arguably.command
def visualize(
    checkpoint_path: str,
    *,
    output_indices: list[int] | None = None,
    mod_basis: int = 113,
    num_components: int = 5,
    output_dir: str = "fig",
):
    """Visualize interaction matrices and singular vector components from a trained bilinear model.

    Args:
        checkpoint_path: Path to the model checkpoint file
        output_indices: List of output indices to visualize (default: [0, 1, mod_basis-1])
        mod_basis: Modular basis (P) used in training
        num_components: Number of top singular vector components to plot
        output_dir: Directory to save figures
    """
    # Setup
    checkpoint_path_obj = Path(checkpoint_path)
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(exist_ok=True, parents=True)

    # Default output indices if not provided
    if not output_indices:
        output_indices = [0, 1, mod_basis - 1]

    logger.info(f"Visualizing interaction matrices for output indices: {output_indices}")

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path_obj}")
    checkpoint = th.load(checkpoint_path_obj, map_location="cpu", weights_only=False)
    logger.info("Checkpoint loaded successfully")

    # Extract config from checkpoint
    config = checkpoint.get("config", {})
    input_dim = config.get("mod_basis", mod_basis)
    hidden_dim = config.get("hidden_dim", 100)
    output_dim = input_dim
    use_output_projection = config.get("use_output_projection", False)

    logger.info(
        f"Creating model with input_dim={input_dim}, hidden_dim={hidden_dim}, "
        f"output_dim={output_dim}, use_output_projection={use_output_projection}"
    )

    # Create model and load state dict
    model = BilinearModularModel(input_dim, hidden_dim, output_dim, use_output_projection)
    state_dict = _remove_compiled_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model.eval()

    # Extract bilinear weights using model method
    bilinear_weights = model.get_interaction_matrices()
    projection_weights = model.output.weight.data if model.use_output_projection else th.eye(output_dim)

    d_out_bilinear, d_in_0, d_in_1 = bilinear_weights.shape
    logger.info(f"Bilinear weights shape: (d_out_bilinear={d_out_bilinear}, d_in_0={d_in_0}, d_in_1={d_in_1})")
    d_out_proj, d_hidden = projection_weights.shape
    logger.info(f"Projection weights shape: (d_out_proj={d_out_proj}, d_hidden={d_hidden})")

    # Process each output index
    for out_idx in output_indices:
        if out_idx >= d_out_proj or out_idx < 0:
            logger.warning(f"Output index {out_idx} out of range [0, {d_out_proj}). Skipping.")
            continue

        logger.info(f"Processing output index {out_idx}")

        # Create output vector (one-hot)
        output_vec = th.zeros(d_out_proj)
        output_vec[out_idx] = 1.0

        # Project output vector if using output projection
        output_vec = projection_weights.T @ output_vec.unsqueeze(1)

        # Interaction matrix is the tensor product along the output dimension
        interaction = th.einsum("ijk,i->jk", bilinear_weights, output_vec.squeeze())

        # Plot interaction matrix
        matrix_path = output_dir_obj / f"interaction_matrix_output_{out_idx}.png"
        plot_interaction_matrix(interaction, out_idx, matrix_path)

        # Compute SVD
        U, S, Vh = th.linalg.svd(interaction, full_matrices=False)  # noqa: N806

        # Plot singular vectors
        svd_path = output_dir_obj / f"singular_vectors_output_{out_idx}.png"
        plot_singular_vectors(U, S, Vh, out_idx, svd_path, num_components=num_components)

    logger.success(f"Visualization complete! Figures saved to {output_dir_obj}")


if __name__ == "__main__":
    arguably.run()
