"""
Eigendecomposition analysis for bilinear layers.

This module computes eigendecompositions of interaction matrices Q = u ·out B
for different output directions u, where B is the bilinear tensor.
"""

import json
from pathlib import Path

import arguably
import torch as th
from loguru import logger


def load_bilinear_checkpoint(checkpoint_path: Path) -> tuple[th.Tensor, dict]:
    """
    Load a bilinear layer checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Tuple of (bilinear_weight, metadata) where bilinear_weight has shape (d_out, d_in_0, d_in_1)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = th.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract the bilinear layer weight
    # Assuming the checkpoint has a key like 'model' or 'bilinear.weight'
    if "model" in checkpoint:
        model_state = checkpoint["model"]
        # Find the bilinear layer weight
        bilinear_weight = None
        for key, value in model_state.items():
            if "bilinear" in key.lower() and "weight" in key.lower():
                bilinear_weight = value
                break
        if bilinear_weight is None:
            raise ValueError("Could not find bilinear weight in checkpoint")
    elif "bilinear.weight" in checkpoint:
        bilinear_weight = checkpoint["bilinear.weight"]
    else:
        # Assume the checkpoint is just the weight tensor
        bilinear_weight = checkpoint

    metadata = {k: v for k, v in checkpoint.items() if k != "model" and not isinstance(v, th.Tensor)}

    logger.info(f"Loaded bilinear weight with shape {bilinear_weight.shape}")
    return bilinear_weight, metadata


def compute_interaction_matrix(bilinear_weight: th.Tensor, output_direction: th.Tensor) -> th.Tensor:
    """
    Compute interaction matrix Q = u ·out B for a given output direction.

    Args:
        bilinear_weight: Bilinear tensor of shape (d_out, d_in_0, d_in_1)
        output_direction: Output direction vector of shape (d_out,)

    Returns:
        Interaction matrix Q of shape (d_in_0, d_in_1)
    """
    # Q = sum_i u_i * B[i, :, :]
    Q = th.einsum("o,oij->ij", output_direction, bilinear_weight)  # noqa: N806
    return Q


def eigendecompose_interaction_matrix(Q: th.Tensor) -> tuple[th.Tensor, th.Tensor]:  # noqa: N803
    """
    Compute eigendecomposition of interaction matrix Q.

    Since Q can be considered symmetric without loss of generality (as per the paper),
    we can use torch.linalg.eigh for efficient symmetric eigendecomposition.

    Args:
        Q: Interaction matrix of shape (d_in_0, d_in_1)

    Returns:
        Tuple of (eigenvalues, eigenvectors) where:
        - eigenvalues: shape (d,) sorted in ascending order
        - eigenvectors: shape (d, d) where eigenvectors[:, i] is the i-th eigenvector
    """
    # Make Q symmetric by averaging with its transpose
    Q_sym = (Q + Q.T) / 2  # noqa: N806

    # Compute eigendecomposition
    # eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = th.linalg.eigh(Q_sym)

    return eigenvalues, eigenvectors


@arguably.command
def eigendecompose(
    checkpoint_path: str,
    mod_basis: int = 113,
    output_dir: str = "exp/eigendecomp",
    top_k: int = 10,
    output_indices: list[int] | None = None,
):
    """
    Perform eigendecomposition on the bilinear layer for specified output directions.

    Args:
        checkpoint_path: Path to the model checkpoint
        mod_basis: The modular arithmetic basis (default: 113)
        output_dir: Directory to save eigendecomposition results
        top_k: Number of top eigenvectors to save (by absolute eigenvalue magnitude)
        output_indices: List of output indices to analyze. If None, analyzes all outputs.
    """
    logger.info(f"Starting eigendecomposition analysis for checkpoint: {checkpoint_path}")
    logger.info(f"Mod basis: {mod_basis}, Top-k: {top_k}")

    # Load checkpoint
    checkpoint_path_obj = Path(checkpoint_path)
    bilinear_weight, metadata = load_bilinear_checkpoint(checkpoint_path_obj)

    d_out, d_in_0, d_in_1 = bilinear_weight.shape
    logger.info(f"Bilinear layer shape: ({d_out}, {d_in_0}, {d_in_1})")

    # Determine which output indices to analyze
    if output_indices is None:
        output_indices = list(range(d_out))
    else:
        output_indices = [idx for idx in output_indices if 0 <= idx < d_out]

    logger.info(f"Analyzing {len(output_indices)} output directions")

    # Create output directory
    output_path = Path(output_dir) / f"mod_{mod_basis}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each output direction
    results = {}
    for output_idx in output_indices:
        logger.info(f"Processing output direction {output_idx}/{d_out-1}")

        # Create one-hot output direction
        output_direction = th.zeros(d_out)
        output_direction[output_idx] = 1.0

        # Compute interaction matrix
        Q = compute_interaction_matrix(bilinear_weight, output_direction)  # noqa: N806

        # Eigendecompose
        eigenvalues, eigenvectors = eigendecompose_interaction_matrix(Q)

        # Sort by absolute eigenvalue magnitude (descending)
        abs_eigenvalues = th.abs(eigenvalues)
        sorted_indices = th.argsort(abs_eigenvalues, descending=True)

        # Keep top-k
        top_indices = sorted_indices[:top_k]
        top_eigenvalues = eigenvalues[top_indices]
        top_eigenvectors = eigenvectors[:, top_indices]  # shape: (d_in, top_k)

        # Store results
        results[output_idx] = {
            "eigenvalues": top_eigenvalues,
            "eigenvectors": top_eigenvectors,
            "eigenvalue_magnitude": abs_eigenvalues[top_indices],
        }

        # Save individual output direction results
        output_file = output_path / f"output_{output_idx}.pt"
        th.save(
            {
                "output_idx": output_idx,
                "eigenvalues": top_eigenvalues,
                "eigenvectors": top_eigenvectors,
                "eigenvalue_magnitude": abs_eigenvalues[top_indices],
                "Q": Q,  # Save interaction matrix for reference
            },
            output_file,
        )
        logger.debug(f"Saved results for output {output_idx} to {output_file}")

    # Save summary
    summary = {
        "mod_basis": mod_basis,
        "checkpoint_path": str(checkpoint_path),
        "bilinear_shape": [d_out, d_in_0, d_in_1],
        "top_k": top_k,
        "output_indices": output_indices,
        "metadata": metadata,
    }

    summary_file = output_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Eigendecomposition complete! Results saved to {output_path}")
    logger.info(f"Processed {len(output_indices)} output directions")

    # Print some statistics
    all_top_eigenvals = th.stack([results[idx]["eigenvalue_magnitude"] for idx in output_indices])
    logger.info("Top eigenvalue statistics:")
    logger.info(f"  Mean: {all_top_eigenvals[:, 0].mean():.4f}")
    logger.info(f"  Std: {all_top_eigenvals[:, 0].std():.4f}")
    logger.info(f"  Max: {all_top_eigenvals[:, 0].max():.4f}")
    logger.info(f"  Min: {all_top_eigenvals[:, 0].min():.4f}")


if __name__ == "__main__":
    arguably.run()

