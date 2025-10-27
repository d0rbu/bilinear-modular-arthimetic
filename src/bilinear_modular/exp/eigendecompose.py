"""Eigendecomposition analysis for bilinear layers via SVD.

This module computes SVD on the flattened bilinear tensor to extract
the most important input and output directions.
"""

import json
from pathlib import Path

import arguably
import torch as th
from loguru import logger

from ..core.train import BilinearModularModel


@arguably.command
def eigendecompose(
    checkpoint_path: str,
    *,
    mod_basis: int = 113,
    output_dir: str = "exp/eigendecomp",
    top_k: int = 10,
):
    """Perform SVD on the bilinear layer to extract top eigenvectors.

    Following the paper's HOSVD approach, we flatten the bilinear tensor
    from (d_out, d_in_0, d_in_1) to (d_out, d_in_0 * d_in_1) and apply
    standard SVD to extract the most important directions.

    Args:
        checkpoint_path: Path to the model checkpoint
        mod_basis: The modular arithmetic basis (default: 113)
        output_dir: Directory to save eigendecomposition results
        top_k: Number of top eigenvectors to save (by singular value magnitude)
    """
    logger.info(f"Starting SVD analysis for checkpoint: {checkpoint_path}")
    logger.info(f"Mod basis: {mod_basis}, Top-k: {top_k}")

    # Load checkpoint
    checkpoint_path_obj = Path(checkpoint_path)
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Extract bilinear weights using model method
    bilinear_weights = model.get_interaction_matrices()

    d_out, d_in_0, d_in_1 = bilinear_weights.shape
    logger.info(f"Bilinear layer shape: ({d_out}, {d_in_0}, {d_in_1})")

    # Flatten the last two dimensions: (d_out, d_in_0, d_in_1) -> (d_out, d_in_0 * d_in_1)
    flattened = bilinear_weights.reshape(d_out, d_in_0 * d_in_1)
    logger.info(f"Flattened to shape: {flattened.shape}")

    # Perform SVD
    logger.info("Computing SVD...")
    U, S, Vh = th.linalg.svd(flattened, full_matrices=False)  # noqa: N806
    logger.info(f"SVD complete. Singular values shape: {S.shape}")

    # Take top-k components
    top_k = min(top_k, S.shape[0])
    U_topk = U[:, :top_k]  # (d_out, top_k) - output directions  # noqa: N806
    S_topk = S[:top_k]  # (top_k,) - singular values  # noqa: N806
    Vh_topk = Vh[:top_k, :]  # (top_k, d_in_0 * d_in_1) - input interaction matrices  # noqa: N806

    logger.info(f"Top {top_k} singular values: {S_topk.tolist()}")

    # Create output directory
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(exist_ok=True, parents=True)

    # Save results
    results = {
        "U": U_topk.cpu(),  # Output directions
        "S": S_topk.cpu(),  # Singular values
        "Vh": Vh_topk.cpu(),  # Input interaction matrices (flattened)
        "shape": {"d_out": d_out, "d_in_0": d_in_0, "d_in_1": d_in_1},
        "top_k": top_k,
    }

    output_file = output_dir_obj / "svd_results.pt"
    th.save(results, output_file)
    logger.success(f"Saved SVD results to {output_file}")

    # Save metadata
    metadata = {
        "checkpoint_path": str(checkpoint_path_obj),
        "mod_basis": mod_basis,
        "top_k": top_k,
        "bilinear_shape": [d_out, d_in_0, d_in_1],
        "singular_values": S_topk.tolist(),
    }

    metadata_file = output_dir_obj / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.success(f"Saved metadata to {metadata_file}")

    logger.success("Eigendecomposition complete!")


if __name__ == "__main__":
    arguably.run()
