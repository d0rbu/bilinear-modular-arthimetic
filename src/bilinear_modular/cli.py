"""Command-line interface for bilinear-modular."""

import arguably

from bilinear_modular.core import generate_dataset as _generate_dataset
from bilinear_modular.core import train as _train
from bilinear_modular.exp import eigendecompose as _eigendecompose
from bilinear_modular.viz.eigenvectors import visualize_eigenvectors as _visualize_eigenvectors
from bilinear_modular.viz.interaction_matrices import visualize as _visualize_interaction_matrices


@arguably.command
def generate_dataset(
    mod_basis: int = 113,
    *,
    output_dir: str | None = None,
) -> None:
    """Generate a complete modular arithmetic dataset.

    Creates all combinations of (a, b) where both range from 0 to mod_basis-1,
    computes c = (a + b) % mod_basis for each pair, and saves as PyTorch tensors.

    Args:
        mod_basis: The modulus for arithmetic operations (default: 113)
        output_dir: Directory to save the dataset (default: data/{mod_basis})
    """
    _generate_dataset(mod_basis=mod_basis, output_dir=output_dir)


@arguably.command
def train(
    *,
    mod_basis: int = 113,
    use_output_projection: bool = False,
    hidden_dim: int = 100,
    batch_size: int = 128,
    learning_rate: float = 3e-3,
    weight_decay: float = 0.5,
    epochs: int = 2000,
    grad_accum_steps: int = 1,
    checkpoint_dir: str = "checkpoints",
    checkpoint_every: int = 100,
    device: str | None = None,
    compile: bool = True,
    seed: int = 0,
    resume_from: str | None = None,
    log_level: str = "INFO",
) -> None:
    """Train a bilinear layer on modular arithmetic.

    Args:
        mod_basis: The modulus for arithmetic (default: 113)
        use_output_projection: Whether to use output projection layer (default: False)
        hidden_dim: Hidden dimension size (default: 100)
        batch_size: Batch size for training (default: 128)
        learning_rate: Learning rate (default: 3e-3)
        weight_decay: Weight decay for AdamW (default: 0.5)
        epochs: Number of training epochs (default: 2000)
        grad_accum_steps: Gradient accumulation steps (default: 1)
        checkpoint_dir: Directory to save checkpoints (default: "checkpoints")
        checkpoint_every: Save checkpoint every N epochs (default: 100)
        device: Device to train on (default: auto-detect cuda/cpu)
        compile: Whether to use torch.compile (default: True)
        seed: Random seed (default: 0)
        resume_from: Path to checkpoint to resume from (optional)
        log_level: Logging level (default: INFO)
    """
    _train(
        mod_basis=mod_basis,
        use_output_projection=use_output_projection,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        grad_accum_steps=grad_accum_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        device=device,
        compile=compile,
        seed=seed,
        resume_from=resume_from,
        log_level=log_level,
    )


@arguably.command
def eigendecompose(
    checkpoint_path: str,
    *,
    mod_basis: int = 113,
    output_dir: str = "exp/eigendecomp",
    top_k: int = 10,
) -> None:
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
    _eigendecompose(
        checkpoint_path=checkpoint_path,
        mod_basis=mod_basis,
        output_dir=output_dir,
        top_k=top_k,
    )


@arguably.command
def visualize_interaction_matrices(
    checkpoint_path: str,
    *,
    output_indices: list[int] | None = None,
    mod_basis: int = 113,
    num_components: int = 5,
    output_dir: str = "fig",
) -> None:
    """Visualize interaction matrices and singular vector components from a trained bilinear model.

    Args:
        checkpoint_path: Path to the model checkpoint file
        output_indices: List of output indices to visualize (default: [0, 1, mod_basis-1])
        mod_basis: Modular basis (P) used in training
        num_components: Number of top singular vector components to plot
        output_dir: Directory to save figures
    """
    _visualize_interaction_matrices(
        checkpoint_path=checkpoint_path,
        output_indices=output_indices,
        mod_basis=mod_basis,
        num_components=num_components,
        output_dir=output_dir,
    )


@arguably.command
def visualize_eigenvectors(
    eigendecomp_dir: str,
    mod_basis: int = 113,
    output_indices: list[int] | None = None,
    n_top: int = 5,
    fig_dir: str = "fig/eigenvectors",
) -> None:
    """Visualize eigenvectors from eigendecomposition results.

    Args:
        eigendecomp_dir: Directory containing eigendecomposition results
        mod_basis: The modular arithmetic basis (default: 113)
        output_indices: List of output indices to visualize. If None, visualizes all available.
        n_top: Number of top eigenvectors to visualize per output
        fig_dir: Directory to save figures
    """
    _visualize_eigenvectors(
        eigendecomp_dir=eigendecomp_dir,
        mod_basis=mod_basis,
        output_indices=output_indices,
        n_top=n_top,
        fig_dir=fig_dir,
    )


def main() -> None:
    """Entry point for the CLI."""
    arguably.run()


if __name__ == "__main__":
    main()
