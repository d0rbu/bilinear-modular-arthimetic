"""Command-line interface for bilinear-modular."""

import arguably

from bilinear_modular.core import generate_dataset as _generate_dataset
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


def main() -> None:
    """Entry point for the CLI."""
    arguably.run()


if __name__ == "__main__":
    main()
