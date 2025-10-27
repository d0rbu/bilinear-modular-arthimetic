"""Command-line interface for bilinear-modular."""

import arguably

from bilinear_modular.core import generate_dataset as _generate_dataset


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


def main() -> None:
    """Entry point for the CLI."""
    arguably.run()


if __name__ == "__main__":
    main()
