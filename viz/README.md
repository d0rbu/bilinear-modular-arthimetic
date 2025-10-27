# Visualization Module

This module provides tools for visualizing interaction matrices and eigenvector components from trained bilinear models.

## Overview

The bilinear layer has weights of shape `(d_out, d_in_0, d_in_1)`. For a given output class `c`, we can visualize how the two inputs interact by computing the interaction matrix:

```
M_c = sum_i L[i] * v[i]
```

where `v` is a one-hot vector with `v[c] = 1`. This gives us a `(d_in_0, d_in_1)` matrix showing the interaction patterns.

## Usage

### Basic Usage

Visualize interaction matrices for default output indices (0, 1, and P-1):

```bash
uv run python viz/interaction_matrices.py visualize checkpoints/model_epoch_2000.pt
```

### Custom Output Indices

Visualize specific output classes:

```bash
uv run python viz/interaction_matrices.py visualize checkpoints/model_epoch_2000.pt --output-indices 0 5 10 50 112
```

### Change Number of Eigenvectors

By default, the top 5 eigenvectors are computed. To change this:

```bash
uv run python viz/interaction_matrices.py visualize checkpoints/model_epoch_2000.pt --num-eigenvectors 10
```

### Different Modular Basis

If you trained with a different modular basis (default is 113):

```bash
uv run python viz/interaction_matrices.py visualize checkpoints/model_epoch_2000.pt --mod-basis 97
```

### Custom Output Directory

Save figures to a different directory:

```bash
uv run python viz/interaction_matrices.py visualize checkpoints/model_epoch_2000.pt --output-dir figures/experiment_1
```

## Output

The tool generates two types of plots for each output index:

1. **Interaction Matrix Heatmap** (`interaction_matrix_output_{idx}.png`)
   - Visualizes the `(d_in_0, d_in_1)` interaction matrix as a heatmap
   - Shows how input features interact to produce the output
   - Uses a diverging colormap (RdBu_r) centered at zero

2. **Top Eigenvector Components** (`eigenvectors_output_{idx}.png`)
   - Shows the top k eigenvectors of the interaction matrix
   - Plots both real and imaginary components
   - Eigenvalues are displayed in the subplot titles

## Understanding the Visualizations

### Interaction Matrix
- The interaction matrix reveals the multiplicative structure learned by the bilinear layer
- For modular addition, we expect to see patterns related to the Discrete Fourier Transform
- Strong diagonal or periodic patterns may indicate frequency-based representations

### Eigenvector Components
- The top eigenvectors capture the dominant modes of interaction
- For modular arithmetic, these often correspond to specific frequencies
- Complex eigenvalues indicate rotational/oscillatory behavior
- The eigenvector components show how the model represents numbers in a transformed space

## Implementation Details

The visualization code:
- Loads checkpoints flexibly (handles various checkpoint formats)
- Uses efficient tensor operations via einsum for computing interaction matrices
- Computes eigendecompositions to reveal the principal components
- Generates high-quality figures (300 DPI) suitable for papers/presentations

## Integration with Training Code

This module is designed to work with checkpoints from the training code in `core/`. It expects:
- A checkpoint file containing model state (typically saved with `torch.save`)
- The bilinear layer weights to be accessible in the state dict
- Standard checkpoint structure (will auto-detect key names)

