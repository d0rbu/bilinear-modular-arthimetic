# Core Training Infrastructure

This directory contains the core training infrastructure for bilinear modular arithmetic experiments.

## Modules

### `train.py`
Main training loop with observability and checkpointing.

**Features:**
- Bilinear layer implementation for modular arithmetic
- Gradient accumulation support
- Automatic checkpointing every N epochs
- Training observability with trackio, tqdm, and loguru
- torch.compile support for faster training
- Configurable hyperparameters via CLI

**Usage:**
```bash
# Basic training
uv run python -m core.train train

# Custom configuration
uv run python -m core.train train \
  --mod-basis 113 \
  --hidden-dim 100 \
  --batch-size 128 \
  --learning-rate 0.003 \
  --weight-decay 0.5 \
  --epochs 2000 \
  --checkpoint-every 100

# Resume from checkpoint
uv run python -m core.train train --resume-from checkpoints/checkpoint_epoch_100.pt

# With gradient accumulation
uv run python -m core.train train --grad-accum-steps 4
```

### `visualize.py`
Visualization utilities for analyzing learned interaction matrices and eigenvectors.

**Features:**
- Plot interaction matrices from bilinear weights
- Compute and visualize eigendecomposition
- Plot eigenvalue spectrum across all hidden units
- Analyze top eigenvector components

**Usage:**
```bash
# Visualize a checkpoint
uv run python -m core.visualize visualize-checkpoint \
  checkpoints/checkpoint_epoch_2000.pt \
  --output-dir visualizations \
  --num-matrices 9 \
  --num-eigenvectors 5
```

## Model Architecture

The bilinear model learns modular addition through a simple architecture:

```
Input A (one-hot) ──┐
                    ├── Bilinear Layer ── ReLU ── Linear ── Output (logits)
Input B (one-hot) ──┘
```

**Key differences from MLP approach:**
- Uses `nn.Bilinear` instead of two separate linear layers
- Direct interaction between inputs through learned weight tensors
- Each hidden unit learns a (input_dim × input_dim) interaction matrix
- Potentially captures modular arithmetic structure more directly

## Model Introspection

The bilinear weights can be extracted for analysis:
```python
# Get interaction matrices from model
interaction_matrices = model.get_interaction_matrices()
# Shape: (hidden_dim, input_dim, input_dim)
```

Each matrix represents how that hidden unit combines the two inputs.

## Observability

Training is instrumented with:
- **trackio**: Experiment tracking and metrics logging
- **tqdm**: Progress bars for epochs and batches
- **loguru**: Structured logging

Metrics logged:
- `train/batch_loss`: Loss per batch
- `train/epoch_loss`: Average loss per epoch
- `val/loss`: Validation loss
- `val/accuracy`: Validation accuracy

## TODO

⚠️ **Data loading is currently a placeholder** - waiting on parallel agent to implement dataset generation.

Expected interface:
```python
train_loader, val_loader = load_data(mod_basis, batch_size)
```

Where each batch yields `(a, b, targets)` with:
- `a`: one-hot encoded first operand
- `b`: one-hot encoded second operand  
- `targets`: integer labels for (a + b) mod P

