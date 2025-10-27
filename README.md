# Bilinear Modular Arithmetic

Training a bilinear layer on modular arithmetic, with visualization of interaction matrices and top eigenvector components.

## Development Setup

This project uses modern Python tooling for a streamlined development experience:

- **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager
- **[ruff](https://docs.astral.sh/ruff/)** - Lightning-fast linting and formatting
- **[ty](https://docs.astral.sh/ty/)** - Type checking
- **[pytest](https://docs.pytest.org/)** - Testing framework
- **[pre-commit](https://pre-commit.com/)** - Git hooks for code quality

### Prerequisites

- Python 3.13
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed

### Installation

1. Clone the repository:
```bash
git clone https://github.com/d0rbu/bilinear-modular-arthimetic.git
cd bilinear-modular-arthimetic
```

2. Install dependencies:
```bash
uv sync --dev
```

3. Install pre-commit hooks:
```bash
uv run pre-commit install
```

## Development Workflow

### Running Tests

```bash
uv run pytest
```

### Linting and Formatting

Check for linting issues:
```bash
uv run ruff check .
```

Auto-fix linting issues:
```bash
uv run ruff check --fix .
```

Format code:
```bash
uv run ruff format .
```

### Type Checking

```bash
uvx ty check
```

### Pre-commit Hooks

Pre-commit hooks are automatically run before each commit. They will:
- Run ruff linting with auto-fixes
- Run ruff formatting
- Run ty type checking

To manually run all pre-commit hooks:
```bash
uv run pre-commit run --all-files
```

## Continuous Integration

GitHub Actions automatically runs the following checks on every push and pull request, split into separate jobs:
- **Linting and Formatting**: Ruff linting and formatting verification
- **Type Checking**: ty type checking
- **Tests**: pytest test suite

All jobs run on Python 3.13.

## Usage

### Dataset Generation

Generate a modular arithmetic dataset for a given modulus:

```python
from bilinear_modular import generate_dataset, ModularArithmeticDataset

# Generate dataset for mod 113 (creates all a+b combinations)
dataset = generate_dataset(mod_basis=113)

# Dataset info
print(f"Total samples: {len(dataset)}")  # 113 * 113 = 12769
print(f"Training samples: {dataset.train_size}")  # 80% = 10215
print(f"Validation samples: {dataset.val_size}")  # 20% = 2554

# Get training batches (returns torch tensors)
inputs, targets = dataset.get_train_batch(batch_size=128)
# inputs: (128, 226) - one-hot encoded [a, b]
# targets: (128, 113) - one-hot encoded c where c = (a + b) % 113

# Get all training data
all_train_inputs, all_train_targets = dataset.get_all_train()

# Use as iterator for training loops
dataset.batch_size = 128
dataset.train()  # Set to training mode
for inputs, targets in dataset:
    # Your training code here
    pass

# Load existing dataset
dataset = ModularArithmeticDataset(mod_basis=113)
```

For a complete example, see `examples/generate_dataset_example.py`.

### Dataset Features

- **Automatic caching**: Datasets are saved to `data/{mod_basis}/` for reuse as .pt files
- **Pure PyTorch**: All data stored and returned as PyTorch tensors (no numpy)
- **One-hot encoding**: Optional one-hot encoding of inputs and outputs
- **Efficient batching**: Simple API for getting training/validation batches
- **Iterator protocol**: Supports `__iter__` and `__next__` for easy training loops
- **Reproducible splits**: Consistent 80/20 train/val split with fixed seed

## Project Structure

```
.
├── src/
│   └── bilinear_modular/     # Main package
│       ├── __init__.py
│       └── core/
│           ├── __init__.py
│           └── dataset.py     # Dataset generation and loading
├── tests/                     # Test files
│   ├── test_placeholder.py
│   └── test_dataset.py        # Dataset tests
├── examples/                  # Example scripts
│   └── generate_dataset_example.py
├── data/                      # Generated datasets (gitignored)
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI
├── .pre-commit-config.yaml   # Pre-commit hooks config
├── pyproject.toml            # Project configuration
└── README.md
```

## Contributing

1. Make sure all tests pass: `uv run pytest`
2. Ensure code is properly formatted: `uv run ruff format .`
3. Check for linting issues: `uv run ruff check .`
4. Verify type checking passes: `uvx ty check`

Pre-commit hooks will automatically run these checks before each commit.
