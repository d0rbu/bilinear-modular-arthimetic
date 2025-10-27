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

### Training

Dataset generation and model training code is located in the `core/` directory (work in progress).

### Visualization

Once you have a trained model checkpoint, you can visualize the interaction matrices:

```bash
# Visualize with default settings (output indices 0, 1, 112 for mod 113)
uv run python viz/interaction_matrices.py visualize checkpoints/model_epoch_2000.pt

# Visualize specific output classes
uv run python viz/interaction_matrices.py visualize checkpoints/model_epoch_2000.pt --output-indices 0 5 10 50 112

# Change number of eigenvectors to plot
uv run python viz/interaction_matrices.py visualize checkpoints/model_epoch_2000.pt --num-eigenvectors 10

# Save to a different directory
uv run python viz/interaction_matrices.py visualize checkpoints/model_epoch_2000.pt --output-dir figures/experiment_1
```

See [viz/README.md](viz/README.md) for detailed documentation on the visualization module.

## Project Structure

```
.
├── src/
│   └── bilinear_modular/     # Main package
│       └── __init__.py
├── core/                      # Training code (in progress)
├── viz/                       # Visualization tools
│   ├── interaction_matrices.py
│   └── README.md
├── fig/                       # Output directory for figures
├── tests/                     # Test files
│   ├── test_placeholder.py
│   └── test_visualization.py
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
