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

- Python 3.11 or higher
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

GitHub Actions automatically runs the following checks on every push and pull request:
- Ruff linting
- Ruff formatting verification
- ty type checking
- pytest test suite

The CI workflow tests against Python 3.11, 3.12, and 3.13.

## Project Structure

```
.
├── src/
│   └── bilinear_modular/     # Main package
│       └── __init__.py
├── tests/                     # Test files
│   └── test_placeholder.py
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

