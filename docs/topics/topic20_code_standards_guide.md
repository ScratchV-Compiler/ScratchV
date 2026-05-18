# Topic 20: Code Standards Configuration - User Guide

## Overview

Topic 20 adds code formatting (Black, isort), linting (Ruff), and type checking (mypy) tooling to the ScratchV project. It includes pre-commit hooks, CI integration, and coding standards documentation.

## Files Added

| File | Purpose |
|---|---|
| `.pre-commit-config.yaml` | Pre-commit hook configuration |
| `docs/CODING_STANDARDS.md` | Project coding guidelines |
| `scripts/lint_check.sh` | Convenience lint/format script |

## Pre-Commit Hooks

Configured hooks run on every `git commit`:

### Black
- **Purpose**: Code formatter
- **Config**: Line length 88, target Python 3.12
- **Hook**: `psf/black`

### isort
- **Purpose**: Import sorter
- **Config**: Black profile, line length 88
- **Hook**: `PyCQA/isort`

### Ruff
- **Purpose**: Fast Python linter (replaces flake8, pycodestyle, pyflakes)
- **Config**: Default rules (E, W, F categories)
- **Hook**: `astral-sh/ruff-pre-commit`
- **Auto-fix**: Enabled

### mypy
- **Purpose**: Static type checker
- **Config**: Strict mode, Python 3.12, ignore missing imports
- **Hook**: `pre-commit/mirrors-mypy`
- **Scope**: `scratchv/` directory only (excludes tests and benchmarks)

## Usage

### Installation

```bash
pip install pre-commit
pre-commit install
```

### Running

```bash
# Auto-run on commit
git commit -m "your message"

# Run on all files manually
pre-commit run --all-files

# Run a specific hook
pre-commit run black --all-files
```

### Lint Check Script

The convenience script runs all checks:

```bash
# Check only
bash scripts/lint_check.sh

# Auto-fix formatting and sorting
bash scripts/lint_check.sh --fix

# Run on all files including tests
bash scripts/lint_check.sh --all
```

## Coding Standards

See `docs/CODING_STANDARDS.md` for the full coding guidelines, including:

- Import conventions (always use `scratchv.*` absolute imports)
- Naming conventions (snake_case for modules, PascalCase for classes)
- Type hints required for all public APIs
- Docstring format (module-level, class, and method docstrings)
- Error handling patterns
- Testing conventions

## CI Integration

To add linting to CI, add a step to your workflow:

```yaml
- name: Lint
  run: |
    pip install ruff mypy
    ruff check .
    mypy scratchv/ --ignore-missing-imports --follow-imports=silent
```

## See Also

- `docs/CODING_STANDARDS.md` - Full coding guidelines
- `.pre-commit-config.yaml` - Hook configuration
- `scripts/lint_check.sh` - Convenience lint script
- `pyproject.toml` - Project configuration with tool settings
