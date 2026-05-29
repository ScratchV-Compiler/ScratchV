# ScratchV Coding Standards

## Code Style

### Formatter: Black

All Python code is formatted with [Black](https://github.com/psf/black) using default settings:

- Line length: 88 characters
- Target Python version: Python 3.8+
- String normalization: enabled

```bash
# Format a file
black path/to/file.py

# Check format without changing files
black --check path/to/file.py

# Format entire project
black .
```

### Import Sorting: isort

Imports are sorted with [isort](https://github.com/PyCQA/isort) using the following profile:

- Profile: black (compatible with Black formatter)
- `from __future__ import annotations` first
- Standard library imports
- Third-party imports
- First-party (`scratchv.*`) imports

```bash
# Sort imports
isort path/to/file.py

# Check imports
isort --check-only path/to/file.py
```

### Linting: Ruff

[Ruff](https://github.com/astral-sh/ruff) is used for fast linting, replacing flake8:

- All pycodestyle (E, W) rules
- All Pyflakes (F) rules
- isort compatibility (I001)
- Unused variables and imports

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Type Checking: mypy

[mypy](http://mypy-lang.org/) is configured for strict type checking:

- Python 3.8+ target
- Strict optional checked
- Disallow untyped defs
- Warn on return Any
- Follow imports

```bash
# Run mypy
mypy scratchv/

# Run on specific module
mypy scratchv/frontend/dsl_parser.py
```

## Project Conventions

### File Organization

```
scratchv/
    __init__.py          # Version info
    frontend/            # DSL and ONNX parsers
    ir/                  # IR types, builder, printer
    optimizer/           # Optimization passes
    analysis/            # CFG analysis, IR verification
    backend/             # Code generation (RISC-V, LLVM)
    verification/        # Runtime verification
    simulator/           # RISC-V and TinyFive simulators
    codegen/             # Code generation interfaces
    utils/               # Logging and utilities
```

### Imports

Always use absolute imports with the `scratchv.*` path:

```python
# Correct
from scratchv.ir.types import Program, Function, OpCode
from scratchv.frontend.dsl_parser import DSLParser

# Incorrect (relative imports)
from .dsl_parser import DSLParser
from ..ir.types import Program
```

### Module Docstrings

Every module should have a docstring describing its purpose:

```python
"""Brief description of the module.

Longer description of the module's purpose, key classes, and usage examples.
"""
```

### Type Hints

Use type hints for all public functions and methods:

```python
def parse(self, text: str) -> Program:
    """Parse DSL text into IR Program.

    Args:
        text: The DSL source code as a string.

    Returns:
        A Program object containing the generated IR.
    """
```

### Error Handling

- Use custom exception classes for domain-specific errors
- Provide clear, actionable error messages
- Include location information (line, column) where applicable

### Naming Conventions

| Element          | Convention                | Example              |
|------------------|---------------------------|----------------------|
| Modules          | snake_case                | `dsl_parser.py`      |
| Classes          | PascalCase                | `DSLParser`          |
| Functions/Methods| snake_case                | `parse_if_block()`   |
| Variables        | snake_case                | `label_counter`      |
| Constants        | UPPER_SNAKE               | `MAX_ERRORS`         |
| Private members  | _underscore prefix        | `_vars`, `_resolve()`|

### Testing

- Tests go in the `tests/` directory
- Use pytest with class-based test organization
- Test file names: `test_<module>.py`
- Test method names: `test_<feature>()`

```python
from scratchv.frontend.dsl_parser import DSLParser

class TestDSLParser:
    def test_parse_simple_add(self):
        dsl = "c = add(a, b)\nreturn c\n"
        parser = DSLParser()
        program = parser.parse(dsl)
        assert len(program.functions[0].blocks[0].instructions) == 2
```

## Pre-Commit Hooks

Install hooks before your first commit:

```bash
pip install pre-commit
pre-commit install
```

Hooks run automatically on `git commit`. To run manually:

```bash
pre-commit run --all-files
```

## Quick Start

```bash
# Install dev dependencies
pip install black isort ruff mypy pre-commit

# Install pre-commit hooks
pre-commit install

# Format and lint
./scripts/lint_check.sh

# Or manually
black .
isort .
ruff check .
mypy scratchv/
```

## CI Integration

The CI pipeline runs:

1. `ruff check .` - Lint
2. `mypy scratchv/` - Type check
3. `black --check .` - Format check
4. `isort --check-only .` - Import order check
5. `pytest tests/` - Tests
6. `python benchmarks/bench_runner.py` - Benchmarks
