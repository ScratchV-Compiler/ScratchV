# Contributing to ScratchV

Thanks for your interest! This is an educational compiler project, and
contributions of all kinds — code, docs, bug reports, teaching materials —
are very welcome.

## Quick Start

```bash
git clone https://github.com/kinsomwang/ScratchV
cd ScratchV
pip install -e .            # install in editable mode
pip install tinyfive        # optional: assembly verification
pytest tests/ -v            # run all tests
```

## Code Style

- **Python version**: 3.8+ compatible (no `|` union syntax in annotations
  unless guarded by `from __future__ import annotations`; no
  `dataclass(slots=True)`).
- **Type hints**: annotate all public functions and methods.
- **Docstrings**: Google or NumPy style is fine — keep them short but useful.
- **No `__pycache__`**: they're gitignored; just don't commit them.

## Pull Request Process

1. **Open an issue** first to discuss the change you'd like to make.
2. Make your changes on a feature branch (`git checkout -b feat/my-thing`).
3. Add or update tests in `tests/`.
4. Run `pytest tests/` — all tests must pass.
5. Run `make check` if available (lint + test).
6. Open a PR with a clear title and description.

## Adding a New IR Opcode

1. Add the opcode to `scratchv/ir/types.py` → `OpCode` enum.
2. (Optional) Add a builder method in `scratchv/ir/builder.py`.
3. Add a selection handler in `scratchv/backend/instruction_select.py`.
4. Add an LLVM codegen handler in `scratchv/backend/llvm_codegen.py`.
5. Add a test case in `tests/`.
6. Run `pytest` to verify.

## Adding a New Optimization Pass

1. Create `scratchv/optimizer/my_pass.py`.
2. Implement a class with a `run(program) → int` method (returns number of
   transformations applied).
3. Register it in `scratchv/main.py` → `run_optimizer()`.
4. Add test cases (positive: should transform; negative: should not).
5. Run `pytest` to verify.

## Documentation

- User-facing docs go in `docs/`.
- Inline code comments are for *why* not *what*.
- The README is the single source of truth for project-wide docs.

## Code of Conduct

Be respectful, assume good faith, and remember that this is a learning project.
Help others level up.
