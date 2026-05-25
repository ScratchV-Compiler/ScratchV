# ScratchV developer makefile
.POSIX:

.PHONY: install test clean lint check docs examples

# ── Installation ──────────────────────────────────────────────────────────────

install:
	pip install -e .
	pip install -e ".[all]" 2>/dev/null || pip install -e .

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	python3 -m pytest tests/ -v --tb=short

test-coverage:
	python3 -m pytest tests/ --cov=scratchv --cov=scratchv_dag --cov-report=term

# ── Lint ──────────────────────────────────────────────────────────────────────

lint:
	-python3 -m flake8 scratchv/ scratchv_dag/ tests/ 2>/dev/null || echo "install flake8: pip install flake8"
	-python3 -m mypy scratchv/ scratchv_dag/ --ignore-missing-imports 2>/dev/null || echo "install mypy: pip install mypy"

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache
	rm -rf scratchv.egg-info scratchv_dag.egg-info
	rm -rf dist build
	rm -f output.s output.ll

# ── Checks (runs before PR) ───────────────────────────────────────────────────

check: clean test

# ── Quick examples ────────────────────────────────────────────────────────────

examples:
	@echo "=== DSL examples ==="
	python3 -m scratchv examples/simple_add.dsl -o /tmp/simple_add.s --dump-ir
	python3 -m scratchv examples/relu_test.dsl -o /tmp/relu.s --optimize all
	python3 -m scratchv examples/matmul_test.dsl -o /tmp/matmul.s --optimize all

# ── Build docs preview (if pandoc is available) ────────────────────────────────

docs:
	@echo "Documentation is markdown — no build required."
	@ls docs/*.md
