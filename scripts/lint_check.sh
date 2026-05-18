#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ScratchV lint and format check script
#
# Runs: black (format), isort (import sort), ruff (lint), mypy (type check)
#
# Usage:
#   ./scripts/lint_check.sh           # Check only (no changes)
#   ./scripts/lint_check.sh --fix     # Auto-fix formatting
#   ./scripts/lint_check.sh --all     # Run on all files including tests
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

MODE="check"
TARGET="scratchv"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fix)
            MODE="fix"
            shift
            ;;
        --all)
            TARGET="."
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--fix] [--all]"
            exit 1
            ;;
    esac
done

echo "==> ScratchV Lint Check <=="
echo "  Mode:   $MODE"
echo "  Target: $TARGET"
echo ""

# -----------------------------------------------------------------------
# 1. Black (format)
# -----------------------------------------------------------------------
echo "[1/4] Running black..."
if [ "$MODE" = "fix" ]; then
    black "$TARGET" || true
    echo "  OK (formatted)"
else
    black --check "$TARGET" 2>&1 || {
        echo "  FAILED: black format check"
        echo "  Run '$0 --fix' to auto-format"
    }
fi

# -----------------------------------------------------------------------
# 2. isort (import sort)
# -----------------------------------------------------------------------
echo "[2/4] Running isort..."
if [ "$MODE" = "fix" ]; then
    isort --profile=black --line-length=88 "$TARGET" || true
    echo "  OK (sorted)"
else
    isort --check-only --profile=black --line-length=88 "$TARGET" 2>&1 || {
        echo "  FAILED: isort import order check"
        echo "  Run '$0 --fix' to auto-fix"
    }
fi

# -----------------------------------------------------------------------
# 3. ruff (lint)
# -----------------------------------------------------------------------
echo "[3/4] Running ruff..."
if [ "$MODE" = "fix" ]; then
    ruff check --fix "$TARGET" || true
    echo "  OK (fixed)"
else
    ruff check "$TARGET" 2>&1 || {
        echo "  FAILED: ruff lint check"
        echo "  Run '$0 --fix' to auto-fix"
    }
fi

# -----------------------------------------------------------------------
# 4. mypy (type check)
# -----------------------------------------------------------------------
echo "[4/4] Running mypy..."
if [ "$TARGET" = "." ]; then
    mypy scratchv/ --ignore-missing-imports --follow-imports=silent 2>&1 || {
        echo "  FAILED: mypy type check"
    }
else
    mypy "$TARGET" --ignore-missing-imports --follow-imports=silent 2>&1 || {
        echo "  FAILED: mypy type check"
    }
fi

echo ""
echo "==> Done <=="
