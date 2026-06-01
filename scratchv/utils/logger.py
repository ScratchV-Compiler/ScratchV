"""Compiler logging infrastructure for ScratchV.

Provides structured, color-coded logging with level-based filtering,
file output, and progress indicators. Replaces ad-hoc print() calls
throughout the compiler pipeline.

Usage::

    from scratchv.utils.logger import init_logger, get_logger

    init_logger(level="DEBUG", log_file="build.log")
    log = get_logger("scratchv.parser")
    log.info("Parsing DSL source (%d lines)", len(lines))

Log levels:
    DEBUG   - Detailed tracing information (gray)
    INFO    - Normal operational messages (green)
    WARNING - Non-critical issues (yellow)
    ERROR   - Critical errors (red)
    CRITICAL- Fatal errors (bold red)
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Optional


# ---------------------------------------------------------------------------
# ANSI color codes for log levels
# ---------------------------------------------------------------------------

_COLORS: dict[str, str] = {
    "DEBUG":    "\033[90m",    # gray
    "INFO":     "\033[32m",    # green
    "WARNING":  "\033[33m",    # yellow
    "ERROR":    "\033[31m",    # red
    "CRITICAL": "\033[1;31m",  # bold red
}

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


# ---------------------------------------------------------------------------
# Custom formatter
# ---------------------------------------------------------------------------

class _ColorFormatter(logging.Formatter):
    """Logging formatter with ANSI color support."""

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with optional color and timing info."""
        levelname = record.levelname
        asctime = time.strftime(
            "%H:%M:%S", time.localtime(record.created),
        )
        name = record.name
        message = record.getMessage()

        if self.use_color and levelname in _COLORS:
            color = _COLORS[levelname]
            level_colored = f"{color}{levelname:<8}{_RESET}"
            time_colored = f"{_DIM}{asctime}{_RESET}"
            name_colored = f"{_BOLD}{name}{_RESET}"
        else:
            level_colored = f"{levelname:<8}"
            time_colored = asctime
            name_colored = name

        return f"{time_colored} {level_colored} [{name_colored}] {message}"


class _PlainFormatter(logging.Formatter):
    """Plain-text formatter for file output (no colors)."""

    def format(self, record: logging.LogRecord) -> str:
        asctime = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(record.created),
        )
        return (f"{asctime} {record.levelname:<8} "
                f"[{record.name}] {record.getMessage()}")


# ---------------------------------------------------------------------------
# Module-level logger registry
# ---------------------------------------------------------------------------

_root_logger: Optional[logging.Logger] = None
_initialized: bool = False
_config: dict = {}


def init_logger(
    level: str = "INFO",
    log_file: str | None = None,
    use_color: bool = True,
) -> None:
    """Initialize the compiler logging system.

    Must be called once at the start of the compiler pipeline. Creates
    console and optional file handlers.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to write log output to (plain text, no color).
        use_color: Enable ANSI color output on console.

    Raises:
        ValueError: If level is not a valid log level string.
    """
    global _root_logger, _initialized, _config

    # Validate level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    _config = {
        "level": level,
        "log_file": log_file,
        "use_color": use_color,
    }

    # Configure root logger
    _root_logger = logging.getLogger("scratchv")
    _root_logger.setLevel(numeric_level)

    # Remove any existing handlers
    _root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(_ColorFormatter(use_color=use_color))
    _root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(
            log_file, mode="w", encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # Always write DEBUG to file
        file_handler.setFormatter(_PlainFormatter())
        _root_logger.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get or create a named logger under the scratchv namespace.

    If no name prefix is given, 'scratchv.' is prepended automatically.

    Args:
        name: Logger name (e.g., 'parser', 'optimizer.constant_folding').

    Returns:
        A logging.Logger instance.

    Raises:
        RuntimeError: If init_logger() has not been called.
    """
    if not _initialized:
        # Auto-initialize with defaults
        init_logger()

    if not name.startswith("scratchv"):
        name = f"scratchv.{name}"
    return logging.getLogger(name)


def set_level(level: str) -> None:
    """Change the log level of the root scratchv logger at runtime.

    Args:
        level: New log level string.

    Raises:
        RuntimeError: If init_logger() has not been called.
    """
    if _root_logger is None:
        raise RuntimeError(
            "Logger not initialized. Call init_logger() first."
        )
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    _root_logger.setLevel(numeric_level)
    for handler in _root_logger.handlers:
        if hasattr(handler, "stream") and handler.stream == sys.stderr:
            handler.setLevel(numeric_level)


def shutdown() -> None:
    """Flush and close all logging handlers."""
    if _root_logger is not None:
        for handler in _root_logger.handlers:
            handler.flush()
            handler.close()
        _root_logger.handlers.clear()


# ---------------------------------------------------------------------------
# Progress indicators
# ---------------------------------------------------------------------------

@contextmanager
def log_phase(name: str, description: str = ""):
    """Log the start and elapsed time of a compiler phase.

    Usage::

        with log_phase("parse", "Parsing DSL input"):
            program = parser.parse(source)

    Output::

        12:34:56 INFO     [scratchv.parse] Parsing DSL input...
        12:34:56 INFO     [scratchv.parse] Parsing DSL input... done (0.032s)
    """
    log = get_logger(name)
    start = time.perf_counter()
    msg = description or name
    log.info("%s...", msg)
    try:
        yield
        elapsed = time.perf_counter() - start
        log.info("%s... done (%.3fs)", msg, elapsed)
    except Exception:
        elapsed = time.perf_counter() - start
        log.error("%s... FAILED (%.3fs)", msg, elapsed)
        raise


def log_progress(
        name: str, current: int, total: int, description: str = "",
) -> None:
    """Log a progress indicator for long-running operations.

    Args:
        name: Logger name.
        current: Current item index (0-based or 1-based).
        total: Total number of items.
        description: Operation description.
    """
    log = get_logger(name)
    pct = (current / total * 100) if total > 0 else 0
    log.info("%s [%d/%d] %.1f%%", description, current, total, pct)


def log_step(name: str, step_name: str) -> None:
    """Log a discrete step within a phase.

    Args:
        name: Logger name.
        step_name: Name of the step being performed.
    """
    log = get_logger(name)
    log.debug("  -> %s", step_name)
