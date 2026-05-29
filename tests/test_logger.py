"""Tests for the compiler logger module."""

import logging
import os
import tempfile

import pytest
from scratchv.utils.logger import (
    init_logger,
    get_logger,
    set_level,
    shutdown,
    log_progress,
    log_phase,
    log_step,
)


class TestInitLogger:
    """Tests for logger initialization."""

    def test_init_default(self):
        init_logger()
        log = get_logger("test_init")
        assert log is not None
        assert isinstance(log, logging.Logger)
        shutdown()

    def test_init_with_level(self):
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            init_logger(level=level)
            log = get_logger("test_level")
            assert log.level <= getattr(logging, level)
            shutdown()

    def test_init_invalid_level(self):
        with pytest.raises(ValueError):
            init_logger(level="INVALID")

    def test_init_with_log_file(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            init_logger(level="DEBUG", log_file=log_path)
            log = get_logger("test_file")
            log.info("test message")
            shutdown()

            # Check file was written
            with open(log_path) as f:
                content = f.read()
            assert "test message" in content
        finally:
            if os.path.exists(log_path):
                os.unlink(log_path)

    def test_init_no_color(self):
        init_logger(use_color=False)
        log = get_logger("test_nocolor")
        log.info("no color message")
        # Should not crash
        shutdown()


class TestGetLogger:
    """Tests for logger factory."""

    def test_auto_prefix(self):
        init_logger()
        log = get_logger("parser")
        assert log.name == "scratchv.parser"
        shutdown()

    def test_already_prefixed(self):
        init_logger()
        log = get_logger("scratchv.codegen")
        assert log.name == "scratchv.codegen"
        shutdown()

    def test_hierarchical_names(self):
        init_logger()
        log = get_logger("optimizer.constant_folding")
        assert log.name == "scratchv.optimizer.constant_folding"
        shutdown()

    def test_returns_logger_instance(self):
        init_logger()
        log = get_logger("test")
        assert isinstance(log, logging.Logger)
        shutdown()

    def test_auto_init_on_first_call(self):
        # If init_logger was never called, get_logger should auto-init
        # Reset by clearing handlers first
        root = logging.getLogger("scratchv")
        root.handlers.clear()
        # Force re-init
        import scratchv.utils.logger as logger_mod
        logger_mod._initialized = False
        log = get_logger("auto")
        assert isinstance(log, logging.Logger)
        shutdown()


class TestSetLevel:
    """Tests for runtime level changes."""

    def test_change_level(self):
        init_logger(level="INFO")
        # Child loggers inherit effective level from parent (NOTSET=0)
        log = get_logger("test_setlevel")
        assert log.getEffectiveLevel() == logging.INFO
        set_level("DEBUG")
        root = logging.getLogger("scratchv")
        assert root.level == logging.DEBUG
        shutdown()

    def test_invalid_level(self):
        init_logger()
        with pytest.raises(ValueError):
            set_level("NOPE")
        shutdown()


class TestLogging:
    """Tests for actual log output."""

    def test_info_logging(self):
        init_logger(level="INFO")
        log = get_logger("test_log_info")
        # Just verify no exception
        log.info("test message %d", 42)
        log.warning("warning message")
        shutdown()

    def test_debug_filtering(self):
        init_logger(level="INFO")
        log = get_logger("test_filter")
        # Debug messages at INFO level should not be emitted
        # (just verify no crash)
        log.debug("this should not appear")
        shutdown()

    def test_error_logging(self):
        init_logger()
        log = get_logger("test_error")
        try:
            raise ValueError("test exception")
        except ValueError:
            log.error("caught exception", exc_info=True)
        shutdown()


class TestProgress:
    """Tests for progress indicators."""

    def test_log_progress(self):
        init_logger(level="INFO")
        log_progress("test_prog", current=5, total=10, description="Testing")
        shutdown()

    def test_log_step(self):
        init_logger(level="DEBUG")
        log_step("test_step", "selecting instructions")
        shutdown()


class TestLogPhase:
    """Tests for the log_phase context manager."""

    def test_phase_success(self):
        init_logger(level="INFO")
        with log_phase("test_phase", "Testing phase"):
            pass  # succeed
        shutdown()

    def test_phase_failure(self):
        init_logger(level="INFO")
        try:
            with log_phase("test_phase_fail", "Failing phase"):
                raise RuntimeError("phase failed")
        except RuntimeError:
            pass  # expected
        shutdown()
