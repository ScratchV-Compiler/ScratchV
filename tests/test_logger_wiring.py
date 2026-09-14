"""Wiring tests for compiler structured logging (Topic 07).

Covers the CLI -> CompilerConfig contract, stderr-only console handler
(R1/R2), end-to-end logging activation (D6) and default-off stability.
"""

import re
import sys
from pathlib import Path

import pytest

import scratchv.utils.logger as logger_mod
from scratchv.main import args_to_config, build_arg_parser, main
from scratchv.utils.logger import shutdown

DSL = (Path(__file__).resolve().parent.parent
       / "benchmarks" / "cases" / "001_simple_add.dsl")


@pytest.fixture(autouse=True)
def _logger_teardown():
    yield
    shutdown()


def test_args_to_config_logging_fields():
    args = build_arg_parser().parse_args(
        ["input.dsl", "--log-level", "DEBUG", "--log-file", "x.log"])
    config = args_to_config(args)
    assert config.use_logger is True
    assert config.log_level == "DEBUG"
    assert config.log_file == "x.log"
    assert isinstance(config.log_color, bool)


def test_log_file_only_implies_logging():
    args = build_arg_parser().parse_args(["input.dsl", "--log-file", "x.log"])
    config = args_to_config(args)
    assert config.use_logger is True
    assert config.log_level == "INFO"
    assert config.log_file == "x.log"


def test_invalid_log_level_rejected_by_cli():
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["input.dsl", "--log-level", "VERBOSE"])


def test_console_handler_targets_stderr():
    logger_mod.init_logger(level="INFO", use_color=False)
    assert logger_mod._console_handler is not None
    assert logger_mod._console_handler.stream is sys.stderr
    shutdown()


def test_cli_logging_end_to_end(tmp_path, capsys):
    out = tmp_path / "out.s"
    log_file = tmp_path / "build.log"
    rc = main([str(DSL), "-o", str(out), "--optimize", "all",
               "--log-level", "DEBUG", "--log-file", str(log_file)])
    captured = capsys.readouterr()

    assert rc == 0
    assert out.exists() and out.stat().st_size > 0
    assert captured.out == ""
    assert "[scratchv.compiler.parse]" in captured.err
    assert "done (" in captured.err
    assert "[scratchv.compiler.codegen]" in captured.err
    assert "[scratchv.compiler.passes]" in captured.err
    assert "constant-folding" in captured.err

    text = log_file.read_text()
    assert "DEBUG" in text
    assert "pass constant-folding" in text
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text)


def test_no_logging_by_default_keeps_outputs_stable(tmp_path, capsys):
    out = tmp_path / "out.s"
    rc = main([str(DSL), "-o", str(out)])
    captured = capsys.readouterr()

    assert rc == 0
    assert out.exists()
    assert captured.out == ""
    assert "scratchv.compiler" not in captured.err
    assert captured.err.strip() == f"OK RISCV output written to {out}"
