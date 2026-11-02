"""Reuse the existing W5 CLI fixtures for focused persistence checks."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[2] / "test_w5_inference_cli.py"
_spec = spec_from_file_location("w5_existing_cli_fixtures", _path)
_existing = module_from_spec(_spec)
_spec.loader.exec_module(_existing)

cli = _existing.cli
runtime = _existing.runtime
reference_suite = _existing.reference_suite


@pytest.fixture
def cli_helpers():
    return _existing
