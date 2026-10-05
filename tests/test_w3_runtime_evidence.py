"""Numerical provenance supplements package versions with actual NumPy details."""
import json

import numpy as np

from probes import w3_common


def test_math_environment_is_captured_without_console_output(monkeypatch, capsys):
    monkeypatch.setattr(np, "show_config", lambda: print("fixture BLAS build"))
    monkeypatch.setattr(np, "show_runtime", lambda: print("fixture SIMD and loaded BLAS"))
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    result = w3_common.numpy_runtime_evidence()
    assert result["numpy_build"] == "fixture BLAS build"
    assert result["numpy_runtime"] == "fixture SIMD and loaded BLAS"
    assert result["thread_environment"]["OPENBLAS_NUM_THREADS"] == "1"
    assert capsys.readouterr().out == ""
    json.dumps(result, allow_nan=False)


def test_unavailable_diagnostics_are_explicit_and_do_not_hide_build_info(monkeypatch):
    monkeypatch.setattr(np, "show_config", lambda: print("known build"))
    monkeypatch.delattr(np, "show_runtime")
    result = w3_common.numpy_runtime_evidence()
    assert result["numpy_build"] == "known build"
    assert result["numpy_runtime"] is None
    assert result["numpy_runtime_error"].startswith("AttributeError:")
