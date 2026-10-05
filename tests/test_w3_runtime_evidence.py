"""Numerical provenance supplements package versions with actual NumPy details."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from probes import w3_common


def test_math_environment_is_captured_without_console_output(monkeypatch, capsys):
    monkeypatch.setattr(np, "show_config", lambda: print("fixture BLAS build"))
    monkeypatch.setattr(np, "show_runtime", lambda: print("fixture SIMD and loaded BLAS"))
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_CORETYPE", "Haswell")
    result = w3_common.numpy_runtime_evidence()
    assert result["numpy_build"] == "fixture BLAS build"
    assert result["numpy_runtime"] == "fixture SIMD and loaded BLAS"
    assert result["thread_environment"]["OPENBLAS_NUM_THREADS"] == "1"
    assert result["blas_environment"] == {"OPENBLAS_CORETYPE": "Haswell"}
    assert capsys.readouterr().out == ""
    json.dumps(result, allow_nan=False)


def test_unavailable_diagnostics_are_explicit_and_do_not_hide_build_info(monkeypatch):
    monkeypatch.setattr(np, "show_config", lambda: print("known build"))
    monkeypatch.delattr(np, "show_runtime")
    result = w3_common.numpy_runtime_evidence()
    assert result["numpy_build"] == "known build"
    assert result["numpy_runtime"] is None
    assert result["numpy_runtime_error"].startswith("AttributeError:")


def private_numpy(tmp_path, *names):
    package = tmp_path / "numpy"
    package.mkdir()
    libraries = tmp_path / "numpy.libs"
    libraries.mkdir()
    for name in names:
        (libraries / name).touch()
    return SimpleNamespace(__file__=str(package / "__init__.py"))


def library_metadata(prefix="scipy_", suffix="64_"):
    return SimpleNamespace(**{
        f"{prefix}openblas_get_corename{suffix}": lambda: b"Haswell",
        f"{prefix}openblas_get_config{suffix}": lambda: b"OpenBLAS fixture USE64BITINT",
    })


@pytest.mark.parametrize("prefix,suffix", [("scipy_", "64_"), ("", "64_"), ("", "_64"), ("", "")])
def test_only_private_openblas_candidates_are_queried(tmp_path, monkeypatch, prefix, suffix):
    numpy = private_numpy(tmp_path, "libopenblas.so.0", "unrelated.dll", "openblas.txt")
    # A similarly named library outside NumPy's private directory is not inspected.
    (tmp_path / "libopenblas-unrelated.so").touch()
    queried = []
    def loaded(path):
        queried.append(path.name)
        return library_metadata(prefix, suffix)
    monkeypatch.setattr(w3_common, "_already_loaded_library", loaded)
    result = w3_common.numpy_openblas_evidence(numpy)
    assert queried == ["libopenblas.so.0"]
    assert result["status"] == "available"
    assert result["libraries"][0]["core_name"] == "Haswell"
    assert result["libraries"][0]["config"] == "OpenBLAS fixture USE64BITINT"
    json.dumps(result, allow_nan=False)


def test_unknown_numpy_blas_remains_unavailable(tmp_path, monkeypatch):
    numpy = private_numpy(tmp_path, "mkl.dll")
    def forbidden(path):
        pytest.fail(f"Must not inspect a different backend: {path}")
    monkeypatch.setattr(w3_common, "_already_loaded_library", forbidden)
    result = w3_common.numpy_openblas_evidence(numpy)
    assert result["status"] == "unavailable"
    assert result["libraries"] == []
    assert "No private" in result["reason"]


@pytest.mark.parametrize("failure", ["not_loaded", "missing_symbol", "empty_metadata"])
def test_optional_metadata_failure_is_recorded(tmp_path, monkeypatch, failure):
    numpy = private_numpy(tmp_path, "libopenblas.so")
    def loaded(path):
        if failure == "not_loaded":
            raise OSError("not loaded")
        if failure == "missing_symbol":
            return SimpleNamespace()
        return SimpleNamespace(openblas_get_corename=lambda: None)
    monkeypatch.setattr(w3_common, "_already_loaded_library", loaded)
    result = w3_common.numpy_openblas_evidence(numpy)
    assert result["status"] == "unavailable"
    assert result["libraries"][0]["status"] == "unavailable"
    assert result["libraries"][0]["error"]
    json.dumps(result, allow_nan=False)


def test_one_unavailable_library_does_not_hide_available_metadata(tmp_path, monkeypatch):
    numpy = private_numpy(tmp_path, "a-openblas.so", "b-openblas.so")
    def loaded(path):
        if path.name.startswith("a-"):
            raise OSError("not loaded")
        return library_metadata()
    monkeypatch.setattr(w3_common, "_already_loaded_library", loaded)
    result = w3_common.numpy_openblas_evidence(numpy)
    assert result["status"] == "available"
    assert [row["status"] for row in result["libraries"]] == ["unavailable", "available"]


def test_missing_numpy_location_is_optional():
    result = w3_common.numpy_openblas_evidence(SimpleNamespace())
    assert result["status"] == "unavailable"
    assert result["error"].startswith("AttributeError:")


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_posix_lookup_requires_noload_and_never_retries_normal_load(monkeypatch, platform):
    monkeypatch.setattr(w3_common.sys, "platform", platform)
    monkeypatch.setattr(w3_common.os, "RTLD_NOLOAD", 4, raising=False)
    monkeypatch.setattr(w3_common.os, "RTLD_LOCAL", 0, raising=False)
    calls = []
    def cdll(path, **options):
        calls.append((path, options))
        raise OSError("not loaded")
    monkeypatch.setattr(w3_common.ctypes, "CDLL", cdll)
    with pytest.raises(OSError, match="not loaded"):
        w3_common._already_loaded_library("/private/libopenblas.so")
    assert calls == [("/private/libopenblas.so", {"mode": 4})]


def test_missing_noload_does_not_fall_back_to_loading(monkeypatch):
    monkeypatch.setattr(w3_common.sys, "platform", "linux")
    monkeypatch.delattr(w3_common.os, "RTLD_NOLOAD", raising=False)
    monkeypatch.setattr(w3_common.ctypes, "CDLL", lambda *a, **k: pytest.fail("No loading fallback"))
    with pytest.raises(OSError, match="unavailable"):
        w3_common._already_loaded_library("/private/libopenblas.so")


@pytest.mark.parametrize("handle", [0, 123])
def test_windows_only_uses_an_existing_handle(monkeypatch, handle):
    monkeypatch.setattr(w3_common.sys, "platform", "win32")
    get_module = lambda path: handle
    kernel = SimpleNamespace(GetModuleHandleW=get_module)
    monkeypatch.setattr(w3_common.ctypes, "WinDLL", lambda *a, **k: kernel, raising=False)
    calls = []
    monkeypatch.setattr(w3_common.ctypes, "CDLL", lambda *a, **k: calls.append((a, k)))
    if handle:
        w3_common._already_loaded_library("private-openblas.dll")
        assert calls == [(("private-openblas.dll",), {"handle": handle})]
    else:
        with pytest.raises(OSError, match="not already loaded"):
            w3_common._already_loaded_library("private-openblas.dll")
        assert calls == []
