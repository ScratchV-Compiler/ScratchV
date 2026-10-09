"""Native C ownership, failure cleanup and explicit non-RV64 diagnostic scope."""

import ctypes
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from probes.w4_qwen3_full import host_diagnostic as host
from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value


def artifact(*, strided=False, policy="sequential"):
    x, w = Value("x", shape=(2, 3)), Value("w", shape=(3, 2))
    builder = IRBuilder()
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.program.global_values.append(w)
    builder.ret(builder.matmul(x, w))
    weight = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2] if strided else np.arange(6, dtype=np.float32).reshape(3, 2)
    weight.setflags(write=False)
    return TensorCCodegen(builder.program, {"w": weight}, constant_storage="external",
                         kernel_calls=True, matmul_policy=policy).generate(), weight


def test_readonly_contiguous_weights_are_borrowed_and_all_native_buffers_stay_alive():
    model, weight = artifact()
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    buffers = host.prepare_buffers(model, {"x": x})
    assert buffers.weights[0] is weight
    assert buffers.inputs[0] is x
    assert buffers.copied_weight_bytes == 0
    assert buffers.weight_pointers[0] == weight.ctypes.data
    assert buffers.workspace.ctypes.data % 64 == buffers.output.ctypes.data % 64 == 0
    assert np.shares_memory(buffers.output, buffers.output_owner)
    assert np.isnan(buffers.output).all()
    assert buffers.arguments(model.workspace_bytes)[3] == model.workspace_bytes


def test_only_noncontiguous_weight_is_materialized_at_native_boundary():
    model, weight = artifact(strided=True)
    buffers = host.prepare_buffers(model, {"x": np.ones((2, 3), np.float32)})
    assert not weight.flags.c_contiguous
    assert buffers.weights[0] is not weight
    assert buffers.weights[0].flags.c_contiguous
    assert buffers.copied_weight_bytes == weight.nbytes
    np.testing.assert_array_equal(buffers.weights[0], weight)
    assert not weight.flags.writeable


@pytest.mark.parametrize("feed", [{}, {"x": np.ones((3, 2), np.float32)},
                                  {"x": np.ones((2, 3), np.float64)},
                                  {"x": np.ones((2, 3), np.float32), "extra": np.ones(1)}])
def test_native_inputs_reject_implicit_conversion_or_binding_errors(feed):
    model, _ = artifact()
    with pytest.raises(ValueError):
        host.prepare_buffers(model, feed)


@pytest.mark.parametrize("policy", ["sequential", "blocked_fma"])
def test_actual_shared_library_executes_kernel_calls_and_unloads(tmp_path, policy):
    try:
        compiler = host.compiler_command()
    except FileNotFoundError:
        pytest.skip("Host clang/gcc or explicit SCRATCHV_ZIG is required")
    model, weight = artifact(policy=policy)
    library, evidence = host.build_library(model, tmp_path / "build", compiler)
    x = np.arange(6, dtype=np.float32).reshape(2, 3) / 4
    buffers = host.prepare_buffers(model, {"x": x})
    result = host.invoke_library(model, buffers, library)
    np.testing.assert_array_equal(result, x @ weight)
    assert evidence["matmul_policy"] == policy
    assert evidence["implicit_fma_contraction"] is False
    assert "-ffp-contract=off" in evidence["command"]
    # This rename is refused by Windows if the DLL is still loaded.
    library.rename(library.with_name("unloaded" + library.suffix))


def test_native_failure_still_releases_dll_handle(monkeypatch):
    model, _ = artifact()
    buffers = host.prepare_buffers(model, {"x": np.ones((2, 3), np.float32)})
    class Function:
        def __call__(self, *_):
            return 7
    library = SimpleNamespace(_handle=12345, scratchv_run_external=Function())
    monkeypatch.setattr(ctypes, "CDLL", lambda _: library)
    released = []
    import _ctypes
    monkeypatch.setattr(_ctypes, "FreeLibrary" if os.name == "nt" else "dlclose", released.append)
    with pytest.raises(RuntimeError, match="failure status 7"):
        host.invoke_library(model, buffers, "unused")
    assert released == [12345]
    assert library._handle == 0


def test_compiler_failure_saves_logs_and_cannot_claim_a_library(tmp_path, monkeypatch):
    model, _ = artifact()
    monkeypatch.setattr(host, "_run_process", lambda *_, **__: subprocess.CompletedProcess([], 1, b"", b"injected compiler failure"))
    with pytest.raises(RuntimeError, match="compiler failure"):
        host.build_library(model, tmp_path / "build", ["fake-cc"])
    logs = list((tmp_path / "build").glob("compiler*"))
    assert logs and any(b"injected compiler failure" in path.read_bytes() for path in logs)


def test_parent_timeout_has_explicit_host_only_failure(tmp_path, monkeypatch):
    def expired(command, **_):
        raise subprocess.TimeoutExpired(command, 0.1, output=b"host began", stderr=b"deadline")
    monkeypatch.setattr(host, "_run_process", expired)
    out = tmp_path / "timeout"
    assert host.main(["--model-dir", str(tmp_path / "model"), "--output-dir", str(out), "--timeout", "0.1"]) == 1
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "HOST_FAIL" and not report["passed"]
    assert report["full_qemu_forward_executed"] is False
    assert not any(report["gates"].values())
    assert "TimeoutExpired" in report["error"]


def test_worker_checks_fixed_assets_before_ort_or_compilation(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "source_evidence", lambda: {})
    def reject(_):
        raise ValueError("fixed model identity mismatch")
    monkeypatch.setattr(host.assets, "verify_files", reject)
    monkeypatch.setattr(host, "ort_reference", lambda *_: pytest.fail("ORT must not run after failed assets"))
    out = tmp_path / "worker"
    args = SimpleNamespace(output_dir=out, model_dir=tmp_path / "model", case="full_seed_0",
                           matmul_policy="sequential", cc=None, compile_timeout=10)
    assert host.worker(args) == 1
    report = json.loads((out / "report.json").read_text())
    assert report["stage"] == "asset_hashes"
    assert report["status"] == "HOST_FAIL"
    assert not report["full_qemu_forward_executed"] and not any(report["gates"].values())


@pytest.mark.parametrize("damage", [None, "qemu", "rv64_gate", "missing_gate", "no_forward", "error", "nonzero",
                                  "team_accepted", "schema_bool", "kind", "negative", "nan", "duplicate"])
def test_parent_never_promotes_worker_claim_to_rv64_or_false_host_pass(tmp_path, monkeypatch, damage):
    out = tmp_path / "attempt"
    def completed(command, **_):
        worker_dir = out / "worker"
        worker_dir.mkdir()
        report = host.new_report("full_seed_0", "sequential")
        report.update(status="HOST_PASS", passed=True, host_diagnostic_passed=True,
                      full_host_forward_executed=True,
                      numeric={"passed": True, "atol": 1e-3, "rtol": 0.0,
                               "elements_compared": 256 * 151936, "max_abs": 0.0001})
        if damage == "qemu":
            report["full_qemu_forward_executed"] = True
        elif damage == "rv64_gate":
            report["gates"]["build:riscv-full"] = True
        elif damage == "missing_gate":
            report["gates"] = {}
        elif damage == "no_forward":
            report["full_host_forward_executed"] = False
        elif damage == "error":
            report["numeric"]["max_abs"] = 1e-3
        elif damage == "team_accepted":
            report["w4_team_accepted"] = True
        elif damage == "schema_bool":
            report["schema_version"] = True
        elif damage == "kind":
            report["diagnostic_kind"] = "qemu_full_graph"
        elif damage == "negative":
            report["numeric"]["max_abs"] = -1
        elif damage == "nan":
            report["numeric"]["max_abs"] = float("nan")
        payload = json.dumps(report)
        if damage == "duplicate":
            payload = payload[:-1] + ', "host_only": true}'
        (worker_dir / "report.json").write_text(payload, encoding="utf-8")
        return subprocess.CompletedProcess(command, int(damage == "nonzero"), b"", b"")
    monkeypatch.setattr(host, "_run_process", completed)
    exit_code = host.main(["--model-dir", str(tmp_path / "model"), "--output-dir", str(out)])
    report = json.loads((out / "report.json").read_text())
    assert (exit_code == 0) is (damage is None)
    assert report["full_qemu_forward_executed"] is False
    assert report["w4_team_accepted"] is False
    assert not any(report["gates"].values())


def test_report_failure_removes_previous_host_pass(tmp_path, monkeypatch):
    report = host.new_report("full_seed_0", "sequential")
    report.update(status="HOST_PASS", passed=True, host_diagnostic_passed=True)
    host.publish(tmp_path, report)
    report["bad"] = float("nan")
    with pytest.raises(ValueError):
        host.publish(tmp_path, report)
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["status"] == "HOST_FAIL" and saved["passed"] is False
    assert saved["w4_team_accepted"] is False
    assert not any(saved["gates"].values())
    assert not (tmp_path / "report.md").exists()
