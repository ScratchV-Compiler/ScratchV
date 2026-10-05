"""Keep reviewable source-identity and failed QEMU execution evidence."""

import json
import subprocess
from types import SimpleNamespace

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from probes.w3_attention import run as attention
from probes.w3_attention.cases import attention_case
from probes.w3_full_preflight import run as preflight
from scratchv.runtime.riscv_tensor import RiscVTensorExecutionError, RiscVTensorTimeoutError


@pytest.mark.parametrize("gate", [attention, preflight], ids=["attention", "full_preflight"])
@pytest.mark.parametrize("failure", [
    PermissionError("source file cannot be read"),
    subprocess.TimeoutExpired(["git", "status"], 15),
], ids=["source_read", "git_timeout"])
def test_source_evidence_failure_keeps_failed_reports(tmp_path, monkeypatch, gate, failure):
    def failed_source():
        raise failure

    def forbidden(*args, **kwargs):
        pytest.fail("Execution must not start without source evidence")

    monkeypatch.setattr(gate, "source_evidence", failed_source)
    monkeypatch.setattr(gate, "run_probe" if gate is attention else "verify_files", forbidden)
    out = tmp_path / "evidence"
    arguments = ["--output-dir", str(out)]
    if gate is preflight:
        arguments += ["--model-dir", str(tmp_path / "unavailable-model")]
    assert gate.main(arguments) == 1
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["passed"] is False and report["status"] == "FAIL"
    assert report["stage"] == "source_evidence"
    assert report["error"].startswith(type(failure).__name__ + ":")
    for filename in ("report.md", "report.html"):
        assert "FAIL" in (out / filename).read_text(encoding="utf-8")


@pytest.mark.parametrize("failure_kind", ["timeout", "runtime_error", "numeric_failed"])
def test_failed_qemu_time_command_and_other_optimization_are_preserved(tmp_path, monkeypatch, failure_kind):
    # Host ONNX/ORT and IR paths stay real. Only compiler/QEMU tools are fake,
    # allowing the test to exercise their actual error object contract.
    case = attention_case("failure17", 17, 17)
    monkeypatch.setattr(attention, "build_cases", lambda: [case])
    monkeypatch.setattr(attention, "source_evidence", lambda: {"git": {"head": "test-only"}})
    monkeypatch.setattr(attention, "discover_toolchain", lambda **kwargs: SimpleNamespace(
        cc=("fake-cc",), qemu="fake-qemu"))

    class FakeDriver:
        def __init__(self, config):
            self.tensor_artifact = SimpleNamespace(level=config.optimize_level)

        def compile(self, model, output):
            return SimpleNamespace(success=True, errors=[])

    def build(artifact, folder, tools):
        return SimpleNamespace(level=artifact.level, elf_sha256="test-elf", compile_command=("fake-cc",),
                               compile_seconds=.25, workspace_bytes=256, tool_versions={})

    attempted = []

    def run(executable, feed, folder, *, timeout):
        attempted.append(executable.level)
        command = ("fake-qemu", executable.level, "model.elf")
        if executable.level == "none":
            if failure_kind == "timeout":
                raise RiscVTensorTimeoutError("injected timeout", elapsed_s=1.75,
                                             command=command, timeout_s=timeout)
            if failure_kind == "runtime_error":
                raise RiscVTensorExecutionError("injected guest failure", elapsed_s=1.75, command=command)
            wrong = case.expected.copy()
            wrong.flat[0] += 0.25
            return SimpleNamespace(output=wrong, elapsed_s=1.75, command=command)
        return SimpleNamespace(output=case.expected.copy(), elapsed_s=.5, command=command)

    monkeypatch.setattr(attention, "CompilerDriver", FakeDriver)
    monkeypatch.setattr(attention, "build_riscv_tensor", build)
    monkeypatch.setattr(attention, "run_riscv_tensor", run)
    out = tmp_path / "evidence"
    assert attention.main(["--output-dir", str(out), "--timeout", "1.5"]) == 1
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert attempted == ["none", "all"]
    assert report["passed"] is False and report["status"] == "FAIL"
    failed, successful = report["cases"][0]["executions"]
    assert failed["status"] == failure_kind and failed["passed"] is False
    assert failed["qemu_process_wall_seconds"] == 1.75
    assert failed["command"] == ["fake-qemu", "none", "model.elf"]
    assert failed["timeout_seconds"] == 1.5
    assert failed["ir_vs_ort"]["passed"] and failed["ir_vs_numpy"]["passed"]
    assert successful["status"] == "success" and successful["passed"] is True
    assert report["passed_executions"] == 1
    assert report["qemu_process_wall_seconds"] == {"total": 2.25, "measured_count": 2}
    if failure_kind == "numeric_failed":
        assert (out / "failure17/none/qemu.npy").is_file()
    else:
        assert failed["stage"] == "qemu" and "injected" in failed["error"]
