"""Mandatory QEMU gate failure/report handling; target execution is a separate job."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from probes.w2_backend_ops import run as probe
from scratchv.runtime.riscv_tensor import RiscVTensorTimeoutError


@pytest.fixture
def fake_target(monkeypatch):
    case = probe.build_cases()[0]
    monkeypatch.setattr(probe, "build_cases", lambda: [case])
    monkeypatch.setattr(probe, "FAMILIES", (case.family,))
    monkeypatch.setattr(probe, "discover_toolchain", lambda **kwargs: SimpleNamespace(cc=("fixture-cc",), qemu="fixture-qemu"))
    monkeypatch.setattr(probe, "build_riscv_tensor", lambda *args: SimpleNamespace(
        elf_sha256="fixture", compile_command=("fixture-cc",), compile_seconds=0.1,
        workspace_bytes=16, tool_versions={"fixture": "unit-test-only"}))
    monkeypatch.setattr(probe, "run_riscv_tensor", lambda *args, **kwargs: SimpleNamespace(
        output=case.expected.copy(), elapsed_s=0.25, command=("fixture-qemu",)))
    return case


def result(tmp_path):
    directory = tmp_path / "report"
    code = probe.main(["--output-dir", str(directory)])
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    assert (directory / "report.md").is_file() and (directory / "report.html").is_file()
    return code, report


def test_missing_tools_fail_with_all_report_formats(tmp_path, monkeypatch):
    def unavailable(**kwargs):
        raise FileNotFoundError("QEMU unavailable")
    monkeypatch.setattr(probe, "discover_toolchain", unavailable)
    code, report = result(tmp_path)
    assert code == 1 and report["passed"] is False and report["stage"] == "toolchain"
    assert "QEMU unavailable" in report["error"]
    assert report["planned_executions"] == 32


def test_success_requires_both_optimization_levels(tmp_path, monkeypatch, fake_target):
    code, report = result(tmp_path)
    assert code == 0 and report["passed"] is True
    assert report["completed_executions"] == report["planned_executions"] == 2
    assert report["passed_executions"] == 2
    assert report["qemu_process_wall_seconds"] == {"measured_count": 2, "total": 0.5}
    assert report["atol"] == 1e-5 and report["rtol"] == 0


@pytest.mark.parametrize("fault", ["numeric", "nan", "shape", "dtype", "second_level"])
def test_incorrect_qemu_output_is_not_a_pass(tmp_path, monkeypatch, fake_target, fault):
    calls = []
    def execute(*args, **kwargs):
        actual = fake_target.expected.copy()
        calls.append(1)
        if fault == "numeric" or (fault == "second_level" and len(calls) == 2):
            actual.flat[0] += np.float32(0.01)
        elif fault == "nan":
            actual.flat[0] = np.nan
        elif fault == "shape":
            actual = actual.flatten()
        elif fault == "dtype":
            actual = actual.astype(np.float64)
        return SimpleNamespace(output=actual, elapsed_s=0.25, command=("fixture-qemu",))
    monkeypatch.setattr(probe, "run_riscv_tensor", execute)
    code, report = result(tmp_path)
    assert code == 1 and report["passed"] is False
    assert report["completed_executions"] == 2
    assert report["passed_executions"] == (1 if fault == "second_level" else 0)


def test_qemu_timeout_preserves_measured_time_and_failures(tmp_path, monkeypatch, fake_target):
    def timeout(*args, **kwargs):
        raise RiscVTensorTimeoutError("QEMU test timeout", elapsed_s=0.125,
                                      command=["fixture-qemu"], timeout_s=0.1)
    monkeypatch.setattr(probe, "run_riscv_tensor", timeout)
    code, report = result(tmp_path)
    assert code == 1
    assert report["completed_executions"] == 0
    assert report["qemu_process_wall_seconds"] == {"measured_count": 2, "total": 0.25}
    assert all(item["status"] == "timeout" for item in report["cases"][0]["executions"])


def test_formula_disagreement_cannot_enter_qemu(tmp_path, monkeypatch, fake_target):
    fake_target.expected.flat[0] += np.float32(1)
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid reference must stop before QEMU")
    monkeypatch.setattr(probe, "run_riscv_tensor", unexpected)
    code, report = result(tmp_path)
    assert code == 1 and report["cases"][0]["executions"] == []
    assert "NumPy formula" in report["cases"][0]["error"]


def test_compiler_failure_cannot_become_qemu_success(tmp_path, monkeypatch, fake_target):
    monkeypatch.setattr(probe.CompilerDriver, "compile", lambda *args: SimpleNamespace(
        success=False, errors=["fixture unsupported operator"]))
    def unexpected(*args, **kwargs):
        pytest.fail("Compiler failure must stop before QEMU")
    monkeypatch.setattr(probe, "run_riscv_tensor", unexpected)
    code, report = result(tmp_path)
    assert code == 1 and report["completed_executions"] == 0
    assert all(item["stage"] == "compile" and "unsupported operator" in item["error"]
               for item in report["cases"][0]["executions"])


@pytest.mark.parametrize("fault", ["empty", "missing_family", "duplicate_name"])
def test_invalid_coverage_fails_before_tools(tmp_path, monkeypatch, fault):
    cases = probe.build_cases()
    selected = [] if fault == "empty" else (cases[:-2] if fault == "missing_family" else cases + [cases[0]])
    monkeypatch.setattr(probe, "build_cases", lambda: selected)
    def unexpected(**kwargs):
        pytest.fail("Invalid coverage must stop before toolchain discovery")
    monkeypatch.setattr(probe, "discover_toolchain", unexpected)
    code, report = result(tmp_path)
    assert code == 1 and "cover every required operator family" in report["error"]


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_invalid_timeout_is_rejected(tmp_path, timeout):
    with pytest.raises(SystemExit, match="2"):
        probe.main(["--output-dir", str(tmp_path / "report"), "--timeout", timeout])


def test_existing_report_is_preserved(tmp_path):
    sentinel = tmp_path / "existing.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        probe.main(["--output-dir", str(tmp_path)])
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("filename", ["report.json", "report.md", "report.html"])
def test_report_failure_invalidates_cli_success(tmp_path, monkeypatch, capsys, filename):
    def successful_probe(out, report, **kwargs):
        report.update(passed=True, stage="complete", cases=[])

    original = Path.write_text

    def blocked(path, data, *args, **kwargs):
        if path.name.removesuffix(".tmp") == filename:
            raise PermissionError("blocked evidence")
        return original(path, data, *args, **kwargs)

    monkeypatch.setattr(probe, "run_probe", successful_probe)
    monkeypatch.setattr(Path, "write_text", blocked)
    directory = tmp_path / "report"
    assert probe.main(["--output-dir", str(directory)]) == 1
    for name in ("report.json", "report.md", "report.html"):
        if name == filename:
            continue
        content = (directory / name).read_text(encoding="utf-8")
        if name == "report.json":
            saved = json.loads(content)
            assert saved["passed"] is False and saved["stage"] == "report-write"
            assert "blocked evidence" in saved["report_write_errors"][0]
        else:
            assert "Result: FAIL" in content
    assert filename in capsys.readouterr().err


def test_report_failure_does_not_replace_primary_error(tmp_path, monkeypatch):
    def failed_probe(out, report, **kwargs):
        report.update(stage="toolchain", cases=[])
        raise RuntimeError("original missing compiler")

    original = Path.write_text

    def blocked(path, data, *args, **kwargs):
        if path.name == "report.md":
            raise OSError("disk full")
        return original(path, data, *args, **kwargs)

    monkeypatch.setattr(probe, "run_probe", failed_probe)
    monkeypatch.setattr(Path, "write_text", blocked)
    directory = tmp_path / "report"
    assert probe.main(["--output-dir", str(directory)]) == 1
    saved = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    assert saved["error"] == "RuntimeError: original missing compiler"
    assert saved["stage"] == "toolchain"
    assert "disk full" in saved["report_write_errors"][0]


def test_json_write_error_after_bytes_written_cannot_publish_pass(tmp_path, monkeypatch):
    original = Path.write_text

    def written_then_failed(path, data, *args, **kwargs):
        result = original(path, data, *args, **kwargs)
        if path.name.removesuffix(".tmp") == "report.json":
            raise OSError("late flush failure")
        return result

    monkeypatch.setattr(Path, "write_text", written_then_failed)
    report = {"passed": True, "stage": "complete", "cases": [], "seconds": 0.1}
    probe.write_reports(tmp_path, report)
    assert report["passed"] is False
    assert not (tmp_path / "report.json").exists()
    assert "Result: FAIL" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "late flush failure" in report["report_write_errors"][0]


def test_json_publish_failure_is_not_success(tmp_path, monkeypatch):
    original = Path.replace

    def fail_publish(path, target):
        if Path(target).name == "report.json":
            raise PermissionError("JSON destination locked")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", fail_publish)
    report = {"passed": True, "stage": "complete", "cases": [], "seconds": 0.1}
    probe.write_reports(tmp_path, report)
    assert report["passed"] is False
    assert report["stage"] == "report-write"
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "report.json.tmp").exists()
    assert "Result: FAIL" in (tmp_path / "report.md").read_text(encoding="utf-8")
