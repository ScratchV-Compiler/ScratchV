"""Real QEMU execution of the external tensor ABI (explicit tool opt-in)."""
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value
from scratchv.runtime.riscv_external import build_external, run_external
from scratchv.runtime.riscv_tensor import discover_toolchain
from scratchv.runtime.weight_bundle import write_weight_bundle
from scratchv.runtime import riscv_external as runtime


def artifact():
    builder = IRBuilder()
    x, w = Value("x", shape=(4, 4)), Value("w", shape=(4, 4))
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.program.global_values.append(w)
    builder.ret(builder.matmul(x, w))
    weights = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
    return TensorCCodegen(builder.program, initializers={"w": weights},
                         constant_storage="external").generate(), weights


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="requires explicit SCRATCHV_QEMU/CC")
def test_external_rv64_matmul_and_reuse(tmp_path):
    model, weights = artifact()
    manifest = write_weight_bundle(model.external_weights, model.external_initializers, tmp_path / "weights")
    exe = build_external(model, tmp_path / "weights", tmp_path / "build", discover_toolchain())
    assert exe.evidence["bytes"] < 100_000_000
    for number in range(2):
        x = np.arange(16, dtype=np.float32).reshape(4, 4) / (number + 1)
        output, report = run_external(exe, {"x": x}, tmp_path / f"run{number}", timeout=30)
        np.testing.assert_array_equal(output, x @ weights)
        assert report["passed"] and report["guest_completed"] and report["exit_code"] == 0
        assert report["weight_sha256"] == manifest["sha256"]
        assert report["elapsed_s"] >= report["output_dump_s"] > 0
        assert report["elapsed_s"] >= report["qemu_wall_s"] > 0
        assert report["qemu_rss_observed_samples"] > 0
        assert report["sampled_peak_qemu_rss_bytes"] > 0
        assert json.loads((tmp_path / f"run{number}/run.json").read_text())["passed"]
    with pytest.raises(FileExistsError):
        run_external(exe, {"x": x}, tmp_path / "run0", timeout=30)
    (tmp_path / "weights/weights.bin").write_bytes(b"broken")
    with pytest.raises(ValueError, match="length"):
        run_external(exe, {"x": x}, tmp_path / "corrupt", timeout=30)
    assert not (tmp_path / "corrupt").exists()


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="requires explicit SCRATCHV_QEMU/CC")
@pytest.mark.parametrize("empty", [False, True])
def test_external_rv64_without_weights(tmp_path, empty):
    builder = IRBuilder()
    x = Value("x", shape=(0,) if empty else (3,))
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.ret(builder.neg(x))
    model = TensorCCodegen(builder.program, constant_storage="external").generate()
    manifest = write_weight_bundle(model.external_weights, model.external_initializers, tmp_path / "weights")
    assert manifest["total_bytes"] == 0
    exe = build_external(model, tmp_path / "weights", tmp_path / "build", discover_toolchain())
    x_data = np.array([] if empty else [1.25, -2.5, 0.0], np.float32)
    output, report = run_external(exe, {"x": x_data}, tmp_path / "run", timeout=30)
    np.testing.assert_array_equal(output, -x_data)
    assert report["passed"] and report["guest_completed"]
    assert not any("loader,file=" in token and "weights.bin" in token for token in report["command"])


def _stub_executable(tmp_path):
    """A valid ELF/container for process-lifecycle tests; no numerical execution."""
    from scratchv.runtime.weight_bundle import plan_guest_memory
    from scratchv.runtime.riscv_tensor import RiscVToolchain
    builder = IRBuilder()
    x = Value("x", shape=(1,))
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.ret(x)
    model = TensorCCodegen(builder.program, constant_storage="external").generate()
    bundle = tmp_path / "weights"
    manifest = write_weight_bundle(model.external_weights, model.external_initializers, bundle)
    layout = plan_guest_memory(input_bytes=4, output_bytes=4)
    elf = tmp_path / "model.elf"
    ident = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    header = struct.pack("<16sHHIQQQIHHHHHH", ident, 2, 243, 1, layout["code_base"],
                         64, 0, 4, 64, 56, 1, 0, 0, 0)
    segment = struct.pack("<IIQQQQQQ", 1, 5, 120, layout["code_base"],
                          layout["code_base"], 4, 4, 1)
    elf.write_bytes(header + segment + b"\x13\x00\x00\x00")
    evidence = runtime.validate_external_elf(elf, layout)
    return runtime.ExternalExecutable(elf, bundle, manifest, layout, model.inputs, model.output,
                                       model.external_weights, RiscVToolchain(("unused",), "stub-qemu"), evidence)


class _Process:
    returncode = None
    pid = 1234

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -9


@pytest.mark.parametrize("outcome", ["success", "nonzero", "timeout", "nonfinite", "cleanup"])
def test_quit_without_reply_still_requires_exit_output_and_cleanup(tmp_path, monkeypatch, outcome):
    """A missing quit acknowledgement never replaces the runner's gates."""
    executable = _stub_executable(tmp_path)
    out = tmp_path / "run"
    events = []
    process = _Process()
    first_wait = True

    def wait(timeout):
        nonlocal first_wait
        events.append("wait")
        if first_wait and outcome == "timeout":
            first_wait = False
            raise subprocess.TimeoutExpired("stub-qemu", timeout)
        if process.returncode is None:
            process.returncode = 7 if outcome == "nonzero" else 0
        return process.returncode

    process.wait = wait

    def launch(*args, cwd, **kwargs):
        (Path(cwd) / "uart.bin").write_bytes(runtime.FRAME.pack(runtime.MAGIC, 0, 4))
        return process

    class QMP:
        def __init__(self, *args):
            pass

        def command(self, command, *args):
            events.append(command)
            if command == "human-monitor-command":
                value = np.nan if outcome == "nonfinite" else 1.0
                (out / "output.bin").write_bytes(np.array([value], dtype="<f4").tobytes())
                return ""
            return None if command == "quit" else {}

        def close(self):
            events.append("close")

    def cleanup(*args):
        events.append("transport-cleanup")
        if outcome == "cleanup":
            raise OSError("injected transport cleanup failure")

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    monkeypatch.setattr(runtime, "QMP", QMP)
    monkeypatch.setattr(runtime, "_windows_job", lambda process: None)
    monkeypatch.setattr(runtime, "_stop_process_tree", lambda process, job: process.kill())
    monkeypatch.setattr(runtime, "_QemuMemoryMonitor", lambda process: SimpleNamespace(
        start=lambda: None, close=lambda: None, report=lambda: {}))
    monkeypatch.setattr(runtime, "remove_transport", cleanup)
    if outcome == "success":
        output, report = run_external(executable, {"x": np.ones(1, np.float32)}, out)
        np.testing.assert_array_equal(output, [1.0])
        assert report["passed"] and report["exit_code"] == 0
        assert "transport-cleanup" in events
    else:
        expected = {"nonzero": RuntimeError, "timeout": subprocess.TimeoutExpired,
                    "nonfinite": ValueError, "cleanup": RuntimeError}[outcome]
        with pytest.raises(expected):
            run_external(executable, {"x": np.ones(1, np.float32)}, out)
    report = json.loads((out / "run.json").read_text())
    assert report["guest_completed"] and report["qmp_quit_reply_received"] is False
    assert report["passed"] is (outcome == "success")
    assert events.index("human-monitor-command") < events.index("quit") < events.index("wait")
    assert "close" in events


def test_qmp_close_failure_keeps_primary_and_attempts_all_cleanup(tmp_path, monkeypatch):
    executable = _stub_executable(tmp_path)
    events = []
    process = _Process()

    def launch(*args, cwd, **kwargs):
        (Path(cwd) / "uart.bin").write_bytes(runtime.FRAME.pack(runtime.MAGIC, 0, 4))
        return process

    class BrokenQMP:
        def __init__(self, *args):
            pass

        def command(self, *args):
            raise ValueError("primary protocol failure")

        def close(self):
            events.append("qmp-close")
            raise OSError("secondary stream failure")

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    monkeypatch.setattr(runtime, "QMP", BrokenQMP)
    monkeypatch.setattr(runtime, "_windows_job", lambda p: (
        SimpleNamespace(CloseHandle=lambda handle: events.append("job-close") or 1), 99))
    monkeypatch.setattr(runtime, "_stop_process_tree", lambda p, job: events.append("tree-kill"))
    with pytest.raises(ValueError, match="primary protocol failure"):
        run_external(executable, {"x": np.ones(1, np.float32)}, tmp_path / "run")
    assert events == ["qmp-close", "tree-kill", "job-close"]
    report = json.loads((tmp_path / "run/run.json").read_text())
    assert report["status"] == "FAIL" and report["guest_completed"]
    assert report["error"] == "ValueError: primary protocol failure"
    assert "secondary stream failure" in report["cleanup_errors"][0]


def test_windows_job_creation_failure_still_reaps_child_and_reports(tmp_path, monkeypatch):
    executable = _stub_executable(tmp_path)
    process = _Process()
    events = []
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **k: process)

    def no_job(process):
        raise OSError("job creation failed")

    monkeypatch.setattr(runtime, "_windows_job", no_job)
    monkeypatch.setattr(runtime, "_stop_process_tree", lambda p, job: events.append("tree-kill"))
    with pytest.raises(OSError, match="job creation failed"):
        run_external(executable, {"x": np.ones(1, np.float32)}, tmp_path / "run")
    assert events == ["tree-kill"] and process.returncode == 0
    report = json.loads((tmp_path / "run/run.json").read_text())
    assert report["status"] == "FAIL" and not report["guest_completed"]


def test_sigterm_unwinds_and_restores_handler(tmp_path, monkeypatch):
    executable = _stub_executable(tmp_path)
    events = []
    previous = object()
    handlers = [previous]
    fake_signal = SimpleNamespace(SIGTERM=15, getsignal=lambda sig: handlers[-1])

    def set_handler(sig, handler):
        handlers.append(handler)

    fake_signal.signal = set_handler
    monkeypatch.setattr(runtime, "signal", fake_signal)
    # Substitute this module's OS view only, not the global os.name used by
    # pathlib or subprocess. This exercises POSIX ownership on Windows too.
    monkeypatch.setattr(runtime, "os", SimpleNamespace(name="posix"))
    process = _Process()
    first = True

    def poll():
        nonlocal first
        if first:
            first = False
            handlers[-1](15, None)
        return process.returncode

    process.poll = poll

    def launch(*a, **kwargs):
        assert kwargs["start_new_session"]
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    monkeypatch.setattr(runtime, "_windows_job", lambda p: None)
    monkeypatch.setattr(runtime, "_stop_process_tree", lambda p, job: events.append("tree-kill"))
    with pytest.raises(SystemExit) as exc:
        run_external(executable, {"x": np.ones(1, np.float32)}, tmp_path / "run")
    assert exc.value.code == 143
    assert events == ["tree-kill"] and handlers[-1] is previous
    report = json.loads((tmp_path / "run/run.json").read_text())
    assert report["error"] == "SystemExit: 143" and report["status"] == "FAIL"


def test_elf_hash_uses_validated_bytes_not_second_read(tmp_path, monkeypatch):
    executable = _stub_executable(tmp_path)
    monkeypatch.setattr(runtime, "sha256_file", lambda path: "wrong second read")
    result = runtime.validate_external_elf(executable.elf_path, executable.layout)
    assert result["sha256"] == executable.evidence["sha256"]


def test_elf_changed_during_validation_is_rejected(tmp_path, monkeypatch):
    executable = _stub_executable(tmp_path)
    original = runtime.struct.unpack_from
    mutated = False

    def unpack(*args, **kwargs):
        nonlocal mutated
        result = original(*args, **kwargs)
        if not mutated:
            mutated = True
            with executable.elf_path.open("ab") as stream:
                stream.write(b"changed")
        return result

    monkeypatch.setattr(runtime.struct, "unpack_from", unpack)
    with pytest.raises(ValueError, match="changed during validation"):
        runtime.validate_external_elf(executable.elf_path, executable.layout)


def test_linux_rss_parser_reports_resident_bytes():
    assert runtime._linux_rss("Name:\tqemu\nVmSize:\t9999999 kB\nVmRSS:\t1234 kB\n") == 1234 * 1024


@pytest.mark.parametrize("status", ["VmSize:\t123 kB", "VmRSS:\t-1 kB", "VmRSS:\t10 MB",
                                    "VmRSS:\tinvalid kB", "VmRSS:\t42"])
def test_linux_rss_parser_rejects_missing_or_invalid_fields(status):
    with pytest.raises(OSError, match="unavailable"):
        runtime._linux_rss(status)


@pytest.mark.skipif(os.name != "nt" and not sys.platform.startswith("linux"),
                    reason="RSS reader supports Windows/Linux")
def test_owned_child_rss_reader_and_monitor_are_real():
    process = subprocess.Popen([sys.executable, "-c", "import time; a=bytearray(8*1024*1024); time.sleep(10)"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    monitor = runtime._QemuMemoryMonitor(process, interval=0.02)
    try:
        monitor.start()
        deadline = time.monotonic() + 3
        while monitor.samples < 2:
            assert time.monotonic() < deadline, monitor.last_error
            time.sleep(0.01)
        monitor.close()
        report = monitor.report()
        assert report["qemu_rss_observed_samples"] >= 2
        assert report["sampled_peak_qemu_rss_bytes"] > 0
        assert report["qemu_rss_availability"] == "available"
    finally:
        monitor.close()
        process.terminate()
        process.wait(timeout=5)


@pytest.mark.skipif(os.name == "nt", reason="real POSIX SIGTERM/process-group test")
def test_posix_sigterm_reaps_owned_process_session(tmp_path):
    executable = _stub_executable(tmp_path)
    fake_qemu = tmp_path / "qemu-stub"
    fake_qemu.write_text(f"#!{sys.executable}\nimport os,time\nfrom pathlib import Path\n"
                         "Path('guest.pid').write_text(str(os.getpid()))\ntime.sleep(300)\n")
    fake_qemu.chmod(0o700)
    worker = tmp_path / "worker.py"
    worker.write_text(
        "from pathlib import Path\nimport numpy as np\n"
        "from scratchv.runtime.riscv_external import ExternalExecutable, run_external\n"
        "from scratchv.runtime.riscv_tensor import RiscVToolchain\n"
        "from scratchv.backend.tensor_c_codegen import TensorSpec\n"
        "from scratchv.ir.types import DataType\n"
        "spec=TensorSpec('x',DataType.FLOAT32,(1,))\n"
        f"exe=ExternalExecutable(Path({str(executable.elf_path)!r}),"
        f"Path({str(executable.bundle_dir)!r}),{executable.manifest!r},{executable.layout!r},"
        f"(spec,),spec,(),RiscVToolchain(('unused',),{str(fake_qemu)!r}),{executable.evidence!r})\n"
        f"run_external(exe,{{'x':np.ones(1,np.float32)}},Path({str(tmp_path / 'run')!r}))\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    process = subprocess.Popen([sys.executable, str(worker)], env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    child_pid = None
    try:
        deadline = time.monotonic() + 15
        pid_path = tmp_path / "run/guest.pid"
        while not pid_path.exists():
            assert process.poll() is None
            if time.monotonic() >= deadline:
                pytest.fail("fake QEMU session did not start")
            time.sleep(0.05)
        child_pid = int(pid_path.read_text())
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=20)
        assert process.returncode == 143
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        report = json.loads((tmp_path / "run/run.json").read_text())
        assert report["error"] == "SystemExit: 143" and report["status"] == "FAIL"
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
