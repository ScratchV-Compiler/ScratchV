"""Real QEMU execution of the external tensor ABI (explicit tool opt-in)."""
import errno
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
from scratchv.runtime.riscv_tensor import RiscVToolchain, discover_toolchain
from scratchv.runtime.weight_bundle import write_weight_bundle
from scratchv.runtime import riscv_external as runtime


def artifact(**codegen_options):
    builder = IRBuilder()
    x, w = Value("x", shape=(4, 4)), Value("w", shape=(4, 4))
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.program.global_values.append(w)
    builder.ret(builder.matmul(x, w))
    weights = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
    return TensorCCodegen(builder.program, initializers={"w": weights},
                         constant_storage="external", **codegen_options).generate(), weights


@pytest.mark.parametrize("project_cache,standard_cache,selected", [
    (None, None, None),
    (None, "standard cache", "standard cache"),
    ("project cache", None, "project cache"),
    ("project cache", "standard cache", "project cache"),
], ids=["default", "standard", "project", "project-overrides-standard"])
def test_external_build_respects_zig_cache_environment(
        tmp_path, monkeypatch, project_cache, standard_cache, selected):
    monkeypatch.chdir(tmp_path)
    for key, value in (("SCRATCHV_ZIG_CACHE", project_cache),
                       ("ZIG_GLOBAL_CACHE_DIR", standard_cache)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    model, _ = artifact()
    write_weight_bundle(model.external_weights, model.external_initializers, tmp_path / "weights")
    received = []

    def compiler(command, *, cwd, timeout, env):
        received.append(env)
        # Inspect the actual subprocess environment without compiling a model.
        return subprocess.CompletedProcess(command, 1, b"", b"controlled compile stop")

    monkeypatch.setattr(runtime, "_run_process", compiler)
    with pytest.raises(RuntimeError, match="controlled compile stop"):
        build_external(model, tmp_path / "weights", tmp_path / "build",
                       RiscVToolchain(("zig", "cc"), "unused-qemu"))
    expected = (tmp_path / selected if selected else
                Path(runtime.__file__).resolve().parents[2] / "output" / ".riscv-zig-cache")
    assert len(received) == 1
    assert received[0]["ZIG_GLOBAL_CACHE_DIR"] == str(expected.resolve())
    assert received[0]["ZIG_LOCAL_CACHE_DIR"] == str(tmp_path / "build" / ".zig-cache")


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


@pytest.mark.parametrize("outcome", ["success", "nonzero", "timeout", "nonfinite", "cleanup", "trailing-uart"])
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
            if command == "stop" and outcome == "trailing-uart":
                with (out / "uart.bin").open("ab") as uart:
                    uart.write(b"unexpected")
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
                    "nonfinite": ValueError, "cleanup": RuntimeError,
                    "trailing-uart": ValueError}[outcome]
        with pytest.raises(expected):
            run_external(executable, {"x": np.ones(1, np.float32)}, out)
    report = json.loads((out / "run.json").read_text())
    assert report["guest_completed"]
    assert report["timing_mode"] in ("event", "polling")
    assert "host validation" in report["forward_wall_scope"]
    qmp_endpoint = report["command"][report["command"].index("-qmp") + 1]
    assert qmp_endpoint.endswith(",server=on,wait=off,nodelay=on")
    if outcome == "trailing-uart":
        assert "human-monitor-command" not in events and "quit" not in events
        assert "trailing completion frame" in report["error"]
        assert "close" in events
        return
    assert report["qmp_quit_reply_received"] is False
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


@pytest.mark.skipif(not sys.platform.startswith("linux") or not hasattr(os, "pidfd_open"),
                    reason="inotify/pidfd event wait is Linux-specific")
@pytest.mark.parametrize("fd_target", ["normal", "inotify", "pidfd"])
@pytest.mark.parametrize("outcome", ["split", "failed", "exit", "timeout"])
def test_uart_event_wait_handles_partial_frames_exit_and_deadline(
        tmp_path, monkeypatch, outcome, fd_target):
    import fcntl
    import resource

    limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if fd_target != "normal" and limit != resource.RLIM_INFINITY and limit <= 1024:
        pytest.skip("high-FD regression requires an existing descriptor limit above 1024")
    waiter = runtime._UartCompletionWaiter(tmp_path, 4)
    assert waiter.timing_mode == "event"
    frame = runtime.FRAME.pack(runtime.MAGIC, 0x80000005 if outcome == "failed" else 0, 4)
    code = {
        "split": ("import time; from pathlib import Path; "
                  f"p=Path('uart.bin'); p.write_bytes({frame[:8]!r}); "
                  f"time.sleep(0.03); p.write_bytes({frame!r})"),
        "failed": f"from pathlib import Path; Path('uart.bin').write_bytes({frame!r})",
        "exit": "pass",
        "timeout": "import time; time.sleep(10)",
    }[outcome]
    process = None
    opened_pidfds, registrations = [], []
    original_pidfd_open, original_poll = os.pidfd_open, runtime.select.poll

    def open_pidfd(pid):
        descriptor = original_pidfd_open(pid)
        if fd_target == "pidfd":
            try:
                high = fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, 1024)
            finally:
                os.close(descriptor)
            descriptor = high
        opened_pidfds.append(descriptor)
        return descriptor

    released = False

    class ReleaseOnPoll:
        def __init__(self):
            self.poller = original_poll()

        def register(self, descriptor, mask):
            registrations.append(descriptor)
            self.poller.register(descriptor, mask)

        def poll(self, timeout):
            nonlocal released
            if not released:
                # Do not let a fast child finish before the high FD is actually
                # registered and waited on. Only one byte, no timing assumption.
                process.stdin.write(b"x")
                process.stdin.flush()
                released = True
            return self.poller.poll(timeout)

    try:
        if fd_target == "inotify":
            old = waiter.fd
            waiter.fd = fcntl.fcntl(old, fcntl.F_DUPFD_CLOEXEC, 1024)
            os.close(old)
        fd = waiter.fd
        process = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1); " + code],
            cwd=tmp_path, stdin=subprocess.PIPE)
        # No completion polling or busy wait is allowed in the event path.
        with monkeypatch.context() as patch:
            patch.setattr(runtime.time, "sleep", lambda _: pytest.fail("event wait polled"))
            patch.setattr(runtime.os, "pidfd_open", open_pidfd)
            patch.setattr(runtime.select, "poll", ReleaseOnPoll)
            deadline = time.monotonic() + (0.05 if outcome == "timeout" else 3)
            if outcome == "split":
                waiter.wait(process, deadline, 3)
                assert (tmp_path / "uart.bin").read_bytes() == frame
            else:
                error, message = {
                    "failed": (RuntimeError, "0x80000005"),
                    "exit": (RuntimeError, "exited before guest completion"),
                    "timeout": (TimeoutError, "forward exceeded"),
                }[outcome]
                with pytest.raises(error, match=message):
                    waiter.wait(process, deadline, 3)
        assert released and fd in registrations and opened_pidfds
        assert opened_pidfds[0] in registrations
        if fd_target != "normal":
            assert (fd if fd_target == "inotify" else opened_pidfds[0]) >= 1024
        assert waiter.timing_mode == "event" and waiter.fallback_reason is None
    finally:
        waiter.close()
        waiter.close()
        if process is not None:
            process.stdin.close()
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
    for descriptor in [fd, *opened_pidfds]:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.skipif(not sys.platform.startswith("linux") or not hasattr(os, "pidfd_open"),
                    reason="inotify/pidfd event wait is Linux-specific")
@pytest.mark.parametrize("failure", ["closed-inotify", "closed-pidfd", "poll-error"])
def test_uart_event_wait_rejects_invalid_descriptors_without_busy_wait(
        tmp_path, monkeypatch, failure):
    waiter = runtime._UartCompletionWaiter(tmp_path, 4)
    process = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
    waiter.pidfd = os.pidfd_open(process.pid)
    invalid_name = {"closed-inotify": "fd", "closed-pidfd": "pidfd"}.get(failure)
    if invalid_name:
        os.close(getattr(waiter, invalid_name))
    original_poll = runtime.select.poll
    calls = []

    class CheckedPoll:
        def __init__(self):
            self.poller = original_poll()

        def register(self, descriptor, mask):
            self.poller.register(descriptor, mask)

        def poll(self, timeout):
            calls.append(1)
            assert len(calls) == 1, "invalid descriptor caused an event busy loop"
            if failure == "poll-error":
                return [(waiter.fd, runtime.select.POLLERR)]
            return self.poller.poll(timeout)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(runtime.select, "poll", CheckedPoll)
            patch.setattr(runtime.time, "sleep", lambda _: pytest.fail("invalid FD used polling fallback"))
            with pytest.raises(OSError) as exc:
                waiter.wait(process, time.monotonic() + 1, 1)
        assert exc.value.errno == (errno.EIO if failure == "poll-error" else errno.EBADF)
        assert calls == [1]
    finally:
        # The injected invalid FD is already closed; release the other owned FD.
        if invalid_name:
            setattr(waiter, invalid_name, None)
        waiter.close()


def test_uart_wait_observes_frame_completed_before_wait(tmp_path, monkeypatch):
    waiter = runtime._UartCompletionWaiter(tmp_path, 4)
    try:
        (tmp_path / "uart.bin").write_bytes(runtime.FRAME.pack(runtime.MAGIC, 0, 4))
        # The frame may already be complete when Popen returns, even before
        # pidfd_open can observe an exited child.
        process = SimpleNamespace(poll=lambda: pytest.fail("already completed"), pid=-1)
        waiter.wait(process, time.monotonic() + 1, 1)
    finally:
        waiter.close()


@pytest.mark.parametrize("fallback", ["platform", "pidfd"])
def test_uart_wait_retains_portable_polling_fallback(tmp_path, monkeypatch, fallback):
    if fallback == "platform":
        monkeypatch.setattr(runtime, "sys", SimpleNamespace(platform="unsupported"))
    elif not sys.platform.startswith("linux") or not hasattr(os, "pidfd_open"):
        pytest.skip("pidfd fallback requires Linux")
    else:
        def unavailable(_pid):
            raise OSError(errno.ENOSYS, "pidfd unsupported")
        monkeypatch.setattr(runtime.os, "pidfd_open", unavailable)
    waiter = runtime._UartCompletionWaiter(tmp_path, 4)
    sleeps = []

    def complete_on_poll(interval):
        sleeps.append(interval)
        (tmp_path / "uart.bin").write_bytes(runtime.FRAME.pack(runtime.MAGIC, 0, 4))

    monkeypatch.setattr(runtime.time, "sleep", complete_on_poll)
    try:
        waiter.wait(_Process(), time.monotonic() + 1, 1)
        assert waiter.timing_mode == "polling" and waiter.fallback_reason
        assert sleeps == [0.05]
        assert waiter.fd is None
    finally:
        waiter.close()


@pytest.mark.skipif(not sys.platform.startswith("linux") or not hasattr(os, "pidfd_open"),
                    reason="inotify watch failure is Linux-specific")
@pytest.mark.parametrize("mask,message", [(0x4000, "overflow"), (0x8000, "lost its directory")])
def test_uart_watch_loss_is_not_silently_accepted(tmp_path, monkeypatch, mask, message):
    waiter = runtime._UartCompletionWaiter(tmp_path, 4)
    try:
        monkeypatch.setattr(runtime.os, "read", lambda *_: struct.pack("iIII", -1, mask, 0, 0))
        with pytest.raises(RuntimeError, match=message):
            waiter._drain()
    finally:
        waiter.close()


@pytest.mark.skipif(not sys.platform.startswith("linux") or not hasattr(os, "pidfd_open"),
                    reason="inotify setup failure is Linux-specific")
def test_uart_watch_setup_failure_closes_fd(tmp_path, monkeypatch):
    read_fd, write_fd = os.pipe()

    def init(_flags):
        return read_fd

    def add_watch(*_args):
        runtime.ctypes.set_errno(errno.ENOSPC)
        return -1

    monkeypatch.setattr(runtime.ctypes, "CDLL", lambda *_args, **_kwargs:
                        SimpleNamespace(inotify_init1=init, inotify_add_watch=add_watch))
    try:
        with pytest.raises(OSError, match="No space"):
            runtime._UartCompletionWaiter(tmp_path, 4)
        with pytest.raises(OSError):
            os.fstat(read_fd)
    finally:
        os.close(write_fd)


def test_uart_wait_accepts_final_frame_written_during_exit_check(tmp_path):
    waiter = runtime._UartCompletionWaiter(tmp_path, 4)

    def exit_after_write():
        (tmp_path / "uart.bin").write_bytes(runtime.FRAME.pack(runtime.MAGIC, 0, 4))
        return 0

    try:
        waiter.wait(SimpleNamespace(poll=exit_after_write, returncode=0),
                    time.monotonic() + 1, 1)
    finally:
        waiter.close()


@pytest.mark.skipif(not sys.platform.startswith("linux") or not hasattr(os, "pidfd_open"),
                    reason="inotify drain is Linux-specific")
def test_uart_watch_drain_is_bounded_under_continuous_events(tmp_path, monkeypatch):
    waiter = runtime._UartCompletionWaiter(tmp_path, 4)
    reads = []

    def continuous_events(*_args):
        reads.append(1)
        assert len(reads) == 1, "unbounded drain starved the outer deadline check"
        return struct.pack("iIII", 1, 0x2, 0, 0)

    try:
        monkeypatch.setattr(runtime.os, "read", continuous_events)
        waiter._drain()
        assert reads == [1]
    finally:
        waiter.close()


def test_uart_cleanup_attempts_both_descriptors_after_close_error(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(platform="unsupported"))
    waiter = runtime._UartCompletionWaiter(tmp_path, 4)
    waiter.pidfd, waiter.fd = 111, 222
    attempts = []

    def close(fd):
        attempts.append(fd)
        if fd == 111:
            raise OSError("injected pidfd close failure")

    monkeypatch.setattr(runtime.os, "close", close)
    with pytest.raises(OSError, match="pidfd close failure"):
        waiter.close()
    assert attempts == [111, 222]
    assert waiter.pidfd is None and waiter.fd is None
    waiter.close()
    assert attempts == [111, 222]


@pytest.mark.parametrize("tile", ["1x4", "2x4", "1x8"])
@pytest.mark.parametrize("rhs_layout", ["original", "column-panel"])
def test_external_build_records_matmul_tile_without_execution(
        tmp_path, monkeypatch, tile, rhs_layout):
    model, _ = artifact(matmul_policy="blocked_fma", matmul_tile=tile,
                        matmul_rhs_layout=rhs_layout)
    write_weight_bundle(model.external_weights, model.external_initializers, tmp_path / "weights")
    monkeypatch.setattr(runtime, "_run_process",
                        lambda command, **kw: subprocess.CompletedProcess(command, 0, b"", b""))
    monkeypatch.setattr(runtime, "validate_external_elf", lambda *a: {})
    monkeypatch.setattr(runtime, "toolchain_versions", lambda *a: {})
    executable = build_external(model, tmp_path / "weights", tmp_path / "build",
                                RiscVToolchain(("unused-cc",), "unused-qemu"))
    saved = json.loads((tmp_path / "build" / "build.json").read_text())
    assert saved["matmul_tile"] == executable.evidence["matmul_tile"] == tile
    assert saved["matmul_rhs_layout"] == executable.evidence["matmul_rhs_layout"] == rhs_layout
    assert saved["matmul_rhs_panels"] == list(model.matmul_rhs_panels)
    assert executable.evidence["matmul_rhs_panels"] == saved["matmul_rhs_panels"]
    assert bool(saved["matmul_rhs_panels"]) is (rhs_layout == "column-panel")
    assert executable.harts == 1
    assert saved["qemu_workspace_layout"] == "none"
