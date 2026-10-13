"""Compiler process setup failures must remain bounded and preserve their cause."""
import io
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from scratchv.runtime import riscv_tensor as runtime


class Process:
    def __init__(self):
        self.returncode = None
        self.stdout, self.stderr = io.BytesIO(), io.BytesIO()
        self.events = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # Real Popen.__exit__ waits without a deadline. Record instead of hanging
        # this test when exercising the pre-fix job-setup exception path.
        self.events.append(("context_exit", self.returncode))

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        assert timeout is not None
        self.events.append(("communicate", timeout))
        self.returncode = 0
        return b"out", b"err"

    def kill(self):
        self.events.append(("kill",))
        self.returncode = -9

    def wait(self, timeout=None):
        assert timeout is not None
        self.events.append(("wait", timeout))
        return self.returncode


def test_job_setup_exception_terminates_child_without_context_wait(monkeypatch):
    process = Process()
    failure = OSError("Job setup failed")
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: process)
    def no_job(child):
        raise failure
    monkeypatch.setattr(runtime, "_windows_job", no_job)
    monkeypatch.setattr(runtime, "_stop_process_tree", lambda child, job: child.kill())
    with pytest.raises(OSError) as caught:
        runtime._run_process(["compiler"], cwd=None, timeout=0.1)
    assert caught.value is failure
    assert ("kill",) in process.events
    assert not any(event[0] == "context_exit" for event in process.events)
    assert process.stdout.closed and process.stderr.closed


def test_cleanup_failures_preserve_primary_and_try_remaining_resources(monkeypatch):
    process = Process()
    class OrderedPipe(io.BytesIO):
        def close(self):
            process.events.append(("close-pipe",))
            super().close()
    process.stdout, process.stderr = OrderedPipe(), OrderedPipe()
    failure = KeyboardInterrupt("original cancellation")
    def broken_communicate(timeout=None):
        process.events.append(("communicate", timeout))
        raise failure
    process.communicate = broken_communicate
    def broken_tree(child, job):
        process.events.append(("stop-tree",))
        raise OSError("tree kill failed")
    kernel = SimpleNamespace(CloseHandle=lambda handle: process.events.append(("close-job",)) or 1)
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(runtime, "_windows_job", lambda child: (kernel, 42))
    monkeypatch.setattr(runtime, "_stop_process_tree", broken_tree)
    with pytest.raises(KeyboardInterrupt) as caught:
        runtime._run_process(["compiler"], cwd=None, timeout=0.1)
    assert caught.value is failure
    assert ("kill",) in process.events and ("close-job",) in process.events
    assert process.stdout.closed and process.stderr.closed
    assert process.events.index(("close-job",)) < process.events.index(("close-pipe",))
    assert any("tree kill failed" in note for note in failure.__notes__)


def test_timeout_keeps_complete_drained_logs(monkeypatch):
    process = Process()
    calls = []
    def communicate(timeout=None):
        calls.append(timeout)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(["compiler"], timeout, output=b"partial")
        process.returncode = -9
        return b"complete stdout", b"complete stderr"
    process.communicate = communicate
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(runtime, "_windows_job", lambda child: None)
    monkeypatch.setattr(runtime, "_stop_process_tree", lambda child, job: child.kill())
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        runtime._run_process(["compiler"], cwd=None, timeout=0.1)
    assert caught.value.output == b"complete stdout"
    assert caught.value.stderr == b"complete stderr"
    assert all(timeout is not None for timeout in calls)


def test_real_child_is_reaped_when_job_setup_raises(monkeypatch, tmp_path):
    processes = []
    def no_job(process):
        processes.append(process)
        raise OSError("injected Job setup failure")
    monkeypatch.setattr(runtime, "_windows_job", no_job)
    with pytest.raises(OSError, match="injected Job"):
        runtime._run_process([sys.executable, "-B", "-c", "import time;time.sleep(60)"],
                             cwd=tmp_path, timeout=0.1)
    assert len(processes) == 1 and processes[0].poll() is not None


def test_windows_blocked_reader_close_cannot_stall_cleanup(monkeypatch):
    process = Process()
    release = threading.Event()
    class BlockedPipe(io.BytesIO):
        def close(self):
            release.wait(timeout=3)
            super().close()
    process.stdout, process.stderr = BlockedPipe(), BlockedPipe()
    def timeout(timeout=None):
        raise subprocess.TimeoutExpired(["compiler"], timeout, output=b"partial")
    process.communicate = timeout
    monkeypatch.setattr(runtime, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(runtime, "_windows_job", lambda child: None)
    monkeypatch.setattr(runtime, "_stop_process_tree", lambda child, job: child.kill())
    started = time.monotonic()
    try:
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            runtime._run_process(["compiler"], cwd=None, timeout=0.1)
        assert time.monotonic() - started < 2
        assert any("Pipe close is pending" in note for note in caught.value.__notes__)
    finally:
        release.set()


def test_popen_failure_restores_posix_handler_without_touching_a_child(monkeypatch):
    events = []
    failure = OSError("compiler launch failed")
    fake_signal = SimpleNamespace(SIGTERM=15, SIGKILL=9, getsignal=lambda sig: "original",
        signal=lambda sig, handler: events.append(("signal", handler)))
    monkeypatch.setattr(runtime, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(runtime, "signal", fake_signal)
    def launch(*args, **kwargs):
        events.append(("launch",))
        raise failure
    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    monkeypatch.setattr(runtime, "_windows_job", lambda child: pytest.fail("no child exists"))
    with pytest.raises(OSError) as caught:
        runtime._run_process(["compiler"], cwd=None, timeout=1)
    assert caught.value is failure
    assert events == [("signal", runtime._terminate_invocation), ("launch",), ("signal", "original")]
