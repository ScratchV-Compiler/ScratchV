"""Real worker-tree memory guards include Windows venv redirector children."""

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from probes.w3_qwen3_full import resources


class _SampleClock:
    def __init__(self, on_sleep=None):
        self.now = 0.0
        self.on_sleep = on_sleep

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep()


class _SampleProcess:
    pid = 42
    returncode = None
    _scratchv_owned_group = True

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake-worker", timeout)
        return self.returncode


def _fake_posix_tree(monkeypatch, process):
    monkeypatch.setattr(resources._PosixGroup, "pids", lambda self: [42] if process.poll() is None else [])
    monkeypatch.setattr(resources._PosixGroup, "identity", staticmethod(
        lambda pid: ("R" if process.poll() is None else "Z", 42, 42, 100)))


@pytest.mark.parametrize("exit_code", [0, 7])
def test_rss_disappearing_before_waitable_exit_preserves_real_exit(monkeypatch, exit_code):
    process = _SampleProcess()
    _fake_posix_tree(monkeypatch, process)
    clock = _SampleClock(lambda: setattr(process, "returncode", exit_code))
    monkeypatch.setattr(resources, "time", clock)

    def missing(pid):
        raise resources._ResidentMemoryUnavailable("exit_mm removed VmRSS before zombie state")

    monkeypatch.setattr(resources, "process_memory", missing)
    monkeypatch.setattr(resources, "terminate_process_tree", lambda _: pytest.fail("Completed worker must not be killed"))
    row = {}
    assert resources.wait_bounded(process, timeout=2, max_memory_bytes=1000, row=row) == exit_code
    assert row["returncode"] == exit_code
    assert row["resource_monitor"]["rss_exit_rechecks"] == 1
    assert row["resource_monitor"]["observed_pids"] == []


@pytest.mark.parametrize("error", ["missing", "permission", "malformed"])
def test_live_worker_with_unobservable_rss_still_fails_and_is_reaped(monkeypatch, error):
    process = _SampleProcess()
    _fake_posix_tree(monkeypatch, process)
    clock = _SampleClock()
    monkeypatch.setattr(resources, "time", clock)
    failure = {"missing": resources._ResidentMemoryUnavailable("missing VmRSS"),
               "permission": PermissionError("status access denied"),
               "malformed": OSError("malformed VmRSS")}[error]

    def missing(pid):
        raise failure

    monkeypatch.setattr(resources, "process_memory", missing)
    monkeypatch.setattr(resources, "terminate_process_tree", lambda child: setattr(child, "returncode", -9))
    row = {}
    with pytest.raises(type(failure)) as caught:
        resources.wait_bounded(process, timeout=2, max_memory_bytes=1000, row=row)
    assert caught.value is failure
    assert process.returncode == -9
    if error == "missing":
        assert resources._RSS_EXIT_GRACE_SECONDS <= clock.now < 0.3
        assert row["resource_monitor"]["rss_exit_rechecks"] > 0
    else:
        assert clock.now == 0  # Unrelated observer failures get no grace period.
        assert row["resource_monitor"]["rss_exit_rechecks"] == 0


def test_rss_recovery_during_grace_still_enforces_memory_limit(monkeypatch):
    process = _SampleProcess()
    _fake_posix_tree(monkeypatch, process)
    clock = _SampleClock()
    monkeypatch.setattr(resources, "time", clock)

    def sample(pid):
        if clock.now == 0:
            raise resources._ResidentMemoryUnavailable("temporarily missing VmRSS")
        return {"rss_bytes": 2000, "private_commit_bytes": None}

    monkeypatch.setattr(resources, "process_memory", sample)
    monkeypatch.setattr(resources, "terminate_process_tree", lambda child: setattr(child, "returncode", -9))
    row = {}
    with pytest.raises(MemoryError):
        resources.wait_bounded(process, timeout=2, max_memory_bytes=1000, row=row)
    assert process.returncode == -9
    assert row["resource_monitor"]["rss_peak_sampled_bytes"] == 2000
    assert row["resource_monitor"]["rss_exit_rechecks"] == 1


@pytest.mark.parametrize("identity", [("Z", 42, 42, 100), ("X", 42, 42, 100),
                                     ("R", 99, 99, 100), ("R", 42, 42, 101)])
def test_missing_rss_recheck_never_attributes_exited_or_reused_pid(monkeypatch, identity):
    tree = resources._PosixGroup(_SampleProcess())
    monkeypatch.setattr(tree, "identity", lambda pid: identity)
    monkeypatch.setattr(resources, "process_memory", lambda pid: (_ for _ in ()).throw(
        resources._ResidentMemoryUnavailable("missing VmRSS")))
    assert tree._member_memory(42, ("R", 42, 42, 100)) is None
    assert tree.rss_exit_rechecks == 0


@pytest.mark.skipif(sys.platform != "linux", reason="Linux real /proc worker-exit sampling")
def test_real_linux_exit_rss_race_is_repeatable_without_false_failures(tmp_path, monkeypatch):
    actual_memory = resources.process_memory
    for attempt in range(20):
        marker = tmp_path / f"exit-{attempt}"
        ready_file = tmp_path / f"ready-{attempt}"
        code = ("import os,time;from pathlib import Path;"
                f"ready=Path({str(ready_file)!r});ready.write_text(str(os.getpid()));"
                f"marker=Path({str(marker)!r});"
                "\nwhile not marker.exists():time.sleep(0.001)\n")
        process = resources.spawn_owned([sys.executable, "-B", "-c", code],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        injected = []

        def sample(pid):
            if not injected:
                assert process.poll() is None
                injected.append(pid)
                marker.touch()
                # Force the otherwise scheduler-dependent kernel exit window.
                raise resources._ResidentMemoryUnavailable("exit-time VmRSS loss")
            return actual_memory(pid)

        try:
            ready(ready_file, process)
            monkeypatch.setattr(resources, "process_memory", sample)
            row = {}
            assert resources.wait_bounded(process, timeout=5, max_memory_bytes=2**30,
                                          row=row, interval=0.01) == 0
            assert injected == [process.pid]
            assert row["returncode"] == 0 and "cleanup_error" not in row
        finally:
            monkeypatch.setattr(resources, "process_memory", actual_memory)
            if process.poll() is None:
                resources.terminate_process_tree(process)


def ready(path, process):
    deadline = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() > deadline or process.poll() is not None:
            raise AssertionError("Owned allocation worker did not become ready")
        time.sleep(0.01)
    return int(path.read_text(encoding="utf-8"))


def allocation_code(path, megabytes=120, sleep=30):
    # Readers use exists() as readiness: atomically publish a completed PID.
    return ("import os,time; from pathlib import Path; "
            f"data=bytearray({megabytes}*1024*1024); "
            "data[::4096]=b'x'*len(data[::4096]); "
            f"marker=Path({str(path)!r}); temporary=marker.with_suffix('.tmp'); "
            "temporary.write_text(str(os.getpid()),encoding='utf-8'); "
            f"temporary.replace(marker); time.sleep({sleep})")


def owned_exit_checker(process, pid):
    """Hold the exact process identity so PID reuse cannot fake successful kill."""
    if os.name == "nt":
        from ctypes import wintypes as w
        job = process._scratchv_job
        assert pid in job.pids()
        handle = job.k.OpenProcess(0x100000 | 0x1000, False, pid)
        assert handle
        member = w.BOOL()
        assert job.k.IsProcessInJob(handle, job.handle, ctypes.byref(member)) and member.value
        job.k.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        job.k.WaitForSingleObject.restype = w.DWORD

        def check():
            try:
                assert job.k.WaitForSingleObject(handle, 5000) == 0, "Actual worker remained alive"
                code = w.DWORD()
                assert job.k.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value != 259
            finally:
                job.k.CloseHandle(handle)
        return check
    identity = resources._PosixGroup.identity(pid)
    assert identity[1] == identity[2] == process.pid

    def check():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                now = resources._PosixGroup.identity(pid)
            except (FileNotFoundError, ProcessLookupError):
                return
            if now[0] == "Z" or now[1:] != identity[1:]:
                return
            time.sleep(0.01)
        raise AssertionError("Actual worker remained alive")
    return check


@pytest.mark.parametrize("cause", ["memory", "timeout", "interrupt"])
def test_venv_real_child_is_observed_and_stopped_without_taskkill(tmp_path, monkeypatch, cause):
    marker = tmp_path / "worker.pid"
    process = resources.spawn_owned([sys.executable, "-B", "-c", allocation_code(marker)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    row = {}
    exit_check = None
    try:
        actual_pid = ready(marker, process)
        exit_check = owned_exit_checker(process, actual_pid)
        tree = process._scratchv_job if os.name == "nt" else resources._PosixGroup(process)
        sample = tree.sample()
        assert actual_pid in sample["pids"]
        assert sample["rss_bytes"] > 100 * 1024**2
        if os.name == "nt" and Path(sys.executable).resolve() != Path(sys._base_executable).resolve():
            assert actual_pid != process.pid  # Exercise the real venv redirector.
            assert len(sample["pids"]) >= 2
        if cause == "interrupt":
            def interrupt(pid):
                raise KeyboardInterrupt("test cancellation")
            monkeypatch.setattr(resources, "process_memory", interrupt)
        # Any attempt to fall back to taskkill would make this test fail.
        if os.name == "nt":
            monkeypatch.setattr(resources, "_legacy_terminate",
                                lambda process: pytest.fail("Job cleanup must not use taskkill"))
        expected = {"memory": MemoryError, "timeout": TimeoutError,
                    "interrupt": KeyboardInterrupt}[cause]
        with pytest.raises(expected):
            resources.wait_bounded(process, timeout=0.08 if cause == "timeout" else 5,
                max_memory_bytes=96*1024**2 if cause == "memory" else 2**40,
                row=row, interval=0.01)
        assert process.poll() is not None
        assert "cleanup_error" not in row
        if cause == "memory":
            observed = row["resource_monitor"]
            assert observed["rss_peak_sampled_bytes"] > 100*1024**2
            assert actual_pid in observed["observed_pids"]
        (tmp_path / "monitor.json").write_text(json.dumps(
            {"launcher_pid": process.pid, "actual_pid": actual_pid, "sample": sample, "guard": row},
            indent=2), encoding="utf-8")
    finally:
        if process.poll() is None or getattr(process, "_scratchv_job", None) is not None:
            resources.terminate_process_tree(process)
        if exit_check:
            exit_check()


def test_descendants_outliving_launcher_are_still_bounded_and_reaped(tmp_path):
    marker = tmp_path / "grandchild.pid"
    child = allocation_code(marker, megabytes=12)
    parent = ("import subprocess,sys,time; from pathlib import Path; "
              f"subprocess.Popen([sys.executable,'-B','-c',{child!r}]); "
              f"marker=Path({str(marker)!r}); "
              "\nwhile not marker.exists(): time.sleep(0.01)\n")
    process = resources.spawn_owned([sys.executable, "-B", "-c", parent],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    checker = None
    try:
        # The launcher may already have exited, so use bounded marker polling
        # that deliberately does not assume launcher liveness.
        deadline = time.monotonic() + 10
        while not marker.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        actual_pid = int(marker.read_text(encoding="utf-8"))
        checker = owned_exit_checker(process, actual_pid)
        assert process.wait(timeout=5) == 0
        row = {}
        with pytest.raises(TimeoutError):
            resources.wait_bounded(process, timeout=0.08, max_memory_bytes=2**40,
                                   row=row, interval=0.01)
        assert actual_pid in row["resource_monitor"]["observed_pids"]
        assert "cleanup_error" not in row
    finally:
        if process.poll() is None or getattr(process, "_scratchv_job", None) is not None:
            resources.terminate_process_tree(process)
        if checker:
            checker()


def test_launch_assignment_failure_never_executes_child_code(tmp_path, monkeypatch):
    marker = tmp_path / "should-not-exist"
    if os.name != "nt":
        # POSIX session creation is atomic in Popen; exercise fail-closed spawn
        # through that platform's creation primitive without Windows API use.
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: (_ for _ in ()).throw(OSError("spawn denied")))
    else:
        def denied(self, process):
            raise OSError("assignment denied")
        monkeypatch.setattr(resources._WindowsJob, "assign_and_resume", denied)
    with pytest.raises(OSError, match="denied"):
        resources.spawn_owned([sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert not marker.exists()
