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
