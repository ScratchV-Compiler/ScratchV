"""Preparation must own descendants even after a Windows launcher exits."""
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from probes.w3_qwen3_full import resources
from scripts import run_w3_preparation as gate


class ExitedLauncher:
    pid = 42
    args = ["fixture-python"]
    returncode = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        return self.returncode


@pytest.mark.parametrize("ownership", ["_scratchv_job", "_scratchv_owned_group"])
def test_preparation_delegates_owned_tree_wait_even_after_launcher_exit(monkeypatch, ownership):
    process, row, calls = ExitedLauncher(), {}, []
    setattr(process, ownership, True)

    def wait(child, *, timeout, max_memory_bytes, row):
        calls.append(child)
        assert timeout == 1.5 and max_memory_bytes == sys.maxsize
        row["resource_monitor"] = {"scope": "owned descendants, no preparation cap"}
        return 0

    monkeypatch.setattr(gate, "wait_bounded", wait, raising=False)
    assert gate.wait_child(process, 1.5, row) == 0
    assert calls == [process] and "resource_monitor" in row


@pytest.mark.parametrize("failure", [TimeoutError, KeyboardInterrupt, OSError])
def test_owned_wait_preserves_timeout_and_interrupt_report_contract(monkeypatch, failure):
    process, row = ExitedLauncher(), {}
    process._scratchv_job = True

    def wait(child, **kwargs):
        kwargs["row"].update(returncode=0, elapsed_seconds=0.01)
        raise failure("owned wait failed")

    monkeypatch.setattr(gate, "wait_bounded", wait, raising=False)
    expected = subprocess.TimeoutExpired if failure is TimeoutError else failure
    with pytest.raises(expected):
        gate.wait_child(process, 0.01, row)
    assert row["returncode"] == 0 and row["elapsed_seconds"] == 0.01


@pytest.mark.skipif(os.name != "nt", reason="Windows Job ownership regression")
def test_real_preparation_timeout_reclaims_grandchild_after_parent_exit(tmp_path):
    import ctypes
    from ctypes import wintypes as w
    child_pid_path = tmp_path / "child-pid.txt"
    child_script = tmp_path / "child.py"
    # Publish readiness only after the complete PID has been written/closed.
    child_script.write_text("import os, pathlib, sys, time\n"
                            "marker = pathlib.Path(sys.argv[1])\n"
                            "temporary = marker.with_suffix('.tmp')\n"
                            "temporary.write_text(str(os.getpid()), encoding='utf-8')\n"
                            "temporary.replace(marker)\n"
                            "time.sleep(30)\n")
    parent_script = tmp_path / "parent.py"
    parent_script.write_text(
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, sys.argv[2], sys.argv[1]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n")
    process = resources.spawn_owned([sys.executable, str(parent_script), str(child_pid_path), str(child_script)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    owned_job = process._scratchv_job
    row, child_handle = {}, None
    try:
        deadline = time.monotonic() + 5
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_pid_path.exists()
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert child_pid in owned_job.pids()
        child_handle = owned_job.k.OpenProcess(0x100000 | 0x1000, False, child_pid)
        assert child_handle
        member = w.BOOL()
        assert owned_job.k.IsProcessInJob(child_handle, owned_job.handle, ctypes.byref(member)) and member.value
        process.wait(timeout=5)
        with pytest.raises(subprocess.TimeoutExpired):
            gate.wait_child(process, 0.1, row)
        assert process._scratchv_job is None
        assert child_pid in row["resource_monitor"]["observed_pids"]
        # The held identity prevents PID reuse from faking successful cleanup.
        owned_job.k.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        owned_job.k.WaitForSingleObject.restype = w.DWORD
        assert owned_job.k.WaitForSingleObject(child_handle, 5000) == 0
        code = w.DWORD()
        assert owned_job.k.GetExitCodeProcess(child_handle, ctypes.byref(code)) and code.value != 259
    finally:
        if getattr(process, "_scratchv_job", None) is not None:
            resources.terminate_process_tree(process)
        if child_handle:
            owned_job.k.CloseHandle(child_handle)
