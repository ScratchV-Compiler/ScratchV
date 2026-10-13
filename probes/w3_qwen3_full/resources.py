"""Observe and bound an owned full-model worker; no extra monitor dependency."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
import time

from scripts.run_w2_acceptance import terminate_process_tree as _legacy_terminate


class _ResidentMemoryUnavailable(OSError):
    """Linux can drop VmRSS while exit is still in progress, before zombie state."""


_RSS_EXIT_GRACE_SECONDS = 0.2
_RSS_EXIT_RECHECK_SECONDS = 0.01


class _VerifiedProcessHandle:
    """A handle already verified as a member of the owned Windows Job."""
    def __init__(self, handle):
        self.handle = handle


class _WindowsJob:
    """Own descendants before the suspended launcher executes any user code.

    Windows 8+ supports nesting under an existing sandbox Job. No breakaway
    flags or UI limits are used. Assignment failure is fatal, not a fallback
    to monitoring only the launcher. Handles are non-inheritable.
    """
    def __init__(self):
        from ctypes import wintypes as w
        self.k = k = ctypes.WinDLL("kernel32", use_last_error=True)
        prototypes = {
            "CreateJobObjectW": ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            "SetInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
            "QueryInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p], w.BOOL),
            "IsProcessInJob": ([w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL)], w.BOOL),
            "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
            "OpenProcess": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            "GetExitCodeProcess": ([w.HANDLE, ctypes.POINTER(w.DWORD)], w.BOOL),
            "CloseHandle": ([w.HANDLE], w.BOOL),
            "CreateToolhelp32Snapshot": ([w.DWORD, w.DWORD], w.HANDLE),
            "Thread32First": ([w.HANDLE, ctypes.c_void_p], w.BOOL),
            "Thread32Next": ([w.HANDLE, ctypes.c_void_p], w.BOOL),
            "OpenThread": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            "GetProcessIdOfThread": ([w.HANDLE], w.DWORD),
            "ResumeThread": ([w.HANDLE], w.DWORD),
        }
        for name, (args, result) in prototypes.items():
            getattr(k, name).argtypes, getattr(k, name).restype = args, result

        class Basic(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                        ("flags", w.DWORD), ("min_working", ctypes.c_size_t),
                        ("max_working", ctypes.c_size_t), ("active", w.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]
        class Limits(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", ctypes.c_ulonglong * 6),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
        self.handle = k.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Limits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign_and_resume(self, process):
        from ctypes import wintypes as w
        if not self.k.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        class ThreadEntry(ctypes.Structure):
            _fields_ = [("size", w.DWORD), ("usage", w.DWORD), ("tid", w.DWORD),
                        ("pid", w.DWORD), ("priority", w.LONG), ("delta", w.LONG), ("flags", w.DWORD)]
        snapshot = self.k.CreateToolhelp32Snapshot(4, 0)  # TH32CS_SNAPTHREAD
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        tids = []
        try:
            entry = ThreadEntry()
            entry.size = ctypes.sizeof(entry)
            ok = self.k.Thread32First(snapshot, ctypes.byref(entry))
            while ok:
                if entry.pid == process.pid:
                    tids.append(entry.tid)
                ok = self.k.Thread32Next(snapshot, ctypes.byref(entry))
            if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.k.CloseHandle(snapshot)
        if len(tids) != 1:
            raise OSError("Suspended launcher must have exactly one primary thread")
        thread = self.k.OpenThread(0x0002 | 0x0040, False, tids[0])
        if not thread:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if self.k.GetProcessIdOfThread(thread) != process.pid:
                raise OSError("Suspended primary thread ownership changed")
            if self.k.ResumeThread(thread) != 1:
                raise OSError("Suspended launcher did not have exactly one suspension")
        finally:
            self.k.CloseHandle(thread)

    def pids(self):
        # QueryInformationJobObject includes descendant nested Jobs. Retry if
        # more processes arrived between sizing and reading the PID list.
        for capacity in (16, 64, 256, 1024, 4096, 16384):
            buffer = ctypes.create_string_buffer(8 + capacity * ctypes.sizeof(ctypes.c_size_t))
            ok = self.k.QueryInformationJobObject(self.handle, 3, buffer, len(buffer), None)
            assigned, count = (ctypes.c_ulong * 2).from_buffer(buffer)
            if ok and count == assigned and count <= capacity:
                return list((ctypes.c_size_t * count).from_buffer(buffer, 8))
            if not ok and ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(ctypes.get_last_error())
        raise OSError("Owned worker Job exceeds bounded process enumeration")

    def sample(self):
        from ctypes import wintypes as w
        total = {"rss_bytes": 0, "private_commit_bytes": 0, "pids": []}
        for pid in self.pids():
            handle = self.k.OpenProcess(0x1010, False, pid)
            if not handle:
                if pid not in self.pids():
                    continue  # Exited after the Job snapshot.
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                member = w.BOOL()
                if not self.k.IsProcessInJob(handle, self.handle, ctypes.byref(member)):
                    raise ctypes.WinError(ctypes.get_last_error())
                if not member.value:
                    # PID reuse cannot substitute an unrelated process: never
                    # inspect or terminate the new process's memory.
                    continue
                try:
                    value = process_memory(_VerifiedProcessHandle(handle))
                except OSError:
                    code = w.DWORD()
                    if self.k.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value != 259:
                        continue
                    raise
                total["rss_bytes"] += value["rss_bytes"]
                total["private_commit_bytes"] += value["private_commit_bytes"] or 0
                total["pids"].append(pid)
            finally:
                self.k.CloseHandle(handle)
        return total

    def terminate(self):
        if not self.k.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())
        deadline = time.monotonic() + 5
        while self.pids():
            if time.monotonic() > deadline:
                raise OSError("Owned worker Job still has active descendants after termination")
            time.sleep(0.01)

    def close(self):
        if self.handle:
            handle, self.handle = self.handle, None
            if not self.k.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())


def spawn_owned(command, **kwargs):
    """Create a worker tree whose ownership precedes all executed child code."""
    if os.name != "nt":
        kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **kwargs)
        process._scratchv_owned_group = True
        return process
    flags = kwargs.pop("creationflags", 0)
    if flags & (0x01000000 | 0x00000004):
        raise ValueError("Caller cannot request breakaway or manage suspended owned workers")
    job, process = _WindowsJob(), None
    try:
        process = subprocess.Popen(command, creationflags=flags | 0x00000004, **kwargs)
        process._scratchv_job = job
        job.assign_and_resume(process)
        return process
    except BaseException as original:
        # Before ResumeThread the launcher cannot have started descendants;
        # after assignment closing the Job owns every descendant regardless.
        try:
            job.close()
        except BaseException as cleanup:
            original.add_note(f"Closing failed worker Job: {cleanup}")
        try:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
        except BaseException as cleanup:
            original.add_note(f"Reaping failed launcher: {cleanup}")
        raise


def terminate_process_tree(process):
    job = getattr(process, "_scratchv_job", None)
    if job is not None:
        try:
            job.terminate()
        finally:
            job.close()
            process._scratchv_job = None
        process.wait(timeout=5)
    else:
        _legacy_terminate(process)


class _PosixGroup:
    """Linux session/group membership, including children outliving launcher."""
    def __init__(self, process):
        self.process = process
        self.rss_exit_rechecks = 0

    @staticmethod
    def identity(pid):
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = text[text.rindex(")") + 2:].split()
        return fields[0], int(fields[2]), int(fields[3]), int(fields[19])

    def pids(self):
        found = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                state, group, session, _ = self.identity(int(entry.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
            if group == session == self.process.pid and state not in {"Z", "X", "x"}:
                found.append(int(entry.name))
        return found

    def _member_memory(self, pid, before):
        # exit_mm() can remove VmRSS before the process becomes a zombie or
        # waitpid observes its exit. Only this specific missing-field case gets
        # a short grace period; permission errors and malformed RSS fail closed.
        deadline = None
        while True:
            try:
                value = process_memory(pid)
            except _ResidentMemoryUnavailable:
                after = self.identity(pid)
                if before[1:] != after[1:] or after[0] in {"Z", "X", "x"}:
                    return None
                if deadline is None:
                    deadline = time.perf_counter() + _RSS_EXIT_GRACE_SECONDS
                if time.perf_counter() >= deadline:
                    raise  # Still the same live process and still unobservable.
                self.rss_exit_rechecks += 1
                time.sleep(_RSS_EXIT_RECHECK_SECONDS)
                continue
            after = self.identity(pid)
            if before[1:] != after[1:] or after[0] in {"Z", "X", "x"}:
                return None
            return value

    def sample(self):
        total = {"rss_bytes": 0, "private_commit_bytes": None, "pids": []}
        for pid in self.pids():
            try:
                before = self.identity(pid)
                if (before[1] != self.process.pid or before[2] != self.process.pid
                        or before[0] in {"Z", "X", "x"}):
                    continue
                value = self._member_memory(pid, before)
                if value is None:
                    continue  # PID was reused/exited during sampling.
            except (FileNotFoundError, ProcessLookupError):
                continue
            except OSError:
                try:
                    after = self.identity(pid)
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if after[0] in {"Z", "X", "x"}:
                    continue
                raise
            total["rss_bytes"] += value["rss_bytes"]
            total["pids"].append(pid)
        return total


def process_memory(pid):
    """Current RSS, and private committed bytes on Windows (not virtual size)."""
    if os.name == "nt":
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                        ("peak_rss", ctypes.c_size_t), ("rss", ctypes.c_size_t),
                        ("peak_paged", ctypes.c_size_t), ("paged", ctypes.c_size_t),
                        ("peak_nonpaged", ctypes.c_size_t), ("nonpaged", ctypes.c_size_t),
                        ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t),
                        ("private", ctypes.c_size_t)]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        borrowed = isinstance(pid, _VerifiedProcessHandle)
        handle = pid.handle if borrowed else kernel.OpenProcess(0x1010, False, pid)
        if not handle:
            raise OSError(ctypes.get_last_error(), "Cannot observe worker memory")
        try:
            values = Counters()
            values.cb = ctypes.sizeof(values)
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(values), values.cb):
                raise OSError(ctypes.get_last_error(), "Cannot read worker memory")
            return {"rss_bytes": int(values.rss), "private_commit_bytes": int(values.private)}
        finally:
            if not borrowed:
                kernel.CloseHandle(handle)
    status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    rows = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    if "VmRSS" not in rows:
        raise _ResidentMemoryUnavailable(f"Cannot observe worker {pid} resident memory")
    try:
        resident = rows["VmRSS"].split()
        if len(resident) != 2 or resident[1] != "kB":
            raise ValueError("VmRSS must use the Linux kB unit")
        rss = int(resident[0]) * 1024
        if rss < 0:
            raise ValueError("VmRSS cannot be negative")
    except ValueError as exc:
        raise OSError(f"Cannot observe worker {pid} resident memory") from exc
    return {"rss_bytes": rss, "private_commit_bytes": None}


def wait_bounded(process, *, timeout, max_memory_bytes, row, interval=0.2):
    """A sampled guard, not an OS hard limit. Stop the owned tree on any failure."""
    started = time.perf_counter()
    job = getattr(process, "_scratchv_job", None)
    group = bool(getattr(process, "_scratchv_owned_group", False))
    tree = job if job is not None else _PosixGroup(process) if group else None
    observation = {"samples": 0, "rss_peak_sampled_bytes": 0, "private_commit_peak_sampled_bytes": None,
                   "interval_seconds": interval, "limit_bytes": max_memory_bytes,
                   "ownership": "windows_job" if job else "posix_process_group",
                   "observed_pids": [], "peak_sampled_processes": 0,
                   "scope": ("Owned Windows Job and descendants; summed sampled RSS/private commit, shared pages may be counted per process, not a hard OS quota"
                             if job else "Owned Linux process group/session; summed sampled RSS, shared pages may be counted per process, not a hard OS quota")}
    row["resource_monitor"] = observation
    try:
        if os.name == "nt" and isinstance(process, subprocess.Popen) and job is None:
            raise ValueError("Windows workers must be created with spawn_owned before execution")
        while process.poll() is None or (tree is not None and tree.pids()):
            if time.perf_counter() - started > timeout:
                raise TimeoutError(f"Full worker exceeded {timeout} seconds")
            try:
                values = tree.sample() if tree is not None else process_memory(process.pid)
            except OSError:
                if process.poll() is not None and (tree is None or not tree.pids()):
                    break
                raise
            observation["samples"] += 1
            observation["rss_peak_sampled_bytes"] = max(observation["rss_peak_sampled_bytes"], values["rss_bytes"])
            private = values["private_commit_bytes"]
            pids = values.get("pids", [process.pid])
            observation["observed_pids"] = sorted(set(observation["observed_pids"]) | set(pids))
            observation["peak_sampled_processes"] = max(observation["peak_sampled_processes"], len(pids))
            if private is not None:
                observation["private_commit_peak_sampled_bytes"] = max(
                    observation["private_commit_peak_sampled_bytes"] or 0, private)
            if max(values["rss_bytes"], private or 0) > max_memory_bytes:
                raise MemoryError(f"Full worker exceeded sampled memory limit {max_memory_bytes} bytes")
            if process.poll() is not None:
                time.sleep(interval)  # Job children can outlive their launcher.
            else:
                try:
                    process.wait(timeout=interval)
                except subprocess.TimeoutExpired:
                    pass
        return process.wait(timeout=5)
    except BaseException:
        if process.poll() is None or job is not None or group:
            try:
                terminate_process_tree(process)
            except (OSError, subprocess.SubprocessError) as cleanup:
                # Preserve timeout/interruption rather than replacing its
                # cause with any secondary cleanup diagnostic.
                row["cleanup_error"] = f"{type(cleanup).__name__}: {cleanup}"
        raise
    finally:
        if isinstance(tree, _PosixGroup):
            observation["rss_exit_rechecks"] = tree.rss_exit_rechecks
        if getattr(process, "_scratchv_job", None) is not None:
            try:
                job.close()
            except OSError as cleanup:
                row["cleanup_error"] = f"{type(cleanup).__name__}: {cleanup}"
            process._scratchv_job = None
        row["returncode"] = process.poll()
        row["elapsed_seconds"] = time.perf_counter() - started
