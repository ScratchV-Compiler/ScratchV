"""Bare-metal RV64 external-weight execution; QMP transports completed output.

QMP only reads guest RAM after the guest's success frame. All model arithmetic
runs in RV64 instructions, without semihosting or host numerical substitution.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import struct
import subprocess
import threading
import time

import numpy as np

from .riscv_tensor import (
    STARTUP_ASSEMBLY, LINKER_SCRIPT, RiscVToolchain, _creation_flags,
    _run_process, _save_process_logs, _spec_layout, _stop_process_tree,
    _terminate_invocation, _windows_job, input_layout, pack_inputs, toolchain_versions,
)
from .weight_bundle import plan_guest_memory, validate_weight_bundle
from .weight_transport import (
    stage_transport, validate_transport, remove_transport, save_transport_report,
    refresh_transport_residuals,
)


FRAME = struct.Struct("<8sIQ")
MAGIC = b"SVEXT001"
MAX_ELF_BYTES = 100_000_000


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_external_elf(path, layout):
    """Validate every load segment, including BSS, against code/stack bounds."""
    path = Path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size >= MAX_ELF_BYTES:
        raise ValueError("Code-only ELF must be smaller than 100,000,000 bytes")
    def identity(info):
        # Windows stat/fstat may expose different ctime meanings. Device,
        # inode, size and mtime remain comparable across both APIs.
        fields = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        return fields + ((info.st_ctime_ns,) if os.name != "nt" else ())
    with path.open("rb") as stream:
        if identity(os.fstat(stream.fileno())) != identity(before):
            raise ValueError("ELF changed while opening for validation")
        data = stream.read(MAX_ELF_BYTES)
        if (len(data) != before.st_size
                or identity(os.fstat(stream.fileno())) != identity(before)):
            raise ValueError("ELF changed while reading for validation")
    if len(data) < 64 or data[:7] != b"\x7fELF\x02\x01\x01":
        raise ValueError("Expected little-endian ELF64")
    kind, machine = struct.unpack_from("<HH", data, 16)
    version = struct.unpack_from("<I", data, 20)[0]
    entry, phoff = struct.unpack_from("<QQ", data, 24)
    flags = struct.unpack_from("<I", data, 48)[0]
    header_size = struct.unpack_from("<H", data, 52)[0]
    phsize, phcount = struct.unpack_from("<HH", data, 54)
    if kind != 2 or machine != 243 or flags & 6 != 4 or flags & 8:
        raise ValueError("Expected executable RISC-V LP64D ELF")
    if version != 1 or header_size != 64:
        raise ValueError("Invalid ELF version or header size")
    if phsize != 56 or not phcount or phoff < 64 or phoff + phsize * phcount > len(data):
        raise ValueError("Invalid ELF program header table")
    segments = []
    for index in range(phcount):
        typ, attrs, offset, vaddr, paddr, filesz, memsz, align = struct.unpack_from(
            "<IIQQQQQQ", data, phoff + index * phsize)
        if typ != 1:
            continue
        if (filesz > memsz or offset + filesz > len(data) or vaddr != paddr
                or paddr < layout["code_base"] or paddr + memsz > layout["code_limit"]):
            raise ValueError("ELF load segment exceeds code region or has invalid sizes")
        if align not in (0, 1) and (align & (align - 1) or (vaddr - offset) % align):
            raise ValueError("Invalid ELF segment alignment")
        segments.append({"address": paddr, "filesz": filesz, "memsz": memsz, "flags": attrs})
    ordered = sorted(segments, key=lambda item: item["address"])
    for left, right in zip(ordered, ordered[1:]):
        if left["address"] + left["memsz"] > right["address"]:
            raise ValueError("ELF load segments overlap")
    if not any(row["flags"] & 1 and row["address"] <= entry < row["address"] + row["filesz"]
               for row in segments):
        raise ValueError("ELF entry is not in an executable file-backed segment")
    if identity(path.stat()) != identity(before):
        raise ValueError("ELF changed during validation")
    # Hash precisely the bytes whose ELF structure was checked, not a second
    # open of the pathname which could describe a different file by now.
    return {"bytes": len(data), "entry": entry, "segments": segments,
            "sha256": hashlib.sha256(data).hexdigest()}


@dataclass(frozen=True)
class ExternalExecutable:
    elf_path: Path
    bundle_dir: Path
    manifest: dict
    layout: dict
    inputs: tuple
    output: object
    weights: tuple
    toolchain: RiscVToolchain
    evidence: dict


def guest_harness(artifact, manifest, layout):
    offsets, _ = input_layout(artifact.inputs)
    function = artifact.function_name
    if not function.isascii() or not function.isidentifier():
        raise ValueError("Invalid C entry point")
    inputs = ",".join(f"(const void *)0x{layout['input_base'] + n:x}UL" for n in offsets) or "0"
    weights = ",".join(f"(const void *)0x{layout['weights_base'] + t['offset']:x}UL"
                       for t in manifest["tensors"]) or "0"
    return r'''
typedef unsigned long usize;
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;
void *memcpy(void *dst, const void *src, usize n) {
    u8 *d=dst; const u8 *s=src; for (usize i=0;i<n;++i) d[i]=s[i]; return dst;
}
void *memset(void *dst, int v, usize n) {
    u8 *d=dst; for (usize i=0;i<n;++i) d[i]=(u8)v; return dst;
}
static void uart_byte(u8 byte) {
    volatile u8 *u=(volatile u8 *)0x10000000UL;
    while (!(u[5]&0x20)) {} u[0]=byte;
}
static void frame(u32 status, u64 size) {
    const char *magic="SVEXT001";
    for (int i=0;i<8;++i) uart_byte((u8)magic[i]);
    for (int i=0;i<4;++i) uart_byte((u8)(status>>(8*i)));
    for (int i=0;i<8;++i) uart_byte((u8)(size>>(8*i)));
    __asm__ volatile("fence rw,rw" ::: "memory");
    for (;;) __asm__ volatile("wfi");
}
void sv_trap_report(u64 cause) { frame(0x80000000U|(u32)cause, 0); }
''' + f'''
extern int {function}(const void *const *, const void *const *, void *, usize, void *);
void sv_main(void) {{
    const void *inputs[]={{ {inputs} }};
    const void *weights[]={{ {weights} }};
    int status={function}(inputs,weights,(void *)0x{layout['workspace_base']:x}UL,
        {artifact.workspace_bytes}UL,(void *)0x{layout['output_base']:x}UL);
    frame((u32)status,{artifact.output.nbytes}ULL);
}}
'''


def build_external(artifact, bundle_dir, build_dir, toolchain, *, timeout=600.0):
    if artifact.constant_storage != "external":
        raise ValueError("External runtime requires external code generation")
    manifest = validate_weight_bundle(bundle_dir, artifact.external_weights)
    _, input_bytes = input_layout(artifact.inputs)
    layout = plan_guest_memory(workspace_bytes=artifact.workspace_bytes,
                               weight_bytes=manifest["total_bytes"], input_bytes=input_bytes,
                               output_bytes=artifact.output.nbytes)
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=False)
    linker = LINKER_SCRIPT.replace("0x1c000000", hex(layout["code_capacity"]))
    sources = {"model.c": artifact.source, "guest.c": guest_harness(artifact, manifest, layout),
               "start.S": STARTUP_ASSEMBLY, "link.ld": linker}
    for name, source in sources.items():
        (build_dir / name).write_text(source, encoding="utf-8")
    target = (["-target", "riscv64-linux-musl", "-static"] if toolchain.is_zig else
              ["--target=riscv64-unknown-elf", "-fuse-ld=lld", "-nostdlib"])
    arch = (["-mcpu=generic_rv64+m+a+f+d+c+zicsr+zifencei"] if toolchain.is_zig else ["-march=rv64gc"])
    command = [*toolchain.cc, *target, "-O2", "-std=c11", *arch, "-mabi=lp64d", "-mcmodel=medany",
               "-msmall-data-limit=0", "-mno-relax", "-ffreestanding", "-fno-builtin",
               "-fno-stack-protector", "-fno-pie", "-fPIC" if toolchain.is_zig else "-fno-pic",
               "-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing",
               "-ffunction-sections", "-fdata-sections", *artifact.compile_flags,
               "model.c", "guest.c", "start.S", "-Wl,-T,link.ld", "-Wl,-e,sv_boot",
               "-Wl,--gc-sections", "-Wl,--build-id=none", *(["-lm"] if toolchain.is_zig else []),
               "-o", "model.elf"]
    env = dict(os.environ)
    temporary = build_dir / "tmp"
    temporary.mkdir()
    env.update(TMP=str(temporary), TEMP=str(temporary), TMPDIR=str(temporary))
    if toolchain.is_zig:
        env["ZIG_GLOBAL_CACHE_DIR"] = str(Path(os.environ.get("SCRATCHV_ZIG_CACHE",
            Path(__file__).resolve().parents[2] / "output/.riscv-zig-cache")).resolve())
        env["ZIG_LOCAL_CACHE_DIR"] = str(build_dir / ".zig-cache")
    start = time.perf_counter()
    try:
        result = _run_process(command, cwd=build_dir, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        _save_process_logs(build_dir, "compiler", exc.stdout, exc.stderr)
        raise TimeoutError(f"External RV64 build exceeded {timeout}s") from exc
    _save_process_logs(build_dir, "compiler", result.stdout, result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-12000:])
    evidence = validate_external_elf(build_dir / "model.elf", layout)
    evidence.update(compile_seconds=time.perf_counter() - start, command=command,
                    versions=toolchain_versions(toolchain),
                    sources={name: sha256_file(build_dir / name) for name in sources})
    (build_dir / "layout.json").write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    (build_dir / "build.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return ExternalExecutable(build_dir / "model.elf", Path(bundle_dir).resolve(), manifest, layout,
                              tuple(artifact.inputs), artifact.output, tuple(artifact.external_weights),
                              toolchain, evidence)


class QMP:
    """A bounded localhost QMP client that ignores asynchronous events."""
    def __init__(self, port, deadline):
        self.deadline = deadline
        self.serial = 0
        self.socket = socket.create_connection(("127.0.0.1", port), timeout=self.remaining())
        self.stream = None
        try:
            self.stream = self.socket.makefile("rwb", buffering=0)
            if "QMP" not in self.read():
                raise RuntimeError("Invalid QMP greeting")
            self.command("qmp_capabilities")
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise

    def remaining(self):
        value = self.deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("QMP deadline exceeded")
        return min(value, 30.0)

    def read(self):
        self.socket.settimeout(self.remaining())
        line = self.stream.readline(1024 * 1024)
        if not line:
            raise EOFError("QMP connection closed before response")
        if len(line) >= 1024 * 1024 or not line.endswith(b"\n"):
            raise RuntimeError("Truncated/oversized QMP response")
        return json.loads(line)

    def command(self, name, arguments=None):
        self.serial += 1
        request = {"execute": name, "id": self.serial}
        if arguments is not None:
            request["arguments"] = arguments
        payload = json.dumps(request).encode() + b"\n"
        self.socket.settimeout(self.remaining())
        if self.stream.write(payload) != len(payload):
            raise RuntimeError("Incomplete QMP command write")
        while True:
            try:
                result = self.read()
            except (EOFError, ConnectionResetError):
                # QMP quit may terminate QEMU before its reply is delivered:
                # https://www.qemu.org/docs/master/interop/qemu-qmp-ref.html#command-quit
                # Only a completely written, argument-free quit gets this
                # exception. The caller must still require a normal process
                # exit and validate the completed output and resource cleanup.
                if name == "quit" and arguments is None:
                    return None
                raise
            if result.get("id") == self.serial:
                if "error" in result:
                    raise RuntimeError(f"QMP {name}: {result['error']}")
                if "return" not in result:
                    raise RuntimeError("Invalid QMP reply")
                return result["return"]

    def close(self):
        first_error = None
        for resource in (self.stream, self.socket):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        if first_error is not None:
            raise first_error


def decode_completion(data, output_bytes):
    if len(data) != FRAME.size:
        raise ValueError("Missing, truncated or trailing completion frame")
    magic, status, size = FRAME.unpack(data)
    if magic != MAGIC:
        raise ValueError("Invalid completion magic")
    if status:
        raise RuntimeError(f"RV64 guest failed with status 0x{status:08x}")
    if size != output_bytes:
        raise ValueError("Guest output length differs from generated ABI")


def _linux_rss(status):
    rows = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    try:
        amount, unit = rows["VmRSS"].split()
        amount = int(amount)
        if amount < 0 or unit != "kB":
            raise ValueError("Invalid VmRSS unit/value")
        return amount * 1024
    except (KeyError, ValueError) as exc:
        raise OSError("QEMU resident memory is unavailable") from exc


def _owned_qemu_rss(process):
    """Read only an owned Popen child, avoiding PID reuse after it is reaped."""
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                        ("peak_rss", ctypes.c_size_t), ("rss", ctypes.c_size_t),
                        ("peak_paged", ctypes.c_size_t), ("paged", ctypes.c_size_t),
                        ("peak_nonpaged", ctypes.c_size_t), ("nonpaged", ctypes.c_size_t),
                        ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]
        get_memory = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_memory.restype = wintypes.BOOL
        values = Counters()
        values.cb = ctypes.sizeof(values)
        if not get_memory(int(process._handle), ctypes.byref(values), values.cb):
            raise OSError(ctypes.get_last_error(), "Cannot read owned QEMU resident memory")
        return int(values.rss)
    # Popen.poll/wait use this lock too. While held, a live child cannot be
    # reaped and replaced by an unrelated process with the same PID.
    with process._waitpid_lock:
        if process.returncode is not None:
            return None
        return _linux_rss(Path(f"/proc/{process.pid}/status").read_text(encoding="utf-8"))


class _QemuMemoryMonitor:
    """Sample process RSS; this is observability, not an enforced RAM limit."""
    def __init__(self, process, interval=0.2):
        self.process, self.interval = process, interval
        self.samples, self.errors, self.peak, self.last_error = 0, 0, None, None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, name="scratchv-qemu-rss", daemon=True)

    def _run(self):
        while not self.stopped.is_set():
            try:
                value = _owned_qemu_rss(self.process)
                if value is not None:
                    self.samples += 1
                    self.peak = value if self.peak is None else max(self.peak, value)
            except (OSError, AttributeError, ValueError) as exc:
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            self.stopped.wait(self.interval)

    def start(self):
        self.thread.start()

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=1)
        if self.thread.is_alive():
            raise RuntimeError("QEMU RSS monitor did not stop")

    def report(self):
        return {
            "sampled_peak_qemu_rss_bytes": self.peak,
            "qemu_rss_observed_samples": self.samples,
            "qemu_rss_availability": ("unavailable" if not self.samples else
                                      "partial" if self.errors else "available"),
            "qemu_rss_sample_errors": self.errors,
            "qemu_rss_last_error": self.last_error,
            "qemu_rss_sample_interval_s": self.interval,
            "qemu_rss_scope": "QEMU process RSS; distinct from guest layout and not a hard limit",
        }


def run_external(executable, inputs, run_dir, *, timeout=3600.0):
    """Run a fresh guest, require completion, stop, dump exact output, then quit."""
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    if sha256_file(executable.elf_path) != executable.evidence["sha256"]:
        raise ValueError("ELF changed after build")
    manifest = validate_weight_bundle(executable.bundle_dir, executable.weights)
    if manifest != executable.manifest:
        raise ValueError("Weight manifest changed after build")
    payload = pack_inputs(executable.inputs, inputs)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "inputs.bin").write_bytes(payload or b"\0")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    layout = executable.layout
    command = [executable.toolchain.qemu, "-machine", "virt", "-cpu", "rv64", "-accel", "tcg",
               "-smp", "1", "-m", str(layout["memory_mib"]), "-bios", "none", "-display", "none",
               "-monitor", "none", "-serial", "file:uart.bin", "-no-reboot",
               "-qmp", f"tcp:127.0.0.1:{port},server=on,wait=off",
               "-kernel", str(executable.elf_path), "-device",
               f"loader,file=inputs.bin,addr=0x{layout['input_base']:x},force-raw=on"]
    options = {"creationflags": _creation_flags()}
    if os.name != "nt":
        options["start_new_session"] = True
    started = time.perf_counter()
    report = {"passed": False, "status": "FAIL", "command": command, "timeout_s": timeout,
              "transport": "QMP physical memory dump after guest completion", "guest_completed": False}
    transport_dir = run_dir / "weight-transport"
    report["weight_transport"] = {}
    qmp, process, job, primary_error, monitor = None, None, None, None, None
    qemu_started, qemu_exited = None, None
    previous_term = None
    try:
        # SIGTERM must unwind through cleanup: QEMU owns a separate POSIX
        # session and would otherwise survive a supervisor cancelling Python.
        if os.name != "nt" and threading.current_thread() is threading.main_thread():
            previous_term = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, _terminate_invocation)
        report["stage"] = "weight_transport"
        stage_transport(executable.bundle_dir / "weights.bin", manifest, transport_dir,
                        report["weight_transport"])
        for row in report["weight_transport"]["files"]:
            path = str(transport_dir / row["file"]).replace(",", ",,")
            command += ["-device", f"loader,file={path},addr=0x{layout['weights_base'] + row['offset']:x},force-raw=on"]
        # Empty bundles intentionally emit no raw loader: QEMU rejects 0-byte files.
        report["stage"] = "qemu"
        deadline = time.monotonic() + timeout
        with (run_dir / "qemu.stdout").open("wb") as stdout, (run_dir / "qemu.stderr").open("wb") as stderr:
            qemu_started = time.perf_counter()
            process = subprocess.Popen(command, cwd=run_dir, stdout=stdout, stderr=stderr, **options)
            job = _windows_job(process)
            monitor = _QemuMemoryMonitor(process)
            monitor.start()
            uart = run_dir / "uart.bin"
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"RV64 forward exceeded {timeout}s")
                if uart.exists() and uart.stat().st_size >= FRAME.size:
                    decode_completion(uart.read_bytes(), executable.output.nbytes)
                    break
                if process.poll() is not None:
                    raise RuntimeError(f"QEMU exited before guest completion ({process.returncode})")
                time.sleep(0.05)
            report.update(guest_completed=True, forward_wall_s=time.perf_counter() - qemu_started)
            qmp = QMP(port, deadline)
            qmp.command("stop")
            dump_started = time.perf_counter()
            output_path = run_dir / "output.bin"
            if executable.output.nbytes:
                reply = qmp.command("human-monitor-command", {"command-line":
                    f"pmemsave 0x{layout['output_base']:x} {executable.output.nbytes} output.bin"})
                if reply.strip():
                    raise RuntimeError(f"QMP memory dump failed: {reply}")
            else:
                output_path.write_bytes(b"")
            if output_path.stat().st_size != executable.output.nbytes:
                raise ValueError("Output dump size mismatch")
            report["output_dump_s"] = time.perf_counter() - dump_started
            report["qmp_quit_reply_received"] = qmp.command("quit") is not None
            process.wait(timeout=min(10, max(0.1, deadline - time.monotonic())))
            qemu_exited = time.perf_counter()
            if process.returncode:
                raise RuntimeError(f"QEMU shutdown failed ({process.returncode})")
            shape, dtype, _ = _spec_layout(executable.output)
            output = (np.memmap(output_path, dtype=dtype.newbyteorder("<"), mode="r", shape=shape)
                      if executable.output.nbytes else np.empty(shape, dtype=dtype))
            for offset in range(0, output.size, 262144):
                if not np.isfinite(output.reshape(-1)[offset:offset + 262144]).all():
                    raise ValueError("Nonfinite RV64 output")
            if validate_weight_bundle(executable.bundle_dir, executable.weights) != manifest:
                raise ValueError("Weight bundle changed during execution")
            report["stage"] = "weight_transport_postcheck"
            report["weight_transport"]["post_validation_elapsed_s"] = validate_transport(
                transport_dir, report["weight_transport"])
            report.update(passed=True, status="PASS", exit_code=process.returncode,
                          output_sha256=sha256_file(output_path), elf_sha256=executable.evidence["sha256"],
                          weight_sha256=manifest["sha256"], input_sha256=sha256_file(run_dir / "inputs.bin"))
            return output, report
    except BaseException as exc:
        primary_error = exc
        report.update(passed=False, status="FAIL", error=f"{type(exc).__name__}: {exc}")
        if report.get("stage") == "weight_transport_postcheck":
            report["weight_transport"].update(status="FAIL", stage="postcheck",
                                              error=report["error"])
        raise
    finally:
        cleanup_errors = []

        def cleanup(label, action):
            try:
                action()
            except BaseException as exc:
                cleanup_errors.append(f"{label}: {type(exc).__name__}: {exc}")

        if qmp is not None:
            cleanup("qmp.close", qmp.close)
        report.update(sampled_peak_qemu_rss_bytes=None, qemu_rss_observed_samples=0,
                      qemu_rss_availability="unavailable", qemu_rss_sample_interval_s=0.2)
        if monitor is not None:
            cleanup("QEMU RSS monitor close", monitor.close)
            cleanup("QEMU RSS monitor report", lambda: report.update(monitor.report()))
        if process is not None:
            # Each cleanup step is attempted even when a preceding one fails.
            # Bounded waits prevent a failed kill from hanging report creation.
            try:
                alive = process.poll() is None
            except BaseException as exc:
                cleanup_errors.append(f"process.poll: {type(exc).__name__}: {exc}")
                alive = True
            if alive:
                cleanup("process-tree termination", lambda: _stop_process_tree(process, job))
                try:
                    process.wait(timeout=10)
                except BaseException as exc:
                    cleanup_errors.append(f"process.wait: {type(exc).__name__}: {exc}")
                    cleanup("process.kill fallback", process.kill)
                    cleanup("process.wait fallback", lambda: process.wait(timeout=5))
            if process.returncode is not None and qemu_exited is None:
                qemu_exited = time.perf_counter()
        report["qemu_wall_s"] = (qemu_exited - qemu_started
                                 if qemu_exited is not None and qemu_started is not None else None)
        if job is not None:
            def close_job():
                if job[0].CloseHandle(job[1]) == 0:
                    raise OSError("CloseHandle failed")
            cleanup("Windows job close", close_job)
        if previous_term is not None:
            cleanup("restore SIGTERM", lambda: signal.signal(signal.SIGTERM, previous_term))
        # Only a fully successful, stopped/reaped guest permits copy removal.
        # Failed/still-running invocations retain transport data for inspection.
        if (report["passed"] and primary_error is None and not cleanup_errors
                and process is not None and process.returncode == 0):
            cleanup("weight transport cleanup", lambda: remove_transport(
                transport_dir, report["weight_transport"]))
        # Staging evidence can become stale during the guest or its postcheck.
        # Observe owned copies again even on failure, without deleting them or
        # allowing a secondary stat error to replace the original exception.
        if "files" in report["weight_transport"]:
            cleanup("weight transport residual observation", lambda: refresh_transport_residuals(
                transport_dir, report["weight_transport"]))
        def preserve_transport_report():
            if (transport_dir / "transport.json").is_file():
                save_transport_report(transport_dir, report["weight_transport"])
        cleanup("weight transport report", preserve_transport_report)
        report["elapsed_s"] = time.perf_counter() - started
        if cleanup_errors:
            report.update(passed=False, status="FAIL", cleanup_errors=cleanup_errors)
            if primary_error is None:
                report["error"] = "RuntimeError: runtime cleanup failed"
        try:
            (run_dir / "run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except BaseException as exc:
            if primary_error is None:
                raise
            if hasattr(primary_error, "add_note"):
                primary_error.add_note(f"Could not preserve run.json: {type(exc).__name__}: {exc}")
        if cleanup_errors and primary_error is None:
            raise RuntimeError("Runtime cleanup failed: " + "; ".join(cleanup_errors))
