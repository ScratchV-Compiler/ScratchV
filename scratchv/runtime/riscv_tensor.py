"""Build and execute generated tensor C as an RV64 ELF on QEMU's virt board.

The guest runs in machine mode without an operating system. Inputs enter through
QEMU's raw loader device; output bytes leave through the emulated UART. No host
floating-point execution or semihosting is used. Zig supplies musl's math routines
when available, while clang users may provide their own freestanding libm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np


RAM_BASE = 0x80000000
MEMORY_MIB = 512
INPUT_BASE = 0x9C000000
INPUT_CAPACITY = 64 * 1024 * 1024
FRAME_MAGIC = b"SVTENS01"
FRAME_END = b"SVEND001"
FRAME_HEADER = struct.Struct("<8sIQ")


@dataclass(frozen=True)
class RiscVToolchain:
    cc: tuple[str, ...]
    qemu: str

    @property
    def is_zig(self) -> bool:
        return Path(self.cc[0]).stem.lower() == "zig"


@dataclass(frozen=True)
class RiscVTensorExecutable:
    elf_path: Path
    inputs: tuple[Any, ...]
    output: Any
    toolchain: RiscVToolchain
    compile_command: tuple[str, ...]
    compile_seconds: float
    workspace_bytes: int
    elf_sha256: str
    tool_versions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RiscVTensorRun:
    output: np.ndarray
    elapsed_s: float
    command: tuple[str, ...]
    uart_path: Path
    stdout_path: Path
    stderr_path: Path


def _creation_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _windows_job(process: subprocess.Popen):
    """Bind a Windows invocation and descendants to a private kill-on-close job."""
    if os.name != "nt":
        return None
    import ctypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                    ("flags", ctypes.c_ulong), ("min_ws", ctypes.c_size_t),
                    ("max_ws", ctypes.c_size_t), ("active", ctypes.c_ulong),
                    ("affinity", ctypes.c_size_t), ("priority", ctypes.c_ulong),
                    ("scheduling", ctypes.c_ulong)]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("basic", BasicLimits), ("io", ctypes.c_ulonglong * 6),
                    ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                    ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel.CreateJobObjectW.restype = ctypes.c_void_p
    kernel.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    kernel.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateJobObjectW(None, None)
    limits = ExtendedLimits()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not handle:
        return None
    if not (kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits))
            and kernel.AssignProcessToJobObject(handle, int(process._handle))):
        kernel.CloseHandle(handle)
        return None
    return kernel, handle


def _stop_process_tree(process: subprocess.Popen, job=None) -> None:
    """Stop only the compiler/emulator tree created for this invocation."""
    if job is not None:
        kernel, handle = job
        if kernel.TerminateJobObject(handle, 1):
            return
    if os.name == "nt":
        if process.poll() is not None:
            return
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, timeout=10, creationflags=_creation_flags())
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_process(command, *, cwd, timeout, env=None) -> subprocess.CompletedProcess:
    options = {"creationflags": _creation_flags()}
    if os.name == "nt":
        options["creationflags"] |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        options["start_new_session"] = True
    with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, **options) as process:
        job = _windows_job(process)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _stop_process_tree(process, job)
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired as drain:
                # A failed OS-level tree termination must not turn a bounded
                # compiler timeout into an unbounded pipe read.
                process.kill()
                if process.stdout:
                    process.stdout.close()
                if process.stderr:
                    process.stderr.close()
                process.wait(timeout=5)
                stdout, stderr = drain.output or b"", drain.stderr or b""
            raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from exc
        finally:
            if job is not None:
                job[0].CloseHandle(job[1])
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def toolchain_versions(toolchain: RiscVToolchain) -> dict[str, str]:
    compiler = ([toolchain.cc[0], "version"] if toolchain.is_zig
                else [*toolchain.cc, "--version"])
    versions = {}
    for name, command in (("compiler", compiler), ("qemu", [toolchain.qemu, "--version"])):
        result = _run_process(command, cwd=None, timeout=15)
        if result.returncode:
            raise RuntimeError(f"Cannot query {name} version: {result.stderr.decode(errors='replace')}")
        lines = result.stdout.decode("utf-8", errors="replace").splitlines()
        versions[name] = lines[0] if lines else "unknown"
    versions["target"] = "riscv64 / RV64GC / LP64D / QEMU virt / bare metal"
    versions["math_library"] = "Zig-bundled musl" if toolchain.is_zig else "caller-supplied"
    return versions


def _resolve_executable(value: str | os.PathLike[str]) -> str:
    value = os.fspath(value)
    resolved = shutil.which(value)
    if resolved:
        return str(Path(resolved).resolve())
    candidate = Path(value)
    if candidate.is_file():
        return str(candidate.resolve())
    raise FileNotFoundError(f"Required executable is unavailable: {value}")


def _workspace_tool(name: str) -> str | None:
    tools = Path(__file__).resolve().parents[2] / "output" / "tools"
    suffix = ".exe" if os.name == "nt" else ""
    if tools.is_dir():
        for candidate in sorted(tools.rglob(name + suffix)):
            if candidate.is_file():
                return str(candidate.resolve())
    return None


def discover_toolchain(
    cc: Sequence[str] | str | os.PathLike[str] | None = None,
    qemu: str | os.PathLike[str] | None = None,
) -> RiscVToolchain:
    """Resolve explicit paths, workspace portable tools, or executables on PATH.

    ``cc`` accepts an executable path or a prefix such as ``[zig, "cc"]``.
    Environment overrides are SCRATCHV_CC and SCRATCHV_QEMU; the
    former is an executable path, not a shell command.
    """
    if cc is None:
        cc = (os.environ.get("SCRATCHV_CC") or os.environ.get("SCRATCHV_RISCV_CC") or _workspace_tool("zig")
              or shutil.which("zig") or shutil.which("clang"))
    if cc is None:
        raise FileNotFoundError("Install portable Zig or provide a RISC-V clang compiler path")
    command = [os.fspath(cc)] if isinstance(cc, (str, os.PathLike)) else list(cc)
    if not command:
        raise ValueError("The compiler command must not be empty")
    command[0] = _resolve_executable(command[0])
    if Path(command[0]).stem.lower() == "zig" and len(command) == 1:
        command.append("cc")
    qemu = (qemu or os.environ.get("SCRATCHV_QEMU") or os.environ.get("SCRATCHV_QEMU_RISCV64")
            or _workspace_tool("qemu-system-riscv64") or shutil.which("qemu-system-riscv64"))
    if qemu is None:
        raise FileNotFoundError("qemu-system-riscv64 is required for real RISC-V execution")
    return RiscVToolchain(tuple(command), _resolve_executable(qemu))


def _spec_layout(spec: Any) -> tuple[tuple[int, ...], np.dtype, int]:
    shape = tuple(int(dimension) for dimension in spec.shape)
    if any(dimension < 0 for dimension in shape):
        raise ValueError(f"Tensor {spec.name!r} requires a fixed non-negative shape")
    dtype = np.dtype(spec.numpy_dtype)
    if dtype.kind not in "fiub" or dtype.itemsize not in (1, 2, 4, 8):
        raise ValueError(f"Unsupported binary tensor dtype: {dtype}")
    nbytes = math.prod(shape) * dtype.itemsize
    if nbytes != int(spec.nbytes):
        raise ValueError(f"Inconsistent byte count for tensor {spec.name!r}")
    return shape, dtype, nbytes


def input_layout(specs: Sequence[Any]) -> tuple[tuple[int, ...], int]:
    """Return aligned offsets and payload size in the fixed loader region."""
    offsets, length = [], 0
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Input tensor names must be unique")
    for spec in specs:
        length = (length + 15) & ~15
        offsets.append(length)
        length += _spec_layout(spec)[2]
    if length > INPUT_CAPACITY:
        raise ValueError("Input payload exceeds the reserved 64 MiB QEMU loader region")
    return tuple(offsets), length


def pack_inputs(specs: Sequence[Any], inputs: Mapping[str, np.ndarray]) -> bytes:
    """Serialize exact-shape, exact-dtype arrays without silently casting IDs."""
    if set(inputs) != {spec.name for spec in specs}:
        raise ValueError("Input names must exactly match the generated tensor ABI")
    offsets, length = input_layout(specs)
    payload = bytearray(length)
    for spec, offset in zip(specs, offsets):
        shape, dtype, nbytes = _spec_layout(spec)
        array = np.asarray(inputs[spec.name])
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"Input {spec.name!r} must be {shape}/{dtype}; "
                             f"received {array.shape}/{array.dtype}")
        raw = array.astype(dtype.newbyteorder("<"), copy=False).tobytes(order="C")
        payload[offset:offset + nbytes] = raw
    return bytes(payload)


def _checksum(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def decode_tensor_frame(data: bytes, output: Any) -> np.ndarray:
    """Reject malformed, truncated, stale, or unsuccessful guest output."""
    if len(data) < FRAME_HEADER.size + 12:
        raise ValueError("QEMU tensor output is truncated or missing")
    magic, status, payload_size = FRAME_HEADER.unpack_from(data)
    if magic != FRAME_MAGIC:
        raise ValueError("QEMU tensor output has invalid protocol magic")
    if status:
        raise RuntimeError(f"RISC-V guest reported failure status {status}")
    shape, dtype, expected_size = _spec_layout(output)
    if payload_size != expected_size:
        raise ValueError(f"Guest output length {payload_size} != expected {expected_size}")
    expected_length = FRAME_HEADER.size + expected_size + 12
    if len(data) != expected_length or data[-8:] != FRAME_END:
        raise ValueError("QEMU tensor output is truncated or has trailing/unframed bytes")
    payload = data[FRAME_HEADER.size:FRAME_HEADER.size + expected_size]
    expected_checksum, = struct.unpack_from("<I", data, FRAME_HEADER.size + expected_size)
    if _checksum(payload) != expected_checksum:
        raise ValueError("QEMU tensor output checksum mismatch")
    result = np.frombuffer(payload, dtype=dtype.newbyteorder("<")).reshape(shape)
    return result.astype(dtype, copy=True)


STARTUP_ASSEMBLY = r"""
.section .text.start,"ax"
.global sv_boot
sv_boot:
    csrr t0, mhartid
    bnez t0, .Lpark
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop
    la sp, __stack_top
    csrw mie, zero
    la t0, sv_trap
    csrw mtvec, t0
    li t0, 0x6000
    csrs mstatus, t0
    csrw fcsr, zero
    la t0, __bss_start
    la t1, __bss_end
.Lzero:
    bgeu t0, t1, .Lmain
    sd zero, 0(t0)
    addi t0, t0, 8
    j .Lzero
.Lmain:
    call sv_main
.Lpark:
    wfi
    j .Lpark
.balign 4
sv_trap:
    csrr a0, mcause
    call sv_trap_report
    j .Lpark
"""


LINKER_SCRIPT = r"""
OUTPUT_ARCH(riscv)
ENTRY(sv_boot)
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 0x1c000000 }
SECTIONS {
    . = ORIGIN(RAM);
    .text : { KEEP(*(.text.start)) *(.text .text.*) } > RAM
    .rodata : ALIGN(16) { *(.rodata .rodata.* .srodata .srodata.*) } > RAM
    .data : ALIGN(16) {
        PROVIDE(__global_pointer$ = . + 0x800);
        *(.data .data.* .sdata .sdata.*)
    } > RAM
    .bss (NOLOAD) : ALIGN(16) {
        __bss_start = .;
        *(.bss .bss.* .sbss .sbss.* COMMON)
        . = ALIGN(16);
        __bss_end = .;
    } > RAM
    __stack_top = ORIGIN(RAM) + LENGTH(RAM);
    ASSERT(__bss_end <= __stack_top - 0x100000, "Model overlaps 1 MiB guest stack")
    /DISCARD/ : { *(.comment .eh_frame .eh_frame_hdr .note .note.*) }
}
"""


def _guest_harness(artifact: Any) -> str:
    offsets, _ = input_layout(artifact.inputs)
    _, _, output_bytes = _spec_layout(artifact.output)
    function = artifact.function_name
    if not function.isidentifier() or not function.isascii():
        raise ValueError("The generated C entry point must be an ASCII identifier")
    addresses = ",".join(f"(const void *)0x{INPUT_BASE + offset:x}UL" for offset in offsets)
    return r"""
typedef unsigned long usize;
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;
void *memcpy(void *dst, const void *src, usize n) {
    u8 *d = dst; const u8 *s = src;
    for (usize i=0; i<n; ++i) d[i] = s[i];
    return dst;
}
void *memset(void *dst, int value, usize n) {
    u8 *d = dst;
    for (usize i=0; i<n; ++i) d[i] = (u8)value;
    return dst;
}
static void uart_byte(u8 byte) {
    volatile u8 *uart = (volatile u8 *)0x10000000UL;
    while (!(uart[5] & 0x20)) {}
    uart[0] = byte;
}
static void uart_data(const u8 *data, usize size) {
    for (usize i=0; i<size; ++i) uart_byte(data[i]);
}
static void uart_u32(u32 value) {
    for (int i=0; i<4; ++i) uart_byte((u8)(value >> (8*i)));
}
static void uart_u64(u64 value) {
    for (int i=0; i<8; ++i) uart_byte((u8)(value >> (8*i)));
}
static void finish(u32 status) {
    *(volatile u32 *)0x100000UL = status ? 0x13333 : 0x5555;
    for (;;) __asm__ volatile("wfi");
}
static void output_frame(u32 status, const u8 *data, usize n) {
    uart_data((const u8 *)"SVTENS01", 8);
    uart_u32(status);
    uart_u64(n);
    u32 checksum = 2166136261U;
    for (usize i=0; i<n; ++i) {
        checksum = (checksum ^ data[i]) * 16777619U;
        uart_byte(data[i]);
    }
    uart_u32(checksum);
    uart_data((const u8 *)"SVEND001", 8);
    finish(status);
}
void sv_trap_report(usize cause) { output_frame(0x100U + (u32)cause, (const u8 *)0, 0); }
""" + f"""
extern int {function}(const void *const inputs[], void *output);
static u8 result[{max(output_bytes, 1)}] __attribute__((aligned(16)));
void sv_main(void) {{
    const void *const inputs[{max(len(offsets), 1)}] = {{{addresses or '0'}}};
    int status = {function}(inputs, result);
    output_frame((u32)status, result, status ? 0 : {output_bytes}UL);
}}
"""


def validate_riscv_elf(path: Path) -> None:
    data = path.read_bytes()[:64]
    if len(data) < 64 or data[:7] != b"\x7fELF\x02\x01\x01":
        raise ValueError("Compiler did not produce a 64-bit little-endian ELF")
    if struct.unpack_from("<H", data, 18)[0] != 243:
        raise ValueError("Compiler output is not a RISC-V ELF")
    entry, = struct.unpack_from("<Q", data, 24)
    if not RAM_BASE <= entry < INPUT_BASE:
        raise ValueError("RISC-V ELF entry point is outside the guest program region")
    flags, = struct.unpack_from("<I", data, 48)
    if flags & 6 != 4:
        raise ValueError("RISC-V ELF must use the LP64D floating-point ABI")


def build_riscv_tensor(
    artifact: Any, build_dir: str | os.PathLike[str], toolchain: RiscVToolchain,
    *, extra_sources: Sequence[str | os.PathLike[str]] = (),
    extra_cflags: Sequence[str] = (), extra_ldflags: Sequence[str] = (),
    timeout: float = 180.0,
) -> RiscVTensorExecutable:
    """Compile one ELF, reusable for all matching inputs through raw loading."""
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    elf = build_dir / "model.elf"
    if elf.exists():
        raise FileExistsError(f"Refusing to overwrite an existing executable: {elf}")
    harness = _guest_harness(artifact)
    for filename, source in (("model.c", artifact.source), ("guest.c", harness),
                             ("start.S", STARTUP_ASSEMBLY), ("link.ld", LINKER_SCRIPT)):
        (build_dir / filename).write_text(source, encoding="utf-8")
    # Zig 0.14 ignores -nostartfiles. An explicit sv_boot entry plus section
    # GC removes the unused CRT entry; the ELF never starts a Linux process.
    # PIC also applies to Zig's automatically built musl, whose default medlow
    # constants cannot address QEMU virt RAM at 0x80000000 on RV64.
    target = (["-target", "riscv64-linux-musl", "-static"]
              if toolchain.is_zig else
              ["--target=riscv64-unknown-elf", "-fuse-ld=lld", "-nostdlib"])
    architecture = (["-mcpu=generic_rv64+m+a+f+d+c+zicsr+zifencei"]
                    if toolchain.is_zig else ["-march=rv64gc"])
    command = [*toolchain.cc, *target, "-O2", "-std=c11", *architecture, "-mabi=lp64d",
               "-mcmodel=medany", "-msmall-data-limit=0", "-mno-relax", "-ffreestanding", "-fno-builtin",
               "-fno-stack-protector", "-fno-pie",
               *(["-fPIC"] if toolchain.is_zig else ["-fno-pic"]), "-fno-fast-math",
               "-ffp-contract=off", "-fno-strict-aliasing", "-ffunction-sections", "-fdata-sections",
               *getattr(artifact, "compile_flags", ()), *extra_cflags,
               "model.c", "guest.c", "start.S", *[str(Path(p).resolve()) for p in extra_sources],
               "-Wl,-T,link.ld", "-Wl,-e,sv_boot", "-Wl,--gc-sections", "-Wl,--build-id=none",
               *extra_ldflags, *(["-lm"] if toolchain.is_zig else []), "-o", "model.elf"]
    started = time.perf_counter()
    environment = os.environ.copy()
    if toolchain.is_zig:
        shared_cache = Path(__file__).resolve().parents[2] / "output" / ".riscv-zig-cache"
        environment["ZIG_GLOBAL_CACHE_DIR"] = str(
            Path(os.environ.get("SCRATCHV_ZIG_CACHE", os.environ.get("ZIG_GLOBAL_CACHE_DIR", shared_cache))).resolve()
        )
        environment["ZIG_LOCAL_CACHE_DIR"] = str(build_dir / ".zig-cache")
    temporary = build_dir / "tmp"
    temporary.mkdir(exist_ok=True)
    environment.update(TMP=str(temporary), TEMP=str(temporary), TMPDIR=str(temporary))
    try:
        process = _run_process(command, cwd=build_dir, timeout=timeout, env=environment)
    except subprocess.TimeoutExpired as exc:
        (build_dir / "compiler.stdout").write_bytes(exc.stdout or b"")
        (build_dir / "compiler.stderr").write_bytes(exc.stderr or b"")
        raise TimeoutError(f"RISC-V compilation exceeded {timeout} seconds") from exc
    elapsed = time.perf_counter() - started
    (build_dir / "compiler.stdout").write_bytes(process.stdout)
    (build_dir / "compiler.stderr").write_bytes(process.stderr)
    if process.returncode:
        message = process.stderr.decode("utf-8", errors="replace")[-12000:]
        raise RuntimeError(f"RISC-V tensor compilation failed ({process.returncode}):\n{message}")
    validate_riscv_elf(elf)
    return RiscVTensorExecutable(elf, tuple(artifact.inputs), artifact.output, toolchain,
                                tuple(command), elapsed, int(artifact.workspace_bytes),
                                hashlib.sha256(elf.read_bytes()).hexdigest(), toolchain_versions(toolchain))


def run_riscv_tensor(
    executable: RiscVTensorExecutable, inputs: Mapping[str, np.ndarray],
    run_dir: str | os.PathLike[str], *, timeout: float = 120.0,
) -> RiscVTensorRun:
    """Execute real RV64 instructions and return the guest's exact tensor bytes."""
    if hashlib.sha256(executable.elf_path.read_bytes()).hexdigest() != executable.elf_sha256:
        raise ValueError("RISC-V ELF changed after it was built; refusing mismatched execution")
    payload = pack_inputs(executable.inputs, inputs)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    uart_path = run_dir / "uart.bin"
    if uart_path.exists():
        raise FileExistsError(f"Refusing to reuse stale QEMU output: {uart_path}")
    (run_dir / "inputs.bin").write_bytes(payload or b"\0")
    command = [executable.toolchain.qemu, "-machine", "virt", "-cpu", "rv64", "-accel", "tcg",
               "-smp", "1", "-m", str(MEMORY_MIB), "-bios", "none", "-display", "none",
               "-monitor", "none", "-serial", "file:uart.bin", "-no-reboot",
               "-kernel", str(executable.elf_path), "-device",
               f"loader,file=inputs.bin,addr=0x{INPUT_BASE:x},force-raw=on"]
    stdout_path, stderr_path = run_dir / "qemu.stdout", run_dir / "qemu.stderr"
    started = time.perf_counter()
    try:
        process = _run_process(command, cwd=run_dir, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_bytes(exc.stdout or b"")
        stderr_path.write_bytes(exc.stderr or b"")
        raise TimeoutError(f"RISC-V QEMU execution exceeded {timeout} seconds") from exc
    elapsed = time.perf_counter() - started
    stdout_path.write_bytes(process.stdout)
    stderr_path.write_bytes(process.stderr)
    if process.returncode:
        diagnostic = process.stderr.decode("utf-8", errors="replace")[-4000:]
        if uart_path.exists():
            # Prefer a structured guest error (e.g. an illegal instruction) to
            # an unexplained QEMU exit code when the UART frame is available.
            decode_tensor_frame(uart_path.read_bytes(), executable.output)
        raise RuntimeError(f"QEMU failed ({process.returncode}): {diagnostic}")
    if not uart_path.is_file():
        raise RuntimeError("QEMU exited without producing UART tensor output")
    result = decode_tensor_frame(uart_path.read_bytes(), executable.output)
    return RiscVTensorRun(result, elapsed, tuple(command), uart_path, stdout_path, stderr_path)
