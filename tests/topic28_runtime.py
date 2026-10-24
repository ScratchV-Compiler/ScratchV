"""Real assembler/ELF/QEMU oracles for Topic 28 review regressions.

These helpers do not use the project's RV32IM-only encoder or a Python FP
implementation as an execution oracle. Missing execution tools skip locally,
but fail when SCRATCHV_REQUIRE_RISCV_EXECUTION=1 (the existing CI contract).
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import struct
import subprocess

import pytest

from scratchv.ir.types import DataType as T


def _required_tool(name: str) -> str:
    tool = shutil.which(name)
    if tool is None:
        pytest.fail(f"Topic 28 cross-assembly requires {name}")
    return tool


def _qemu() -> str:
    tool = shutil.which("qemu-riscv32")
    # Reuse the review's locally extracted package, without a system install.
    local = Path("/tmp/topic28-review-tools/extracted/usr/bin/qemu-riscv32")
    if tool is None and local.is_file() and os.access(local, os.X_OK):
        tool = str(local)
    if tool is None:
        if os.environ.get("SCRATCHV_REQUIRE_RISCV_EXECUTION") == "1":
            pytest.fail("required execution tool missing: qemu-riscv32")
        pytest.skip("binary execution requires qemu-riscv32 (not found)")
    return tool


def execute(assembly, harness, tmp_path, isa="rv32imfd", abi="ilp32d", *, separate=False):
    """Never patch product assembly; optionally compile caller independently."""
    qemu = _qemu()
    clang, linker = _required_tool("clang"), _required_tool("ld.lld")
    source = tmp_path / "product.s"
    source.write_text(assembly if separate else assembly + "\n" + harness)
    inputs = [str(source)]
    if separate:
        obj = tmp_path / "product.o"
        assembled = subprocess.run([clang, "--target=riscv32", f"-march={isa}",
            f"-mabi={abi}", "-c", str(source), "-o", str(obj)],
            capture_output=True, text=True, timeout=20)
        assert assembled.returncode == 0, assembled.stderr
        caller = tmp_path / "caller.s"
        caller.write_text(harness)
        inputs = [str(obj), str(caller)]
    binary = tmp_path / "execution"
    linked = subprocess.run([clang, "--target=riscv32", f"-march={isa}", f"-mabi={abi}",
        "-nostdlib", "-static", f"-fuse-ld={linker}", "-Wl,-e,_start", *inputs,
        "-o", str(binary)], capture_output=True, text=True, timeout=20)
    assert linked.returncode == 0, linked.stderr
    run = subprocess.run([qemu, str(binary)], capture_output=True, timeout=2)
    assert run.returncode == 0, (run.returncode, run.stderr, run.stdout)
    return run.stdout


def caller_start(argument=None):
    # tp is ABI-reserved, so unlike a temporary it survives the callee.
    return (".text\n.globl _start\n_start:\naddi sp,sp,-32\nmv tp,sp\n"
        + "".join(f"li s{i},{0x123400 + i}\n" for i in range(12))
        + (f"li a0,{argument}\n" if argument is not None else ""))


def check_caller():
    return ("bne sp,tp,.Lreview_bad_abi\n"
        + "".join(f"li t0,{0x123400 + i}\nbne s{i},t0,.Lreview_bad_abi\n" for i in range(12)))


def write_exit(size=12):
    return (f"li a0,1\nmv a1,sp\nli a2,{size}\nli a7,64\necall\n"
        "li a0,0\nli a7,93\necall\n.Lreview_bad_abi:\nli a0,101\nli a7,93\necall\n")


def observe(assembly, tmp_path, dtype=T.INT32, *, argument=None, function="main",
            isa="rv32imfd", abi="ilp32d", separate=False):
    store = {T.INT32: "sw a0,0(sp)\nsw zero,4(sp)\n",
             T.FLOAT32: "fsw fa0,0(sp)\nsw zero,4(sp)\n",
             T.FLOAT64: "fsd fa0,0(sp)\n"}[dtype]
    harness = (caller_start(argument) + ("csrw fflags,zero\n" if "f" in isa else "")
        + f"call {function}\n" + check_caller() + store
        + ("csrr t0,fflags\n" if "f" in isa else "li t0,0\n")
        + "sw t0,8(sp)\n" + write_exit())
    data = execute(assembly, harness, tmp_path, isa, abi, separate=separate)
    assert len(data) == 12
    width = 8 if dtype is T.FLOAT64 else 4
    return int.from_bytes(data[:width], "little", signed=dtype is T.INT32), int.from_bytes(data[8:], "little")


def observe_abi(assembly, path, dtype, function="main", isa="rv32imfd", abi="ilp32d"):
    """Independent caller checks original sp/ra/GPR/fs before storing results.

    Product and caller are separate objects. On D, store the entire fa0 even
    for f32 so a successful low-word result cannot hide incorrect NaN-boxing.
    """
    fp, double = "f" in isa, "d" in isa
    harness = ".text\n.globl _start\n_start:\naddi sp,sp,-64\nmv tp,sp\n"
    harness += "".join(f"li s{i},{0x123400 + i}\n" for i in range(12))
    if fp:
        harness += "la t0,.Ltopic28_seed\n"
        harness += "".join(f"{'fld' if double else 'flw'} fs{i},0(t0)\n" for i in range(12))
        harness += "csrw fflags,zero\n"
    harness += (f"call {function}\n.Ltopic28_after:\nbne sp,tp,.Ltopic28_bad\n"
                "la t0,.Ltopic28_after\nbne ra,t0,.Ltopic28_bad\n")
    harness += "".join(f"li t0,{0x123400 + i}\nbne s{i},t0,.Ltopic28_bad\n" for i in range(12))
    if fp:
        for i in range(12):
            harness += (f"{'fsd' if double else 'fsw'} fs{i},16(sp)\n"
                        "lw t0,16(sp)\nli t1,0x89abcdef\nbne t0,t1,.Ltopic28_bad\n")
            if double:
                harness += "lw t0,20(sp)\nli t1,0x3ff12345\nbne t0,t1,.Ltopic28_bad\n"
    if dtype is T.INT32:
        harness += "sw a0,0(sp)\nsw zero,4(sp)\n"
    elif double:
        harness += "fsd fa0,0(sp)\n"
    else:
        harness += "fsw fa0,0(sp)\nsw zero,4(sp)\n"
    harness += ("csrr t0,fflags\n" if fp else "li t0,0\n")
    harness += ("sw t0,8(sp)\nli a0,1\nmv a1,sp\nli a2,12\nli a7,64\necall\n"
                "li a0,0\nli a7,93\necall\n.Ltopic28_bad:\nli a0,102\nli a7,93\necall\n")
    if fp:
        harness += ".section .rodata\n.p2align 3\n.Ltopic28_seed:\n.quad 0x3ff1234589abcdef\n"
    output = execute(assembly, harness, path, isa, abi, separate=True)
    assert len(output) == 12
    if dtype is T.FLOAT32 and double:
        assert output[4:8] == b"\xff" * 4, "f32 return must be NaN-boxed on D"
    width = 8 if dtype is T.FLOAT64 else 4
    return int.from_bytes(output[:width], "little", signed=dtype is T.INT32), int.from_bytes(output[8:], "little")


def _execute(assembly: str, harness: str, tmp_path: Path,
             isa: str, abi: str) -> bytes:
    if shutil.which("ld.lld") is None:
        if os.environ.get("SCRATCHV_REQUIRE_RISCV_EXECUTION") == "1":
            pytest.fail("required execution linker missing: ld.lld")
        pytest.skip("ELF execution requires ld.lld")
    output = execute(assembly, harness, tmp_path, isa, abi)
    assert len(output) == 12, output
    return output


def _store_bits(value: float, width: int) -> str:
    bits = int.from_bytes(struct.pack("<f" if width == 32 else "<d", value), "little")
    result = f"li t0,{bits & 0xffffffff}\nsw t0,0(sp)\n"
    if width == 64:
        result += f"li t0,{bits >> 32}\nsw t0,4(sp)\n"
    return result


def execute_fp(assembly: str, tmp_path: Path, width: int,
               argument: float | None = None, *,
               function: str = "main") -> tuple[int, int]:
    """Return raw return bits and fflags; seed registers to expose undefined uses."""
    assert width in {32, 64}
    load, store = ("flw", "fsw") if width == 32 else ("fld", "fsd")
    seeds = _store_bits(9.0, width) + "".join(
        f"{load} ft{index},0(sp)\n" for index in range(12)
    ) + f"{load} fa0,0(sp)\n"
    parameter = "" if argument is None else _store_bits(argument, width) + f"{load} fa0,0(sp)\n"
    upper = "sw zero,4(sp)\n" if width == 32 else ""
    harness = (
        ".text\n.globl _start\n_start:\naddi sp,sp,-16\n" + seeds + parameter
        + f"csrw fflags,zero\ncall {function}\n" + f"{store} fa0,0(sp)\n" + upper
        + "csrr t0,fflags\nsw t0,8(sp)\nli a0,1\nmv a1,sp\nli a2,12\nli a7,64\necall\n"
        + "li a0,0\nli a7,93\necall\n"
    )
    output = _execute(assembly, harness, tmp_path, "rv32imfd", "ilp32d")
    return int.from_bytes(output[:width // 8], "little"), int.from_bytes(output[8:12], "little")


def execute_i32(assembly: str, tmp_path: Path, *,
                argument: int | None = None, function: str = "main") -> int:
    """Execute a RV32IM scalar function and return signed a0."""
    setup = "addi sp,sp,-16\n"
    if argument is not None:
        # Seed the pool and nearby frame to make missing ABI definitions
        # observable, rather than accidentally inheriting the right value.
        setup = (
            "addi sp,sp,-2048\naddi t0,sp,-1024\nli t1,512\n"
            ".Lreview_clear:\nsw zero,0(t0)\naddi t0,t0,4\n"
            "addi t1,t1,-1\nbnez t1,.Lreview_clear\n"
            + "".join(f"li {reg},9\n" for reg in (
                *(f"t{i}" for i in range(7)), *(f"s{i}" for i in range(12)),
            )) + f"li a0,{argument}\n"
        )
    harness = (
        ".text\n.globl _start\n_start:\n" + setup + f"call {function}\n"
        "sw a0,0(sp)\nsw zero,4(sp)\nsw zero,8(sp)\n"
        "li a0,1\nmv a1,sp\nli a2,12\nli a7,64\necall\n"
        "li a0,0\nli a7,93\necall\n"
    )
    return int.from_bytes(_execute(assembly, harness, tmp_path, "rv32im", "ilp32")[:4], "little", signed=True)
