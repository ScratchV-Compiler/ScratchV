#!/usr/bin/env python3
"""RV32 full benchmark: ScratchV vs LLVM on a unified RV32 target.

Pipeline:
  1. ONNX -> ScratchV RV32IM binary (Q16.16 fixed point)
  2. ONNX -> LLVM IR -> RV32IMF assembly (float32, via optional llvmlite)
  3. ScratchV binary loaded into TinyFive ProfiledMachine (code + weights +
     deterministic input) and simulated in explicit, budgeted chunks
  4. Honest report (schema ``rv32-bench/2``): every number carries its
     provenance; static counts never masquerade as dynamic ones.

Only measured TinyFive execution fills ``scratchv.dynamic``.  When the
simulator is missing, or a run is truncated by a budget/timeout, the report
says so instead of inventing numbers.

Usage:
    python scratchv/standalone/rv32_bench.py models/graph/cnn.onnx \\
        --full --output-dir benchmark_reports
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import random
import re
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np

PROJ = Path(__file__).resolve().parent.parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scratchv.simulator.tinyfive import ProfiledMachine  # noqa: E402
from scratchv.standalone.onnx_to_riscv_standalone import (  # noqa: E402
    ONNXModel,
    _disasm_one,
)
from scratchv.standalone import bench_report  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Constants and exit codes
# ═══════════════════════════════════════════════════════════════════════════

SCHEMA_VERSION = "rv32-bench/2"
EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_NO_SIMULATOR, EXIT_LAYOUT, EXIT_INCOMPLETE = (
    0, 1, 2, 3, 4, 7,
)

SP_ADDR = 128 * 1024 * 1024
INPUT_ADDR = 160 * 1024 * 1024
OUTPUT_ADDR = 192 * 1024 * 1024
GUARD_BYTES = 1024 * 1024
OUTPUT_GUARD_BYTES = 4096

DEFAULT_MEM_SIZE = 268435456
DEFAULT_TIMEOUT_S = 900.0
DEFAULT_CHUNK_INSTRUCTIONS = 10_000_000
DEFAULT_INPUT_SEED = 42

DATA_OFFSET_RE = re.compile(r"Data offset:\s*0x([0-9A-Fa-f]+)")
WORKSPACE_RE = re.compile(r"Workspace:\s*([\d,]+)\s+bytes")
CODE_SIZE_RE = re.compile(r"Code size:\s*([\d,]+)\s+bytes")

TARGETS = {
    "scratchv": {"isa": "rv32im", "abi": "ilp32", "numeric_format": "q16.16"},
    "llvm": {
        "isa": "rv32imf", "abi": "ilp32", "numeric_format": "float32",
        "triple": "riscv32-unknown-elf", "opt_level": 2,
    },
}

MODEL_FIELDS = (
    "path", "sha256", "bytes", "input_name", "input_shape", "output_name",
    "output_shape", "initializer_count", "weight_bytes",
)

OPS_KEYS = ("total", "load", "store", "mul", "add", "madd", "branch")

RV64_ONLY_MNEMONICS = frozenset({
    "ld", "sd", "lwu", "addw", "subw", "addiw", "sllw", "srlw", "sraw",
    "slliw", "srliw", "sraiw", "mulw", "divw", "divuw", "remw", "remuw",
    "fld", "fsd", "fcvt.l.s", "fcvt.s.l",
})

_NON_INSTRUCTION_PSEUDO = frozenset({"li", "mv", "nop", "ret", "j", "jr"})

_LOAD_OPS = frozenset({"lw", "lh", "lb", "lbu", "lhu", "flw", "flw.s"})
_STORE_OPS = frozenset({"sw", "sh", "sb", "fsw", "fsw.s"})
_MUL_OPS = frozenset({"mul", "mulh", "mulhsu", "mulhu"})
_MADD_OPS = frozenset({
    "fmadd.s", "fmsub.s", "fnmadd.s", "fnmsub.s", "fmul.s", "fadd.s",
    "fsub.s", "fdiv.s",
})
_ARITH_OPS = frozenset({
    "add", "addi", "sub", "slt", "sltu", "slti", "sltiu", "slli", "srli",
    "srai", "and", "andi", "or", "ori", "xor", "xori", "lui", "auipc",
    "div", "divu", "rem", "remu", "sll", "srl", "sra",
})
_BRANCH_OPS = frozenset({
    "beq", "bne", "blt", "bge", "bltu", "bgeu", "jal", "jalr",
})


class LayoutError(RuntimeError):
    """Image or memory layout cannot satisfy the bare-metal ABI."""


class LabelParseError(RuntimeError):
    """The assembly listing and the binary code size disagree."""


class SimulationTimeout(RuntimeError):
    """Wall-clock timeout fired during simulation."""


_timed_out = False


def _alarm_handler(signum, frame):  # pragma: no cover - signal glue
    global _timed_out
    _timed_out = True
    raise SimulationTimeout("simulation wall-clock timeout")


# ═══════════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════════

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def _run_py(args: list[str], timeout: float = 120.0):
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJ) + (os.pathsep + existing if existing else "")
    proc = subprocess.run(
        [sys.executable] + args, capture_output=True, text=True,
        cwd=str(PROJ), timeout=timeout, env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _to_int(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        return int(text.replace(",", "").strip())
    except ValueError:
        return None


def parse_data_offset(stdout: str) -> int | None:
    """Parse ``Data offset: 0x...`` from the compiler stdout."""
    m = DATA_OFFSET_RE.search(stdout)
    return int(m.group(1), 16) if m else None


def parse_workspace_bytes(stdout: str) -> int | None:
    """Parse ``Workspace: N bytes`` from the compiler stdout."""
    m = WORKSPACE_RE.search(stdout)
    return _to_int(m.group(1)) if m else None


def parse_code_bytes(stdout: str) -> int | None:
    """Parse ``Code size: N bytes`` from the compiler stdout."""
    m = CODE_SIZE_RE.search(stdout)
    return _to_int(m.group(1)) if m else None


def _iter_asm_lines(asm_text: str, stop_at_data: bool = False):
    """Yield ``(pc_index, mnemonic, operands)`` for instruction lines."""
    pc = 0
    for raw in asm_text.splitlines():
        line = raw.split("#", 1)[0]
        line = line.split(";", 1)[0]
        line = line.strip()
        if not line:
            continue
        if line.startswith("."):
            if stop_at_data and re.match(r"\.(section|data|rodata|bss)\b", line):
                break
            continue
        if ":" in line:
            head, _, tail = line.partition(":")
            if " " not in head and tail.strip() == "":
                continue
            if " " not in head and tail.strip():
                line = tail.strip()
        tokens = line.replace(",", " ").split()
        if not tokens:
            continue
        yield pc, tokens[0].lower(), tokens[1:]
        pc += 4


def _scan_asm(asm_text: str, stop_at_data: bool = False) -> tuple[int, dict]:
    count = 0
    for _pc, _op, _ops in _iter_asm_lines(asm_text, stop_at_data=stop_at_data):
        count += 1
    return count, static_instruction_mix(asm_text, stop_at_data=stop_at_data)


def static_instruction_mix(asm_text: str, stop_at_data: bool = False) -> dict:
    """Static mnemonic mix; never fed into ``dynamic.ops``."""
    mix = {k: 0 for k in ("load", "store", "mul", "add", "madd", "branch", "other")}
    for _pc, op, _operands in _iter_asm_lines(asm_text, stop_at_data=stop_at_data):
        if op in _LOAD_OPS:
            mix["load"] += 1
        elif op in _STORE_OPS:
            mix["store"] += 1
        elif op in _MUL_OPS:
            mix["mul"] += 1
        elif op in _MADD_OPS:
            mix["madd"] += 1
        elif op in _BRANCH_OPS or op in ("j", "jr", "ret"):
            mix["branch"] += 1
        elif op in _ARITH_OPS or op in ("li", "mv", "nop"):
            mix["add"] += 1
        else:
            mix["other"] += 1
    return {"source": "asm_scan", **mix}


def parse_labels(asm_text: str, expected_code_bytes: int) -> dict[int, str]:
    """Map PC -> label from a compiled ``.s`` listing.

    Every branch/jump target must resolve to a label; a missing target raises
    ``LabelParseError`` instead of silently degrading to a self-jump.
    """
    pc = 0
    labels: dict[int, str] = {}
    instructions: list[tuple[int, str, list[str]]] = []
    for raw in asm_text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("."):
            continue
        if stripped.endswith(":") and "(" not in stripped:
            labels[pc] = stripped[:-1]
            continue
        tokens = stripped.replace(",", " ").split()
        instructions.append((pc, tokens[0].lower(), tokens[1:]))
        pc += 4

    if pc != expected_code_bytes:
        raise LabelParseError(
            f"asm/code-size mismatch: {pc} != {expected_code_bytes}"
        )
    if 0 not in labels:
        raise LabelParseError("missing _start label at pc=0")

    for ipc, op, operands in instructions:
        if op in ("beq", "bne", "blt", "bge", "bltu", "bgeu", "jal", "j"):
            if not operands:
                continue
            target = operands[-1]
            if re.match(r"^[+-]?\d+$", target):
                target_pc = ipc + int(target)
            elif target in labels:
                target_pc = labels[target]
            else:
                raise LabelParseError(
                    f"unresolved branch target {target!r} at pc={ipc}"
                )
            if target_pc not in labels:
                raise LabelParseError(
                    f"branch at pc={ipc} targets pc={target_pc} without a label"
                )
    return labels


def detect_isa_mismatch(asm_text: str) -> list[str]:
    """Return RV64-only mnemonics found in an assembly listing."""
    found = set()
    for _pc, op, _operands in _iter_asm_lines(asm_text, stop_at_data=True):
        if op in RV64_ONLY_MNEMONICS:
            found.add(op)
    return sorted(found)


def _tinyfive_supported_mnemonics() -> frozenset[str]:
    try:
        from tinyfive.machine import dec_dict  # type: ignore
    except Exception:
        return frozenset()
    return frozenset(entry[1] for entry in dec_dict.values())


def check_mnemonics(code_words: list[int]) -> list[str]:
    """Return code words whose mnemonic TinyFive cannot execute."""
    supported = _tinyfive_supported_mnemonics()
    if not supported:
        return []
    unsupported = set()
    for word in code_words:
        text = _disasm_one(word)
        op = text.split(" ", 1)[0].strip().lower()
        if op.startswith(".word"):
            unsupported.add(f".word (0x{word:08x})")
            continue
        if op in supported or op in _NON_INSTRUCTION_PSEUDO:
            continue
        unsupported.add(f"{op} (0x{word:08x})")
    return sorted(unsupported)


def build_input_q16(elements: int, seed: int) -> bytes:
    """Deterministic Q16.16 input identical to the compiler self-check."""
    rng = random.Random(seed)
    values = [int((rng.random() - 0.5) * 0.2 * 65536) for _ in range(elements)]
    return struct.pack(f"<{elements}i", *values)


# ═══════════════════════════════════════════════════════════════════════════
# Model / environment metadata
# ═══════════════════════════════════════════════════════════════════════════

def get_model_info(model_path: str) -> dict:
    """Model identity, shapes and element counts (no hardcoded descriptions)."""
    model = ONNXModel.from_file(model_path)
    input_t = model.inputs[0] if model.inputs else None
    output_t = model.outputs[0] if model.outputs else None

    def _numel(t):
        if t is None:
            return 0
        n = 1
        for d in t.shape:
            n *= int(d)
        return n

    return {
        "path": str(model_path),
        "sha256": sha256_file(model_path),
        "bytes": os.path.getsize(model_path),
        "input_name": input_t.name if input_t else None,
        "input_shape": list(input_t.shape) if input_t else [],
        "output_name": output_t.name if output_t else None,
        "output_shape": list(output_t.shape) if output_t else [],
        "initializer_count": len(model.initializers),
        "weight_bytes": sum(t.size_bytes for t in model.initializers.values()),
        "input_elements": _numel(input_t),
        "output_elements": _numel(output_t),
    }


def get_environment() -> dict:
    def _version(name: str):
        try:
            import importlib.metadata as md
            return md.version(name)
        except Exception:
            pass
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", None)
        except Exception:
            return None

    return {
        "python": "%d.%d.%d" % sys.version_info[:3],
        "numpy": _version("numpy"),
        "tinyfive": _version("tinyfive"),
        "llvmlite": _version("llvmlite"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Layout / image loading
# ═══════════════════════════════════════════════════════════════════════════

def compute_layout(*, data_offset: int, data_size: int, workspace_bytes: int,
                   input_elements: int, output_elements: int,
                   mem_size: int) -> dict:
    """Validate the bare-metal layout and return the addresses."""
    if data_offset <= 0 or data_offset % 4 != 0:
        raise LayoutError(f"data_offset is not a positive 4-byte multiple: {data_offset}")

    halt_addr = _align_up(data_offset + data_size, 16)
    in_bytes = input_elements * 4
    out_bytes = output_elements * 4

    if halt_addr + 4 > mem_size:
        raise LayoutError(
            f"image does not fit: halt_addr=0x{halt_addr:x} + 4 > mem_size={mem_size}"
        )
    min_out = OUTPUT_ADDR + out_bytes + OUTPUT_GUARD_BYTES
    if min_out > mem_size:
        raise LayoutError(
            f"memory_layout_invalid: need mem_size >= {min_out} bytes "
            f"(192MiB output base + {out_bytes} bytes output + "
            f"{OUTPUT_GUARD_BYTES} bytes guard); got {mem_size}"
        )
    if SP_ADDR + workspace_bytes + GUARD_BYTES > INPUT_ADDR:
        raise LayoutError(
            f"workspace collides with the input region: sp=0x{SP_ADDR:x} + "
            f"{workspace_bytes} + {GUARD_BYTES} guard > input=0x{INPUT_ADDR:x}"
        )
    if INPUT_ADDR + in_bytes > OUTPUT_ADDR:
        raise LayoutError(
            f"input region overruns output base: input=0x{INPUT_ADDR:x} + "
            f"{in_bytes} > output=0x{OUTPUT_ADDR:x}"
        )

    return {
        "sp": SP_ADDR,
        "input_addr": INPUT_ADDR,
        "output_addr": OUTPUT_ADDR,
        "halt_addr": halt_addr,
        "mem_size": mem_size,
    }


def load_scratchv_image(binary_path: str, data_offset: int) -> tuple[list[int], bytes]:
    """Split the flat ``.bin`` into code words and weight bytes."""
    binary = Path(binary_path).read_bytes()
    if data_offset is None or data_offset <= 0 or data_offset % 4 != 0:
        raise LayoutError(f"invalid data_offset for image: {data_offset}")
    if data_offset >= len(binary):
        raise LayoutError(
            f"image has no weight section: data_offset={data_offset}, "
            f"binary={len(binary)}"
        )
    code_words = [
        int.from_bytes(binary[i:i + 4], "little")
        for i in range(0, data_offset, 4)
    ]
    return code_words, binary[data_offset:]


# ═══════════════════════════════════════════════════════════════════════════
# Compilation
# ═══════════════════════════════════════════════════════════════════════════

def compile_scratchv(onnx_path: str, output_bin: str, output_asm: str,
                     timeout_s: float = 120.0) -> dict:
    """Compile ONNX -> ScratchV RV32IM binary and parse its layout."""
    t0 = time.perf_counter()
    try:
        rc, stdout, stderr = _run_py([
            "scratchv/standalone/onnx_to_riscv_standalone.py",
            onnx_path, "-o", output_bin, "--asm", output_asm,
        ], timeout=timeout_s)
    except Exception as exc:  # subprocess.TimeoutExpired, OSError
        return {
            "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300],
            "reason": "compiler_invocation_failed", "static_insns": 0,
            "static_source": "asm_scan", "data_offset": None,
            "elapsed_s": time.perf_counter() - t0,
        }

    elapsed = time.perf_counter() - t0
    base = {
        "status": "failed", "static_insns": 0, "static_source": "asm_scan",
        "data_offset": None, "elapsed_s": elapsed,
    }
    if rc != 0:
        base.update({
            "error": (stderr or stdout)[-300:],
            "reason": "compiler_returned_nonzero",
        })
        return base

    data_offset = parse_data_offset(stdout)
    workspace_bytes = parse_workspace_bytes(stdout)
    code_bytes = parse_code_bytes(stdout)
    if data_offset is None or workspace_bytes is None:
        base.update({
            "error": "binary_layout_unparsed",
            "reason": "binary_layout_unparsed",
        })
        return base

    try:
        binary = Path(output_bin).read_bytes()
    except OSError as exc:
        base.update({"error": f"binary_unreadable: {exc}", "reason": "binary_unreadable"})
        return base

    asm_text = Path(output_asm).read_text() if Path(output_asm).exists() else ""
    try:
        parse_labels(asm_text, data_offset)
    except LabelParseError as exc:
        base.update({"error": str(exc), "reason": "label_parse_failed"})
        return base

    static_insns, _mix = _scan_asm(asm_text)
    if code_bytes is not None and code_bytes != data_offset:
        base.update({
            "error": f"code size mismatch: stdout={code_bytes} data_offset={data_offset}",
            "reason": "binary_layout_unparsed",
        })
        return base
    if static_insns != data_offset // 4:
        base.update({
            "error": f"asm/code-size mismatch: {static_insns} != {data_offset // 4}",
            "reason": "label_parse_failed",
        })
        return base

    return {
        "status": "success",
        "binary": os.path.basename(output_bin),
        "binary_bytes": len(binary),
        "binary_sha256": hashlib.sha256(binary).hexdigest(),
        "code_bytes": data_offset,
        "data_offset": data_offset,
        "data_offset_source": "compiler_stdout",
        "data_bytes": len(binary) - data_offset,
        "workspace_bytes": workspace_bytes,
        "static_insns": static_insns,
        "static_source": "asm_scan",
        "elapsed_s": elapsed,
    }


def compile_llvm_rv32(onnx_path: str, output_asm: str, *,
                      triple: str = "riscv32-unknown-elf",
                      opt_level: int = 2) -> dict:
    """Compile ONNX -> LLVM IR -> assembly with an explicit RV32 triple."""
    t0 = time.perf_counter()

    def _skipped(reason: str) -> dict:
        return {
            "status": "skipped", "reason": reason, "isa_detected": None,
            "isa_mismatch": False, "static_insns": 0,
            "static_source": "asm_scan",
            "elapsed_s": time.perf_counter() - t0,
        }

    try:
        from llvmlite import binding  # type: ignore
    except ImportError:
        return _skipped("llvmlite not available")

    ir_path = str(Path(output_asm).with_suffix(".ll"))
    try:
        rc, _stdout, stderr = _run_py([
            "scratchv/standalone/onnx_to_llvm_standalone.py",
            onnx_path, "-o", ir_path, "--opt-level", str(opt_level),
        ], timeout=120.0)
    except Exception as exc:
        return {
            "status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:300],
            "isa_detected": None, "isa_mismatch": False, "static_insns": 0,
            "static_source": "asm_scan", "elapsed_s": time.perf_counter() - t0,
        }
    if rc != 0 or not Path(ir_path).exists():
        return {
            "status": "failed",
            "reason": f"LLVM IR generation failed: {stderr[-200:]}",
            "isa_detected": None, "isa_mismatch": False, "static_insns": 0,
            "static_source": "asm_scan", "elapsed_s": time.perf_counter() - t0,
        }

    try:
        binding.initialize()
        binding.initialize_all_targets()
        binding.initialize_all_asmprinters()
        llmod = binding.parse_assembly(Path(ir_path).read_text())
        llmod.triple = triple
        target = binding.Target.from_triple(triple)
        tm = target.create_target_machine(
            cpu="generic-rv32", features="+m,+f", codemodel="small",
            opt=opt_level,
        )
        llmod.data_layout = str(tm.target_data)
        llmod.verify()
        asm = tm.emit_assembly(llmod)
        Path(output_asm).write_text(asm)
    except Exception as exc:
        return {
            "status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:300],
            "isa_detected": None, "isa_mismatch": False, "static_insns": 0,
            "static_source": "asm_scan", "elapsed_s": time.perf_counter() - t0,
        }

    mismatches = detect_isa_mismatch(asm)
    static_insns, _mix = _scan_asm(asm, stop_at_data=True)
    return {
        "status": "success",
        "reason": (
            f"rv64 mnemonics detected: {', '.join(mismatches)}"
            if mismatches else None
        ),
        "isa_detected": "riscv64" if mismatches else "rv32",
        "isa_mismatch": bool(mismatches),
        "static_insns": static_insns,
        "static_source": "asm_scan",
        "elapsed_s": time.perf_counter() - t0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════

# TinyFive 1.0.0 compatibility shims, applied to machine *instances* only.
#
# 1. ``machine.LW``/``LH`` build words by shifting ``numpy.uint8`` scalars,
#    which overflows at eight bits under NumPy 2.x and silently drops bytes
#    1..3 (the adapter already patches ``read_i32`` for instruction fetch but
#    not these data loads).
# 2. ``machine.exe(start, end, instructions)`` ignores ``instructions``
#    whenever ``end`` is given and has no way to stop at both.  Without an
#    instruction budget tinyfive would keep decoding zeroed memory past the
#    halt address; every decode increments ``ops['total']``, which silently
#    inflates the measured dynamic count.  The shim below stops at whichever
#    comes first: the halt address or the budget.
#
# Neither patch modifies ``scratchv/simulator/tinyfive.py``.

def _tinyfive_lw_compat(s, rd, imm, rs1):
    addr = int(s.x[rs1]) + int(imm)
    if addr < 0 or addr + 4 > len(s.mem):
        raise IndexError(f"tinyfive lw out of bounds: {addr}")
    s.x[rd] = int.from_bytes(bytes(s.mem[addr:addr + 4]), "little", signed=True)
    s.ipc()


def _tinyfive_lh_compat(s, rd, imm, rs1):
    addr = int(s.x[rs1]) + int(imm)
    if addr < 0 or addr + 2 > len(s.mem):
        raise IndexError(f"tinyfive lh out of bounds: {addr}")
    s.x[rd] = int.from_bytes(bytes(s.mem[addr:addr + 2]), "little", signed=True)
    s.ipc()


def _install_tinyfive_compat(machine, halt_addr: int) -> None:
    """Bind NumPy-safe loads and a halt-aware bounded ``exe`` to *machine*."""
    raw = getattr(machine, "_machine", None)
    if raw is None:
        return
    raw.LW = MethodType(_tinyfive_lw_compat, raw)
    raw.LH = MethodType(_tinyfive_lh_compat, raw)

    def exe_with_halt(s, start, end=None, instructions=0):
        target = halt_addr if end is None else end
        s.pc = s.look_up_label(start)
        end_pc = s.look_up_label(target)
        executed = 0
        while s.pc != end_pc:
            if instructions and executed >= instructions:
                break
            s.dec(np.binary_repr(s.u(s.read_i32(s.pc)), 32))
            executed += 1

    raw.exe = MethodType(exe_with_halt, raw)


def _read_pc(machine) -> int:
    """Read PC without tripping over TinyFive's scalar-PC after JALR."""
    try:
        return int(machine.pc)
    except Exception:
        pass
    raw = getattr(getattr(machine, "_machine", None), "pc", None)
    if raw is None:
        return 0
    if getattr(raw, "ndim", 0):
        return int(raw[0])
    return int(raw)


def _read_register_usage(machine) -> tuple[int, int, int]:
    raw = getattr(machine, "_machine", None)
    if raw is None:
        return 0, 0, 0
    x_usage = getattr(raw, "x_usage", None)
    f_usage = getattr(raw, "f_usage", None)
    x_used = int((x_usage > 0).sum()) if x_usage is not None else 0
    x_total = int(x_usage.sum()) if x_usage is not None else 0
    f_used = int((f_usage > 0).sum()) if f_usage is not None else 0
    return x_used, x_total, f_used


def _unavailable_dynamic(*, reason: str, mem_size: int, timeout_s: float,
                         input_seed: int, input_elements: int,
                         halt_addr: int) -> dict:
    return {
        "source": "unavailable",
        "simulator": "tinyfive",
        "simulator_version": get_environment().get("tinyfive"),
        "completion": "not_run",
        "reason": reason,
        "limit": None,
        "executed": None,
        "timeout_s": float(timeout_s),
        "elapsed_s": 0.0,
        "memory_size_bytes": mem_size,
        "input_seed": input_seed,
        "input_elements": input_elements,
        "halt_addr": halt_addr,
        "ops": None,
        "x_registers_used": None,
        "x_usage_total": None,
        "f_registers_used": None,
        "per_label": None,
        "per_label_note": "tinyfive exe() exposes no per-PC trace",
        "last_error": None,
    }


def _unavailable_output(elements: int, completion: str) -> dict:
    """Output block for runs that produced no (or no complete) values."""
    return {
        "addr": OUTPUT_ADDR,
        "elements": elements,
        "raw_hex": None,
        "q16_16": None,
        "completion": completion,
        "partial": completion != "halted",
    }


def _read_output(machine, addr: int, elements: int) -> dict:
    values = []
    if machine.available and elements > 0:
        for i in range(elements):
            values.append(int(machine.read_mem_i32(addr + i * 4)) & 0xFFFFFFFF)
    raw_hex = "0x" + "".join(f"{v:08x}" for v in values) if values else ""
    q16 = [
        (v if v < 0x80000000 else v - 0x100000000) / 65536.0 for v in values
    ]
    return {
        "addr": addr,
        "elements": elements,
        "raw_hex": raw_hex,
        "q16_16": q16,
    }


def run_simulation(*, asm_path: str, binary_path: str, data_offset: int,
                   workspace_bytes: int, input_elements: int,
                   output_elements: int, max_instructions: int = 0,
                   mem_size: int = DEFAULT_MEM_SIZE,
                   timeout_s: float = DEFAULT_TIMEOUT_S,
                   chunk_instructions: int = DEFAULT_CHUNK_INSTRUCTIONS,
                   input_seed: int = DEFAULT_INPUT_SEED) -> dict:
    """Execute the compiled binary with TinyFive and report honest counters."""
    global _timed_out

    t0 = time.perf_counter()
    limit = None if max_instructions in (0, None) else int(max_instructions)
    if limit is not None and limit < 0:
        raise ValueError(f"max_instructions must be >= 0, got {max_instructions}")
    if chunk_instructions <= 0:
        raise ValueError(f"chunk_instructions must be > 0, got {chunk_instructions}")

    binary = Path(binary_path).read_bytes()
    data_size = len(binary) - data_offset
    layout = compute_layout(
        data_offset=data_offset, data_size=data_size,
        workspace_bytes=workspace_bytes, input_elements=input_elements,
        output_elements=output_elements, mem_size=mem_size,
    )
    halt_addr = layout["halt_addr"]

    m = ProfiledMachine(mem_size=mem_size)
    if not m.available:
        return {
            "dynamic": _unavailable_dynamic(
                reason="tinyfive not installed (ProfiledMachine.available=False)",
                mem_size=mem_size, timeout_s=timeout_s, input_seed=input_seed,
                input_elements=input_elements, halt_addr=halt_addr,
            ),
            "output": _unavailable_output(output_elements, "not_run"),
        }
    _install_tinyfive_compat(m, halt_addr)

    try:
        code_words, weights = load_scratchv_image(binary_path, data_offset)
    except LayoutError as exc:
        return {
            "dynamic": _unavailable_dynamic(
                reason=f"image_load_failed: {exc}", mem_size=mem_size,
                timeout_s=timeout_s, input_seed=input_seed,
                input_elements=input_elements, halt_addr=halt_addr,
            ),
            "output": _unavailable_output(output_elements, "not_run"),
        }

    unsupported = check_mnemonics(code_words)
    if unsupported:
        dyn = _unavailable_dynamic(
            reason="unsupported_mnemonics: " + ", ".join(unsupported[:8]),
            mem_size=mem_size, timeout_s=timeout_s, input_seed=input_seed,
            input_elements=input_elements, halt_addr=halt_addr,
        )
        return {
            "dynamic": dyn,
            "output": _unavailable_output(output_elements, "not_run"),
        }

    m.load_binary(code_words, origin=0)
    m.load_data(weights, data_offset)
    input_blob = build_input_q16(input_elements, input_seed)
    if input_blob:
        m.load_data(input_blob, INPUT_ADDR)
    m.set_reg(2, layout["sp"])
    m.set_reg(10, layout["input_addr"])
    m.set_reg(11, layout["output_addr"])
    m.set_reg(1, halt_addr)

    _timed_out = False
    old_handler = None
    use_alarm = hasattr(signal, "SIGALRM") and timeout_s is not None and timeout_s > 0
    if use_alarm:
        try:
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        except (ValueError, OSError):  # not in main thread
            use_alarm = False
    deadline = (
        time.monotonic() + float(timeout_s)
        if timeout_s and timeout_s > 0 else float("inf")
    )

    executed = 0
    completion = "error"
    error_msg: str | None = None
    devnull = open(os.devnull, "w")
    try:
        while True:
            pc = _read_pc(m)
            if limit is not None and executed >= limit:
                completion = "budget_exhausted"
                break
            if pc == halt_addr:
                completion = "halted"
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                completion = "timeout"
                break
            chunk = chunk_instructions
            if limit is not None:
                chunk = min(chunk, limit - executed)
            if chunk <= 0:
                completion = "budget_exhausted"
                break

            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, max(remaining, 1e-6))
            try:
                with contextlib.redirect_stdout(devnull):
                    m.run(instructions=chunk, start=pc, strict=True)
            except RuntimeError as exc:
                if _timed_out:
                    completion = "timeout"
                else:
                    completion = "error"
                    error_msg = m.last_error or str(exc)[:300]
                break
            finally:
                if use_alarm:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)

            after = _perf_total(m)
            if after == executed and _read_pc(m) != halt_addr:
                completion = "error"
                error_msg = (
                    f"simulation stalled at pc=0x{pc:x} "
                    "(unsupported instruction or wedged decoder)"
                )
                break
            executed = after
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        devnull.close()

    elapsed = time.perf_counter() - t0
    perf = m.get_perf()
    executed = _perf_total(m)
    x_used, x_total, f_used = _read_register_usage(m)

    if completion == "error":
        dyn = _unavailable_dynamic(
            reason=error_msg or m.last_error or "simulation error",
            mem_size=mem_size, timeout_s=timeout_s, input_seed=input_seed,
            input_elements=input_elements, halt_addr=halt_addr,
        )
        dyn["completion"] = "error"
        dyn["executed"] = executed
        dyn["elapsed_s"] = elapsed
        dyn["last_error"] = m.last_error
        return {
            "dynamic": dyn,
            "output": _unavailable_output(output_elements, "error"),
        }

    ops = {key: int(perf.get(key, 0)) for key in OPS_KEYS}
    output = _read_output(m, layout["output_addr"], output_elements)
    output["completion"] = completion
    output["partial"] = completion != "halted"
    return {
        "dynamic": {
            "source": "simulated",
            "simulator": "tinyfive",
            "simulator_version": get_environment().get("tinyfive"),
            "completion": completion,
            "limit": limit,
            "executed": executed,
            "timeout_s": float(timeout_s),
            "elapsed_s": elapsed,
            "memory_size_bytes": mem_size,
            "input_seed": input_seed,
            "input_elements": input_elements,
            "halt_addr": halt_addr,
            "ops": ops,
            "x_registers_used": x_used,
            "x_usage_total": x_total,
            "f_registers_used": f_used,
            "per_label": None,
            "per_label_note": "tinyfive exe() exposes no per-PC trace",
            "last_error": m.last_error,
        },
        "output": output,
    }


def _perf_total(machine) -> int:
    try:
        return int(machine.get_perf().get("total", 0))
    except Exception:
        return int(getattr(machine, "instr_count", 0))


# ═══════════════════════════════════════════════════════════════════════════
# Report assembly and audit
# ═══════════════════════════════════════════════════════════════════════════

def _simulated_halted(side: dict, name: str):
    dyn = side.get("dynamic") or {}
    if dyn.get("source") != "simulated":
        return f"{name}.dynamic.source!='simulated'"
    if dyn.get("completion") != "halted":
        return f"{name}.completion=='{dyn.get('completion')}'"
    ops = dyn.get("ops")
    if not isinstance(ops, dict) or ops.get("total") is None:
        return f"{name}.dynamic.ops missing"
    return True


def _comparison(scratchv: dict, llvm: dict) -> dict:
    reasons = []
    sv_state = _simulated_halted(scratchv, "scratchv")
    ll_state = _simulated_halted(llvm, "llvm")
    if sv_state is not True:
        reasons.append(sv_state)
    if ll_state is not True:
        reasons.append(ll_state)
    if reasons:
        return {
            "dynamic_instruction_ratio": None,
            "incomparable_reason": "; ".join(reasons),
        }
    sv_total = scratchv["dynamic"]["ops"]["total"]
    ll_total = llvm["dynamic"]["ops"]["total"]
    if ll_total <= 0:
        return {
            "dynamic_instruction_ratio": None,
            "incomparable_reason": "llvm.dynamic.ops.total==0",
        }
    return {
        "dynamic_instruction_ratio": round(sv_total / ll_total, 6),
        "incomparable_reason": None,
    }


def build_report(model: dict, environment: dict, scratchv: dict, llvm: dict) -> dict:
    """Assemble the schema v2 report dictionary."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": {"script": "rv32_bench.py"},
        "model": {key: model.get(key) for key in MODEL_FIELDS},
        "environment": environment,
        "targets": TARGETS,
        "scratchv": scratchv,
        "llvm": llvm,
        "comparison": _comparison(scratchv, llvm),
        "warnings": [],
        "errors": [],
    }


def audit_provenance(report: dict) -> list[str]:
    """Return the list of honesty violations; empty means the report is clean."""
    violations: list[str] = []

    if report.get("schema_version") != SCHEMA_VERSION:
        violations.append(
            f"schema_version must be {SCHEMA_VERSION!r}, got "
            f"{report.get('schema_version')!r}"
        )

    model = report.get("model") or {}
    sha = str(model.get("sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        violations.append("model.sha256 missing or not a 64-char hex digest")
    if not model.get("path"):
        violations.append("model.path missing")

    env = report.get("environment") or {}
    for key in ("python", "numpy", "tinyfive", "llvmlite"):
        if key not in env:
            violations.append(f"environment.{key} missing")

    targets = report.get("targets") or {}
    for side in ("scratchv", "llvm"):
        target = targets.get(side) or {}
        if not target.get("isa"):
            violations.append(f"targets.{side}.isa missing")

    for side_name in ("scratchv", "llvm"):
        side = report.get(side_name) or {}
        compile_info = side.get("compile") or {}
        if compile_info.get("status") == "success" and \
                compile_info.get("static_source") != "asm_scan":
            violations.append(
                f"{side_name}.compile.static_source must be 'asm_scan'"
            )
        dyn = side.get("dynamic")
        if dyn is None:
            violations.append(f"{side_name}.dynamic missing")
            continue
        source = dyn.get("source")
        completion = dyn.get("completion")
        if source == "simulated":
            for key in ("simulator", "simulator_version", "executed",
                        "memory_size_bytes", "input_seed"):
                if dyn.get(key) is None:
                    violations.append(
                        f"{side_name}.dynamic.{key} missing for simulated data"
                    )
            if completion not in ("halted", "budget_exhausted", "timeout"):
                violations.append(
                    f"{side_name}.dynamic.completion invalid: "
                    f"{dyn.get('completion')!r}"
                )
            ops = dyn.get("ops")
            if not isinstance(ops, dict):
                violations.append(f"{side_name}.dynamic.ops missing")
            else:
                for key in OPS_KEYS:
                    if not isinstance(ops.get(key), int):
                        violations.append(
                            f"{side_name}.dynamic.ops.{key} must be an int"
                        )
                executed = dyn.get("executed")
                if isinstance(ops.get("total"), int) and \
                        isinstance(executed, int) and ops["total"] != executed:
                    violations.append(
                        f"{side_name}.dynamic: executed={executed} != "
                        f"ops.total={ops['total']} (inconsistent counters)"
                    )
            limit = dyn.get("limit")
            executed = dyn.get("executed")
            if isinstance(limit, int) and isinstance(executed, int) and \
                    executed > limit:
                violations.append(
                    f"{side_name}.dynamic: executed={executed} > "
                    f"limit={limit} (budget overrun)"
                )
            if completion == "halted" and limit is not None and \
                    executed == limit:
                violations.append(
                    f"{side_name}.dynamic: halted but executed==limit=="
                    f"{limit} (unverified halt)"
                )
        elif source == "unavailable":
            if not dyn.get("reason"):
                violations.append(f"{side_name}.dynamic.reason missing")
            if dyn.get("ops") is not None:
                violations.append(
                    f"{side_name}.dynamic.ops must be null when unavailable"
                )
        else:
            violations.append(
                f"{side_name}.dynamic.source invalid: {source!r}"
            )

        out = side.get("output")
        if out is None:
            if side_name == "scratchv":
                violations.append("scratchv.output missing")
        elif not isinstance(out, dict):
            violations.append(f"{side_name}.output must be a dict")
        else:
            partial = out.get("partial")
            if not isinstance(partial, bool):
                violations.append(f"{side_name}.output.partial must be a bool")
            elif isinstance(completion, str) and \
                    partial != (completion != "halted"):
                violations.append(
                    f"{side_name}.output.partial={partial} inconsistent with "
                    f"dynamic.completion={completion!r}"
                )
            out_completion = out.get("completion")
            if not isinstance(out_completion, str) or not out_completion:
                violations.append(f"{side_name}.output.completion missing")
            if partial is False and out.get("raw_hex") is None:
                violations.append(
                    f"{side_name}.output.raw_hex missing for a complete result"
                )
            q16 = out.get("q16_16")
            if q16 is not None and not isinstance(q16, list):
                violations.append(
                    f"{side_name}.output.q16_16 must be a list or null"
                )
            elif isinstance(q16, list) and isinstance(out.get("elements"), int) \
                    and len(q16) != out["elements"]:
                violations.append(
                    f"{side_name}.output.q16_16 has {len(q16)} elements, "
                    f"expected {out['elements']}"
                )

    comparison = report.get("comparison") or {}
    ratio = comparison.get("dynamic_instruction_ratio")
    if ratio is not None:
        if not isinstance(ratio, (int, float)):
            violations.append("comparison.dynamic_instruction_ratio not numeric")
        for side_name in ("scratchv", "llvm"):
            dyn = (report.get(side_name) or {}).get("dynamic") or {}
            if dyn.get("source") != "simulated" or \
                    dyn.get("completion") != "halted":
                violations.append(
                    "comparison.dynamic_instruction_ratio computed from an "
                    f"incomplete/non-simulated side ({side_name})"
                )
                break
    elif not comparison.get("incomparable_reason"):
        violations.append("comparison.incomparable_reason missing when ratio is null")

    for key in ("warnings", "errors"):
        if not isinstance(report.get(key), list):
            violations.append(f"{key} must be a list")

    return violations


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="RV32 full benchmark: ScratchV vs LLVM on TinyFive "
                    "(honest provenance, explicit budgets)"
    )
    p.add_argument("model", help="Path to ONNX model")
    p.add_argument("--output-dir", default="benchmark_reports")
    p.add_argument("--html", default="rv32_bench.html")
    p.add_argument("--json", default="rv32_bench.json")
    p.add_argument("--md", default="rv32_bench.md")
    p.add_argument("--max-instructions", type=int, default=0,
                   help="Instruction budget; 0 means full simulation (default: 0)")
    p.add_argument("--full", action="store_true",
                   help="Explicitly request a full (untruncated) simulation")
    p.add_argument("--mem-size", type=int, default=DEFAULT_MEM_SIZE,
                   help="TinyFive memory size in bytes (default: 268435456)")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                   help="Wall-clock timeout in seconds (default: 900)")
    p.add_argument("--chunk-instructions", type=int,
                   default=DEFAULT_CHUNK_INSTRUCTIONS,
                   help="Instructions per simulation chunk (default: 10000000)")
    p.add_argument("--input-seed", type=int, default=DEFAULT_INPUT_SEED)
    p.add_argument("--skip-llvm", action="store_true")
    p.add_argument("--allow-missing-simulator", action="store_true")
    p.add_argument("--fail-on-incomplete", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def _unavailable_llvm(reason: str) -> dict:
    return {
        "compile": {
            "status": "skipped" if reason == "--skip-llvm" else "unavailable",
            "reason": reason,
            "isa_detected": None,
            "isa_mismatch": False,
            "static_insns": 0,
            "static_source": "asm_scan",
            "elapsed_s": 0.0,
        },
        "dynamic": {
            "source": "unavailable",
            "simulator": "tinyfive",
            "completion": "not_run",
            "reason": reason,
            "ops": None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.full and args.max_instructions > 0:
        parser.error(
            "--full conflicts with --max-instructions N>0: choose one"
        )
    if args.max_instructions < 0:
        parser.error("--max-instructions must be >= 0")
    if args.chunk_instructions <= 0:
        parser.error("--chunk-instructions must be > 0")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    effective_limit = 0 if args.full else args.max_instructions
    warnings: list[str] = []
    errors: list[str] = []

    if not hasattr(signal, "SIGALRM") and effective_limit == 0 and \
            not args.allow_missing_simulator:
        parser.error(
            "this platform has no SIGALRM timeout; "
            "provide --max-instructions N or --allow-missing-simulator"
        )

    try:
        model_info = get_model_info(args.model)
    except Exception as exc:
        print(f"error: model_parse_failed: {exc}", file=sys.stderr)
        return EXIT_LAYOUT

    out_bytes = model_info["output_elements"] * 4
    min_mem = OUTPUT_ADDR + out_bytes + OUTPUT_GUARD_BYTES
    if args.mem_size < min_mem:
        print(
            f"error: memory_layout_invalid: need mem_size >= {min_mem} bytes "
            f"(192MiB output base + {out_bytes} bytes output + "
            f"{OUTPUT_GUARD_BYTES} bytes guard); got {args.mem_size}",
            file=sys.stderr,
        )
        return EXIT_LAYOUT

    print(f"RV32 benchmark: {args.model}", file=sys.stderr)
    print("  [1/4] ScratchV compilation (RV32IM, Q16.16)", file=sys.stderr)
    sv_compile = compile_scratchv(
        args.model, str(out / "_sv.bin"), str(out / "_sv.s"),
    )
    if sv_compile["status"] != "success":
        print(
            f"error: scratchv_compile_failed: "
            f"{sv_compile.get('reason') or sv_compile.get('error')}",
            file=sys.stderr,
        )
        return EXIT_LAYOUT

    asm_text = (out / "_sv.s").read_text()
    static_mix = static_instruction_mix(asm_text)

    if args.skip_llvm:
        llvm = _unavailable_llvm("--skip-llvm")
    else:
        print("  [2/4] LLVM compilation (RV32IMF, float32)", file=sys.stderr)
        llvm_compile = compile_llvm_rv32(args.model, str(out / "_ll_rv32.s"))
        if llvm_compile["status"] == "failed":
            llvm_reason = (
                "llvm compile failed: "
                f"{llvm_compile.get('reason') or llvm_compile.get('error')}"
            )
        elif llvm_compile["status"] == "skipped":
            llvm_reason = (
                f"llvm compile skipped: {llvm_compile.get('reason')}"
            )
        elif llvm_compile.get("isa_mismatch"):
            llvm_reason = (
                f"llvm isa mismatch: {llvm_compile.get('reason')} "
                "(dynamic comparison disabled)"
            )
        else:
            llvm_reason = (
                "llvm executable image pipeline not implemented "
                "(topic 25 boundary)"
            )
        llvm = {
            "compile": llvm_compile,
            "dynamic": {
                "source": "unavailable",
                "simulator": "tinyfive",
                "completion": "not_run",
                "reason": llvm_reason,
                "ops": None,
            },
        }
        if llvm_compile["status"] == "failed":
            warnings.append(
                "LLVM compilation failed: "
                f"{llvm_compile.get('reason') or llvm_compile.get('error')}"
            )
        elif llvm_compile["status"] == "skipped":
            warnings.append(
                f"LLVM side skipped: {llvm_compile.get('reason')}"
            )
        elif llvm_compile.get("isa_mismatch"):
            warnings.append(
                "LLVM assembly contains RV64-only mnemonics; dynamic "
                "comparison disabled"
            )

    if effective_limit == 0:
        warnings.append(
            "full simulation requested; no automatic instruction-count or "
            "wall-clock estimate is available (use --max-instructions N to "
            "calibrate); results are marked partial if the wall-clock "
            "timeout fires"
        )

    print("  [3/4] TinyFive simulation", file=sys.stderr)
    try:
        sim = run_simulation(
            asm_path=str(out / "_sv.s"),
            binary_path=str(out / "_sv.bin"),
            data_offset=sv_compile["data_offset"],
            workspace_bytes=sv_compile["workspace_bytes"],
            input_elements=model_info["input_elements"],
            output_elements=model_info["output_elements"],
            max_instructions=effective_limit,
            mem_size=args.mem_size,
            timeout_s=args.timeout,
            chunk_instructions=args.chunk_instructions,
            input_seed=args.input_seed,
        )
    except LayoutError as exc:
        print(f"error: memory_layout_invalid: {exc}", file=sys.stderr)
        return EXIT_LAYOUT

    dynamic = sim["dynamic"]
    if dynamic["source"] == "unavailable" and \
            dynamic["completion"] == "not_run":
        if dynamic.get("reason", "").startswith("unsupported_mnemonics") or \
                dynamic.get("reason", "").startswith("image_load_failed"):
            print(f"error: {dynamic['reason']}", file=sys.stderr)
            return EXIT_LAYOUT
        if not args.allow_missing_simulator:
            print(
                f"error: simulator_unavailable: {dynamic['reason']} "
                "(use --allow-missing-simulator for a static-only report)",
                file=sys.stderr,
            )
            return EXIT_NO_SIMULATOR
        warnings.append(
            "simulator unavailable; dynamic section omitted "
            f"({dynamic['reason']})"
        )
    elif dynamic.get("completion") == "budget_exhausted":
        warnings.append(
            f"budget exhausted at {dynamic['limit']} instructions; "
            "dynamic counts are partial"
        )
    elif dynamic.get("completion") == "timeout":
        warnings.append(
            f"wall-clock timeout after {dynamic['elapsed_s']:.1f}s; "
            "dynamic counts are partial"
        )
    elif dynamic.get("completion") == "error":
        errors.append(dynamic.get("reason") or "simulation error")

    scratchv = {
        "compile": sv_compile,
        "static_instruction_mix": static_mix,
        "dynamic": dynamic,
        "output": sim["output"],
    }

    print("  [4/4] Report", file=sys.stderr)
    report = build_report(model_info, get_environment(), scratchv, llvm)
    report["warnings"] = warnings
    report["errors"] = errors

    violations = audit_provenance(report)
    if violations:
        for violation in violations:
            print(f"provenance_violation: {violation}", file=sys.stderr)
        return EXIT_ERROR

    markdown = bench_report.render_markdown(report)
    (out / args.md).write_text(markdown, encoding="utf-8")
    (out / args.html).write_text(bench_report.render_html(report), encoding="utf-8")
    (out / args.json).write_text(bench_report.render_bench_json(report), encoding="utf-8")
    if not args.quiet:
        print(markdown)
    print(
        f"Reports: {out / args.html} | {out / args.md} | {out / args.json}",
        file=sys.stderr,
    )

    if args.fail_on_incomplete and dynamic.get("completion") != "halted":
        return EXIT_INCOMPLETE
    if dynamic.get("completion") == "error":
        return EXIT_ERROR
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
