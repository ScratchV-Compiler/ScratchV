"""TinyFive adapter for verifying and profiling generated RISC-V binaries.

Uses the ``tinyfive`` package's ``machine`` class for cycle-accurate RV32IM
simulation.  Binary code (32-bit instruction words) is loaded directly into
the simulated memory — no assembly parsing needed.

Requires: pip install tinyfive numpy
"""

from __future__ import annotations

import numpy as np
from types import MethodType
from typing import Optional


def _tinyfive_read_i32_compat(machine_obj, addr: int) -> np.int32:
    """Read an i32 without relying on NumPy uint8 shift semantics.

    TinyFive 1.0.0 builds a word by shifting ``numpy.uint8`` scalars.  With
    current NumPy versions those shifts overflow at eight bits, so instruction
    fetches lose bytes 1..3.  Bind this compatible implementation to the
    TinyFive instance used by the adapter.
    """
    address = int(addr)
    if address < 0 or address + 4 > len(machine_obj.mem):
        raise IndexError(f"TinyFive i32 read out of bounds: {address}")
    raw = bytes(machine_obj.mem[address:address + 4])
    # Keep TinyFive's original signed-int32 contract.  Returning a Python
    # negative int breaks ``np.uint32(inst)`` on NumPy 2.x, whereas np.int32
    # preserves the same bit pattern when instruction decoding converts it.
    return np.int32(int.from_bytes(raw, "little", signed=True))


class ProfiledMachine:
    """TinyFive machine wrapper for benchmark-quality RISC-V simulation.

    Loads pre-compiled binary code (32-bit words) directly into memory
    and executes via TinyFive's ``exe()``.  Provides register/memory I/O
    helpers and exposes TinyFive's built-in performance counters.

    Usage::

        m = ProfiledMachine(mem_size=128 * 1024 * 1024)
        m.load_binary(code_words, origin=0)
        m.write_mem_i32(data_addr, value)
        m.set_reg(10, input_ptr)   # a0
        m.set_reg(11, output_ptr)  # a1
        m.run(instructions=100_000_000)
        print(m.get_perf())
    """

    def __init__(self, mem_size: int = 4096):
        self._m = None
        self.mem_size = mem_size
        self.instr_count = 0
        self.last_error: Optional[str] = None
        self._available = False
        self._init_machine()

    def _init_machine(self):
        try:
            from tinyfive.machine import machine
            self._m = machine(mem_size=self.mem_size)
            self._m.read_i32 = MethodType(
                _tinyfive_read_i32_compat,
                self._m,
            )
            self._available = True
        except ImportError:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def _machine(self):
        """Compatibility view used by older standalone benchmark helpers."""
        return self._m

    # ── Binary loading (primary path) ───────────────────────────────────

    def load_binary(self, words: list[int], origin: int = 0):
        """Load raw 32-bit instruction words into memory at *origin*.

        Each word is written as 4 little-endian bytes.  PC is set to *origin*.
        This is the preferred loading method — it uses the compiler's own
        binary output, avoiding TinyFive's limited asm() parser.
        """
        if not self._available:
            return
        if origin < 0 or origin + len(words) * 4 > self.mem_size:
            raise ValueError(
                f"binary does not fit TinyFive memory: origin={origin}, "
                f"bytes={len(words) * 4}, mem_size={self.mem_size}"
            )
        for i, word in enumerate(words):
            addr = origin + i * 4
            self._write_u32(addr, word)
        self._set_pc(origin)

    def load_data(self, data: bytes, addr: int):
        """Load raw byte data into memory at *addr*."""
        if not self._available:
            return
        if addr < 0 or addr + len(data) > self.mem_size:
            raise ValueError(
                f"data does not fit TinyFive memory: address={addr}, "
                f"bytes={len(data)}, mem_size={self.mem_size}"
            )
        self._m.mem[addr:addr + len(data)] = np.frombuffer(data, dtype=np.uint8)

    # ── Assembly loading (fallback for simple snippets) ─────────────────

    def load_asm(self, asm_lines: list[str], origin: int = 0x200):
        """Assemble source with ScratchV's encoder and load the binary.

        TinyFive's text assembler only accepts numeric registers and silently
        skipping unsupported source would make a simulation report invalid.
        The shared encoder gives both production code and verification the
        same explicit failure behavior.
        """
        if not self._available:
            return
        from scratchv.backend.riscv_encoder import assemble_to_binary

        binary = assemble_to_binary("\n".join(asm_lines))
        words = [
            int.from_bytes(binary[i:i + 4], "little")
            for i in range(0, len(binary), 4)
        ]
        self.load_binary(words, origin=origin)

    # ── Execution ───────────────────────────────────────────────────────

    def run(
        self,
        instructions: Optional[int] = None,
        start: Optional[int] = None,
        *,
        n: Optional[int] = None,
        strict: bool = False,
    ):
        """Execute a bounded number of instructions with TinyFive.

        Uses TinyFive's ``exe(start, instructions=N)`` directly.
        ``n`` is retained as a compatibility alias for older benchmark code.
        With ``strict=True`` execution errors propagate instead of being
        converted to ``last_error``.
        """
        if not self._available:
            return
        if instructions is not None and n is not None:
            raise ValueError("specify either instructions or n, not both")
        limit = n if n is not None else instructions
        limit = 100_000_000 if limit is None else max(0, limit)
        start_pc = self.pc if start is None else start
        before = int(self._m.ops.get('total', 0))
        self.last_error = None
        try:
            # RV32 arithmetic intentionally wraps modulo 2^32; NumPy reports
            # that architectural behavior as an overflow warning.
            with np.errstate(over="ignore"):
                self._m.exe(start=start_pc, instructions=limit)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            if strict:
                raise RuntimeError(self.last_error) from exc
        finally:
            # TinyFive does not enforce the architectural x0 invariant itself.
            self._m.x[0] = 0
            after = int(self._m.ops.get('total', 0))
            self.instr_count = after - before

    def run_until(
        self,
        end: int,
        max_instructions: int = 100_000_000,
        start: Optional[int] = None,
        *,
        strict: bool = False,
    ):
        """Execute until the program counter reaches *end* or the limit."""
        if not self._available:
            return
        start_pc = self.pc if start is None else start
        before = int(self._m.ops.get('total', 0))
        self.last_error = None
        try:
            # TinyFive's exe() also normalizes pc to a scalar before decoding.
            # Keeping its initial one-element NumPy array breaks branch updates.
            self._m.pc = int(start_pc)
            with np.errstate(over="ignore"):
                for _ in range(max(0, max_instructions)):
                    if self.pc == end:
                        break
                    inst = self._m.read_i32(self.pc)
                    self._m.dec(np.binary_repr(self._m.u(inst), 32))
                else:
                    raise RuntimeError(
                        f"instruction limit exceeded: {max_instructions}"
                    )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            if strict:
                raise RuntimeError(self.last_error) from exc
        finally:
            self._m.x[0] = 0
            after = int(self._m.ops.get('total', 0))
            self.instr_count = after - before

    # ── Register access ─────────────────────────────────────────────────

    def get_reg(self, idx: int) -> int:
        """Read an RV32 register; x0 is hardwired and invalid indices are 0.

        The bounds check fixes accidental NumPy negative indexing (for
        example, ``x[-1]`` reading x31) at the adapter boundary.
        """
        if not self._available or not 0 <= idx < 32 or idx == 0:
            return 0
        return int(self._m.x[idx])

    def set_reg(self, idx: int, value: int):
        """Write an RV32 register while preserving the architectural x0."""
        if not self._available or not 0 < idx < 32:
            return
        signed = value & 0xFFFFFFFF
        if signed >= 0x80000000:
            signed -= 0x100000000
        self._m.x[idx] = np.int32(signed)

    # ── Memory I/O ──────────────────────────────────────────────────────

    def write_mem_i32(self, addr: int, value: int):
        """Write a 32-bit signed integer (little-endian)."""
        # Convert signed int32 to uint32 bit pattern
        self._write_u32(addr, np.uint32(value & 0xFFFFFFFF))

    def read_mem_i32(self, addr: int) -> int:
        """Read a 32-bit signed integer (little-endian)."""
        if not self._available:
            return 0
        raw = self._m.mem[addr:addr + 4]
        if len(raw) < 4:
            return 0
        val = int(raw.view(np.uint32)[0])
        return val if val < 0x80000000 else val - 0x100000000

    # ── Performance counters ────────────────────────────────────────────

    def get_perf(self) -> dict:
        """Return TinyFive's built-in performance counters."""
        if not self._available:
            return {}
        return {
            "total": int(self._m.ops.get('total', 0)),
            "load": int(self._m.ops.get('load', 0)),
            "store": int(self._m.ops.get('store', 0)),
            "mul": int(self._m.ops.get('mul', 0)),
            "add": int(self._m.ops.get('add', 0)),
            "madd": int(self._m.ops.get('madd', 0)),
            "branch": int(self._m.ops.get('branch', 0)),
        }

    def print_perf(self):
        """Print TinyFive's performance report."""
        if self._available:
            try:
                self._m.print_perf()
            except AttributeError:
                pass
        print(f"  Instruction count: {self.instr_count}")

    # ── Internal helpers ────────────────────────────────────────────────

    def _write_u32(self, addr: int, value):
        """Write uint32 as 4 little-endian bytes to mem."""
        if not self._available:
            return
        if addr < 0 or addr + 4 > self.mem_size:
            raise IndexError(f"TinyFive u32 write out of bounds: {addr}")
        self._m.mem[addr:addr + 4] = np.array(
            [value], dtype=np.uint32
        ).view(np.uint8)

    def _set_pc(self, val: int):
        if not self._available:
            return
        if isinstance(self._m.pc, np.ndarray):
            self._m.pc[0] = np.uint32(val)  # type: ignore[index]
        else:
            self._m.pc = int(val)

    @property
    def pc(self) -> int:
        if not self._available:
            return 0
        pc_val = self._m.pc
        if isinstance(pc_val, np.ndarray):
            return int(pc_val[0])  # type: ignore[index]
        return int(pc_val)


class StubProfiledMachine(ProfiledMachine):
    """Non-executing test double used when unit tests do not need TinyFive.

    This class only stores state and counts parsed source instructions.  It
    must never be used as evidence of RISC-V execution or semantic equality.
    """

    def __init__(self):
        # Do not initialize and then discard a real TinyFive machine.  The
        # stub owns simple Python state and must remain independent of whether
        # the optional dependency happens to be installed.
        self._available = True
        self._m = None
        self.mem_size = 4096
        self.regs = [0] * 32
        self.memory: dict[int, int] = {}
        self._pc = 0
        self.instr_count = 0
        self.last_error = None
        self._code_words: list[int] = []

    def load_binary(self, words: list[int], origin: int = 0):
        self._pc = origin
        self._code_words = words

    def load_data(self, data: bytes, addr: int):
        for i, b in enumerate(data):
            self.memory[addr + i] = b

    def load_asm(self, asm_lines: list[str], origin: int = 0x200):
        """Record executable source lines for deterministic stub counting."""
        from scratchv.backend._asm_parser import parse_line

        self._pc = origin
        self._code_words = []
        for source in asm_lines:
            parsed = parse_line(source)
            if parsed.opcode is not None and not parsed.is_directive:
                self._code_words.append(0)

    def run(self, instructions=None, start=0, *, n=None, strict=False):
        # Count words as executed instructions
        if instructions is not None and n is not None:
            raise ValueError("specify either instructions or n, not both")
        words = getattr(self, '_code_words', [])
        requested = n if n is not None else instructions
        limit = len(words) if requested is None else max(0, requested)
        self.instr_count = min(len(words), limit)

    def get_reg(self, idx: int) -> int:
        if idx == 0:
            return 0
        return self.regs[idx] if 0 <= idx < len(self.regs) else 0

    def set_reg(self, idx: int, value: int):
        if 0 < idx < len(self.regs):
            self.regs[idx] = value

    def write_mem_i32(self, addr: int, value: int):
        b = (value & 0xFFFFFFFF).to_bytes(4, "little")
        for i, byte in enumerate(b):
            self.memory[addr + i] = byte

    def read_mem_i32(self, addr: int) -> int:
        raw = bytes(self.memory.get(addr + i, 0) for i in range(4))
        value = int.from_bytes(raw, "little")
        return value if value < 0x80000000 else value - 0x100000000

    @property
    def pc(self) -> int:
        return self._pc

    @pc.setter
    def pc(self, val: int):
        self._pc = val


_REG_NUMS: dict[str, int] = {
    "zero": 0, "x0": 0,
    "ra": 1, "x1": 1,
    "sp": 2, "x2": 2,
    "gp": 3, "x3": 3,
    "tp": 4, "x4": 4,
    "t0": 5, "x5": 5,
    "t1": 6, "x6": 6,
    "t2": 7, "x7": 7,
    "s0": 8, "fp": 8, "x8": 8,
    "s1": 9, "x9": 9,
    "a0": 10, "x10": 10,
    "a1": 11, "x11": 11,
    "a2": 12, "x12": 12,
    "a3": 13, "x13": 13,
    "a4": 14, "x14": 14,
    "a5": 15, "x15": 15,
    "a6": 16, "x16": 16,
    "a7": 17, "x17": 17,
    "s2": 18, "x18": 18,
    "s3": 19, "x19": 19,
    "s4": 20, "x20": 20,
    "s5": 21, "x21": 21,
    "s6": 22, "x22": 22,
    "s7": 23, "x23": 23,
    "s8": 24, "x24": 24,
    "s9": 25, "x25": 25,
    "s10": 26, "x26": 26,
    "s11": 27, "x27": 27,
    "t3": 28, "x28": 28,
    "t4": 29, "x29": 29,
    "t5": 30, "x30": 30,
    "t6": 31, "x31": 31,
}


def verify_assembly(
    asm_code: str,
    verbose: bool = False,
    initial_registers: Optional[dict[str, int]] = None,
) -> dict:
    """Verify generated assembly by running it in TinyFive.

    Uses ``RISCVAEncoder`` to assemble text → binary, then executes each
    encoded instruction once.  Encoding and execution failures are reported;
    this function never substitutes a static-analysis fallback for simulation.

    Args:
        asm_code: RISC-V assembly text.
        verbose: Print performance info.
        initial_registers: Optional ABI register values to set before execution.

    Returns:
        dict with keys: success, instr_count, return_value, error
    """
    m = ProfiledMachine(mem_size=128 * 1024 * 1024)
    if not m.available:
        return {
            "success": False,
            "instr_count": 0,
            "return_value": None,
            "backend": "tinyfive",
            "error": "tinyfive not installed",
        }

    try:
        from scratchv.backend.riscv_encoder import assemble_to_binary

        binary = assemble_to_binary(asm_code)
        if not binary:
            raise ValueError("assembler produced empty binary")
        words = [
            int.from_bytes(binary[i:i + 4], "little")
            for i in range(0, len(binary), 4)
        ]
        m.load_binary(words, origin=0)

        for reg_name, value in (initial_registers or {}).items():
            reg_num = _REG_NUMS.get(reg_name)
            if reg_num is not None:
                m.set_reg(reg_num, int(value))

        return_address = len(binary)
        m.set_reg(1, return_address)
        m.run_until(
            end=return_address,
            max_instructions=100_000_000,
            start=0,
            strict=True,
        )
    except Exception as e:
        return {
            "success": False,
            "instr_count": m.instr_count,
            "return_value": None,
            "backend": "tinyfive",
            "error": str(e),
        }

    if verbose:
        m.print_perf()
    return {
        "success": True,
        "instr_count": m.instr_count,
        "return_value": m.get_reg(10),
        "backend": "tinyfive",
        "error": None,
    }
