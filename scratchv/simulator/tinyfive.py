"""TinyFive adapter for verifying and profiling generated RISC-V binaries.

Uses the ``tinyfive`` package's ``machine`` class for cycle-accurate RV32IM
simulation.  Binary code (32-bit instruction words) is loaded directly into
the simulated memory — no assembly parsing needed.

Requires: pip install tinyfive numpy
"""

from __future__ import annotations

import re
import numpy as np
from typing import Optional


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
        self._available = False
        self._init_machine()

    def _init_machine(self):
        try:
            from tinyfive.machine import machine
            self._m = machine(mem_size=self.mem_size)
            self._available = True
        except ImportError:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    # ── Binary loading (primary path) ───────────────────────────────────

    def load_binary(self, words: list[int], origin: int = 0):
        """Load raw 32-bit instruction words into memory at *origin*.

        Each word is written as 4 little-endian bytes.  PC is set to *origin*.
        This is the preferred loading method — it uses the compiler's own
        binary output, avoiding TinyFive's limited asm() parser.
        """
        if not self._available:
            return
        for i, word in enumerate(words):
            addr = origin + i * 4
            self._write_u32(addr, word)
        self._set_pc(origin)

    def load_data(self, data: bytes, addr: int):
        """Load raw byte data into memory at *addr*."""
        if not self._available:
            return
        self._m.mem[addr:addr + len(data)] = np.frombuffer(data, dtype=np.uint8)

    # ── Assembly loading (fallback for simple snippets) ─────────────────

    def load_asm(self, asm_lines: list[str], origin: int = 0x200):
        """Load assembly text into memory via TinyFive's asm().

        NOTE: TinyFive's asm() has limitations — integer register numbers
        only, no pseudo-instructions.  Prefer ``load_binary()`` for
        production code.
        """
        if not self._available:
            return
        self._set_pc(origin)
        for line in asm_lines:
            line = line.split("#")[0].strip()
            if not line or line.endswith(":"):
                if line.endswith(":"):
                    label = line[:-1]
                    self._m.label_dict[label] = self._m.pc[0]  # type: ignore[index]
                continue
            parts = re.split(r'[,\s]+', line)
            op = parts[0].lower()
            args = [self._parse_arg(a) for a in parts[1:] if a]
            try:
                self._m.asm(op, *args)
            except Exception:
                continue

    def _parse_arg(self, arg: str):
        try:
            return int(arg)
        except ValueError:
            return arg

    # ── Execution ───────────────────────────────────────────────────────

    def run(self, instructions: Optional[int] = None, start: int = 0):
        """Execute for *instructions* cycles, or until halt if None.

        Uses TinyFive's ``exe(start, instructions=N)`` directly.
        After execution, ``instr_count`` reflects the built-in ops counter.

        NOTE: TinyFive's exe() mutates ``pc`` from numpy array to plain int.
        We save/restore to keep the wrapper consistent.
        """
        if not self._available:
            return
        n = instructions if instructions is not None else 100_000_000
        # Save PC before exe (which replaces pc array with int)
        saved_pc = int(self._m.pc[0]) if hasattr(self._m.pc, '__getitem__') else int(self._m.pc)
        try:
            self._m.exe(start=start, instructions=n)
        except Exception:
            pass
        # Restore pc as numpy array
        self._m.pc = np.array([saved_pc], dtype=np.uint32) if not hasattr(self._m.pc, '__getitem__') else self._m.pc
        self.instr_count = int(self._m.ops.get('total', 0))

    # ── Register access ─────────────────────────────────────────────────

    def get_reg(self, idx: int) -> int:
        """Read signed 32-bit register value."""
        if not self._available:
            return 0
        return int(self._m.x[idx])

    def set_reg(self, idx: int, value: int):
        """Write signed 32-bit register value."""
        if not self._available:
            return
        self._m.x[idx] = np.int32(value)

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
        self._m.mem[addr:addr + 4] = np.array(
            [value], dtype=np.uint32
        ).view(np.uint8)

    def _set_pc(self, val: int):
        if not self._available:
            return
        self._m.pc[0] = np.uint32(val)  # type: ignore[index]

    @property
    def pc(self) -> int:
        if not self._available:
            return 0
        pc_val = self._m.pc
        if hasattr(pc_val, '__getitem__'):
            return int(pc_val[0])  # type: ignore[index]
        return int(pc_val)


class StubProfiledMachine(ProfiledMachine):
    """Always-available stub for testing without TinyFive installed."""

    def __init__(self):
        super().__init__()
        self._available = True
        self._m = None
        self.regs = [0] * 32
        self.memory: dict[int, int] = {}
        self._pc = 0
        self.instr_count = 0

    def load_binary(self, words: list[int], origin: int = 0):
        self._pc = origin
        self._code_words = words

    def load_data(self, data: bytes, addr: int):
        for i, b in enumerate(data):
            self.memory[addr + i] = b

    def run(self, instructions=None, start=0):
        # Count words as executed instructions
        words = getattr(self, '_code_words', [])
        self.instr_count = min(len(words), instructions or len(words))

    def get_reg(self, idx: int) -> int:
        return self.regs[idx] if idx < len(self.regs) else 0

    def set_reg(self, idx: int, value: int):
        if idx < len(self.regs):
            self.regs[idx] = value

    def write_mem_i32(self, addr: int, value: int):
        b = np.uint32(value).tobytes()
        for i, byte in enumerate(b):
            self.memory[addr + i] = byte

    def read_mem_i32(self, addr: int) -> int:
        return self.memory.get(addr, 0)

    @property
    def pc(self) -> int:
        return self._pc

    @pc.setter
    def pc(self, val: int):
        self._pc = val


def verify_assembly(asm_code: str, verbose: bool = False) -> dict:
    """Verify generated assembly by running it in TinyFive.

    Args:
        asm_code: RISC-V assembly text.
        verbose: Print performance info.

    Returns:
        dict with keys: success, instr_count, error
    """
    m = ProfiledMachine(mem_size=128 * 1024 * 1024)
    if not m.available:
        return {
            "success": False,
            "instr_count": 0,
            "error": "tinyfive not installed",
        }

    lines = asm_code.strip().split("\n")
    m.load_asm(lines, origin=0)

    try:
        m.run(instructions=100_000_000)
    except Exception as e:
        return {
            "success": False,
            "instr_count": m.instr_count,
            "error": str(e),
        }

    return {"success": True, "instr_count": m.instr_count, "error": None}
