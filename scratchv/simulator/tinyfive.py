"""TinyFive adapter for verifying and profiling generated RISC-V assembly.

Requires ``tinyfive`` package:
    pip install tinyfive numpy
"""

from __future__ import annotations

import re
from typing import Optional


class ProfiledMachine:
    """A TinyFive Machine wrapper that counts executed instructions.

    Falls back to a stub implementation when TinyFive is not installed,
    so tests and basic verification work without the dependency.
    """

    def __init__(self, mem_size: int = 4096):
        self._machine = None
        self.mem_size = mem_size
        self.instr_count = 0
        self._available = False
        self._init_machine()

    def _init_machine(self):
        try:
            from tinyfive.machine import Machine
            self._machine = Machine(mem_size=self.mem_size)
            self._available = True
        except ImportError:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def load_asm(self, asm_lines: list[str], origin: int = 0x200):
        """Load assembly instructions into the machine."""
        if not self._available:
            return
        self._machine.pc = origin
        for line in asm_lines:
            line = line.split("#")[0].strip()
            if not line or line.endswith(":"):
                continue
            parts = re.split(r'[,\s]+', line)
            op = parts[0].lower()
            args = [self._parse_arg(a) for a in parts[1:] if a]
            self._machine.asm(op, *args)

    def _parse_arg(self, arg: str):
        try:
            return int(arg)
        except ValueError:
            return arg

    def run(self, n: Optional[int] = None, start: Optional[str] = None):
        """Execute loaded code and count instructions."""
        if not self._available:
            return

        # Wrap the execute loop with a counter
        original_exe = self._machine.exe
        self.instr_count = 0

        def counted_exe(*args, **kwargs):
            self.instr_count += 1
            return original_exe(*args, **kwargs)

        self._machine.exe = counted_exe
        try:
            self._machine.exe(n=n, start=start)
        finally:
            self._machine.exe = original_exe

    def get_reg(self, idx: int) -> int:
        """Read register value."""
        if not self._available:
            return 0
        return self._machine.x[idx]

    def write_mem_i32(self, addr: int, value: int):
        """Write a 32-bit integer to memory."""
        if not self._available:
            return
        self._machine.write_i32(value, addr)

    def read_mem_i32(self, addr: int) -> int:
        """Read a 32-bit integer from memory."""
        if not self._available:
            return 0
        return self._machine.read_i32(addr)

    def print_perf(self):
        """Print performance counters if available."""
        if self._available:
            try:
                self._machine.print_perf()
            except AttributeError:
                pass
        print(f"  Instruction count: {self.instr_count}")


class StubProfiledMachine(ProfiledMachine):
    """Always-available stub that records instruction calls for testing."""

    def __init__(self):
        super().__init__()
        self._available = True
        self._machine = None
        self.regs = [0] * 32
        self.memory = {}
        self.pc = 0
        self.done = False
        self.instr_count = 0

    def load_asm(self, asm_lines: list[str], origin: int = 0x200):
        self.pc = origin
        self._code = asm_lines

    def run(self, n=None, start=None):
        # Stub: just increment counter for each "instruction"
        code = getattr(self, '_code', [])
        for line in code:
            line = line.split("#")[0].strip()
            if line and not line.endswith(":"):
                self.instr_count += 1

    def get_reg(self, idx: int) -> int:
        return self.regs[idx]

    def write_mem_i32(self, addr: int, value: int):
        self.memory[addr] = value

    def read_mem_i32(self, addr: int) -> int:
        return self.memory.get(addr, 0)


def verify_assembly(asm_code: str, verbose: bool = False) -> dict:
    """Verify generated assembly by running it in TinyFive.

    Args:
        asm_code: RISC-V assembly text.
        verbose: Print performance info.

    Returns:
        dict with keys: success, instr_count, error
    """
    try:
        from tinyfive.machine import Machine
        m = Machine(mem_size=4096)
    except ImportError:
        return {"success": False, "instr_count": 0, "error": "tinyfive not installed"}

    lines = asm_code.strip().split("\n")
    m.pc = 4 * 128

    for line in lines:
        line = line.split("#")[0].strip()
        if not line or line.endswith(":") or line.startswith("."):
            continue
        line = line.lstrip()
        parts = re.split(r'[,\s]+', line)
        if not parts:
            continue
        op = parts[0].lower()
        args = []
        for a in parts[1:]:
            if not a:
                continue
            try:
                args.append(int(a))
            except ValueError:
                args.append(a)
        try:
            m.asm(op, *args)
        except Exception as e:
            return {"success": False, "instr_count": 0, "error": str(e)}

    # Count instructions
    instr_count = [0]

    original_exe = m.exe
    def counted_exe(*a, **kw):
        instr_count[0] += 1
        return original_exe(*a, **kw)

    m.exe = counted_exe
    try:
        m.exe()
    except Exception as e:
        return {"success": False, "instr_count": instr_count[0], "error": str(e)}
    finally:
        m.exe = original_exe

    result = {"success": True, "instr_count": instr_count[0], "error": None}
    if verbose:
        print(f"Instructions executed: {instr_count[0]}")
        try:
            m.print_perf()
        except AttributeError:
            pass
    return result
