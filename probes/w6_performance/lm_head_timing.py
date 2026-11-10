"""Guest-clock instrumentation for the isolated W6 LM Head experiment on Linux."""

from contextlib import contextmanager
import ctypes
import os
from dataclasses import replace
from pathlib import Path
import re
import struct
import subprocess
from unittest.mock import patch

from scratchv.runtime import riscv_external


_SYMBOL = "sv_w6_lm_head_ticks"
_TIMER = r"""
static volatile uint64_t sv_w6_lm_head_ticks[2];
static inline uint64_t sv_w6_read_time(void) {
    uint64_t t;
    __asm__ volatile("fence iorw,iorw\n\trdtime %0\n\tfence iorw,iorw"
                     : "=r"(t) :: "memory");
    return t;
}
"""


def instrument(artifact):
    """Time the transpose (if present), projection, checks and final logits copy."""
    before, entry, body = artifact.source.rpartition(f"int {artifact.function_name}(")
    if not entry or _SYMBOL in artifact.source:
        raise ValueError("Missing or already instrumented model entry")
    # Insert helpers above the entry's optional platform-specific export decoration.
    include = "#include <stdint.h>"
    if before.count(include) != 1:
        raise ValueError("Expected the generated FP32 prelude")
    before = before.replace(include, include + "\n" + _TIMER, 1)
    fused = "/* w6_lm_head_begin */"
    if fused in body:
        if body.count(fused) != 1:
            raise ValueError("Ambiguous candidate timing marker")
        start = body.index(fused)
    else:
        matches = list(re.finditer(r"/\* \d+: transpose \*/", body))
        if len(matches) != 1 or not re.search(r"/\* \d+: matmul \*/", body[matches[0].end():]):
            raise ValueError("Expected the isolated transpose followed by MatMul")
        start = matches[0].start()
    end = re.search(r"(?m)^  return 0;\n}\s*\Z", body)
    if not end or start >= end.start():
        raise ValueError("Missing final model return after logits write-back")
    body = (body[:start] + f"{_SYMBOL}[0] = sv_w6_read_time();\n  " +
            body[start:end.start()] + f"  {_SYMBOL}[1] = sv_w6_read_time();\n" + body[end.start():])
    return replace(artifact, source=before + entry + body)


def timing_address(executable):
    """Resolve the 16-byte counter pair in the exact ELF used for this case."""
    listing = subprocess.check_output(["nm", "-S", str(executable.elf_path)], text=True)
    matches = re.findall(rf"^([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+[bBdD]\s+{_SYMBOL}$",
                         listing, flags=re.MULTILINE)
    if len(matches) != 1 or int(matches[0][1], 16) != 16:
        raise ValueError(f"Expected a single 16-byte {_SYMBOL} in {executable.elf_path}")
    return int(matches[0][0], 16)


@contextmanager
def capture_ticks(address, *, short_qemu_highwater=False):
    """Reuse the existing QMP stop. Dump counters only after it succeeds."""
    if address <= 0 or address % 8 != 0:
        raise ValueError("Invalid guest counter address")

    observation = {}
    class TimingQMP(riscv_external.QMP):
        def command(self, name, arguments=None):
            if name == "quit" and short_qemu_highwater:
                # A short guest may finish between periodic RSS samples. Read
                # the kernel's high-water mark after the dump, before exit.
                children = Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()
                guests = [Path("/proc") / pid for pid in children
                          if (Path("/proc") / pid / "comm").read_text().startswith("qemu-system")]
                if len(guests) != 1:
                    raise RuntimeError("Expected the single owned QEMU child")
                value = next(line.split(":", 1)[1] for line in (guests[0] / "status").read_text().splitlines()
                             if line.startswith("VmHWM:"))
                amount, unit = value.split()
                if unit != "kB" or int(amount) <= 0:
                    raise ValueError("Invalid QEMU VmHWM")
                observation["qemu_highwater_bytes"] = int(amount) * 1024
            response = super().command(name, arguments)
            if name == "stop":
                reply = super().command("human-monitor-command", {
                    "command-line": f"pmemsave 0x{address:x} 16 lm-head-ticks.bin",
                })
                if reply.strip():
                    raise RuntimeError(f"QMP could not dump LM Head ticks: {reply}")
            return response

    # This standalone probe runs one guest at a time in the current Python process.
    with patch.object(riscv_external, "QMP", TimingQMP):
        yield observation


def read_timebase(qemu, directory):
    """Read /cpus/timebase-frequency from the installed QEMU virt machine DTB."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(qemu), "-machine", "virt,dumpdtb=virt.dtb", "-cpu", "rv64",
                    "-accel", "tcg,thread=multi", "-smp", "4", "-bios", "none",
                    "-display", "none", "-serial", "none", "-monitor", "none"],
                   cwd=directory, check=True, timeout=20, capture_output=True)
    dtb = ctypes.create_string_buffer((directory / "virt.dtb").read_bytes())
    # libfdt is the device-tree library used by Ubuntu's qemu-system-misc.
    fdt = ctypes.CDLL("libfdt.so.1")
    fdt.fdt_check_header.argtypes = [ctypes.c_void_p]
    fdt.fdt_check_header.restype = ctypes.c_int
    fdt.fdt_path_offset.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    fdt.fdt_path_offset.restype = ctypes.c_int
    fdt.fdt_getprop.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p,
                                ctypes.POINTER(ctypes.c_int)]
    fdt.fdt_getprop.restype = ctypes.c_void_p
    if fdt.fdt_check_header(dtb) != 0:
        raise ValueError("Invalid QEMU virt DTB")
    cpus = fdt.fdt_path_offset(dtb, b"/cpus")
    size = ctypes.c_int()
    if cpus < 0:
        raise ValueError("Missing /cpus in QEMU virt DTB")
    address = fdt.fdt_getprop(dtb, cpus, b"timebase-frequency", ctypes.byref(size))
    if not address or size.value != 4:
        raise ValueError("Missing 32-bit /cpus/timebase-frequency in QEMU virt DTB")
    hz, = struct.unpack(">I", ctypes.string_at(address, size.value))
    if hz <= 0:
        raise ValueError("Invalid QEMU virt timebase")
    return hz


def read_timing(run_dir, timebase_hz):
    start, end = struct.unpack("<QQ", (Path(run_dir) / "lm-head-ticks.bin").read_bytes())
    if not 0 < start < end or timebase_hz <= 0:
        raise ValueError(f"Invalid guest timer values: {start}, {end}, {timebase_hz}")
    return {"guest_start_ticks": start, "guest_end_ticks": end,
            "guest_elapsed_ticks": end - start, "timebase_hz": timebase_hz,
            "guest_lm_head_s": (end - start) / timebase_hz}
