#!/usr/bin/env python3
"""Spike RISC-V simulator harness for ScratchV-generated CNN binary.

Wraps a ScratchV flat binary into a minimal ELF32, runs it through Spike
with cache simulation and instrumentation, then parses the output to
collect:

  - Dynamic instruction count (committed + categorized via opcode stats)
  - I-cache hit/miss rates
  - D-cache hit/miss rates
  - PC histogram (hotspots)
  - Instruction trace samples
  - Spike execution time

Spike binary:  /home/kinsomwang/workspace/coralnpu-spike-rv32/bin/spike
ScratchV binary: output.bin

Usage:
    python scratchv/standalone/spike_sim.py                    \\
        --binary output.bin --code-size 3140                    \\
        [--max-instr 50000000] [--ic 64:2:32] [--dc 128:4:32]
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field

# ── Paths ──────────────────────────────────────────────────────────────────
SPIKE = "/home/kinsomwang/workspace/coralnpu-spike-rv32/bin/spike"
SPIKE_DASM = "/home/kinsomwang/workspace/coralnpu-spike-rv32/bin/spike-dasm"
SPIKE_LOG_PARSER = "/home/kinsomwang/workspace/coralnpu-spike-rv32/bin/spike-log-parser"

# ── Constants ──────────────────────────────────────────────────────────────
ELF_BASE = 0x80000000       # RISC-V DRAM base (Spike default)
STARTUP_SIZE = 20           # 5 instructions × 4 bytes
STACK_TOP = 0x85000000      # sp = stack (~80 MB above ELF, grows down)
INPUT_BUF  = 0x86000000     # a0 = input buffer
OUTPUT_BUF = 0x87000000     # a1 = output buffer
# All addresses are within [ELF_BASE, ELF_BASE + 512MB) for -m512

# RISC-V opcode encoding helpers
RV_OP_LUI   = 0b0110111
RV_OP_JAL   = 0b1101111
RV_OP_ECALL = 0b1110011     # ECALL is SYSTEM opcode with funct12=0, rd=0, rs1=0


def _sext(v: int, bits: int) -> int:
    mask = (1 << bits) - 1
    v &= mask
    if v >> (bits - 1):
        v -= 1 << bits
    return v


def rv_lui(rd: int, imm20: int) -> int:
    """LUI rd, imm20 (upper 20 bits of 32-bit immediate)."""
    return ((imm20 & 0xFFFFF) << 12) | ((rd & 0x1F) << 7) | RV_OP_LUI


def rv_jal(rd: int, offset: int) -> int:
    """JAL rd, offset (signed 21-bit, byte-aligned)."""
    offset &= 0x1FFFFE  # 21 bits, LSB forced to 0
    imm20 = (offset >> 20) & 1
    imm10_1 = (offset >> 1) & 0x3FF
    imm11 = (offset >> 11) & 1
    imm19_12 = (offset >> 12) & 0xFF
    return (
        (imm20 << 31) |
        (imm10_1 << 21) |
        (imm11 << 20) |
        (imm19_12 << 12) |
        ((rd & 0x1F) << 7) |
        RV_OP_JAL
    )


def rv_jal_x0(offset: int) -> int:
    """JAL x0, offset — unconditional jump (j pseudo-instruction)."""
    return rv_jal(0, offset)


def emit_startup_stub(cnn_entry_offset: int) -> bytes:
    """Emit a minimal startup stub (5 instructions, 20 bytes).

    Sets up stack, input/output pointers, calls the CNN function, then spins.

    Layout:
      [0]  lui sp, STACK_TOP[31:12]
      [4]  lui a0, INPUT_BUF[31:12]
      [8]  lui a1, OUTPUT_BUF[31:12]
      [12] jal x1, cnn_entry     → call main_graph
      [16] j .                   → spin forever (Spike stops at --instructions limit)

    Args:
        cnn_entry_offset: Byte offset from the JAL instruction to the CNN entry.
                          JAL is at offset 12 in the stub, CNN at offset 20.
                          So offset = 20 - 12 = 8.
    """
    insns = []

    # lui sp, upper20     → sp = upper20 << 12 (stack grows down from here)
    sp_upper = (STACK_TOP >> 12) & 0xFFFFF
    insns.append(rv_lui(2, sp_upper))  # x2 = sp

    # lui a0, upper20     → a0 = input buffer
    a0_upper = (INPUT_BUF >> 12) & 0xFFFFF
    insns.append(rv_lui(10, a0_upper))  # x10 = a0

    # lui a1, upper20     → a1 = output buffer
    a1_upper = (OUTPUT_BUF >> 12) & 0xFFFFF
    insns.append(rv_lui(11, a1_upper))  # x11 = a1

    # jal ra, cnn_entry   → call CNN function (standard ABI: ra = x1)
    # JAL is at byte offset 12 in the stub, CNN entry at offset 20
    jal_offset = cnn_entry_offset - 12  # 20 - 12 = 8
    insns.append(rv_jal(1, jal_offset))  # x1 = ra

    # j .  → infinite spin loop (JAL x0, 0)
    insns.append(rv_jal_x0(0))

    return struct.pack(f"<{len(insns)}I", *insns)


# ═══════════════════════════════════════════════════════════════════════════
# ELF32 builder (minimal, zero-dependency)
# ═══════════════════════════════════════════════════════════════════════════

def build_minimal_elf32(
    code: bytes,
    data: bytes,
    entry_offset: int = 0,
) -> bytes:
    """Build a minimal RV32 ELF executable.

    ELF layout:
      ELF header (52 B)
      Program header (32 B)
      [padding to 0x1000]
      .text (startup stub + code + data)

    The entire file is mapped as a single PT_LOAD segment at ELF_BASE.
    """
    # ELF header constants
    EM_RISCV = 243
    ET_EXEC = 2
    EV_CURRENT = 1
    ELFCLASS32 = 1
    ELFDATA2LSB = 1

    # Offsets
    eh_size = 52
    ph_size = 32
    ph_off = eh_size

    # Page-align the loadable content
    page_size = 0x1000
    text_file_off = page_size
    text_vaddr = ELF_BASE + text_file_off

    # Startup stub + CNN code + weight data
    startup = emit_startup_stub(STARTUP_SIZE)
    payload = startup + code + data
    # Round up to page alignment for file size
    padded_size = (text_file_off + len(payload) + page_size - 1) & ~(page_size - 1)
    file_size = padded_size
    mem_size = text_file_off + len(payload)

    # Build ELF header (52 bytes)
    elf = bytearray()
    # e_ident[16]
    elf.extend(b'\x7fELF')                          # magic
    elf.append(ELFCLASS32)                           # ei_class: 32-bit
    elf.append(ELFDATA2LSB)                          # ei_data: little-endian
    elf.append(EV_CURRENT)                           # ei_version
    elf.append(0)                                    # ei_osabi: System V
    elf.append(0)                                    # ei_abiversion
    elf.extend(b'\x00' * 7)                          # padding
    # e_type
    elf.extend(struct.pack('<H', ET_EXEC))
    # e_machine
    elf.extend(struct.pack('<H', EM_RISCV))
    # e_version
    elf.extend(struct.pack('<I', EV_CURRENT))
    # e_entry: entry point at startup stub
    elf.extend(struct.pack('<I', text_vaddr))
    # e_phoff
    elf.extend(struct.pack('<I', ph_off))
    # e_shoff (no section headers)
    elf.extend(struct.pack('<I', 0))
    # e_flags
    elf.extend(struct.pack('<I', 0))
    # e_ehsize
    elf.extend(struct.pack('<H', eh_size))
    # e_phentsize
    elf.extend(struct.pack('<H', ph_size))
    # e_phnum
    elf.extend(struct.pack('<H', 1))
    # e_shentsize
    elf.extend(struct.pack('<H', 0))
    # e_shnum
    elf.extend(struct.pack('<H', 0))
    # e_shstrndx
    elf.extend(struct.pack('<H', 0))
    assert len(elf) == eh_size, f"ELF header size mismatch: {len(elf)} != {eh_size}"

    # Build program header (PT_LOAD for the entire file)
    PT_LOAD = 1
    PF_R = 4
    PF_W = 2
    PF_X = 1
    ph = bytearray()
    ph.extend(struct.pack('<I', PT_LOAD))           # p_type
    ph.extend(struct.pack('<I', text_file_off))      # p_offset (file offset)
    ph.extend(struct.pack('<I', text_vaddr))         # p_vaddr
    ph.extend(struct.pack('<I', text_vaddr))         # p_paddr
    ph.extend(struct.pack('<I', len(payload)))       # p_filesz
    ph.extend(struct.pack('<I', len(payload)))       # p_memsz
    ph.extend(struct.pack('<I', PF_R | PF_W | PF_X)) # p_flags
    ph.extend(struct.pack('<I', page_size))          # p_align
    assert len(ph) == ph_size, f"PH size mismatch: {len(ph)} != {ph_size}"

    elf.extend(ph)

    # Pad to page alignment, then append payload
    pad_needed = text_file_off - len(elf)
    if pad_needed > 0:
        elf.extend(b'\x00' * pad_needed)
    elf.extend(payload)

    # Pad to aligned file size
    remaining = file_size - len(elf)
    if remaining > 0:
        elf.extend(b'\x00' * remaining)

    return bytes(elf)


# ═══════════════════════════════════════════════════════════════════════════
# Spike runner
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SpikeResult:
    """Parsed Spike simulation output."""
    # Basic stats
    committed_insns: int = 0
    total_insns: int = 0
    wall_time_s: float = 0.0
    exit_code: int = 0

    # Cache stats (set if --ic/--dc/--l2 used)
    icache_hits: int = 0
    icache_misses: int = 0
    icache_miss_rate: float = 0.0
    dcache_hits: int = 0
    dcache_misses: int = 0
    dcache_miss_rate: float = 0.0

    # PC histogram (if -g used)
    pc_histogram: dict[int, int] = field(default_factory=dict)

    # Raw stdout/stderr
    stdout: str = ""
    stderr: str = ""

    # Spike internal struct data (from stderr)
    commited_insns_per_sec: float = 0.0


def run_spike(
    elf_path: str,
    max_instr: int = 100_000_000,
    ic_config: str = "64:2:32",
    dc_config: str = "128:4:32",
    track_pc: bool = True,
    log_commits: bool = False,
    timeout_s: int = 600,
    isa: str = "rv32im",
    mem_mb: int = 512,
) -> SpikeResult:
    """Run Spike on an ELF binary and parse results.

    Args:
        elf_path: Path to the ELF binary.
        max_instr: Stop after this many committed instructions.
        ic_config: I-cache config "sets:ways:blocksize".
        dc_config: D-cache config "sets:ways:blocksize".
        track_pc: Enable PC histogram tracking (-g flag).
        log_commits: Enable per-instruction commit logging (slow).
        timeout_s: Wall-clock timeout.
        isa: RISC-V ISA string.
        mem_mb: Target memory in MiB.

    Returns: SpikeResult with parsed data.
    """
    cmd = [
        SPIKE,
        f"--isa={isa}",
        f"-m{mem_mb}",
        f"--ic={ic_config}",
        f"--dc={dc_config}",
        f"--instructions={max_instr}",
    ]
    if track_pc:
        cmd.append("-g")
    if log_commits:
        cmd.append("--log-commits")

    cmd.append(elf_path)

    print(f"  Running Spike: {' '.join(cmd)}", file=sys.stderr)
    print(f"  Max instructions: {max_instr:,}", file=sys.stderr)

    result = SpikeResult()
    t_start = time.perf_counter()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        result.stdout = proc.stdout
        result.stderr = proc.stderr
        result.exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        result.stderr = "TIMEOUT: Spike did not finish within time limit"
        result.exit_code = -1
        result.wall_time_s = timeout_s
        return result
    except FileNotFoundError:
        result.stderr = f"ERROR: Spike not found at {SPIKE}"
        result.exit_code = -2
        return result

    result.wall_time_s = time.perf_counter() - t_start
    result.total_insns = max_instr  # We set the limit

    # ── Parse committed instruction count from stderr ──
    # Spike stderr format: "Commited 100000000 instructions"
    for line in result.stderr.splitlines():
        if "Commited" in line and "instructions" in line:
            parts = line.strip().split()
            for i, p in enumerate(parts):
                if p == "Commited" or p == "Committed":
                    try:
                        result.committed_insns = int(parts[i + 1])
                    except (IndexError, ValueError):
                        pass
                if "MIPS" in p or "mips" in p.lower():
                    try:
                        # Extract the number before MIPS
                        result.commited_insns_per_sec = float(parts[i - 1])
                    except (IndexError, ValueError):
                        pass

    # ── Parse cache stats from stderr ──
    # Format:
    # I$: 16 sets × 4 ways × 32 B = 2048 B
    #   hits: 98234567    misses: 12345    miss rate: 0.01%
    # D$: 32 sets × 4 ways × 32 B = 4096 B
    #   hits: 87654321    misses: 23456    miss rate: 0.03%
    current_cache = None
    for line in result.stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("I$:"):
            current_cache = "icache"
        elif stripped.startswith("D$:"):
            current_cache = "dcache"
        elif current_cache and "hits:" in stripped:
            import re
            hits_m = re.search(r'hits:\s+(\d+)', stripped)
            misses_m = re.search(r'misses:\s+(\d+)', stripped)
            rate_m = re.search(r'miss rate:\s+([\d.]+)%', stripped)
            if current_cache == "icache":
                if hits_m:
                    result.icache_hits = int(hits_m.group(1))
                if misses_m:
                    result.icache_misses = int(misses_m.group(1))
                if rate_m:
                    result.icache_miss_rate = float(rate_m.group(1))
            elif current_cache == "dcache":
                if hits_m:
                    result.dcache_hits = int(hits_m.group(1))
                if misses_m:
                    result.dcache_misses = int(misses_m.group(1))
                if rate_m:
                    result.dcache_miss_rate = float(rate_m.group(1))

    # ── Parse PC histogram from stdout (-g flag) ──
    # Format (from spike source code):
    # PC histogram (number of commits per PC):
    # 0x80000014: 12345678
    # 0x80000018: 23456789
    # ...
    in_histogram = False
    for line in result.stdout.splitlines():
        if "PC histogram" in line or ("histogram" in line.lower() and "pc" in line.lower()):
            in_histogram = True
            continue
        if in_histogram and ':' in line:
            parts = line.strip().split(':')
            if len(parts) >= 2:
                try:
                    pc_str = parts[0].strip()
                    count_str = parts[1].strip()
                    if pc_str.startswith('0x'):
                        pc = int(pc_str, 16)
                        count = int(count_str)
                        result.pc_histogram[pc] = count
                except (ValueError, IndexError):
                    pass
        elif in_histogram and line.strip() == "":
            in_histogram = False

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Instruction trace analysis (for --log-commits output)
# ═══════════════════════════════════════════════════════════════════════════

def run_spike_with_log(
    elf_path: str,
    max_instr: int = 10_000_000,
    log_path: str = "",
    isa: str = "rv32im",
    mem_mb: int = 512,
    timeout_s: int = 600,
) -> tuple[SpikeResult, str]:
    """Run Spike with --log-commits and save the log.

    WARNING: Logging every committed instruction is very slow (100-1000× slower).
    Only use for small instruction counts (e.g., 1M-10M).
    """
    cmd = [
        SPIKE,
        f"--isa={isa}",
        f"-m{mem_mb}",
        f"--instructions={max_instr}",
        "-l",                          # enable instruction logging
    ]
    if log_path:
        cmd.append(f"--log={log_path}")

    cmd.append(elf_path)

    print(f"  Running Spike with instruction log...", file=sys.stderr)
    print(f"  Max instructions: {max_instr:,}", file=sys.stderr)
    print(f"  Log file: {log_path}", file=sys.stderr)

    result = SpikeResult()
    t_start = time.perf_counter()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        result.stdout = proc.stdout
        result.stderr = proc.stderr
        result.exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        result.stderr = "TIMEOUT"
        result.exit_code = -1
        result.wall_time_s = timeout_s
        return result, ""

    result.wall_time_s = time.perf_counter() - t_start

    # If log was written to file, read it
    log_content = ""
    if log_path and os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_content = f.read()

    return result, log_content


# ═══════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_spike_report(
    result: SpikeResult,
    binary_path: str,
    code_size: int,
    ic_config: str,
    dc_config: str,
    max_instr: int,
    log_data: dict | None = None,
) -> str:
    """Format a detailed Spike simulation report."""
    lines = []
    sep = "=" * 76
    lines.append(sep)
    lines.append("  Spike RISC-V Simulator — CNN Benchmark Report")
    lines.append(sep)
    lines.append(f"  Binary:        {binary_path}")
    lines.append(f"  Code size:     {code_size:,} B ({code_size // 4} static insns)")
    lines.append(f"  Spike:         {SPIKE}")
    lines.append(f"  ISA:           rv32im")
    lines.append("")

    # ── Timing ──
    lines.append("  ── Simulation Execution ──")
    lines.append(f"  Wall time:        {result.wall_time_s:.2f} s")
    lines.append(f"  Instr limit:      {max_instr:,}")
    lines.append(f"  Committed insns:  {result.committed_insns:,}")
    if result.wall_time_s > 0 and result.committed_insns > 0:
        mips = result.committed_insns / result.wall_time_s / 1_000_000
        lines.append(f"  Simulated MIPS:   {mips:.1f}")
    lines.append(f"  Exit code:        {result.exit_code}")
    if result.exit_code != 0:
        lines.append(f"  Stderr: {result.stderr[:500]}")
    lines.append("")

    # ── Cache Simulation ──
    lines.append("  ── Cache Performance (Spike model) ──")
    lines.append(f"  I$ config:    {ic_config}")
    total_i = result.icache_hits + result.icache_misses
    if total_i > 0:
        hit_rate_i = result.icache_hits / total_i * 100
        lines.append(f"  I$ hits:      {result.icache_hits:>15,}")
        lines.append(f"  I$ misses:    {result.icache_misses:>15,}")
        lines.append(f"  I$ miss rate: {result.icache_miss_rate:>14.3f}% (parsed)")
        lines.append(f"  I$ hit rate:  {hit_rate_i:>14.2f}% (calculated)")
        if result.committed_insns > 0:
            misses_per_kilo = result.icache_misses / result.committed_insns * 1000
            lines.append(f"  I$ MPKI:      {misses_per_kilo:>14.3f}")
        lines.append("")

    lines.append(f"  D$ config:    {dc_config}")
    total_d = result.dcache_hits + result.dcache_misses
    if total_d > 0:
        hit_rate_d = result.dcache_hits / total_d * 100
        lines.append(f"  D$ hits:      {result.dcache_hits:>15,}")
        lines.append(f"  D$ misses:    {result.dcache_misses:>15,}")
        lines.append(f"  D$ miss rate: {result.dcache_miss_rate:>14.3f}% (parsed)")
        lines.append(f"  D$ hit rate:  {hit_rate_d:>14.2f}% (calculated)")
        if result.committed_insns > 0:
            misses_per_kilo = result.dcache_misses / result.committed_insns * 1000
            lines.append(f"  D$ MPKI:      {misses_per_kilo:>14.3f}")
        lines.append("")

    # Ratio
    if total_i > 0 and total_d > 0:
        icache_miss_pct = result.icache_misses / (total_i + total_d) * 100
        dcache_miss_pct = result.dcache_misses / (total_i + total_d) * 100
        lines.append(f"  I$ miss share:  {icache_miss_pct:.1f}% of all misses")
        lines.append(f"  D$ miss share:  {dcache_miss_pct:.1f}% of all misses")
        lines.append("")

    # ── PC Histogram ──
    if result.pc_histogram:
        lines.append("  ── Top-15 Hottest PCs ──")
        top_pcs = sorted(result.pc_histogram.items(), key=lambda x: -x[1])[:15]
        for i, (pc, count) in enumerate(top_pcs):
            pct = count / max(result.committed_insns, 1) * 100
            lines.append(f"  {i+1:2d}. 0x{pc:08x}  {count:>12,} commits ({pct:5.1f}%)")
        lines.append("")

    # ── Raw stderr (for debugging) ──
    if result.stderr and result.exit_code != 0:
        lines.append("  ── Spike stderr ──")
        for line in result.stderr.splitlines()[:30]:
            lines.append(f"  | {line}")
        lines.append("")

    lines.append(sep)
    lines.append(f"  Simulation complete. {result.committed_insns:,} instructions "
                 f"in {result.wall_time_s:.1f}s")
    lines.append(sep)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main entry
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Spike RISC-V Simulator — ScratchV CNN Benchmark"
    )
    parser.add_argument(
        "--binary", default="output.bin",
        help="Path to ScratchV-generated .bin file",
    )
    parser.add_argument(
        "--code-size", type=int, required=True,
        help="Code section size in bytes",
    )
    parser.add_argument(
        "--max-instr", type=int, default=50_000_000,
        help="Max instructions to simulate [default: 50M]",
    )
    parser.add_argument(
        "--ic", default="64:2:32",
        help="I-cache config \"sets:ways:blocksize\" [default: 64:2:32]",
    )
    parser.add_argument(
        "--dc", default="128:4:32",
        help="D-cache config \"sets:ways:blocksize\" [default: 128:4:32]",
    )
    parser.add_argument(
        "--isa", default="rv32im",
        help="RISC-V ISA string [default: rv32im]",
    )
    parser.add_argument(
        "--mem", type=int, default=512,
        help="Target memory in MiB [default: 512]",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Wall-clock timeout in seconds [default: 600]",
    )
    parser.add_argument(
        "--no-pc-histogram", action="store_true",
        help="Disable PC histogram tracking",
    )
    parser.add_argument(
        "--log-commits", action="store_true",
        help="Enable per-instruction commit logging (VERY slow)",
    )
    parser.add_argument(
        "--log-instr-limit", type=int, default=100_000,
        help="Instruction limit when using --log-commits [default: 100K]",
    )
    parser.add_argument(
        "--keep-elf", action="store_true",
        help="Keep the generated ELF file (default: temp file)",
    )
    parser.add_argument(
        "--elf-output", default="",
        help="Path to save the generated ELF (implies --keep-elf)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # ── Load binary ────────────────────────────────────────────────────
    if not os.path.exists(args.binary):
        print(f"ERROR: binary not found: {args.binary}", file=sys.stderr)
        return 1

    with open(args.binary, "rb") as f:
        raw_binary = f.read()

    code_size = args.code_size
    if code_size % 4 != 0:
        code_size += 4 - (code_size % 4)

    code = raw_binary[:code_size]
    data = raw_binary[code_size:]

    print(f"  Loaded: {len(raw_binary):,} bytes "
          f"(code: {len(code):,}, data: {len(data):,})", file=sys.stderr)

    # ── Build ELF ──────────────────────────────────────────────────────
    print(f"  Building minimal ELF32 for RV32...", file=sys.stderr)
    elf_bytes = build_minimal_elf32(code, data)

    # Decide where to put the ELF
    elf_path = args.elf_output
    if not elf_path:
        elf_path = args.binary.replace(".bin", "_spike.elf")
    if not elf_path.endswith(".elf"):
        elf_path += ".elf"

    with open(elf_path, "wb") as f:
        f.write(elf_bytes)
    print(f"  ELF written: {elf_path} ({len(elf_bytes):,} bytes)", file=sys.stderr)
    if args.keep_elf or args.elf_output:
        print(f"  (ELF file preserved)", file=sys.stderr)

    # ── Run Spike ──────────────────────────────────────────────────────
    if args.log_commits:
        log_file = args.binary.replace(".bin", "_spike.log")
        max_for_log = min(args.max_instr, args.log_instr_limit)
        result, log_content = run_spike_with_log(
            elf_path,
            max_instr=max_for_log,
            log_path=log_file,
            isa=args.isa,
            mem_mb=args.mem,
            timeout_s=args.timeout,
        )
    else:
        result = run_spike(
            elf_path,
            max_instr=args.max_instr,
            ic_config=args.ic,
            dc_config=args.dc,
            track_pc=not args.no_pc_histogram,
            log_commits=False,
            timeout_s=args.timeout,
            isa=args.isa,
            mem_mb=args.mem,
        )

    # ── Generate report ────────────────────────────────────────────────
    if args.json:
        import json
        report = {
            "binary": args.binary,
            "code_size": code_size,
            "static_insns": code_size // 4,
            "max_instr": args.max_instr,
            "committed_insns": result.committed_insns,
            "wall_time_s": result.wall_time_s,
            "exit_code": result.exit_code,
            "icache": {
                "config": args.ic,
                "hits": result.icache_hits,
                "misses": result.icache_misses,
                "miss_rate_pct": result.icache_miss_rate,
            },
            "dcache": {
                "config": args.dc,
                "hits": result.dcache_hits,
                "misses": result.dcache_misses,
                "miss_rate_pct": result.dcache_miss_rate,
            },
            "top_pcs": sorted(
                [{"pc": f"0x{pc:08x}", "count": cnt}
                 for pc, cnt in result.pc_histogram.items()],
                key=lambda x: -x["count"]
            )[:15],
            "stderr_tail": result.stderr[-2000:] if result.stderr else "",
        }
        print(json.dumps(report, indent=2))
    else:
        print(generate_spike_report(
            result,
            args.binary,
            code_size,
            args.ic,
            args.dc,
            args.max_instr,
        ))

    # Cleanup temp ELF if not keeping
    if not (args.keep_elf or args.elf_output) and os.path.exists(elf_path):
        try:
            os.unlink(elf_path)
        except OSError:
            pass

    return 0 if result.exit_code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
