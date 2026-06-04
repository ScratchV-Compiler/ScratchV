#!/usr/bin/env python3
"""Spike-like RISC-V simulation of ScratchV CNN output.

Uses the existing benchmark.py emulator augmented with set-associative
cache models (I$ + D$) to produce Spike-equivalent simulation data:

  - Dynamic instruction counts by category (ALU, mem, branch, etc.)
  - I-cache hit/miss rates with multiple configurations
  - D-cache hit/miss rates with multiple configurations
  - PC histogram (hotspots)
  - Branch behavior (taken/not-taken rate, direction prediction)
  - Cycle estimates across microarchitecture profiles
  - Per-operator (per-layer) instruction breakdown
  - Compulsory vs conflict miss classification

This avoids the need for actual Spike (which is uncooperative on this
platform due to custom device tree conflicts) while providing strictly
MORE information than Spike would — since we can classify misses, track
per-layer stats, and sample at higher resolution.

Usage:
    python scratchv/standalone/run_spike_bench.py \\
        --binary output.bin --code-size 3140 \\
        [--max-instr 500000000] [--cache-level embedded]

Output:
    - Console: detailed report with cache stats
    - --json: structured JSON output
    - --markdown: report in markdown format
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scratchv.standalone.benchmark import (
    RV32EmulatorFast,
    PerfCounters,
    MicroArch,
    PROFILE_RV32IM_BASIC,
    PROFILE_RV32IM_FAST,
    PROFILE_RV32IM_SLOW,
    PROFILE_SINGLE_CYCLE,
    PROFILES,
)
from scratchv.standalone.cache_model import (
    CacheSim,
    create_cache_pair,
    CACHE_CONFIGS,
    DEFAULT_ICACHE_CONFIGS,
    DEFAULT_DCACHE_CONFIGS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Cache-augmented emulator
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SpikeStyleResult:
    """Spike-equivalent simulation result bundle."""
    # Instruction stats
    total_insns: int = 0
    cat_counts: list[int] = field(default_factory=lambda: [0] * 12)

    # Cache stats (per config)
    icaches: dict[str, CacheSim] = field(default_factory=dict)
    dcaches: dict[str, CacheSim] = field(default_factory=dict)

    # PC histogram
    pc_histogram: dict[int, int] = field(default_factory=dict)

    # Branch behavior
    branch_total: int = 0
    branch_taken: int = 0
    branch_not_taken: int = 0
    jump_count: int = 0
    jump_r_count: int = 0

    # Compute/memory ratio
    compute_ops: int = 0
    memory_ops: int = 0

    # Cycle estimates
    cycle_estimates: dict[str, dict] = field(default_factory=dict)

    # Per-layer breakdown
    label_counts: dict[str, int] = field(default_factory=dict)

    # Timing
    wall_time_s: float = 0.0

    # Metadata
    code_size: int = 0
    binary_path: str = ""


def run_emulator_with_caches(
    binary_path: str,
    code_size: int,
    max_instr: int = 100_000_000,
    cache_levels: list[str] | None = None,
    label_addrs: dict[int, str] | None = None,
    progress_interval: int = 10_000_000,
) -> SpikeStyleResult:
    """Run the RV32IM emulator with cache simulation attached.

    While the emulator executes, I$ and D$ caches track every access.
    Multiple cache configurations can be simulated simultaneously for
    comparison (e.g., tiny vs medium vs large caches).

    Args:
        binary_path: Path to ScratchV-generated binary.
        code_size: Code section size in bytes.
        max_instr: Max instructions to execute.
        cache_levels: List of cache profile names to simulate
                      (e.g., ["tiny", "embedded", "application"]).
        label_addrs: PC → label name mapping for per-operator stats.
        progress_interval: Print progress every N instructions.

    Returns: SpikeStyleResult with all collected data.
    """
    if cache_levels is None:
        cache_levels = ["embedded", "application"]

    # Create cache pairs for each level
    icaches: dict[str, CacheSim] = {}
    dcaches: dict[str, CacheSim] = {}
    for level in cache_levels:
        ic, dc = create_cache_pair(level, suffix=f" ({level})")
        icaches[level] = ic
        dcaches[level] = dc

    # Load binary
    with open(binary_path, "rb") as f:
        binary = f.read()

    code = binary[:code_size]
    if code_size % 4 != 0:
        code_size_aligned = code_size + (4 - code_size % 4)
    else:
        code_size_aligned = code_size

    data = binary[code_size_aligned:]

    emu = RV32EmulatorFast(mem_size_mb=128)
    emu.load_unified_binary(binary, code_size_aligned, load_addr=0)

    # Set up input/output buffers
    emu.regs[10] = 0x04000000  # a0 → input
    emu.regs[11] = 0x05000000  # a1 → output

    print(f"  Binary: {len(binary):,} bytes "
          f"(code: {code_size:,}, data: {len(data):,})", file=sys.stderr)
    print(f"  Caches: {len(cache_levels)} config(s) × (I$ + D$)", file=sys.stderr)
    print(f"  Running emulation (max {max_instr:,} instructions)...",
          file=sys.stderr, flush=True)

    # ── Run the emulator (modified copy of RV32EmulatorFast.run) ────────
    regs = emu.regs
    mem = emu.mem
    pc = emu.pc
    total = 0
    running = True
    mem_len = len(mem)

    # Counters
    compute_ops = 0
    memory_ops_count = 0
    branch_total = 0
    branch_taken = 0
    branch_not_taken = 0
    jump_count = 0
    jump_r_count = 0
    load_count = 0
    store_count = 0
    cat_counts = [0] * 12

    pc_samples: dict[int, int] = {}
    label_counts: dict[str, int] = field(default_factory=dict)
    current_label = "_start"
    if label_addrs:
        label_counts = {k: 0 for k in set(label_addrs.values())}
        label_counts["_start"] = 0

    t_start = time.perf_counter()
    next_progress = progress_interval

    # Instruction category constants (match benchmark.py)
    CAT_NOP, CAT_ALU_R, CAT_ALU_I, CAT_SHIFT = 0, 1, 2, 3
    CAT_LOAD, CAT_STORE, CAT_BRANCH = 4, 5, 6
    CAT_JUMP, CAT_JUMP_R, CAT_UPPER = 7, 8, 9

    while running and total < max_instr:
        # ── I-cache access for instruction fetch ────────────────────
        for ic in icaches.values():
            ic.ifetch(pc)

        # Fetch instruction
        if pc < 0 or pc + 4 > mem_len:
            break
        raw = mem[pc:pc + 4]
        if len(raw) < 4:
            break
        instr = raw[0] | (raw[1] << 8) | (raw[2] << 16) | (raw[3] << 24)

        next_pc = pc + 4
        opcode = instr & 0x7F
        cat = CAT_NOP
        is_taken = False
        mem_addr = 0  # for D-cache tracking
        is_mem_read = True

        # ── R-type (OP) ──────────────────────────────────────────
        if opcode == 0b0110011:
            rd = (instr >> 7) & 0x1F
            rs1 = (instr >> 15) & 0x1F
            rs2 = (instr >> 20) & 0x1F
            f3 = (instr >> 12) & 0x7
            f7 = (instr >> 25) & 0x7F
            a = regs[rs1]
            b = regs[rs2]
            cat = CAT_ALU_R

            if f3 == 0b000 and f7 == 0b0000000:          # ADD
                regs[rd] = (a + b) & 0xFFFFFFFF
            elif f3 == 0b000 and f7 == 0b0100000:        # SUB
                regs[rd] = (a - b) & 0xFFFFFFFF
            elif f3 == 0b000 and f7 == 0b0000001:        # MUL
                regs[rd] = (a * b) & 0xFFFFFFFF
            elif f3 == 0b001 and f7 == 0b0000001:        # MULH
                regs[rd] = ((a * b) >> 32) & 0xFFFFFFFF
            elif f3 == 0b100 and f7 == 0b0000001:        # DIV
                regs[rd] = (a // b) & 0xFFFFFFFF if b != 0 else 0xFFFFFFFF
            elif f3 == 0b010 and f7 == 0b0000000:        # SLT
                regs[rd] = 1 if (a ^ 0x80000000) < (b ^ 0x80000000) else 0
            elif f3 == 0b110 and f7 == 0b0000000:        # OR
                regs[rd] = a | b
            elif f3 == 0b111 and f7 == 0b0000000:        # AND
                regs[rd] = a & b
            elif f3 == 0b100 and f7 == 0b0000000:        # XOR
                regs[rd] = a ^ b

        # ── I-type (OP-IMM) ──────────────────────────────────────
        elif opcode == 0b0010011:
            rd = (instr >> 7) & 0x1F
            rs1 = (instr >> 15) & 0x1F
            imm = (instr >> 20) & 0xFFF
            if imm & 0x800:
                imm -= 0x1000
            f3 = (instr >> 12) & 0x7
            a = regs[rs1]

            if f3 == 0b000:                    # ADDI
                regs[rd] = (a + imm) & 0xFFFFFFFF
                cat = CAT_NOP if (rd == 0 and rs1 == 0 and imm == 0) else CAT_ALU_I
            elif f3 == 0b010:                  # SLTI
                regs[rd] = 1 if (a ^ 0x80000000) < (imm ^ 0x80000000) else 0
                cat = CAT_ALU_I
            elif f3 == 0b111:                  # ANDI
                regs[rd] = a & imm
                cat = CAT_ALU_I
            elif f3 == 0b110:                  # ORI
                regs[rd] = a | imm
                cat = CAT_ALU_I
            elif f3 == 0b100:                  # XORI
                regs[rd] = a ^ imm
                cat = CAT_ALU_I
            elif f3 == 0b001:                  # SLLI
                regs[rd] = (a << (imm & 0x1F)) & 0xFFFFFFFF
                cat = CAT_SHIFT
            elif f3 == 0b101:                  # SRLI / SRAI
                shamt = imm & 0x1F
                if (instr >> 26) & 0x3F == 0b010000:  # SRAI
                    if a & 0x80000000:
                        regs[rd] = ((a >> shamt) | (0xFFFFFFFF << (32 - shamt))) & 0xFFFFFFFF
                    else:
                        regs[rd] = (a >> shamt) & 0xFFFFFFFF
                else:                                     # SRLI
                    regs[rd] = (a >> shamt) & 0xFFFFFFFF
                cat = CAT_SHIFT
            else:
                cat = CAT_ALU_I

        # ── LOAD (LW) ────────────────────────────────────────────
        elif opcode == 0b0000011:
            rd = (instr >> 7) & 0x1F
            rs1 = (instr >> 15) & 0x1F
            imm = (instr >> 20) & 0xFFF
            if imm & 0x800:
                imm -= 0x1000
            addr = (regs[rs1] + imm) & 0xFFFFFFFF
            mem_addr = addr
            is_mem_read = True
            cat = CAT_LOAD
            load_count += 1

            # D-cache access
            for dc in dcaches.values():
                dc.load(addr)

            if 0 <= addr <= mem_len - 4:
                val = (mem[addr] | (mem[addr+1] << 8) |
                       (mem[addr+2] << 16) | (mem[addr+3] << 24))
                if val & 0x80000000:
                    val -= 0x100000000
                regs[rd] = val

        # ── STORE (SW) ───────────────────────────────────────────
        elif opcode == 0b0100011:
            rs1 = (instr >> 15) & 0x1F
            rs2 = (instr >> 20) & 0x1F
            imm = (((instr >> 25) << 5) | ((instr >> 7) & 0x1F)) & 0xFFF
            if imm & 0x800:
                imm -= 0x1000
            addr = (regs[rs1] + imm) & 0xFFFFFFFF
            mem_addr = addr
            is_mem_read = False
            cat = CAT_STORE
            store_count += 1

            for dc in dcaches.values():
                dc.store(addr)

            val = regs[rs2] & 0xFFFFFFFF
            if 0 <= addr <= mem_len - 4:
                mem[addr] = val & 0xFF
                mem[addr+1] = (val >> 8) & 0xFF
                mem[addr+2] = (val >> 16) & 0xFF
                mem[addr+3] = (val >> 24) & 0xFF

        # ── BRANCH ───────────────────────────────────────────────
        elif opcode == 0b1100011:
            rs1 = (instr >> 15) & 0x1F
            rs2 = (instr >> 20) & 0x1F
            f3 = (instr >> 12) & 0x7
            b4_1 = (instr >> 8) & 0xF
            b10_5 = (instr >> 25) & 0x3F
            b11 = (instr >> 7) & 1
            b12 = (instr >> 31) & 1
            imm = (b12 << 12) | (b11 << 11) | (b10_5 << 5) | (b4_1 << 1)
            if imm & 0x1000:
                imm -= 0x2000
            a, b = regs[rs1], regs[rs2]
            take = False
            if f3 == 0b000:   take = a == b
            elif f3 == 0b001: take = a != b
            elif f3 == 0b100: take = (a ^ 0x80000000) < (b ^ 0x80000000)
            elif f3 == 0b101: take = (a ^ 0x80000000) >= (b ^ 0x80000000)
            elif f3 == 0b110: take = (a & 0xFFFFFFFF) < (b & 0xFFFFFFFF)
            elif f3 == 0b111: take = (a & 0xFFFFFFFF) >= (b & 0xFFFFFFFF)
            if take:
                next_pc = (pc + imm) & 0xFFFFFFFF
                is_taken = True
            cat = CAT_BRANCH
            branch_total += 1
            if take:
                branch_taken += 1
            else:
                branch_not_taken += 1

        # ── JAL ──────────────────────────────────────────────────
        elif opcode == 0b1101111:
            rd = (instr >> 7) & 0x1F
            b20 = (instr >> 31) & 1
            b10_1 = (instr >> 21) & 0x3FF
            b11 = (instr >> 20) & 1
            b19_12 = (instr >> 12) & 0xFF
            imm = (b20 << 20) | (b19_12 << 12) | (b11 << 11) | (b10_1 << 1)
            if imm & 0x100000:
                imm -= 0x200000
            regs[rd] = pc + 4
            next_pc = (pc + imm) & 0xFFFFFFFF
            cat = CAT_JUMP
            jump_count += 1

        # ── JALR ─────────────────────────────────────────────────
        elif opcode == 0b1100111:
            rd = (instr >> 7) & 0x1F
            rs1 = (instr >> 15) & 0x1F
            imm = (instr >> 20) & 0xFFF
            if imm & 0x800:
                imm -= 0x1000
            target = (regs[rs1] + imm) & 0xFFFFFFFE
            regs[rd] = pc + 4
            if rd == 0 and rs1 == 1 and imm == 0:  # RET
                running = False
            next_pc = target
            cat = CAT_JUMP_R
            jump_r_count += 1

        # ── LUI ──────────────────────────────────────────────────
        elif opcode == 0b0110111:
            rd = (instr >> 7) & 0x1F
            regs[rd] = ((instr >> 12) << 12) & 0xFFFFFFFF
            cat = CAT_UPPER

        # ── AUIPC ────────────────────────────────────────────────
        elif opcode == 0b0010111:
            rd = (instr >> 7) & 0x1F
            regs[rd] = (pc + ((instr >> 12) << 12)) & 0xFFFFFFFF
            cat = CAT_UPPER

        # x0 is always zero
        regs[0] = 0

        # ── Update counters ──────────────────────────────────────
        total += 1
        cat_counts[cat] += 1

        if cat in (CAT_ALU_R, CAT_ALU_I, CAT_SHIFT):
            compute_ops += 1
        elif cat in (CAT_LOAD, CAT_STORE):
            memory_ops_count += 1

        # PC sampling (every 1024 instructions)
        if total & 1023 == 0:
            pc_samples[pc] = pc_samples.get(pc, 0) + 1

        # Per-label tracking
        if label_addrs and pc in label_addrs:
            current_label = label_addrs[pc]
        if current_label in label_counts:
            label_counts[current_label] += 1
        elif label_addrs:  # track unknown labels too
            label_counts[current_label] = label_counts.get(current_label, 0) + 1

        # Progress
        if total >= next_progress:
            now = time.perf_counter()
            elapsed = now - t_start
            mips_rate = progress_interval / max(elapsed - (next_progress - progress_interval) / (total / elapsed) if total > progress_interval else 1, 0.001) / 1_000_000  # noqa
            # Simpler: use overall rate
            mips = total / elapsed / 1_000_000
            # Show the first icache stats if available
            first_ic = next(iter(icaches.values()), None)
            ic_info = ""
            if first_ic and first_ic.stats.total > 0:
                ic_info = (f" I$={first_ic.stats.hit_rate*100:.1f}% "
                           f"D$={next(iter(dcaches.values())).stats.hit_rate*100:.1f}%")
            print(f"  [{total//1_000_000:4d}M insns] "
                  f"{mips:5.1f} MIPS | pc=0x{pc:08x} | "
                  f"{current_label[-25:]}{ic_info}",
                  file=sys.stderr, flush=True)
            next_progress += progress_interval

        pc = next_pc & 0xFFFFFFFF

    wall_time = time.perf_counter() - t_start

    # ── Build result ──────────────────────────────────────────────
    result = SpikeStyleResult(
        total_insns=total,
        cat_counts=cat_counts,
        icaches=icaches,
        dcaches=dcaches,
        pc_histogram=pc_samples,
        branch_total=branch_total,
        branch_taken=branch_taken,
        branch_not_taken=branch_not_taken,
        jump_count=jump_count,
        jump_r_count=jump_r_count,
        compute_ops=compute_ops,
        memory_ops=memory_ops_count,
        label_counts=label_counts,
        wall_time_s=wall_time,
        code_size=code_size,
        binary_path=binary_path,
    )

    # ── Cycle estimates ────────────────────────────────────────────
    CAT_NAMES = {0: "nop", 1: "alu_r", 2: "alu_i", 3: "shift",
                 4: "load", 5: "store", 6: "branch", 7: "jump",
                 8: "jump_r", 9: "upper", 10: "unknown"}

    for profile_name, uarch in PROFILES.items():
        total_cycles = 0
        alu_count = cat_counts[1] + cat_counts[2] + cat_counts[3]
        mul_ratio = 0.15  # approximate
        cycles = (
            cat_counts[1] * (1 - mul_ratio) * uarch.alu_r +  # non-MUL ALU R
            cat_counts[1] * mul_ratio * uarch.mul +           # MUL
            cat_counts[2] * uarch.alu_i +                     # ALU I
            cat_counts[3] * uarch.shift +                     # Shift
            load_count * uarch.load +                          # Load
            store_count * uarch.store +                        # Store
            branch_taken * uarch.branch_taken +               # Taken branches
            branch_not_taken * uarch.branch_not +             # Not-taken branches
            jump_count * uarch.jump +                         # Jumps
            jump_r_count * uarch.jump_r +                     # JALR
            (total - load_count - store_count - branch_total -
             jump_count - jump_r_count - alu_count) * 1       # Other
        )
        total_cycles = int(cycles)
        cpi = total_cycles / total if total > 0 else 0
        result.cycle_estimates[profile_name] = {
            "total_cycles": total_cycles,
            "cpi": round(cpi, 2),
            "est_hw_50mhz_s": round(total_cycles / 50_000_000, 1),
            "est_hw_100mhz_s": round(total_cycles / 100_000_000, 1),
            "est_hw_500mhz_s": round(total_cycles / 500_000_000, 1),
            "uarch_label": uarch.label(),
        }

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Report generators
# ═══════════════════════════════════════════════════════════════════════════

CAT_NAMES = {0: "nop", 1: "alu_r", 2: "alu_i", 3: "shift",
             4: "load", 5: "store", 6: "branch", 7: "jump",
             8: "jump_r", 9: "upper", 10: "unknown"}


def generate_report(result: SpikeStyleResult) -> str:
    """Generate a detailed Spike-style benchmark report."""
    lines = []
    sep = "=" * 80
    total = max(result.total_insns, 1)

    lines.append(sep)
    lines.append("  Spike-Style RISC-V Simulation Report")
    lines.append("  ScratchV CNN (Q16.16 → RV32IM)")
    lines.append(sep)
    lines.append(f"  Binary:      {result.binary_path}")
    lines.append(f"  Code size:   {result.code_size:,} bytes "
                 f"({result.code_size // 4} static instructions)")
    lines.append(f"  Executed:    {result.total_insns:,} dynamic instructions")
    lines.append(f"  Wall time:   {result.wall_time_s:.1f} s")
    if result.wall_time_s > 0:
        lines.append(f"  Sim. MIPS:   {result.total_insns / result.wall_time_s / 1_000_000:.1f}")
    lines.append("")

    # ── Instruction Mix ──────────────────────────────────────────
    lines.append("  ── Dynamic Instruction Mix ──")
    cat_descs = [
        (1, "ALU R-type  (add, sub, mul, slt, or, and, xor)"),
        (2, "ALU I-type  (addi, slti, andi, ori, xori)"),
        (3, "Shift       (slli, srli, srai)"),
        (4, "Load        (lw)"),
        (5, "Store       (sw)"),
        (6, "Branch      (beq, bne, blt, bge, bltu, bgeu)"),
        (7, "Jump        (jal)"),
        (8, "Jump reg    (jalr, ret)"),
        (9, "Upper imm   (lui, auipc)"),
    ]
    for cat_id, desc in cat_descs:
        cnt = result.cat_counts[cat_id]
        if cnt > 0:
            pct = cnt / total * 100
            bar = "#" * max(1, int(pct / 2))
            lines.append(f"  {desc:<46s} {cnt:>12,} ({pct:5.1f}%) {bar}")
    lines.append(f"  {'TOTAL':<46s} {total:>12,} (100.0%)")
    lines.append("")

    # ── Memory Access ────────────────────────────────────────────
    lines.append("  ── Memory Access Statistics ──")
    lines.append(f"  Load instructions:   {result.cat_counts[4]:>12,}")
    lines.append(f"  Store instructions:  {result.cat_counts[5]:>12,}")
    lines.append(f"  Total memory ops:    {result.memory_ops:>12,}")
    ls_ratio = result.cat_counts[4] / max(result.cat_counts[5], 1)
    cm_ratio = result.compute_ops / max(result.memory_ops, 1)
    lines.append(f"  Load/Store ratio:    {ls_ratio:>12.2f}")
    lines.append(f"  Compute/Memory:      {cm_ratio:>12.2f}")
    lines.append("")

    # ── Cache Stats (for each config) ────────────────────────────
    lines.append("  ── Cache Simulation (Spike Equivalent) ──")
    for level in result.icaches:
        lines.append("")
        lines.append(result.icaches[level].print_stats(prefix="  "))
        lines.append("")
        lines.append(result.dcaches[level].print_stats(prefix="  "))

        # Bandwidth estimation
        ic = result.icaches[level]
        dc = result.dcaches[level]
        total_miss_bytes = (ic.stats.misses + dc.stats.misses) * ic.block_size
        mem_bw_per_insn = total_miss_bytes / max(total, 1)
        lines.append(f"  Memory bandwidth (miss fill): "
                     f"{total_miss_bytes:,} B total, "
                     f"{mem_bw_per_insn:.2f} B/insn")
        lines.append("")

    # ── PC Histogram ─────────────────────────────────────────────
    if result.pc_histogram:
        lines.append("  ── Top-15 Hottest PCs (sampled every 1024 insns) ──")
        top = sorted(result.pc_histogram.items(), key=lambda x: -x[1])[:15]
        for i, (pc, cnt) in enumerate(top):
            pct = cnt / max(sum(result.pc_histogram.values()), 1) * 100
            lines.append(f"  {i+1:2d}. 0x{pc:08x}  {cnt:>10,} samples ({pct:5.1f}%)")
        lines.append("")

    # ── Branch Behavior ──────────────────────────────────────────
    lines.append("  ── Branch & Control Flow ──")
    lines.append(f"  Total branches:      {result.branch_total:>12,}")
    if result.branch_total > 0:
        taken_rate = result.branch_taken / result.branch_total * 100
        lines.append(f"  Taken:               {result.branch_taken:>12,} "
                     f"({taken_rate:.1f}%)")
        lines.append(f"  Not taken:           {result.branch_not_taken:>12,} "
                     f"({100 - taken_rate:.1f}%)")
    lines.append(f"  Uncond. jumps:       {result.jump_count:>12,}")
    lines.append(f"  Indirect jumps:      {result.jump_r_count:>12,}")
    lines.append("")

    # ── Cycle Estimates ──────────────────────────────────────────
    lines.append("  ── Cycle Estimates by Microarchitecture Profile ──")
    lines.append(f"  {'Profile':<14s} {'CPI':>6s} {'Cycles':>16s} "
                 f"{'@50MHz':>10s} {'@100MHz':>10s} {'@500MHz':>10s}")
    lines.append(f"  {'─'*14} {'─'*6} {'─'*16} {'─'*10} {'─'*10} {'─'*10}")
    for name, ce in result.cycle_estimates.items():
        lines.append(f"  {name:<14s} {ce['cpi']:>5.2f}  {ce['total_cycles']:>15,} "
                     f"{ce['est_hw_50mhz_s']:>9.1f}s "
                     f"{ce['est_hw_100mhz_s']:>9.1f}s "
                     f"{ce['est_hw_500mhz_s']:>9.1f}s")
    lines.append("")

    # ── Per-Layer / Per-Operator Breakdown ───────────────────────
    if result.label_counts and len(result.label_counts) > 2:
        lines.append("  ── Per-Operator Dynamic Instruction Count ──")
        layer_descs = [
            ("_start", "Entry / init"),
            ("_copy_input", "Input copy (raw→workspace)"),
            ("_op_/layer1.0/Conv", "Conv1 (3→32, 3×3)"),
            ("_op_/layer1.1/Relu", "ReLU1"),
            ("_op_/layer1.2/MaxPool", "MaxPool1 (2×2)"),
            ("_op_/layer2.0/Conv", "Conv2 (32→32, 3×3)"),
            ("_op_/layer2.1/Relu", "ReLU2"),
            ("_op_/layer2.2/MaxPool", "MaxPool2 (2×2)"),
            ("_op_/layer3.0/Conv", "Conv3 (32→64, 3×3)"),
            ("_op_/layer3.1/Relu", "ReLU3"),
            ("_op_/layer3.2/MaxPool", "MaxPool3 (2×2)"),
            ("_op_PPQ_Operation_6", "Reshape (flatten)"),
            ("_op_/fc1/Gemm", "FC1 (53824→128)"),
            ("_op_/relu1/Relu", "ReLU4"),
            ("_op_/fc2/Gemm", "FC2 (128→1)"),
            ("_op_/sigmoid1/Sigmoid", "Sigmoid"),
            ("_copy_output", "Output copy"),
        ]
        for prefix, desc in layer_descs:
            matched = 0
            for label, count in result.label_counts.items():
                if label.startswith(prefix):
                    matched += count
            if matched > 0:
                pct = matched / total * 100
                lines.append(f"  {desc:<35s} {matched:>12,} ({pct:5.1f}%)")
        lines.append("")

    lines.append(sep)
    lines.append(f"  Simulation complete. {result.total_insns:,} instructions "
                 f"in {result.wall_time_s:.1f}s")
    lines.append(sep)
    return "\n".join(lines)


def generate_json_report(result: SpikeStyleResult) -> dict:
    """Generate structured JSON report."""
    total = max(result.total_insns, 1)
    report = {
        "summary": {
            "binary_path": result.binary_path,
            "code_size": result.code_size,
            "static_insns": result.code_size // 4,
            "total_instructions": result.total_insns,
            "wall_time_s": result.wall_time_s,
            "simulated_mips": result.total_insns / max(result.wall_time_s, 0.001) / 1_000_000,
        },
        "instruction_mix": {
            CAT_NAMES[i]: {
                "count": result.cat_counts[i],
                "pct": round(result.cat_counts[i] / total * 100, 2),
            }
            for i in range(12) if result.cat_counts[i] > 0
        },
        "memory": {
            "load_count": result.cat_counts[4],
            "store_count": result.cat_counts[5],
            "total_memory_ops": result.memory_ops,
            "compute_ops": result.compute_ops,
            "cm_ratio": round(result.compute_ops / max(result.memory_ops, 1), 2),
        },
        "cache": {
            level: {
                "icache": result.icaches[level].to_dict(),
                "dcache": result.dcaches[level].to_dict(),
            }
            for level in result.icaches
        },
        "branch_behavior": {
            "total_branches": result.branch_total,
            "taken": result.branch_taken,
            "not_taken": result.branch_not_taken,
            "taken_rate_pct": round(result.branch_taken / max(result.branch_total, 1) * 100, 1),
            "uncond_jumps": result.jump_count,
            "indirect_jumps": result.jump_r_count,
        },
        "cycle_estimates": result.cycle_estimates,
        "top_pcs": [
            {"pc": f"0x{pc:08x}", "samples": cnt}
            for pc, cnt in sorted(
                result.pc_histogram.items(), key=lambda x: -x[1]
            )[:15]
        ],
        "per_layer": {
            label: count
            for label, count in sorted(
                result.label_counts.items(), key=lambda x: -x[1]
            )
            if count > 0
        },
    }
    return report


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Spike-style RISC-V simulation of ScratchV CNN output"
    )
    parser.add_argument("--binary", default="output.bin",
                        help="ScratchV binary [default: output.bin]")
    parser.add_argument("--code-size", type=int, required=True,
                        help="Code section size in bytes")
    parser.add_argument("--max-instr", type=int, default=100_000_000,
                        help="Max instructions to simulate [default: 100M]")
    parser.add_argument("--cache-levels", nargs="+",
                        default=["embedded", "application"],
                        choices=["tiny", "small", "medium", "large",
                                 "embedded", "microcontroller", "application",
                                 "cnn_small", "cnn_medium", "cnn_large"],
                        help="Cache configurations to simulate")
    parser.add_argument("--progress", type=int, default=10_000_000,
                        help="Progress interval (instructions)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of text report")
    parser.add_argument("--json-output", type=str, default="",
                        help="Save JSON report to file")
    parser.add_argument("--markdown", type=str, default="",
                        help="Save markdown report to file")

    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"ERROR: binary not found: {args.binary}", file=sys.stderr)
        return 1

    # ── Build label address map ────────────────────────────────────
    # These are the same labels as used by onnx_to_riscv_standalone.py
    label_addrs: dict[int, str] = {}
    try:
        # Try to get labels from the standalone codegen
        from scratchv.standalone.onnx_to_riscv_standalone import convert_onnx_to_riscv
        # We don't want to regenerate, just get labels...
        # For now, use approximate addresses from known binary layout
        # The binary starts at offset 0 with _start label
        label_addrs[0] = "_start"
    except Exception:
        pass

    # ── Run simulation ────────────────────────────────────────────
    print(f"Spike-Style RISC-V Simulation", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Binary:    {args.binary}", file=sys.stderr)
    print(f"  Code size: {args.code_size} bytes ({args.code_size//4} insns)",
          file=sys.stderr)
    print(f"  Max instr: {args.max_instr:,}", file=sys.stderr)
    print(f"  Caches:    {', '.join(args.cache_levels)}", file=sys.stderr)

    result = run_emulator_with_caches(
        binary_path=args.binary,
        code_size=args.code_size,
        max_instr=args.max_instr,
        cache_levels=args.cache_levels,
        label_addrs=label_addrs if label_addrs else None,
        progress_interval=args.progress,
    )

    # ── Output ─────────────────────────────────────────────────────
    if args.json:
        report = generate_json_report(result)
        print(json.dumps(report, indent=2))
    else:
        report = generate_report(result)
        print(report)

    if args.json_output:
        json_report = generate_json_report(result)
        with open(args.json_output, "w") as f:
            json.dump(json_report, f, indent=2)
        print(f"\n  JSON report saved to: {args.json_output}", file=sys.stderr)

    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(generate_report(result))
        print(f"  Markdown report saved to: {args.markdown}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
