#!/usr/bin/env python3
"""RISC-V RV32IM performance benchmark emulator (library-free, optimized).

Executes a ScratchV-generated RISC-V binary and collects detailed per-instruction
performance metrics. Python 3.8+ stdlib only, zero external dependencies.

Metrics:
  - Dynamic instruction counts by category (ALU, mem, branch, jump, upper)
  - Instruction mix percentages with visual bar chart
  - Memory access stats (loads/stores/ratio)
  - Branch behaviour (taken / not-taken rate)
  - Compute-to-memory ratio
  - Host execution time & simulated MIPS
  - Per-operator (per-label) instruction counts
  - Top-10 hottest PC addresses (sampled)
  - Progress every 10M instructions

Usage:
    python benchmark.py output.bin --code-size 3140 [--max-instr 50000000]
"""

from __future__ import annotations

import struct
import sys
import os
import time
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════
# Optimized RV32IM emulator with inline performance counters
# ═══════════════════════════════════════════════════════════════════════════


def _sext(v: int, bits: int) -> int:
    mask = (1 << bits) - 1
    v &= mask
    if v >> (bits - 1):
        v -= 1 << bits
    return v


def _u32(v: int) -> int:
    return v & 0xFFFFFFFF


# Instruction category constants
CAT_NOP = 0
CAT_ALU_R = 1
CAT_ALU_I = 2
CAT_SHIFT = 3
CAT_LOAD = 4
CAT_STORE = 5
CAT_BRANCH = 6
CAT_JUMP = 7
CAT_JUMP_R = 8
CAT_UPPER = 9
CAT_UNKNOWN = 10

CAT_NAMES = {
    0: "nop", 1: "alu_r", 2: "alu_i", 3: "shift", 4: "load",
    5: "store", 6: "branch", 7: "jump", 8: "jump_r", 9: "upper",
    10: "unknown",
}


class PerfCounters:
    """Fast performance counters updated inline during emulation."""

    __slots__ = (
        "total", "cat_counts", "compute_ops", "memory_ops",
        "load_count", "store_count", "branch_total", "branch_taken",
        "branch_not_taken", "jump_count", "jump_r_count", "ret_count",
        "pc_samples", "label_counts", "label_addrs",
        "t_start", "t_end",
    )

    def __init__(self, label_addrs: dict[int, str] | None = None):
        self.total: int = 0
        self.cat_counts: list[int] = [0] * 12
        self.compute_ops: int = 0
        self.memory_ops: int = 0
        self.load_count: int = 0
        self.store_count: int = 0
        self.branch_total: int = 0
        self.branch_taken: int = 0
        self.branch_not_taken: int = 0
        self.jump_count: int = 0
        self.jump_r_count: int = 0
        self.ret_count: int = 0
        self.pc_samples: dict[int, int] = {}
        self.label_counts: dict[str, int] = defaultdict(int)
        self.label_addrs: dict[int, str] = label_addrs or {}
        self.t_start: float = 0.0
        self.t_end: float = 0.0

    @property
    def elapsed(self) -> float:
        return self.t_end - self.t_start

    @property
    def mips(self) -> float:
        e = self.elapsed
        return self.total / e / 1_000_000 if e > 0 else 0.0

    @property
    def cm_ratio(self) -> float:
        return self.compute_ops / self.memory_ops if self.memory_ops > 0 else float("inf")

    @property
    def ls_ratio(self) -> float:
        return self.load_count / self.store_count if self.store_count > 0 else float("inf")

    @property
    def branch_rate(self) -> float:
        return self.branch_taken / self.branch_total if self.branch_total > 0 else 0.0


class RV32EmulatorFast:
    """Optimized RV32IM emulator with inline performance counting.

    Executes a flat RISC-V binary (code + embedded Q16.16 weight data).
    """

    def __init__(self, mem_size_mb: int = 128):
        self.mem = bytearray(mem_size_mb * 1024 * 1024)
        self.regs = [0] * 32
        self.pc = 0
        self.regs[2] = (mem_size_mb // 2) * 1024 * 1024
        self._running = False

    def load_unified_binary(self, binary: bytes, code_size: int,
                            load_addr: int = 0) -> None:
        """Load code+data binary. Code at load_addr, data immediately after (4B aligned)."""
        data_offset = code_size
        if data_offset % 4 != 0:
            data_offset += 4 - (data_offset % 4)
        self.mem[load_addr:load_addr + code_size] = binary[:code_size]
        data = binary[code_size:]
        self.data_addr = load_addr + data_offset
        self.mem[self.data_addr:self.data_addr + len(data)] = data
        self.pc = load_addr

    def run(self, max_instr: int = 2_000_000_000,
            label_addrs: dict[int, str] | None = None,
            progress_interval: int = 10_000_000) -> PerfCounters:
        """Execute until RET or max_instr. Collects all performance metrics.

        Optimizations for speed:
          - Inline instruction classification in the execution switch
          - Local variable caching for regs[], mem, pc
          - Avoid function calls in the hot loop
          - Only sample PC every 1000 instructions for histogram
        """
        p = PerfCounters(label_addrs)
        p.t_start = time.perf_counter()

        regs = self.regs
        mem = self.mem
        pc = self.pc
        total = 0
        cat_counts = p.cat_counts
        running = True
        mem_len = len(mem)

        # Local bindings for speed
        cat_alu_r = CAT_ALU_R
        cat_alu_i = CAT_ALU_I
        cat_shift = CAT_SHIFT
        cat_load = CAT_LOAD
        cat_store = CAT_STORE
        cat_branch = CAT_BRANCH
        cat_jump = CAT_JUMP
        cat_jump_r = CAT_JUMP_R
        cat_upper = CAT_UPPER
        cat_nop = CAT_NOP

        # Per-layer tracking: current label name
        label_addrs_map = label_addrs or {}
        current_label = "_start"
        p.label_counts[current_label] = 0

        # Progress tracking
        next_progress = progress_interval
        last_progress_time = time.perf_counter()

        while running and total < max_instr:
            # Fetch
            if pc < 0 or pc + 4 > mem_len:
                break
            raw = mem[pc:pc + 4]
            if len(raw) < 4:
                break
            instr = raw[0] | (raw[1] << 8) | (raw[2] << 16) | (raw[3] << 24)

            next_pc = pc + 4
            opcode = instr & 0x7F
            cat = cat_nop  # default

            # ── R-type (OP) ──────────────────────────────────────────
            if opcode == 0b0110011:
                rd = (instr >> 7) & 0x1F
                rs1 = (instr >> 15) & 0x1F
                rs2 = (instr >> 20) & 0x1F
                f3 = (instr >> 12) & 0x7
                f7 = (instr >> 25) & 0x7F
                a = regs[rs1]
                b = regs[rs2]
                cat = cat_alu_r

                if f3 == 0b000 and f7 == 0b0000000:          # ADD
                    regs[rd] = _u32(a + b)
                elif f3 == 0b000 and f7 == 0b0100000:        # SUB
                    regs[rd] = _u32(a - b)
                elif f3 == 0b000 and f7 == 0b0000001:        # MUL
                    regs[rd] = _u32(a * b)
                elif f3 == 0b001 and f7 == 0b0000001:        # MULH
                    regs[rd] = _u32((a * b) >> 32)
                elif f3 == 0b100 and f7 == 0b0000001:        # DIV
                    regs[rd] = _u32(a // b) if b != 0 else 0xFFFFFFFF
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
                imm = _sext((instr >> 20) & 0xFFF, 12)
                f3 = (instr >> 12) & 0x7
                a = regs[rs1]

                if f3 == 0b000:                    # ADDI
                    regs[rd] = _u32(a + imm)
                    cat = cat_nop if (rd == 0 and rs1 == 0 and imm == 0) else cat_alu_i
                elif f3 == 0b010:                  # SLTI
                    regs[rd] = 1 if (a ^ 0x80000000) < (imm ^ 0x80000000) else 0
                    cat = cat_alu_i
                elif f3 == 0b111:                  # ANDI
                    regs[rd] = a & imm
                    cat = cat_alu_i
                elif f3 == 0b110:                  # ORI
                    regs[rd] = a | imm
                    cat = cat_alu_i
                elif f3 == 0b100:                  # XORI
                    regs[rd] = a ^ imm
                    cat = cat_alu_i
                elif f3 == 0b001:                  # SLLI
                    regs[rd] = _u32(a << (imm & 0x1F))
                    cat = cat_shift
                elif f3 == 0b101:                  # SRLI / SRAI
                    shamt = imm & 0x1F
                    if (instr >> 26) & 0x3F == 0b010000:  # SRAI
                        if a & 0x80000000:
                            regs[rd] = _u32((a >> shamt) | (0xFFFFFFFF << (32 - shamt)))
                        else:
                            regs[rd] = _u32(a >> shamt)
                    else:                                     # SRLI
                        regs[rd] = _u32(a >> shamt)
                    cat = cat_shift
                else:
                    cat = cat_alu_i

            # ── LOAD (LW) ────────────────────────────────────────────
            elif opcode == 0b0000011:
                rd = (instr >> 7) & 0x1F
                rs1 = (instr >> 15) & 0x1F
                imm = _sext((instr >> 20) & 0xFFF, 12)
                addr = _u32(regs[rs1] + imm)
                if 0 <= addr <= mem_len - 4:
                    regs[rd] = (mem[addr] | (mem[addr+1] << 8) |
                               (mem[addr+2] << 16) | (mem[addr+3] << 24))
                    # Sign-extend: if bit 31 set, make negative
                    if regs[rd] & 0x80000000:
                        regs[rd] -= 0x100000000
                cat = cat_load
                p.load_count += 1

            # ── STORE (SW) ───────────────────────────────────────────
            elif opcode == 0b0100011:
                rs1 = (instr >> 15) & 0x1F
                rs2 = (instr >> 20) & 0x1F
                imm = _sext(((instr >> 25) << 5) | ((instr >> 7) & 0x1F), 12)
                addr = _u32(regs[rs1] + imm)
                if 0 <= addr <= mem_len - 4:
                    val = regs[rs2] & 0xFFFFFFFF
                    mem[addr] = val & 0xFF
                    mem[addr+1] = (val >> 8) & 0xFF
                    mem[addr+2] = (val >> 16) & 0xFF
                    mem[addr+3] = (val >> 24) & 0xFF
                cat = cat_store
                p.store_count += 1

            # ── BRANCH ───────────────────────────────────────────────
            elif opcode == 0b1100011:
                rs1 = (instr >> 15) & 0x1F
                rs2 = (instr >> 20) & 0x1F
                f3 = (instr >> 12) & 0x7
                # Decode B-immediate
                b4_1 = (instr >> 8) & 0xF
                b10_5 = (instr >> 25) & 0x3F
                b11 = (instr >> 7) & 1
                b12 = (instr >> 31) & 1
                imm = _sext((b12 << 12) | (b11 << 11) | (b10_5 << 5) | (b4_1 << 1), 13)
                a, b = regs[rs1], regs[rs2]
                take = False
                if f3 == 0b000:   take = a == b                                 # BEQ
                elif f3 == 0b001: take = a != b                                 # BNE
                elif f3 == 0b100: take = (a ^ 0x80000000) < (b ^ 0x80000000)    # BLT
                elif f3 == 0b101: take = (a ^ 0x80000000) >= (b ^ 0x80000000)   # BGE
                elif f3 == 0b110: take = (a & 0xFFFFFFFF) < (b & 0xFFFFFFFF)    # BLTU
                elif f3 == 0b111: take = (a & 0xFFFFFFFF) >= (b & 0xFFFFFFFF)   # BGEU
                if take:
                    next_pc = pc + imm
                cat = cat_branch
                p.branch_total += 1
                if take:
                    p.branch_taken += 1
                else:
                    p.branch_not_taken += 1

            # ── JAL ──────────────────────────────────────────────────
            elif opcode == 0b1101111:
                rd = (instr >> 7) & 0x1F
                b20 = (instr >> 31) & 1
                b10_1 = (instr >> 21) & 0x3FF
                b11 = (instr >> 20) & 1
                b19_12 = (instr >> 12) & 0xFF
                imm = _sext((b20 << 20) | (b19_12 << 12) | (b11 << 11) | (b10_1 << 1), 21)
                regs[rd] = pc + 4
                next_pc = pc + imm
                cat = cat_jump
                p.jump_count += 1

            # ── JALR ─────────────────────────────────────────────────
            elif opcode == 0b1100111:
                rd = (instr >> 7) & 0x1F
                rs1 = (instr >> 15) & 0x1F
                imm = _sext((instr >> 20) & 0xFFF, 12)
                target = _u32(regs[rs1] + imm) & 0xFFFFFFFE
                regs[rd] = pc + 4
                if rd == 0 and rs1 == 1 and imm == 0:  # RET
                    running = False
                    p.ret_count += 1
                next_pc = target
                cat = cat_jump_r
                p.jump_r_count += 1

            # ── LUI ──────────────────────────────────────────────────
            elif opcode == 0b0110111:
                rd = (instr >> 7) & 0x1F
                regs[rd] = (instr >> 12) << 12
                cat = cat_upper

            # ── AUIPC ────────────────────────────────────────────────
            elif opcode == 0b0010111:
                rd = (instr >> 7) & 0x1F
                regs[rd] = _u32(pc + ((instr >> 12) << 12))
                cat = cat_upper

            # x0 always zero
            regs[0] = 0

            # ── Update counters ──────────────────────────────────────
            total += 1
            cat_counts[cat] += 1

            if cat in (CAT_ALU_R, CAT_ALU_I, CAT_SHIFT):
                p.compute_ops += 1
            elif cat in (CAT_LOAD, CAT_STORE):
                p.memory_ops += 1

            # PC sampling (every 1024 instructions for histogram)
            if total & 1023 == 0:
                addr = pc
                p.pc_samples[addr] = p.pc_samples.get(addr, 0) + 1

            # Per-label tracking
            if pc in label_addrs_map:
                current_label = label_addrs_map[pc]
            p.label_counts[current_label] += 1

            # Progress reporting
            if total >= next_progress:
                now = time.perf_counter()
                elapsed = now - last_progress_time
                mips_rate = progress_interval / elapsed / 1_000_000 if elapsed > 0 else 0
                label_short = current_label[-30:] if len(current_label) > 30 else current_label
                print(f"  [{total//1_000_000:4d}M insns] "
                      f"{mips_rate:5.1f} MIPS | pc=0x{pc:08x} | {label_short}",
                      file=sys.stderr, flush=True)
                next_progress += progress_interval
                last_progress_time = now

            pc = next_pc

        # Store back
        self.regs[0] = 0
        self.pc = pc
        p.total = total
        p.t_end = time.perf_counter()
        return p

    def read_mem_i32(self, addr: int) -> int:
        if addr + 4 > len(self.mem):
            return 0
        return (self.mem[addr] | (self.mem[addr+1] << 8) |
                (self.mem[addr+2] << 16) | (self.mem[addr+3] << 24))


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════════


def run_benchmark(binary_path: str, code_size: int,
                  input_data: bytes = b"",
                  load_addr: int = 0,
                  max_instr: int = 2_000_000_000,
                  label_addrs: dict[int, str] | None = None,
                  verbose: bool = True) -> PerfCounters:
    """Run a ScratchV-generated RISC-V binary through the emulator.

    Args:
        binary_path: Path to unified binary (code + data).
        code_size: Size of code section in bytes.
        input_data: Q16.16 input tensor bytes (optional, placed at a0).
        load_addr: Memory address where binary is loaded.
        max_instr: Max instructions to execute.
        label_addrs: Dict mapping PC addresses → label names for per-operator stats.
        verbose: Print setup info.
    """
    with open(binary_path, "rb") as f:
        binary = f.read()

    emu = RV32EmulatorFast(mem_size_mb=128)
    emu.load_unified_binary(binary, code_size, load_addr)

    # Place input at a0
    if input_data:
        input_buf = 0x04000000
        emu.regs[10] = input_buf
        for i, b in enumerate(input_data):
            emu.mem[input_buf + i] = b

    # a1 → output buffer
    emu.regs[11] = 0x05000000

    if verbose:
        print(f"  Binary:   {len(binary):,} B (code: {code_size:,} B, "
              f"data: {len(binary) - code_size:,} B)")
        print(f"  Memory:   {len(emu.mem) // 1024 // 1024} MB")
        print(f"  Entry PC: 0x{load_addr:08x}  SP: 0x{emu.regs[2]:08x}")
        print(f"  Input @ 0x{emu.regs[10]:08x}  Output @ 0x{emu.regs[11]:08x}")
        print(f"  Running (max {max_instr:,} instructions)...",
              flush=True)

    perf = emu.run(max_instr=max_instr, label_addrs=label_addrs)

    if verbose:
        result_addr = emu.regs[11]  # a1
        result_val = emu.read_mem_i32(result_addr)
        result_float = result_val / 65536.0
        print(f"\n  Output Q16.16: {result_val}  (≈ {result_float:.6f} float)",
              flush=True)

    return perf


# ═══════════════════════════════════════════════════════════════════════════
# Report formatter
# ═══════════════════════════════════════════════════════════════════════════


def format_benchmark_report(perf: PerfCounters,
                            binary_path: str = "",
                            code_size: int = 0) -> str:
    """Generate a detailed human-readable benchmark report."""
    lines = []
    sep = "=" * 72
    total = max(perf.total, 1)
    cat = perf.cat_counts

    lines.append(sep)
    lines.append("  ScratchV CNN RISC-V Performance Benchmark Report")
    lines.append(sep)
    if binary_path:
        lines.append(f"  Binary:     {binary_path}")
    if code_size:
        lines.append(f"  Code size:  {code_size:,} B ({code_size // 4} static insns)")
    lines.append(f"  Executed:   {total:,} dynamic instructions")
    lines.append("")

    # Timing
    lines.append("  ── Host Execution Timing ──")
    lines.append(f"  Wall time:     {perf.elapsed:.2f} s")
    lines.append(f"  Simulated MIPS: {perf.mips:.1f}")
    est_hw_50mhz = total / 50_000_000
    lines.append(f"  Est. HW time:  {est_hw_50mhz:.2f} s (@ 50 MHz, CPI=1)")
    lines.append("")

    # Instruction Mix
    lines.append("  ── Dynamic Instruction Mix ──")
    categories = [
        (CAT_ALU_R, "ALU (R-type: add/sub/mul/div/slt/or/and/xor)"),
        (CAT_ALU_I, "ALU immediate (addi/slti/andi/ori/xori)"),
        (CAT_SHIFT, "Shift (slli/srli/srai)"),
        (CAT_LOAD, "Load (lw)"),
        (CAT_STORE, "Store (sw)"),
        (CAT_BRANCH, "Branch (beq/bne/blt/bge)"),
        (CAT_JUMP, "Jump (jal/j)"),
        (CAT_JUMP_R, "Jump register (jalr/ret)"),
        (CAT_UPPER, "Upper immediate (lui/auipc)"),
        (CAT_NOP, "NOP"),
    ]
    for cat_id, desc in categories:
        count = cat[cat_id]
        pct = count / total * 100
        bar = "#" * int(pct / 2) if pct >= 0.5 else ""
        lines.append(f"  {CAT_NAMES[cat_id]:<10s} {count:>12,} ({pct:5.1f}%) {bar}")
    lines.append(f"  {'TOTAL':<10s} {total:>12,} (100.0%)")
    lines.append("")

    # Memory Access
    lines.append("  ── Memory Access Statistics ──")
    lines.append(f"  Load instructions:  {perf.load_count:>12,}")
    lines.append(f"  Store instructions: {perf.store_count:>12,}")
    lines.append(f"  Total memory ops:   {perf.memory_ops:>12,}")
    lines.append(f"  Load/Store ratio:   {perf.ls_ratio:>12.2f}")
    lines.append("")

    # Compute/Memory Ratio
    lines.append("  ── Compute-to-Memory Ratio ──")
    cpu_pct = perf.compute_ops / total * 100
    mem_pct = perf.memory_ops / total * 100
    classification = ("compute-heavy" if perf.cm_ratio > 2 else
                      "memory-heavy" if perf.cm_ratio < 0.5 else "balanced")
    lines.append(f"  Compute ops:  {perf.compute_ops:>12,} ({cpu_pct:.1f}%)")
    lines.append(f"  Memory ops:   {perf.memory_ops:>12,} ({mem_pct:.1f}%)")
    lines.append(f"  C/M ratio:    {perf.cm_ratio:>12.2f} ({classification})")
    lines.append("")

    # Branch Behavior
    lines.append("  ── Branch & Control Flow ──")
    lines.append(f"  Total branches:    {perf.branch_total:>12,}")
    lines.append(f"  Taken:             {perf.branch_taken:>12,} "
                 f"({perf.branch_rate * 100:.1f}%)")
    lines.append(f"  Not taken:         {perf.branch_not_taken:>12,} "
                 f"({(1 - perf.branch_rate) * 100:.1f}%)")
    lines.append(f"  Uncond. jumps:     {perf.jump_count:>12,}")
    lines.append(f"  Indirect jumps:    {perf.jump_r_count:>12,}")
    lines.append(f"  RET instructions:  {perf.ret_count:>12,}")
    lines.append("")

    # Hot PCs (sampled)
    if perf.pc_samples:
        lines.append("  ── Top-10 Hottest PCs (sampled every 1024 insns) ──")
        top_pc = sorted(perf.pc_samples.items(), key=lambda x: -x[1])[:10]
        for i, (pc, cnt) in enumerate(top_pc):
            lines.append(f"  {i+1:2d}. 0x{pc:08x}  {cnt:>8,} samples")
        lines.append("")

    # Per-operator breakdown
    if perf.label_counts and len(perf.label_counts) > 2:
        lines.append("  ── Per-Operator Dynamic Instruction Count ──")
        label_descs = [
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
            ("_op_PPQ_Operation_6", "Reshape (flatten 53824 el)"),
            ("_op_/fc1/Gemm", "FC1 (53824→128)"),
            ("_op_/relu1/Relu", "ReLU4 (128 el)"),
            ("_op_/fc2/Gemm", "FC2 (128→1)"),
            ("_op_/sigmoid1/Sigmoid", "Sigmoid"),
            ("_op_PPQ_Operation_12", "Reshape (final)"),
            ("_copy_output", "Output copy (workspace→out)"),
        ]
        for prefix, desc in label_descs:
            matched = 0
            for label, count in perf.label_counts.items():
                if label.startswith(prefix):
                    matched += count
            if matched > 0:
                pct = matched / total * 100
                lines.append(f"  {desc:<35s} {matched:>12,} ({pct:5.1f}%)")
        lines.append(f"  {'─'*35}  {'─'*12}")
        lines.append(f"  {'TOTAL':<35s} {total:>12,} (100.0%)")
        lines.append("")

    lines.append(sep)
    lines.append(f"  Benchmark complete. {total:,} instructions in {perf.elapsed:.2f}s")
    lines.append(f"  Host: {perf.mips:.1f} MIPS  |  "
                 f"Est. HW @50MHz CPI=1: {total / 50_000_000:.1f}s")
    lines.append(sep)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry
# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="ScratchV RISC-V Performance Benchmark (library-free)"
    )
    parser.add_argument("binary", help="Path to ScratchV-generated .bin file")
    parser.add_argument("--code-size", type=int, required=True,
                        help="Code section size in bytes")
    parser.add_argument("--max-instr", type=int, default=2_000_000_000,
                        help="Max instructions before timeout")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Error: binary not found: {args.binary}", file=sys.stderr)
        return 1

    perf = run_benchmark(
        binary_path=args.binary,
        code_size=args.code_size,
        max_instr=args.max_instr,
        verbose=True,
    )

    if args.json:
        import json
        result = {
            "total_instructions": perf.total,
            "host_elapsed_sec": perf.elapsed,
            "mips": perf.mips,
            "category_counts": {CAT_NAMES[i]: c for i, c in enumerate(perf.cat_counts) if c > 0},
            "load_count": perf.load_count,
            "store_count": perf.store_count,
            "compute_memory_ratio": perf.cm_ratio,
            "load_store_ratio": perf.ls_ratio,
            "branch_total": perf.branch_total,
            "branch_taken": perf.branch_taken,
            "branch_taken_rate": perf.branch_rate,
        }
        print(json.dumps(result, indent=2))
    else:
        print(format_benchmark_report(perf, args.binary, args.code_size))

    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Analytical instruction estimator (instant, no emulation needed)
# ═══════════════════════════════════════════════════════════════════════════


def estimate_cnn_instructions(
    input_shape: tuple[int, ...],
    conv_layers: list[dict],
    fc_layers: list[dict],
) -> dict:
    """Estimate dynamic instruction counts analytically from CNN dimensions.

    Args:
        input_shape: (N, C, H, W) input tensor shape.
        conv_layers: List of dicts with keys:
            out_c, kernel, stride, pad, has_relu, has_maxpool, pool_kernel, pool_stride
        fc_layers: List of dicts with keys: in_dim, out_dim, has_relu, has_sigmoid

    Returns:
        Dict with estimated instruction counts by category.
    """
    N, C, H, W = input_shape

    # Instruction counts per MAC (inner loop body)
    # For Conv: ~30 insns per MAC iteration (addr calc + load + mul + srai + add + loop)
    # For FC:   ~15 insns per MAC iteration
    CONV_INSNS_PER_MAC = 30
    FC_INSNS_PER_MAC = 15

    # Per-element ops (ReLU, MaxPool, Sigmoid, Reshape)
    RELU_INSNS_PER_EL = 8
    MAXPOOL_INSNS_PER_OUT_EL = 12
    SIGMOID_INSNS_PER_EL = 20
    RESHAPE_INSNS_PER_EL = 8

    total_compute = 0
    total_memory = 0
    total_branch = 0
    total_insns = 0
    layer_insns: dict[str, int] = {}

    current_shape = (N, C, H, W)

    for i, cl in enumerate(conv_layers):
        name = f"Conv{i+1}"
        out_c = cl["out_c"]
        K = cl.get("kernel", 3)
        stride = cl.get("stride", 1)
        pad = cl.get("pad", 0)
        _, c_in, h_in, w_in = current_shape

        h_out = (h_in + 2 * pad - K) // stride + 1
        w_out = (w_in + 2 * pad - K) // stride + 1
        macs = out_c * h_out * w_out * c_in * K * K
        insns = macs * CONV_INSNS_PER_MAC
        total_insns += insns
        total_compute += int(macs * 22)  # ~22 compute insns per MAC
        total_memory += int(macs * 6)    # ~6 memory insns per MAC
        total_branch += int(macs * 2)    # ~2 branches per MAC
        layer_insns[name] = insns

        total_insns += out_c * h_out * w_out * 4  # bias load + loop increments
        current_shape = (N, out_c, h_out, w_out)

        if cl.get("has_relu", True):
            el = out_c * h_out * w_out
            insns = el * RELU_INSNS_PER_EL
            total_insns += insns
            total_compute += insns
            layer_insns[f"ReLU{i+1}"] = insns

        if cl.get("has_maxpool", False):
            pk = cl.get("pool_kernel", 2)
            ps = cl.get("pool_stride", 2)
            ph_out = (h_out - pk) // ps + 1
            pw_out = (w_out - pk) // ps + 1
            out_el = N * out_c * ph_out * pw_out
            insns = out_el * pk * pk * MAXPOOL_INSNS_PER_OUT_EL
            total_insns += insns
            total_compute += int(insns * 0.7)
            total_memory += int(insns * 0.3)
            layer_insns[f"MaxPool{i+1}"] = insns
            current_shape = (N, out_c, ph_out, pw_out)

    # Reshape (flatten)
    flat_el = 1
    for d in current_shape:
        flat_el *= d
    total_insns += flat_el * RESHAPE_INSNS_PER_EL
    total_memory += flat_el * RESHAPE_INSNS_PER_EL
    layer_insns["Reshape (flatten)"] = flat_el * RESHAPE_INSNS_PER_EL

    # FC layers
    fc_input_dim = flat_el
    for i, fc in enumerate(fc_layers):
        name = f"FC{i+1}"
        in_dim = fc.get("in_dim", fc_input_dim)
        out_dim = fc["out_dim"]
        macs = in_dim * out_dim
        insns = macs * FC_INSNS_PER_MAC
        total_insns += insns
        total_compute += int(macs * 10)
        total_memory += int(macs * 4)
        total_branch += int(macs * 1)
        layer_insns[name] = insns

        if fc.get("has_relu", False):
            el = out_dim
            insns = el * RELU_INSNS_PER_EL
            total_insns += insns
            layer_insns[f"ReLU (after {name})"] = insns

        if fc.get("has_sigmoid", False):
            el = out_dim
            insns = el * SIGMOID_INSNS_PER_EL
            total_insns += insns
            layer_insns[f"Sigmoid"] = insns

        fc_input_dim = out_dim

    # Input/output copy overhead
    total_insns += flat_el * 6  # input copy

    return {
        "total_estimated": total_insns,
        "total_compute": total_compute,
        "total_memory": total_memory,
        "total_branch": total_branch,
        "compute_ratio": total_compute / max(total_insns, 1) * 100,
        "memory_ratio": total_memory / max(total_insns, 1) * 100,
        "branch_ratio": total_branch / max(total_insns, 1) * 100,
        "cm_ratio": total_compute / max(total_memory, 1),
        "per_layer": layer_insns,
        "est_hw_time_50mhz": total_insns / 50_000_000,
        "est_hw_time_100mhz": total_insns / 100_000_000,
    }


def estimate_cnn_model() -> dict:
    """Estimate instruction counts for the specific cnn.onnx model."""
    return estimate_cnn_instructions(
        input_shape=(1, 3, 250, 250),
        conv_layers=[
            {"out_c": 32, "kernel": 3, "stride": 1, "pad": 0,
             "has_relu": True, "has_maxpool": True, "pool_kernel": 2, "pool_stride": 2},
            {"out_c": 32, "kernel": 3, "stride": 1, "pad": 0,
             "has_relu": True, "has_maxpool": True, "pool_kernel": 2, "pool_stride": 2},
            {"out_c": 64, "kernel": 3, "stride": 1, "pad": 0,
             "has_relu": True, "has_maxpool": True, "pool_kernel": 2, "pool_stride": 2},
        ],
        fc_layers=[
            {"in_dim": 53824, "out_dim": 128, "has_relu": True},
            {"in_dim": 128, "out_dim": 1, "has_sigmoid": True},
        ],
    )


def print_estimate(est: dict) -> None:
    """Print analytical estimation as a formatted table."""
    sep = "=" * 72
    print(sep)
    print("  CNN Dynamic Instruction Count Estimation (Analytical)")
    print(sep)
    print(f"  Estimated total instructions: {est['total_estimated']:,.0f}")
    print(f"  Compute:  {est['compute_ratio']:.1f}%  "
          f"Memory: {est['memory_ratio']:.1f}%  "
          f"Branch: {est['branch_ratio']:.1f}%")
    print(f"  C/M ratio: {est['cm_ratio']:.1f}")
    print(f"  Est. HW time @ 50 MHz (CPI=1): {est['est_hw_time_50mhz']:.2f} s")
    print(f"  Est. HW time @ 100 MHz (CPI=1): {est['est_hw_time_100mhz']:.2f} s")
    print()
    print(f"  {'Layer':<30s} {'Instructions':>15s} {'%':>8s}")
    print(f"  {'─'*30} {'─'*15} {'─'*8}")
    total = max(est['total_estimated'], 1)
    for name, insns in est["per_layer"].items():
        pct = insns / total * 100
        print(f"  {name:<30s} {insns:>15,.0f} {pct:>7.1f}%")
    print(f"  {'─'*30} {'─'*15} {'─'*8}")
    print(f"  {'TOTAL':<30s} {est['total_estimated']:>15,.0f} {100.0:>7.1f}%")
    print(sep)


if __name__ == "__main__":
    sys.exit(main())
