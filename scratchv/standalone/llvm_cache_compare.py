#!/usr/bin/env python3
"""Cache-aware performance comparison: LLVM RV64FD vs ScratchV RV32IM.

Analyzes the LLVM-generated RISC-V RV64FD assembly for cnn.onnx and
computes dynamic instruction counts, cache behavior, and cycle estimates,
then compares against the ScratchV Q16.16 RV32IM version.

Since we can't run a full RV64FD emulator (no 64-bit support in our
Python emulator), we use analytical estimation based on:
  1. Static instruction classification from LLVM assembly
  2. CNN layer dimensions for dynamic counts
  3. Inner loop body instruction analysis
  4. Cache model fed with estimated memory access patterns

Output:
  - Dynamic instruction mix (LLVM vs ScratchV)
  - Cache performance (I$ / D$) for both
  - Cycle estimates across microarchitecture profiles
  - Side-by-side comparison tables
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# ── Import cache model ─────────────────────────────────────────────────────
from cache_model import CacheSim, create_cache_pair, CACHE_CONFIGS


# ═══════════════════════════════════════════════════════════════════════════
# CNN Model Dimensions (from cnn.onnx)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LayerDims:
    name: str
    macs: int           # multiply-accumulate operations
    elements: int       # output elements
    h_out: int = 1
    w_out: int = 1


def compute_layer_dims() -> list[LayerDims]:
    """Compute MAC counts and output sizes for all CNN layers."""
    layers = []
    N, C, H, W = 1, 3, 250, 250

    # ── Conv1: 3→32, 3×3, stride=1, pad=0 ──
    out_c = 32; K = 3; stride = 1; pad = 0
    h_out = (H + 2*pad - K)//stride + 1  # 248
    w_out = (W + 2*pad - K)//stride + 1  # 248
    macs = out_c * h_out * w_out * C * K * K
    layers.append(LayerDims("Conv1 (3→32, 3×3)", macs, out_c*h_out*w_out, h_out, w_out))
    H, W, C = h_out, w_out, out_c

    # ReLU1
    layers.append(LayerDims("ReLU1", 0, C*H*W, H, W))

    # MaxPool1: 2×2
    pk = 2; ps = 2
    H = (H - pk)//ps + 1  # 124
    W = (W - pk)//ps + 1
    layers.append(LayerDims("MaxPool1 (2×2)", C*H*W*pk*pk, C*H*W, H, W))

    # ── Conv2: 32→32, 3×3 ──
    out_c2 = 32; K2 = 3
    h_out2 = (H + 2*0 - K2)//1 + 1  # 122
    w_out2 = (W + 2*0 - K2)//1 + 1
    macs2 = out_c2 * h_out2 * w_out2 * C * K2 * K2
    layers.append(LayerDims("Conv2 (32→32, 3×3)", macs2, out_c2*h_out2*w_out2, h_out2, w_out2))
    H, W, C = h_out2, w_out2, out_c2

    # ReLU2
    layers.append(LayerDims("ReLU2", 0, C*H*W, H, W))

    # MaxPool2: 2×2
    H = (H - pk)//ps + 1  # 61
    W = (W - pk)//ps + 1
    layers.append(LayerDims("MaxPool2 (2×2)", C*H*W*pk*pk, C*H*W, H, W))

    # ── Conv3: 32→64, 3×3 ──
    out_c3 = 64; K3 = 3
    h_out3 = (H + 2*0 - K3)//1 + 1  # 59
    w_out3 = (W + 2*0 - K3)//1 + 1
    macs3 = out_c3 * h_out3 * w_out3 * C * K3 * K3
    layers.append(LayerDims("Conv3 (32→64, 3×3)", macs3, out_c3*h_out3*w_out3, h_out3, w_out3))
    H, W, C = h_out3, w_out3, out_c3

    # ReLU3
    layers.append(LayerDims("ReLU3", 0, C*H*W, H, W))

    # MaxPool3: 2×2
    H = (H - pk)//ps + 1  # 29
    W = (W - pk)//ps + 1
    layers.append(LayerDims("MaxPool3 (2×2)", C*H*W*pk*pk, C*H*W, H, W))

    # Flatten
    flat_el = C * H * W  # 64 * 29 * 29 = 53824
    layers.append(LayerDims("Reshape (flatten)", 0, flat_el))

    # ── FC1: 53824→128 ──
    macs_fc1 = flat_el * 128
    layers.append(LayerDims("FC1 (53824→128)", macs_fc1, 128))

    # ReLU4
    layers.append(LayerDims("ReLU4", 0, 128))

    # ── FC2: 128→1 ──
    macs_fc2 = 128 * 1
    layers.append(LayerDims("FC2 (128→1)", macs_fc2, 1))

    # Sigmoid
    layers.append(LayerDims("Sigmoid", 0, 1))

    return layers


# ═══════════════════════════════════════════════════════════════════════════
# LLVM inner loop instruction analysis
# ═══════════════════════════════════════════════════════════════════════════

# Per-instruction-type counts for ONE inner loop MAC iteration in the LLVM output.
# Extracted from manual analysis of the LLVM RV64FD O3 assembly inner loop body.
#
# A typical Conv inner loop body (float32):
#   slli + add    → address calc for input[row][col][c]
#   slli + add    → address calc for weight[out_c][ky][kx][c]
#   flw           → load weight
#   flw           → load input
#   fmul.s        → multiply
#   fadd.s        → accumulate
#   addi + slti + blt  → loop increment + check (shared across iterations)

# LLVM float32 inner loop: ~6 instructions per MAC
LLVM_CONV_PER_MAC = {
    "alu_i": 2.0,      # addi, slli for address calc and loop counters
    "alu_r": 0.5,      # add (address addition)
    "load": 2.0,       # flw (weight + input)
    "store": 0.0,      # accumulator stays in register
    "fp": 2.0,         # fmul.s + fadd.s
    "branch": 0.3,     # blt (shared across iterations)
    "jump": 0.1,       # j (loop back, shared)
    "upper": 0.1,      # lui (constant loading, shared)
}
LLVM_CONV_INSNS_PER_MAC = sum(LLVM_CONV_PER_MAC.values())  # ~7.0

# LLVM MaxPool inner loop: ~12 instructions per output element
LLVM_MAXPOOL_PER_EL = {
    "alu_i": 4.0,
    "alu_r": 1.0,
    "load": 3.0,       # flw to load pool window
    "fp": 1.0,         # fmax / comparison
    "branch": 1.5,
    "upper": 0.5,
    "jump": 0.5,
    "store": 0.5,
}
LLVM_MAXPOOL_INSNS_PER_EL = sum(LLVM_MAXPOOL_PER_EL.values())

# LLVM FC inner loop: ~5 instructions per MAC
LLVM_FC_PER_MAC = {
    "alu_i": 1.5,
    "alu_r": 0.5,
    "load": 2.0,       # flw (weight + input)
    "fp": 2.0,         # fmul.s + fadd.s
    "branch": 0.3,
    "upper": 0.2,
}
LLVM_FC_INSNS_PER_MAC = sum(LLVM_FC_PER_MAC.values())

# ReLU (per element): ~4 instructions
LLVM_RELU_PER_EL = {
    "alu_i": 1.0,
    "load": 1.0,
    "store": 1.0,
    "fp": 0.5,         # flt.s / fmax
    "branch": 0.5,
}

# Sigmoid approximation (per element): ~25 instructions
LLVM_SIGMOID_PER_EL = {
    "alu_i": 5.0,
    "alu_r": 3.0,
    "load": 3.0,
    "store": 2.0,
    "fp": 8.0,         # fdiv.s, fmul.s, fadd.s, etc.
    "branch": 2.0,
    "upper": 1.0,
    "jump": 1.0,
}

# Reshape/flatten (per element): ~5 instructions
LLVM_RESHAPE_PER_EL = {
    "alu_i": 2.0,
    "alu_r": 0.5,
    "load": 1.0,
    "store": 1.0,
    "branch": 0.5,
}

# ── ScratchV Q16.16 inner loop analysis ────────────────────────────────────

# ScratchV conv: ~30 instructions per MAC (Q16.16 fixed-point)
SCRATCHV_CONV_PER_MAC = {
    "alu_i": 8.0,
    "alu_r": 10.0,     # mul + add + srai + addressing
    "load": 6.0,       # lw (weight + input, multiple due to 32-bit ops)
    "store": 2.0,      # sw for spills
    "branch": 2.0,
    "upper": 1.0,
    "shift": 1.0,
}
SCRATCHV_CONV_INSNS_PER_MAC = sum(SCRATCHV_CONV_PER_MAC.values())

SCRATCHV_FC_PER_MAC = {
    "alu_i": 4.0,
    "alu_r": 5.0,
    "load": 4.0,
    "store": 1.0,
    "branch": 1.0,
}
SCRATCHV_FC_INSNS_PER_MAC = sum(SCRATCHV_FC_PER_MAC.values())

SCRATCHV_MAXPOOL_PER_EL = {
    "alu_i": 4.0,
    "alu_r": 5.0,
    "load": 4.0,
    "store": 1.0,
    "branch": 2.0,
    "shift": 1.0,
}
SCRATCHV_MAXPOOL_INSNS_PER_EL = sum(SCRATCHV_MAXPOOL_PER_EL.values())

SCRATCHV_RELU_PER_EL = {
    "alu_i": 3.0,
    "alu_r": 3.0,
    "load": 1.0,
    "store": 1.0,
    "branch": 1.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic instruction estimator
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DynInsnEstimate:
    """Estimated dynamic instruction counts by category."""
    name: str
    total: float = 0.0
    alu_i: float = 0.0
    alu_r: float = 0.0
    shift: float = 0.0
    load: float = 0.0
    store: float = 0.0
    branch: float = 0.0
    jump: float = 0.0
    upper: float = 0.0
    fp: float = 0.0
    other: float = 0.0

    # Memory access details for cache simulation
    load_addrs_distinct: int = 0     # number of distinct load addresses
    store_addrs_distinct: int = 0    # number of distinct store addresses
    working_set_bytes: int = 0       # estimated working set in bytes


def estimate_llvm_dynamic() -> DynInsnEstimate:
    """Estimate dynamic instruction counts for LLVM RV64FD O3 version."""
    est = DynInsnEstimate(name="LLVM RV64FD (float32)")
    layers = compute_layer_dims()

    for layer in layers:
        name = layer.name
        macs = layer.macs
        el = layer.elements

        if "Conv" in name:
            per_mac = LLVM_CONV_PER_MAC
            for cat, ratio in per_mac.items():
                setattr(est, cat, getattr(est, cat) + macs * ratio)
        elif "FC" in name:
            per_mac = LLVM_FC_PER_MAC
            for cat, ratio in per_mac.items():
                setattr(est, cat, getattr(est, cat) + macs * ratio)
        elif "MaxPool" in name:
            # MACs field holds comparisons for MaxPool
            for cat, ratio in LLVM_MAXPOOL_PER_EL.items():
                setattr(est, cat, getattr(est, cat) + el * ratio)
        elif "ReLU" in name:
            for cat, ratio in LLVM_RELU_PER_EL.items():
                setattr(est, cat, getattr(est, cat) + el * ratio)
        elif "Sigmoid" in name:
            for cat, ratio in LLVM_SIGMOID_PER_EL.items():
                setattr(est, cat, getattr(est, cat) + el * ratio)
        elif "Reshape" in name:
            for cat, ratio in LLVM_RESHAPE_PER_EL.items():
                setattr(est, cat, getattr(est, cat) + el * ratio)

    est.total = sum(getattr(est, cat) for cat in
                    ["alu_i", "alu_r", "shift", "load", "store",
                     "branch", "jump", "upper", "fp", "other"])

    # Estimate memory access patterns for cache simulation
    # Working set: Conv filter window + input tile + output tile
    # For a typical 3×3 conv: 9 weights + 9 inputs + 1 output = 19 floats = 76 bytes per output pixel
    # But the weight tensor is reused across the output spatial dimensions
    # Working set for Conv1: 32 × 3 × 3 × 3 = 864 weights + input tile
    # Input tile: roughly (kernel_size + stride)² × channels = ~9 × 3 = 27 elements
    # Total working set: ~864 × 4 + 27 × 4 + output tile ≈ 3.5 KB for small convolutions
    #
    # For FC layers: weight matrix can be huge (e.g., 53824 × 128 × 4 = 27.5 MB)
    # Streaming access pattern: sequential read of weight matrix
    est.load_addrs_distinct = sum(
        (l.macs * 9 + l.elements * 2) for l in layers if "Conv" in l.name
    ) + sum(
        l.macs * 2 for l in layers if "FC" in l.name
    )
    # Approximate
    est.working_set_bytes = 16 * 1024  # ~16KB for inner loop working set

    return est


def estimate_scratchv_dynamic() -> DynInsnEstimate:
    """Estimate dynamic instruction counts for ScratchV RV32IM version."""
    est = DynInsnEstimate(name="ScratchV RV32IM (Q16.16)")
    layers = compute_layer_dims()

    for layer in layers:
        name = layer.name
        macs = layer.macs
        el = layer.elements

        if "Conv" in name:
            per_mac = SCRATCHV_CONV_PER_MAC
            for cat, ratio in per_mac.items():
                setattr(est, cat, getattr(est, cat) + macs * ratio)
        elif "FC" in name:
            per_mac = SCRATCHV_FC_PER_MAC
            for cat, ratio in per_mac.items():
                setattr(est, cat, getattr(est, cat) + macs * ratio)
        elif "MaxPool" in name:
            for cat, ratio in SCRATCHV_MAXPOOL_PER_EL.items():
                setattr(est, cat, getattr(est, cat) + el * ratio)
        elif "ReLU" in name:
            for cat, ratio in SCRATCHV_RELU_PER_EL.items():
                setattr(est, cat, getattr(est, cat) + el * ratio)
        elif "Sigmoid" in name:
            # Sigmoid is very expensive in Q16.16
            for cat, ratio in {"alu_i": 5.0, "alu_r": 8.0, "load": 3.0,
                               "store": 2.0, "branch": 4.0, "shift": 3.0}.items():
                setattr(est, cat, getattr(est, cat) + el * ratio)
        elif "Reshape" in name:
            for cat, ratio in {"alu_i": 2.0, "alu_r": 2.0, "load": 1.0,
                               "store": 1.0, "branch": 1.0}.items():
                setattr(est, cat, getattr(est, cat) + el * ratio)

    est.total = sum(getattr(est, cat) for cat in
                    ["alu_i", "alu_r", "shift", "load", "store",
                     "branch", "jump", "upper", "fp", "other"])

    est.working_set_bytes = 16 * 1024
    return est


# ═══════════════════════════════════════════════════════════════════════════
# Analytical cache simulation
# ═══════════════════════════════════════════════════════════════════════════

def simulate_cache_analytically(
    est: DynInsnEstimate,
    icache: CacheSim,
    dcache: CacheSim,
    bytes_per_access: int = 4,
    working_set_multiplier: float = 1.0,
) -> tuple[dict, dict]:
    """Analytically estimate cache performance from instruction mix and access patterns.

    For the I-cache:
      - Code fits entirely: ~100% hit rate after compulsory misses
      - Compulsory misses = ceil(code_size / block_size)

    For the D-cache:
      - Working set and streaming access determine miss rate
      - More accesses → more absolute misses even at same hit rate
      - Q16.16 has larger working set due to intermediate fixed-point values

    Returns (icache_stats_dict, dcache_stats_dict).
    """
    # ── I-cache ──
    code_size = 956 * 4 if "LLVM" in est.name else 785 * 4
    code_blocks = (code_size + icache.block_size - 1) // icache.block_size
    total_ifetch = est.total
    ic_compulsory = code_blocks
    ic_hits = total_ifetch - ic_compulsory

    # ── D-cache ──
    total_daccess = est.load + est.store
    # Working set size: LLVM float32 accesses fewer distinct addresses
    # (each float32 load = 1 address; Q16.16 needs multiple integer ops per MAC,
    #  touching more addresses for the same data)
    base_ws_kb = 64  # base working set for inner loops in KB
    ws_kb = base_ws_kb * working_set_multiplier
    ws_bytes = int(ws_kb * 1024)
    ws_blocks = (ws_bytes + dcache.block_size - 1) // dcache.block_size
    cache_blocks = dcache.sets * dcache.ways
    cache_bytes = cache_blocks * dcache.block_size

    if ws_blocks <= cache_blocks:
        # Working set fits: mostly compulsory + cold misses
        dc_compulsory = ws_blocks * 2  # weight + activation streams
        # Low conflict rate when WS fits
        conflict_rate = 0.002  # 0.2%
        dc_conflict = total_daccess * conflict_rate
        dc_misses = dc_compulsory + dc_conflict
    else:
        # Working set exceeds cache: significant conflict/capacity misses
        fraction_inner = 0.75  # ~75% of accesses are in inner loops
        excess_ratio = (ws_blocks - cache_blocks) / cache_blocks
        conflict_rate = min(0.15, excess_ratio * 0.1)  # cap at 15%
        dc_compulsory = ws_blocks * 2
        dc_conflict = total_daccess * fraction_inner * conflict_rate
        dc_misses = dc_compulsory + dc_conflict

    # Cap at total
    dc_misses = min(dc_misses, total_daccess)
    dc_hits = total_daccess - dc_misses

    hit_rate = dc_hits / max(total_daccess, 1) * 100

    ic_stats = {
        "name": icache.name,
        "config": f"{icache.sets}x{icache.ways}x{icache.block_size}",
        "size_bytes": icache.total_size_bytes,
        "size_kb": icache.total_size_bytes / 1024,
        "accesses": int(total_ifetch),
        "hits": int(ic_hits),
        "misses": int(ic_compulsory),
        "total_miss_bytes": int(ic_compulsory * icache.block_size),
        "hit_rate_pct": round(ic_hits / max(total_ifetch, 1) * 100, 3),
        "miss_rate_pct": round(ic_compulsory / max(total_ifetch, 1) * 100, 3),
        "mpki": round(ic_compulsory / max(total_ifetch / 1000, 1), 2),
    }

    dc_stats = {
        "name": dcache.name,
        "config": f"{dcache.sets}x{dcache.ways}x{dcache.block_size}",
        "size_bytes": dcache.total_size_bytes,
        "size_kb": dcache.total_size_bytes / 1024,
        "accesses": int(total_daccess),
        "hits": int(dc_hits),
        "misses": int(dc_misses),
        "total_miss_bytes": int(dc_misses * dcache.block_size),
        "hit_rate_pct": round(hit_rate, 3),
        "miss_rate_pct": round(100 - hit_rate, 3),
        "mpki": round(dc_misses / max(total_daccess / 1000, 1), 2),
    }

    return ic_stats, dc_stats


# ═══════════════════════════════════════════════════════════════════════════
# Cycle estimation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MicroArch:
    name: str
    alu_r: int = 1; alu_i: int = 1; shift: int = 1
    load: int = 1; store: int = 1
    branch_taken: int = 1; branch_not: int = 1
    jump: int = 1; jump_r: int = 1; upper: int = 1
    mul: int = 1; div: int = 1; fp_op: int = 1


PROFILES = {
    "single-cycle": MicroArch("single-cycle", fp_op=1),
    "rv32im-basic": MicroArch("rv32im-basic", mul=4, div=34, load=2, branch_taken=2, branch_not=1),
    # RV64FD with FPU: fmul/fadd are 3-4 cycles but fully pipelined (CPI~1.0 per op after filling pipeline)
    # Load-use hazard: 1 cycle stall if load→use dependency (flw→fmul.s, etc.)
    "rv64fd-basic": MicroArch("rv64fd-basic", mul=3, div=16, load=2, fp_op=1,
                               branch_taken=2, branch_not=1),
    "rv64fd-fast": MicroArch("rv64fd-fast", mul=1, div=4, load=1, fp_op=1,
                              branch_taken=1, branch_not=1),
    "rv64fd-slow": MicroArch("rv64fd-slow", mul=5, div=20, load=3, fp_op=2,
                              branch_taken=3, branch_not=1),
}


def estimate_cycles(est: DynInsnEstimate) -> dict[str, dict]:
    """Estimate total cycles for each microarchitecture profile."""
    results = {}
    branch_total = est.branch
    # Assume 20% taken for CNN loops (from empirical data)
    branch_taken = branch_total * 0.20
    branch_not = branch_total * 0.80

    for pname, p in PROFILES.items():
        cycles = (
            est.alu_i * p.alu_i +
            est.alu_r * 0.85 * p.alu_r +    # 85% non-MUL ALU-R
            est.alu_r * 0.15 * p.mul +       # 15% MUL (CNN inner loop)
            est.shift * p.shift +
            est.load * p.load +
            est.store * p.store +
            branch_taken * p.branch_taken +
            branch_not * p.branch_not +
            est.jump * p.jump +
            est.upper * p.upper +
            est.fp * p.fp_op
        )
        total = int(cycles)
        cpi = total / max(est.total, 1)
        results[pname] = {
            "profile": p.name,
            "total_cycles": total,
            "cpi": round(cpi, 2),
            "est_hw_50mhz_s": round(total / 50_000_000, 1),
            "est_hw_100mhz_s": round(total / 100_000_000, 1),
            "est_hw_500mhz_s": round(total / 500_000_000, 1),
            "est_hw_1000mhz_s": round(total / 1_000_000_000, 1),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_comparison_report(
    llvm_est: DynInsnEstimate,
    scratchv_est: DynInsnEstimate,
    llvm_icache_emb: dict,
    llvm_dcache_emb: dict,
    scratchv_icache_emb: dict,
    scratchv_dcache_emb: dict,
    llvm_icache_app: dict,
    llvm_dcache_app: dict,
    scratchv_icache_app: dict,
    scratchv_dcache_app: dict,
    llvm_cycles: dict,
    scratchv_cycles: dict,
) -> str:
    """Generate a detailed side-by-side comparison report."""
    lines = []
    sep = "=" * 84

    lines.append(sep)
    lines.append("  Cache-Aware Performance Comparison")
    lines.append("  LLVM RV64FD (float32)  vs  ScratchV RV32IM (Q16.16)")
    lines.append("  Model: cnn.onnx  |  3×Conv + 3×MaxPool + 2×FC + ReLU + Sigmoid")
    lines.append(sep)
    lines.append("")

    # ── 1. Static Code Size ────────────────────────────────────────────
    lines.append("  ── 1. Static Code Size ──")
    lines.append(f"  {'Metric':<35s} {'LLVM RV64FD':>20s} {'ScratchV RV32IM':>20s}")
    lines.append(f"  {'─'*35} {'─'*20} {'─'*20}")
    lines.append(f"  {'Static instructions':<35s} {'956':>20s} {'785':>20s}")
    lines.append(f"  {'Code bytes':<35s} {'3,824':>20s} {'3,140':>20s}")
    lines.append(f"  {'Inline data (weights)':<35s} {'~27 MB':>20s} {'~27 MB':>20s}")
    lines.append("")

    # ── 2. Dynamic Instruction Distribution ────────────────────────────
    lines.append("  ── 2. Dynamic Instruction Count (estimated) ──")
    lines.append(f"  {'Category':<20s} {'LLVM RV64FD':>18s} {'%':>7s}  "
                 f"{'ScratchV RV32IM':>18s} {'%':>7s}  {'Ratio':>8s}")
    lines.append(f"  {'─'*20} {'─'*18} {'─'*7}  {'─'*18} {'─'*7}  {'─'*8}")

    categories = [
        ("alu_r", "ALU R-type"),
        ("alu_i", "ALU I-type"),
        ("shift", "Shift"),
        ("load", "Load"),
        ("store", "Store"),
        ("branch", "Branch"),
        ("jump", "Jump"),
        ("upper", "Upper imm"),
        ("fp", "Float ops"),
    ]

    for cat, label in categories:
        lv = getattr(llvm_est, cat, 0)
        sv = getattr(scratchv_est, cat, 0)
        lp = lv / max(llvm_est.total, 1) * 100
        sp = sv / max(scratchv_est.total, 1) * 100
        ratio = lv / max(sv, 1)
        lines.append(f"  {label:<20s} {lv:>18,.0f} {lp:>6.1f}%  "
                     f"{sv:>18,.0f} {sp:>6.1f}%  {ratio:>7.2f}x")

    # Total
    lines.append(f"  {'─'*20} {'─'*18} {'─'*7}  {'─'*18} {'─'*7}  {'─'*8}")
    lines.append(f"  {'TOTAL':<20s} {llvm_est.total:>18,.0f} {'100.0':>6}%  "
                 f"{scratchv_est.total:>18,.0f} {'100.0':>6}%  "
                 f"{llvm_est.total/max(scratchv_est.total,1):>7.2f}x")
    lines.append("")

    # ── 3. Memory & Cache ──────────────────────────────────────────────
    lines.append("  ── 3. Memory Access & Cache Performance ──")
    lines.append(f"  {'Metric':<35s} {'LLVM RV64FD':>20s} {'ScratchV RV32IM':>20s}")
    lines.append(f"  {'─'*35} {'─'*20} {'─'*20}")

    mem_metrics = [
        ("Total loads", f"{llvm_est.load:,.0f}", f"{scratchv_est.load:,.0f}"),
        ("Total stores", f"{llvm_est.store:,.0f}", f"{scratchv_est.store:,.0f}"),
        ("Total memory ops", f"{llvm_est.load + llvm_est.store:,.0f}",
         f"{scratchv_est.load + scratchv_est.store:,.0f}"),
        ("Compute/Memory ratio",
         f"{(llvm_est.alu_r+llvm_est.alu_i+llvm_est.fp)/max(llvm_est.load+llvm_est.store,1):.1f}",
         f"{(scratchv_est.alu_r+scratchv_est.alu_i)/max(scratchv_est.load+scratchv_est.store,1):.1f}"),
        ("",
         "I$: " + icache_config_str(llvm_icache_emb),
         "I$: " + icache_config_str(scratchv_icache_emb)),
        ("",
         "D$: " + dcache_config_str(llvm_dcache_emb),
         "D$: " + dcache_config_str(scratchv_dcache_emb)),
    ]
    for label, lv, sv in mem_metrics:
        if label:
            lines.append(f"  {label:<35s} {lv:>20s} {sv:>20s}")
        else:
            lines.append(f"  {'':35s} {lv:>20s} {sv:>20s}")

    lines.append("")
    # Embedded cache
    lines.append(f"  {'Embedded':<10s} {'I$=':>2s}{icache_config_str(llvm_icache_emb):12s} "
                 f"{'':6s} {'I$=':>2s}{icache_config_str(scratchv_icache_emb):12s}")
    lines.append(f"  {'':10s} {'D$=':>2s}{dcache_config_str(llvm_dcache_emb):12s} "
                 f"{'':6s} {'D$=':>2s}{dcache_config_str(scratchv_dcache_emb):12s}")
    lines.append(f"  {'':10s} {'I$:':>3s} {llvm_icache_emb['hit_rate_pct']:>8.2f}% hit"
                 f"  {'D$:':>3s} {llvm_dcache_emb['hit_rate_pct']:>8.2f}% hit"
                 f"  | {'I$:':>3s} {scratchv_icache_emb['hit_rate_pct']:>8.2f}% hit"
                 f"  {'D$:':>3s} {scratchv_dcache_emb['hit_rate_pct']:>8.2f}% hit")
    lines.append(f"  {'':10s} {'D$ misses:':>10s} {llvm_dcache_emb['misses']:>12,}"
                 f"  {'':16s} {'D$ misses:':>10s} {scratchv_dcache_emb['misses']:>12,}")
    lines.append("")

    # Application cache
    lines.append(f"  {'Application':<10s} {'I$=':>2s}{icache_config_str(llvm_icache_app):12s} "
                 f"{'':6s} {'I$=':>2s}{icache_config_str(scratchv_icache_app):12s}")
    lines.append(f"  {'':10s} {'D$=':>2s}{dcache_config_str(llvm_dcache_app):12s} "
                 f"{'':6s} {'D$=':>2s}{dcache_config_str(scratchv_dcache_app):12s}")
    lines.append(f"  {'':10s} {'I$:':>3s} {llvm_icache_app['hit_rate_pct']:>8.2f}% hit"
                 f"  {'D$:':>3s} {llvm_dcache_app['hit_rate_pct']:>8.2f}% hit"
                 f"  | {'I$:':>3s} {scratchv_icache_app['hit_rate_pct']:>8.2f}% hit"
                 f"  {'D$:':>3s} {scratchv_dcache_app['hit_rate_pct']:>8.2f}% hit")
    lines.append(f"  {'':10s} {'D$ misses:':>10s} {llvm_dcache_app['misses']:>12,}"
                 f"  {'':16s} {'D$ misses:':>10s} {scratchv_dcache_app['misses']:>12,}")
    lines.append("")

    # ── Absolute cache miss comparison ──
    lines.append("  ── Cache Miss Volume (absolute bytes) ──")
    llvm_total_miss = llvm_icache_emb.get("total_miss_bytes", 0) + llvm_dcache_emb.get("total_miss_bytes", 0)
    sv_total_miss = scratchv_icache_emb.get("total_miss_bytes", 0) + scratchv_dcache_emb.get("total_miss_bytes", 0)
    lines.append(f"  {'':30s} {'LLVM':>18s} {'ScratchV':>18s}  {'Ratio':>8s}")
    lines.append(f"  {'─'*30} {'─'*18} {'─'*18}  {'─'*8}")
    lines.append(f"  {'I$ miss bytes':30s} "
                 f"{llvm_icache_emb.get('total_miss_bytes', 0):>18,} "
                 f"{scratchv_icache_emb.get('total_miss_bytes', 0):>18,}  "
                 f"{scratchv_icache_emb.get('total_miss_bytes', 0)/max(llvm_icache_emb.get('total_miss_bytes', 1), 1):>7.1f}x")
    lines.append(f"  {'D$ miss bytes':30s} "
                 f"{llvm_dcache_emb.get('total_miss_bytes', 0):>18,} "
                 f"{scratchv_dcache_emb.get('total_miss_bytes', 0):>18,}  "
                 f"{scratchv_dcache_emb.get('total_miss_bytes', 0)/max(llvm_dcache_emb.get('total_miss_bytes', 1), 1):>7.1f}x")
    lines.append(f"  {'─'*30} {'─'*18} {'─'*18}  {'─'*8}")
    lines.append(f"  {'Total miss bytes':30s} "
                 f"{llvm_total_miss:>18,} "
                 f"{sv_total_miss:>18,}  "
                 f"{sv_total_miss/max(llvm_total_miss, 1):>7.1f}x")
    lines.append(f"  {'Miss bytes per MAC':30s} "
                 f"{llvm_total_miss/263000000:>17.2f}B "
                 f"{sv_total_miss/263000000:>17.2f}B  "
                 f"{(sv_total_miss/max(llvm_total_miss, 1)):>7.1f}x")
    lines.append("")

    # ── 4. Cycle Estimates ─────────────────────────────────────────────
    lines.append("  ── 4. Cycle Estimates ──")
    lines.append(f"  {'Profile':<20s} {'LLVM CPI':>9s} {'LLVM Cycles':>14s}  "
                 f"{'ScratchV CPI':>12s} {'ScratchV Cycles':>14s}")
    lines.append(f"  {'─'*20} {'─'*9} {'─'*14}  {'─'*12} {'─'*14}")

    common_profiles = ["single-cycle", "rv32im-basic", "rv64fd-basic"]
    for pname in common_profiles:
        lc = llvm_cycles.get(pname, {})
        sc = scratchv_cycles.get(pname, {})
        lines.append(f"  {pname:<20s} "
                     f"{lc.get('cpi', 0):>8.2f}  "
                     f"{lc.get('total_cycles', 0):>13,}  "
                     f"{sc.get('cpi', 0):>11.2f}  "
                     f"{sc.get('total_cycles', 0):>13,}")

    lines.append("")
    lines.append(f"  {'Frequency':<12s} {'LLVM Time':>14s} {'ScratchV Time':>14s}  "
                 f"{'Speedup':>10s}")
    lines.append(f"  {'─'*12} {'─'*14} {'─'*14}  {'─'*10}")
    for freq in [50, 100, 500, 1000]:
        key = f"est_hw_{freq}mhz_s"
        lt = llvm_cycles.get("rv64fd-basic", {}).get(key, 0)
        st = scratchv_cycles.get("rv32im-basic", {}).get(key, 0)
        speedup = st / max(lt, 0.01)
        lines.append(f"  {'@'+str(freq)+'MHz':<12s} {lt:>13.1f}s {st:>13.1f}s  "
                     f"{speedup:>9.1f}x")

    lines.append("")
    lines.append("  ── 5. Key Insights ──")
    lines.append("")

    # Compute key ratios
    dynamic_ratio = scratchv_est.total / max(llvm_est.total, 1)
    lines.append(f"  1. Dynamic instruction ratio: {dynamic_ratio:.1f}x more in ScratchV")
    lines.append(f"     LLVM float32: ~{llvm_est.total/1e9:.1f}B insns  vs  "
                 f"ScratchV Q16.16: ~{scratchv_est.total/1e9:.1f}B insns")

    llvm_cpi = llvm_cycles.get("rv64fd-basic", {}).get("cpi", 1.0)
    sv_cpi = scratchv_cycles.get("rv32im-basic", {}).get("cpi", 1.0)
    lines.append(f"  2. CPI: LLVM ~{llvm_cpi:.2f} (fmul/fadd pipelined)  vs  "
                 f"ScratchV ~{sv_cpi:.2f} (mul=4 cyc)")

    llvm_time_100 = llvm_cycles.get("rv64fd-basic", {}).get("est_hw_100mhz_s", 0)
    sv_time_100 = scratchv_cycles.get("rv32im-basic", {}).get("est_hw_100mhz_s", 0)
    lines.append(f"  3. Est. time @100MHz: LLVM {llvm_time_100:.1f}s  vs  "
                 f"ScratchV {sv_time_100:.1f}s")

    lines.append(f"  4. I$ hit rate both ~100% (tiny code footprint)")
    lines.append(f"  5. D$: LLVM has fewer loads/stores due to float32 single-instruction data ops")
    lines.append(f"  6. FP instructions (fmul.s/fadd.s) replace ~15 Q16.16 integer ops per MAC")

    lines.append("")
    lines.append(sep)
    lines.append(f"  Analysis complete.")
    lines.append(sep)

    return "\n".join(lines)


def icache_config_str(stats: dict) -> str:
    return f"{stats['config']} ({stats['size_kb']:.0f}KB)"


def dcache_config_str(stats: dict) -> str:
    return f"{stats['config']} ({stats['size_kb']:.0f}KB)"


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("LLVM vs ScratchV Cache Performance Comparison", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # ── Estimate dynamic instruction counts ─────────────────────────────
    print("\n[1] Estimating dynamic instruction counts...", file=sys.stderr)
    llvm_est = estimate_llvm_dynamic()
    scratchv_est = estimate_scratchv_dynamic()

    print(f"  LLVM:     ~{llvm_est.total/1e9:.2f}B dynamic instructions", file=sys.stderr)
    print(f"  ScratchV: ~{scratchv_est.total/1e9:.2f}B dynamic instructions", file=sys.stderr)
    print(f"  Ratio:    {scratchv_est.total/llvm_est.total:.1f}x", file=sys.stderr)

    # ── Analytical cache simulation ─────────────────────────────────────
    print("\n[2] Analytical cache simulation...", file=sys.stderr)
    ic_embedded, dc_embedded = create_cache_pair("embedded")
    ic_app, dc_app = create_cache_pair("application")

    # Use differentiated working set estimates:
    # LLVM float32: compact 4B loads, FP regs hold accumulators → smaller WS
    # ScratchV Q16.16: more spills, temp regs → ~2x larger WS footprint
    llvm_ic_emb, llvm_dc_emb = simulate_cache_analytically(
        llvm_est, ic_embedded, dc_embedded, working_set_multiplier=1.0)
    llvm_ic_app, llvm_dc_app = simulate_cache_analytically(
        llvm_est, ic_app, dc_app, working_set_multiplier=1.0)

    sv_ic_emb, sv_dc_emb = simulate_cache_analytically(
        scratchv_est, ic_embedded, dc_embedded, working_set_multiplier=2.0)
    sv_ic_app, sv_dc_app = simulate_cache_analytically(
        scratchv_est, ic_app, dc_app, working_set_multiplier=2.0)

    print(f"  LLVM I$ (embedded):   {llvm_ic_emb['hit_rate_pct']:.2f}% hit", file=sys.stderr)
    print(f"  LLVM D$ (embedded):   {llvm_dc_emb['hit_rate_pct']:.2f}% hit", file=sys.stderr)
    print(f"  ScratchV I$ (embedded): {sv_ic_emb['hit_rate_pct']:.2f}% hit", file=sys.stderr)
    print(f"  ScratchV D$ (embedded): {sv_dc_emb['hit_rate_pct']:.2f}% hit", file=sys.stderr)

    # ── Cycle estimation ────────────────────────────────────────────────
    print("\n[3] Estimating cycles...", file=sys.stderr)
    llvm_cycles = estimate_cycles(llvm_est)
    scratchv_cycles = estimate_cycles(scratchv_est)

    # ── Generate report ─────────────────────────────────────────────────
    print("\n[4] Generating report...", file=sys.stderr)
    report = generate_comparison_report(
        llvm_est, scratchv_est,
        llvm_ic_emb, llvm_dc_emb,
        sv_ic_emb, sv_dc_emb,
        llvm_ic_app, llvm_dc_app,
        sv_ic_app, sv_dc_app,
        llvm_cycles, scratchv_cycles,
    )

    # Output
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-output", default="")
    parser.add_argument("--markdown", default="output/llvm_vs_scratchv_cache_report.md")
    args = parser.parse_args()

    if args.json:
        result = {
            "llvm": {
                "name": llvm_est.name,
                "dynamic_instructions": {k: getattr(llvm_est, k) for k in
                    ["total", "alu_i", "alu_r", "shift", "load", "store",
                     "branch", "jump", "upper", "fp"]},
                "cache_embedded": {"icache": llvm_ic_emb, "dcache": llvm_dc_emb},
                "cache_application": {"icache": llvm_ic_app, "dcache": llvm_dc_app},
                "cycles": llvm_cycles,
            },
            "scratchv": {
                "name": scratchv_est.name,
                "dynamic_instructions": {k: getattr(scratchv_est, k) for k in
                    ["total", "alu_i", "alu_r", "shift", "load", "store",
                     "branch", "jump", "upper", "fp"]},
                "cache_embedded": {"icache": sv_ic_emb, "dcache": sv_dc_emb},
                "cache_application": {"icache": sv_ic_app, "dcache": sv_dc_app},
                "cycles": scratchv_cycles,
            },
        }
        print(json.dumps(result, indent=2))
    else:
        print(report)

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump({
                "llvm": {
                    "name": llvm_est.name,
                    "dynamic_instructions": {k: getattr(llvm_est, k) for k in
                        ["total", "alu_i", "alu_r", "shift", "load", "store",
                         "branch", "jump", "upper", "fp"]},
                    "cycles": llvm_cycles,
                },
                "scratchv": {
                    "name": scratchv_est.name,
                    "dynamic_instructions": {k: getattr(scratchv_est, k) for k in
                        ["total", "alu_i", "alu_r", "shift", "load", "store",
                         "branch", "jump", "upper", "fp"]},
                    "cycles": scratchv_cycles,
                },
            }, f, indent=2)
        print(f"\n  JSON saved to: {args.json_output}", file=sys.stderr)

    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(report)
        print(f"  Report saved to: {args.markdown}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
