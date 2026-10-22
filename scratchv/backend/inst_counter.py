"""RISC-V Instruction Counter.

Parses RISC-V assembly text and counts instructions by category,
producing text tables, matplotlib charts, and HTML reports.

Usage as module::

    from scratchv.backend.inst_counter import count_instructions
    stats = count_instructions(asm_text)
    print(stats)  # dict: {category: count}

Usage as CLI::

    python -m scratchv.backend.inst_counter file1.s file2.s --chart output.png
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional

from scratchv.backend._asm_parser import parse_line

# ---------------------------------------------------------------------------
# Instruction categories
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
CLASSIFICATION_VERSION = "1"

# Standard 9 functional categories (in canonical presentation order):
# ALU (integer arithmetic/logic), FP (floating-point arithmetic),
# MEM (memory load/store), BRANCH (conditional branches),
# JUMP (unconditional jumps/calls/returns), ATOMIC (atomic memory ops/AMOs),
# SYSTEM (CSR/system/fences/traps), PSEUDO (assembler pseudo-instructions),
# MISC (unrecognized/custom instructions).
_CATEGORY_ORDER = [
    "ALU", "FP", "MEM", "BRANCH", "JUMP", "ATOMIC", "SYSTEM", "PSEUDO", "MISC",
]
_LEGACY_CATEGORY_ORDER = ["ALU", "MEM", "BRANCH", "JUMP", "PSEUDO", "MISC"]


# ---------------------------------------------------------------------------
# Opcode Classification Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OpcodeInfo:
    """Metadata describing an instruction opcode.

    Attributes:
        category: One of the 9 canonical categories (ALU, FP, MEM, BRANCH,
            JUMP, ATOMIC, SYSTEM, PSEUDO, MISC).
        extension: Source ISA extension mnemonic (I, M, F, D, A, C, Zbb,
            System, Pseudo, etc.).
        is_pseudo: Whether this mnemonic represents an assembler pseudo-instruction
            (orthogonal to functional category).
    """
    category: str
    extension: str
    is_pseudo: bool = False


# Canonical registry mapping lowercase opcode mnemonics to classification metadata.
_OPCODE_REGISTRY: dict[str, OpcodeInfo] = {}


def _register(
    category: str,
    extension: str,
    opcodes: Sequence[str],
    is_pseudo: bool = False,
) -> None:
    """Register a collection of opcodes into the global classification registry."""
    info = OpcodeInfo(category=category, extension=extension, is_pseudo=is_pseudo)
    for op in opcodes:
        _OPCODE_REGISTRY[op.lower()] = info



# 1. ALU (Integer arithmetic, logic, shift, compare, upper immediate)
_register("ALU", "I", [
    "add", "addi", "sub", "sll", "slli", "srl", "srli", "sra", "srai",
    "xor", "xori", "or", "ori", "and", "andi",
    "slt", "slti", "sltu", "sltiu",
    "lui", "auipc",
    # 64-bit integer
    "addw", "addiw", "subw", "sllw", "slliw", "srlw", "srliw", "sraw", "sraiw",
])
_register("ALU", "M", [
    "mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu",
    "mulw", "divw", "divuw", "remw", "remuw",
])
_register("ALU", "C", [
    "c.addi", "c.addiw", "c.addi16sp", "c.addi4spn",
    "c.slli", "c.srli", "c.srai", "c.andi",
    "c.sub", "c.xor", "c.or", "c.and",
    "c.subw", "c.addw", "c.add", "c.lui",
])
_register("ALU", "Zbb", [
    "clz", "ctz", "cpop", "sext.b", "sext.h", "zext.h",
    "min", "max", "minu", "maxu", "andn", "orn", "xnor",
    "rol", "ror", "rori",
])

# 2. FP (Floating-point arithmetic, conversion, move, compare)
_register("FP", "F", [
    "fadd.s", "fsub.s", "fmul.s", "fdiv.s", "fsqrt.s",
    "fmadd.s", "fmsub.s", "fnmadd.s", "fnmsub.s",
    "fsgnj.s", "fsgnjn.s", "fsgnjx.s",
    "fmin.s", "fmax.s",
    "feq.s", "flt.s", "fle.s", "fclass.s",
    "fcvt.w.s", "fcvt.wu.s", "fcvt.s.w", "fcvt.s.wu",
    "fcvt.l.s", "fcvt.lu.s", "fcvt.s.l", "fcvt.s.lu",
    "fmv.x.w", "fmv.w.x",
])
_register("FP", "D", [
    "fadd.d", "fsub.d", "fmul.d", "fdiv.d", "fsqrt.d",
    "fmadd.d", "fmsub.d", "fnmadd.d", "fnmsub.d",
    "fsgnj.d", "fsgnjn.d", "fsgnjx.d",
    "fmin.d", "fmax.d",
    "feq.d", "flt.d", "fle.d", "fclass.d",
    "fcvt.w.d", "fcvt.wu.d", "fcvt.d.w", "fcvt.d.wu",
    "fcvt.l.d", "fcvt.lu.d", "fcvt.d.l", "fcvt.d.lu",
    "fcvt.s.d", "fcvt.d.s",
    "fmv.x.d", "fmv.d.x",
])
_register("FP", "FP-Pseudo", [
    "fmv.s", "fabs.s", "fneg.s",
    "fmv.d", "fabs.d", "fneg.d",
], is_pseudo=True)

# 3. MEM (Loads and stores - integer, float, compressed)
_register("MEM", "I", [
    "lb", "lbu", "lh", "lhu", "lw", "lwu", "ld",
    "sb", "sh", "sw", "sd",
])
_register("MEM", "F", ["flw", "fsw"])
_register("MEM", "D", ["fld", "fsd"])
_register("MEM", "C", [
    "c.lw", "c.sw", "c.ld", "c.sd",
    "c.lwsp", "c.swsp", "c.ldsp", "c.sdsp",
    "c.flw", "c.fsw", "c.fld", "c.fsd",
    "c.flwsp", "c.fswsp", "c.fldsp", "c.fsdsp",
])

# 4. BRANCH (Conditional branches)
_register("BRANCH", "I", [
    "beq", "bne", "blt", "bge", "bltu", "bgeu",
])
_register("BRANCH", "Pseudo", [
    "beqz", "bnez", "blez", "bgtz", "bltz", "bgez",
    "bgt", "ble", "bgtu", "bleu",
], is_pseudo=True)
_register("BRANCH", "C", ["c.beqz", "c.bnez"])

# 5. JUMP (Unconditional jumps, calls, returns)
_register("JUMP", "I", ["jal", "jalr"])
_register("JUMP", "Pseudo", [
    "j", "jr", "ret", "call", "tail",
], is_pseudo=True)
_register("JUMP", "C", ["c.j", "c.jal", "c.jr", "c.jalr"])

# 6. ATOMIC (Load-reserved, store-conditional, AMOs)
_register("ATOMIC", "A", [
    "lr.w", "sc.w", "lr.d", "sc.d",
    "amoadd.w", "amoswap.w", "amoand.w", "amoor.w", "amoxor.w",
    "amomin.w", "amomax.w", "amominu.w", "amomaxu.w",
    "amoadd.d", "amoswap.d", "amoand.d", "amoor.d", "amoxor.d",
    "amomin.d", "amomax.d", "amominu.d", "amomaxu.d",
])

# 7. SYSTEM (CSR, fence, traps, privilege)
_register("SYSTEM", "I", [
    "fence", "fence.i",
    "csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi", "csrrci",
    "ecall", "ebreak", "wfi", "mret", "sret", "sfence.vma",
])
_register("SYSTEM", "Pseudo", [
    "csrr", "csrw", "csrs", "csrc", "csrwi", "csrsi", "csrci",
    "rdcycle", "rdtime", "rdinstret",
], is_pseudo=True)

# 8. PSEUDO (Non-control flow assembler pseudo-instructions)
_register("PSEUDO", "Pseudo", [
    "li", "la", "mv", "not", "neg", "negw", "sext.w",
    "seqz", "snez", "sltz", "sgtz", "nop",
], is_pseudo=True)
_register("PSEUDO", "C", ["c.nop", "c.mv"], is_pseudo=True)


# Suffixes for atomic memory ordering
_ATOMIC_SUFFIX_RE = re.compile(r'^(amo\w+\.[wd]|(?:lr|sc)\.[wd])\.(?:aqrl|aq|rl)$')


def classify_opcode(opcode: str) -> OpcodeInfo:
    """Classify a normalized opcode mnemonic.

    Returns the OpcodeInfo from the registry if matched, or OpcodeInfo for MISC.
    """
    op = opcode.strip().lower()

    # 1. Exact match in canonical registry
    if op in _OPCODE_REGISTRY:
        return _OPCODE_REGISTRY[op]

    # 2. Atomic ordering suffixes (.aq, .rl, .aqrl)
    m = _ATOMIC_SUFFIX_RE.match(op)
    if m:
        base_op = m.group(1)
        if base_op in _OPCODE_REGISTRY:
            base_info = _OPCODE_REGISTRY[base_op]
            return OpcodeInfo(category=base_info.category, extension=base_info.extension, is_pseudo=base_info.is_pseudo)

    # 3. Fallback to MISC
    return OpcodeInfo(category="MISC", extension="unknown", is_pseudo=False)


def _classify_opcode(opcode: str) -> str:
    """Legacy helper: return category name for opcode."""
    return classify_opcode(opcode).category


def _extract_opcode(line: str) -> Optional[str]:
    """Legacy helper: extract opcode from line using shared assembly parser."""
    parsed = parse_line(line)
    if parsed is None or parsed.is_directive or parsed.opcode is None:
        return None
    return parsed.opcode.lower()


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstructionStats:
    """Immutable statistics of RISC-V instructions."""
    category_counts: Mapping[str, int]
    opcode_counts: Mapping[str, int]
    unknown_opcodes: Mapping[str, int]
    total: int
    source: str | None = None
    count_mode: str = "textual"
    isa: str = "unknown"
    schema_version: int = SCHEMA_VERSION
    classification_version: str = CLASSIFICATION_VERSION

    def __post_init__(self) -> None:
        # Defensive copy
        cat_copy = {cat: int(self.category_counts.get(cat, 0)) for cat in _CATEGORY_ORDER}
        op_copy = {str(k): int(v) for k, v in self.opcode_counts.items()}
        unk_copy = {str(k): int(v) for k, v in self.unknown_opcodes.items()}

        # Verify invariants
        for cat, v in cat_copy.items():
            if v < 0:
                raise ValueError(f"Category count cannot be negative: {cat}={v}")
        for op, v in op_copy.items():
            if v < 0:
                raise ValueError(f"Opcode count cannot be negative: {op}={v}")
        for op, v in unk_copy.items():
            if v < 0:
                raise ValueError(f"Unknown opcode count cannot be negative: {op}={v}")
            if op not in op_copy:
                raise ValueError(f"Unknown opcode '{op}' must be present in opcode_counts")

        cat_sum = sum(cat_copy.values())
        op_sum = sum(op_copy.values())

        if self.total != cat_sum:
            raise ValueError(f"total ({self.total}) != sum(category_counts) ({cat_sum})")
        if self.total != op_sum:
            raise ValueError(f"total ({self.total}) != sum(opcode_counts) ({op_sum})")

        # Freeze dictionaries
        object.__setattr__(self, "category_counts", MappingProxyType(cat_copy))
        object.__setattr__(self, "opcode_counts", MappingProxyType(op_copy))
        object.__setattr__(self, "unknown_opcodes", MappingProxyType(unk_copy))

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary following Topic 12 schema."""
        return {
            "schema_version": self.schema_version,
            "kind": "single",
            "classification_version": self.classification_version,
            "source": self.source,
            "count_mode": self.count_mode,
            "isa": self.isa,
            "total": self.total,
            "category_counts": dict(self.category_counts),
            "opcode_counts": dict(self.opcode_counts),
            "unknown_opcodes": dict(self.unknown_opcodes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstructionStats:
        """Create InstructionStats from dictionary."""
        return cls(
            category_counts=data["category_counts"],
            opcode_counts=data["opcode_counts"],
            unknown_opcodes=data.get("unknown_opcodes", {}),
            total=data["total"],
            source=data.get("source"),
            count_mode=data.get("count_mode", "textual"),
            isa=data.get("isa", "unknown"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            classification_version=data.get("classification_version", CLASSIFICATION_VERSION),
        )


@dataclass(frozen=True)
class CategoryDelta:
    """Delta calculation for an instruction category."""
    baseline: int
    current: int
    absolute: int
    percent: float | None


def _calc_percent(absolute: int, baseline: int) -> float | None:
    if baseline > 0:
        return round(absolute / baseline * 100, 4)
    if baseline == 0 and absolute == 0:
        return 0.0
    return None


@dataclass(frozen=True)
class ComparisonResult:
    """Comparison results across multiple assembly files."""
    baseline_label: str
    stats: Mapping[str, InstructionStats]
    category_deltas: Mapping[str, Mapping[str, CategoryDelta]]
    opcode_deltas: Mapping[str, Mapping[str, int]]

    def __post_init__(self) -> None:
        if self.baseline_label not in self.stats:
            raise ValueError(
                f"baseline_label '{self.baseline_label}' must be present in stats: {list(self.stats.keys())}"
            )
        stats_copy = {k: v for k, v in self.stats.items()}
        cat_deltas_copy = {
            label: MappingProxyType({c: cd for c, cd in deltas.items()})
            for label, deltas in self.category_deltas.items()
        }
        op_deltas_copy = {
            label: MappingProxyType({op: cnt for op, cnt in deltas.items()})
            for label, deltas in self.opcode_deltas.items()
        }

        object.__setattr__(self, "stats", MappingProxyType(stats_copy))
        object.__setattr__(self, "category_deltas", MappingProxyType(cat_deltas_copy))
        object.__setattr__(self, "opcode_deltas", MappingProxyType(op_deltas_copy))

    # --- Legacy compatibility properties ---

    @property
    def files(self) -> list[str]:
        """List of file labels."""
        return list(self.stats.keys())

    @property
    def counts(self) -> dict[str, dict[str, Any]]:
        """Per-file legacy dictionary of counts."""
        return {label: _stats_to_legacy_dict(s) for label, s in self.stats.items()}

    @property
    def diffs(self) -> dict[str, dict[str, int]]:
        """Per-file differences relative to baseline (excludes baseline)."""
        res: dict[str, dict[str, int]] = {}
        for label, deltas in self.category_deltas.items():
            if label != self.baseline_label:
                res[label] = {cat: deltas[cat].absolute for cat in _CATEGORY_ORDER}
        return res

    def to_dataframe(self):  # type: ignore[name-defined]  # noqa: F821
        """Convert to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame conversion.")

        rows = []
        for label in self.files:
            row = {"File": label}
            s = self.stats[label]
            row.update({c: s.category_counts.get(c, 0) for c in _CATEGORY_ORDER})
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary following Topic 12 comparison schema."""
        baseline_stats = self.stats[self.baseline_label]
        file_entries = []

        for label, s in self.stats.items():
            total_abs = s.total - baseline_stats.total
            total_pct = _calc_percent(total_abs, baseline_stats.total)

            cat_deltas = {}
            for cat in _CATEGORY_ORDER:
                cd = self.category_deltas[label][cat]
                cat_deltas[cat] = {
                    "baseline": cd.baseline,
                    "current": cd.current,
                    "absolute": cd.absolute,
                    "percent": cd.percent,
                }

            file_entries.append({
                "label": label,
                "stats": {
                    "source": s.source,
                    "total": s.total,
                    "category_counts": dict(s.category_counts),
                    "opcode_counts": dict(s.opcode_counts),
                    "unknown_opcodes": dict(s.unknown_opcodes),
                },
                "delta": {
                    "total": {
                        "baseline": baseline_stats.total,
                        "current": s.total,
                        "absolute": total_abs,
                        "percent": total_pct,
                    },
                    "category_counts": cat_deltas,
                    "opcode_counts": dict(self.opcode_deltas[label]),
                },
            })

        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "comparison",
            "classification_version": baseline_stats.classification_version,
            "count_mode": baseline_stats.count_mode,
            "isa": baseline_stats.isa,
            "baseline": self.baseline_label,
            "files": file_entries,
        }


# ---------------------------------------------------------------------------
# Core Analysis Functions
# ---------------------------------------------------------------------------

def analyze_instructions(
    asm_text: str,
    *,
    source: str | None = None,
    isa: str = "unknown",
) -> InstructionStats:
    """Analyze assembly text and return structured InstructionStats."""
    cat_counts = {cat: 0 for cat in _CATEGORY_ORDER}
    opcode_counts: Counter[str] = Counter()
    unknown_opcodes: Counter[str] = Counter()
    total = 0

    for line in asm_text.splitlines():
        parsed = parse_line(line)
        if parsed is None or parsed.is_directive or parsed.opcode is None:
            continue

        op = parsed.opcode.lower()
        info = classify_opcode(op)

        cat_counts[info.category] += 1
        opcode_counts[op] += 1
        total += 1

        if info.category == "MISC":
            unknown_opcodes[op] += 1

    return InstructionStats(
        category_counts=cat_counts,
        opcode_counts=dict(opcode_counts),
        unknown_opcodes=dict(unknown_opcodes),
        total=total,
        source=source,
        count_mode="textual",
        isa=isa,
        schema_version=SCHEMA_VERSION,
        classification_version=CLASSIFICATION_VERSION,
    )


def analyze_file(filepath: str, *, isa: str = "unknown") -> InstructionStats:
    """Analyze an assembly file from disk and return InstructionStats."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return analyze_instructions(content, source=filepath, isa=isa)


def from_emulator_counts(
    opcode_counts: Mapping[str, int],
    *,
    source: str = "emulator",
    isa: str = "rv32im",
) -> InstructionStats:
    """Construct an InstructionStats object from emulator dynamic opcode execution counts (count_mode='dynamic')."""
    cat_counts = {cat: 0 for cat in _CATEGORY_ORDER}
    norm_opcodes: dict[str, int] = {}
    unknown_opcodes: dict[str, int] = {}
    total = 0

    for op, cnt in opcode_counts.items():
        op_norm = str(op).strip().lower()
        cnt_int = int(cnt)
        norm_opcodes[op_norm] = norm_opcodes.get(op_norm, 0) + cnt_int
        info = classify_opcode(op_norm)
        cat_counts[info.category] += cnt_int
        total += cnt_int
        if info.category == "MISC":
            unknown_opcodes[op_norm] = unknown_opcodes.get(op_norm, 0) + cnt_int

    return InstructionStats(
        category_counts=cat_counts,
        opcode_counts=norm_opcodes,
        unknown_opcodes=unknown_opcodes,
        total=total,
        source=source,
        count_mode="dynamic",
        isa=isa,
        schema_version=SCHEMA_VERSION,
        classification_version=CLASSIFICATION_VERSION,
    )



# ---------------------------------------------------------------------------
# Comparison Functions
# ---------------------------------------------------------------------------

def compare_stats(
    items: Mapping[str, InstructionStats] | Sequence[tuple[str, InstructionStats]],
    *,
    baseline_label: str | None = None,
) -> ComparisonResult:
    """Compare multiple InstructionStats objects."""
    if isinstance(items, Mapping):
        item_list = list(items.items())
    else:
        item_list = list(items)

    if not item_list:
        raise ValueError("Cannot compare empty items")

    labels = [k for k, _ in item_list]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate labels found in comparison: {labels}")

    stats_dict = {k: v for k, v in item_list}

    # Resolve baseline
    if baseline_label is None:
        baseline_label = labels[0]
    elif baseline_label not in stats_dict:
        # Try matching source path
        found = None
        for lbl, s in stats_dict.items():
            if s.source == baseline_label:
                found = lbl
                break
        if found:
            baseline_label = found
        else:
            raise ValueError(f"Baseline label '{baseline_label}' not found among {labels}")

    base_s = stats_dict[baseline_label]

    # Verify compatibility
    for lbl, s in stats_dict.items():
        if s.schema_version != base_s.schema_version:
            raise ValueError(
                f"not comparable: schema_version mismatch ({s.schema_version} vs {base_s.schema_version})"
            )
        if s.classification_version != base_s.classification_version:
            raise ValueError(
                f"not comparable: classification_version mismatch "
                f"({s.classification_version} vs {base_s.classification_version})"
            )
        if s.count_mode != base_s.count_mode:
            raise ValueError(
                f"not comparable: count_mode mismatch ({s.count_mode} vs {base_s.count_mode})"
            )
        if s.isa != base_s.isa:
            raise ValueError(
                f"not comparable: isa mismatch ('{s.isa}' vs '{base_s.isa}')"
            )

    # Union of all opcodes
    all_opcodes: set[str] = set()
    for s in stats_dict.values():
        all_opcodes.update(s.opcode_counts.keys())

    category_deltas: dict[str, dict[str, CategoryDelta]] = {}
    opcode_deltas: dict[str, dict[str, int]] = {}

    for lbl, s in stats_dict.items():
        c_deltas: dict[str, CategoryDelta] = {}
        for cat in _CATEGORY_ORDER:
            base_cnt = base_s.category_counts.get(cat, 0)
            cur_cnt = s.category_counts.get(cat, 0)
            diff = cur_cnt - base_cnt
            pct = _calc_percent(diff, base_cnt)
            c_deltas[cat] = CategoryDelta(baseline=base_cnt, current=cur_cnt, absolute=diff, percent=pct)
        category_deltas[lbl] = c_deltas

        o_deltas: dict[str, int] = {}
        for op in sorted(all_opcodes):
            base_cnt = base_s.opcode_counts.get(op, 0)
            cur_cnt = s.opcode_counts.get(op, 0)
            o_deltas[op] = cur_cnt - base_cnt
        opcode_deltas[lbl] = o_deltas

    return ComparisonResult(
        baseline_label=baseline_label,
        stats=stats_dict,
        category_deltas=category_deltas,
        opcode_deltas=opcode_deltas,
    )


def _make_unique_labels(filepaths: Sequence[str]) -> list[str]:
    """Generate unique labels for filepaths, defaulting to basename."""
    basenames = [os.path.basename(p) for p in filepaths]
    if len(set(basenames)) == len(basenames):
        return basenames

    # If duplicate basenames, disambiguate
    counts = Counter(basenames)
    labels = []
    seen: dict[str, int] = {}
    for p, b in zip(filepaths, basenames):
        if counts[b] == 1:
            labels.append(b)
        else:
            idx = seen.get(b, 0) + 1
            seen[b] = idx
            labels.append(f"{b}#{idx}")
    return labels


def compare_files(
    filepaths: list[str],
    *,
    labels: list[str] | None = None,
    baseline: str | None = None,
    isa: str = "unknown",
) -> ComparisonResult:
    """Compare multiple assembly files."""
    if not filepaths:
        raise ValueError("filepaths cannot be empty")

    if labels is not None:
        if len(labels) != len(filepaths):
            raise ValueError("Number of labels must match number of filepaths")
        if len(set(labels)) != len(labels):
            raise ValueError("Labels must be unique")
    else:
        labels = _make_unique_labels(filepaths)

    items = []
    for lbl, path in zip(labels, filepaths):
        stats = analyze_file(path, isa=isa)
        items.append((lbl, stats))

    return compare_stats(items, baseline_label=baseline)


# ---------------------------------------------------------------------------
# Legacy Compatibility Adapters
# ---------------------------------------------------------------------------

def _stats_to_legacy_dict(stats: InstructionStats) -> dict[str, Any]:
    """Convert InstructionStats to legacy dictionary format."""
    counts: dict[str, Any] = {cat: stats.category_counts.get(cat, 0) for cat in _CATEGORY_ORDER}
    counts["_detailed"] = Counter(stats.opcode_counts)
    return counts


def count_instructions(asm_text: str) -> dict[str, Any]:
    """Legacy API: Count instructions by category returning a dict."""
    stats = analyze_instructions(asm_text)
    return _stats_to_legacy_dict(stats)


def count_instructions_file(filepath: str) -> dict[str, Any]:
    """Legacy API: Count instructions from file on disk returning a dict."""
    stats = analyze_file(filepath)
    return _stats_to_legacy_dict(stats)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render_text(
    target: InstructionStats | ComparisonResult | dict[str, Any],
    *,
    verbose: bool = False,
) -> str:
    """Render single stats or comparison as plain text table."""
    if isinstance(target, dict):
        # Adapt legacy dict
        cat_counts = {c: target.get(c, 0) for c in _CATEGORY_ORDER}
        detailed = target.get("_detailed", {})
        total = sum(v for k, v in cat_counts.items() if isinstance(v, int))
        target = InstructionStats(
            category_counts=cat_counts,
            opcode_counts=detailed,
            unknown_opcodes={k: v for k, v in detailed.items() if classify_opcode(k).category == "MISC"},
            total=total,
        )

    if isinstance(target, InstructionStats):
        lines = [
            "=" * 60,
            "RISC-V Instruction Statistics",
            "=" * 60,
            f"{'Category':<10} {'Count':>8} {'Percent':>10} {'Bar'}",
            "-" * 60,
        ]
        for cat in _CATEGORY_ORDER:
            cnt = target.category_counts.get(cat, 0)
            pct = (cnt / target.total * 100) if target.total > 0 else 0.0
            bar = "#" * max(1, int(pct / 2)) if cnt > 0 else ""
            lines.append(f"{cat:<10} {cnt:>8} {pct:>9.1f}% {bar}")

        lines.append("-" * 60)
        lines.append(f"{'TOTAL':<10} {target.total:>8}")
        lines.append("")

        if verbose and target.opcode_counts:
            lines.append("-" * 60)
            lines.append("Per-instruction Breakdown")
            lines.append("-" * 60)
            sorted_ops = sorted(target.opcode_counts.items(), key=lambda x: (-x[1], x[0]))
            for op, cnt in sorted_ops:
                cat = classify_opcode(op).category
                lines.append(f"  {op:<16} {cnt:>6}  ({cat})")
            lines.append("")

        if target.unknown_opcodes:
            lines.append("-" * 60)
            lines.append("WARNING: Unknown Instructions (MISC)")
            lines.append("-" * 60)
            for op, cnt in sorted(target.unknown_opcodes.items(), key=lambda x: (-x[1], x[0])):
                lines.append(f"  {op:<16} {cnt:>6}")
            lines.append("")

        return "\n".join(lines).rstrip()

    elif isinstance(target, ComparisonResult):
        lines = [
            "=" * 80,
            "RISC-V Instruction Count Comparison",
            "=" * 80,
        ]
        header = f"{'Category':<10}"
        for f in target.files:
            header += f" {f:>12}"
        if target.diffs:
            header += "  |" + "".join(f" {'diff ' + f:>12}" for f in target.diffs.keys())
        lines.append(header)
        lines.append("-" * 80)

        for cat in _CATEGORY_ORDER:
            line = f"{cat:<10}"
            for f in target.files:
                line += f" {target.stats[f].category_counts.get(cat, 0):>12}"
            if target.diffs:
                line += "  |"
                for f in target.diffs.keys():
                    val = target.category_deltas[f][cat].absolute
                    sign = "+" if val > 0 else ""
                    line += f" {sign}{val:>11}"
            lines.append(line)

        lines.append("-" * 80)
        tot_line = f"{'TOTAL':<10}"
        for f in target.files:
            tot_line += f" {target.stats[f].total:>12}"
        if target.diffs:
            tot_line += "  |"
            base_tot = target.stats[target.baseline_label].total
            for f in target.diffs.keys():
                cur_tot = target.stats[f].total
                diff = cur_tot - base_tot
                sign = "+" if diff > 0 else ""
                tot_line += f" {sign}{diff:>11}"
        lines.append(tot_line)
        lines.append("")

        if verbose:
            lines.append("-" * 80)
            lines.append("Per-instruction Breakdown Diff (relative to baseline)")
            lines.append("-" * 80)
            for f in target.diffs.keys():
                lines.append(f"Changes for {f}:")
                for op, delta in target.opcode_deltas[f].items():
                    if delta != 0:
                        sign = "+" if delta > 0 else ""
                        lines.append(f"  {op:<16} {sign}{delta:>6}")
            lines.append("")

        has_unk = any(s.unknown_opcodes for s in target.stats.values())
        if has_unk:
            lines.append("-" * 80)
            lines.append("WARNING: Files contain unknown instructions (MISC):")
            for f in target.files:
                unks = target.stats[f].unknown_opcodes
                if unks:
                    unk_str = ", ".join(f"{op}({cnt})" for op, cnt in unks.items())
                    lines.append(f"  {f}: {unk_str}")
            lines.append("")

        return "\n".join(lines).rstrip()

    raise TypeError(f"Unsupported target type: {type(target)}")


def format_table(counts: dict[str, Any] | InstructionStats) -> str:
    """Legacy API: Format instruction counts as text table."""
    return render_text(counts, verbose=True)


def render_json(
    target: InstructionStats | ComparisonResult | dict[str, Any],
    *,
    indent: int = 2,
) -> str:
    """Render single stats or comparison as JSON string."""
    if isinstance(target, dict):
        cat_counts = {c: target.get(c, 0) for c in _CATEGORY_ORDER}
        detailed = target.get("_detailed", {})
        total = sum(v for k, v in cat_counts.items() if isinstance(v, int))
        target = InstructionStats(
            category_counts=cat_counts,
            opcode_counts=detailed,
            unknown_opcodes={k: v for k, v in detailed.items() if classify_opcode(k).category == "MISC"},
            total=total,
        )
    return json.dumps(target.to_dict(), indent=indent)


def render_markdown(
    target: InstructionStats | ComparisonResult | dict[str, Any],
    *,
    verbose: bool = False,
) -> str:
    """Render single stats or comparison as GitHub Flavored Markdown."""
    if isinstance(target, dict):
        cat_counts = {c: target.get(c, 0) for c in _CATEGORY_ORDER}
        detailed = target.get("_detailed", {})
        total = sum(v for k, v in cat_counts.items() if isinstance(v, int))
        target = InstructionStats(
            category_counts=cat_counts,
            opcode_counts=detailed,
            unknown_opcodes={k: v for k, v in detailed.items() if classify_opcode(k).category == "MISC"},
            total=total,
        )

    if isinstance(target, InstructionStats):
        lines = [
            "# RISC-V Instruction Statistics",
            "",
            f"- **Total Instructions**: `{target.total}`",
            f"- **Source**: `{target.source or 'inline'}`",
            f"- **ISA**: `{target.isa}`",
            "",
            "| Category | Count | Percent |",
            "| :--- | ---: | ---: |",
        ]
        for cat in _CATEGORY_ORDER:
            cnt = target.category_counts.get(cat, 0)
            pct = (cnt / target.total * 100) if target.total > 0 else 0.0
            lines.append(f"| {cat} | {cnt} | {pct:.1f}% |")
        lines.append(f"| **TOTAL** | **{target.total}** | **100.0%** |")
        lines.append("")

        if target.unknown_opcodes:
            lines.append("> [!WARNING]")
            lines.append("> Contains unknown opcodes classified as MISC:")
            for op, cnt in sorted(target.unknown_opcodes.items()):
                lines.append(f"> - `{op}`: {cnt}")
            lines.append("")

        if verbose and target.opcode_counts:
            lines.append("## Per-instruction Breakdown")
            lines.append("")
            lines.append("| Opcode | Category | Count |")
            lines.append("| :--- | :--- | ---: |")
            for op, cnt in sorted(target.opcode_counts.items(), key=lambda x: (-x[1], x[0])):
                cat = classify_opcode(op).category
                lines.append(f"| `{op}` | {cat} | {cnt} |")
            lines.append("")

        return "\n".join(lines).rstrip()

    elif isinstance(target, ComparisonResult):
        def _esc(s: str) -> str:
            return s.replace("|", r"\|")

        lines = [
            "# RISC-V Instruction Count Comparison",
            "",
            f"- **Baseline**: `{target.baseline_label}`",
            "",
        ]
        header = "| Category |" + "".join(f" {_esc(f)} |" for f in target.files)
        align = "| :--- |" + "".join(" ---: |" for _ in target.files)
        if target.diffs:
            header += "".join(f" diff {_esc(f)} |" for f in target.diffs.keys())
            align += "".join(" ---: |" for _ in target.diffs)

        lines.append(header)
        lines.append(align)

        for cat in _CATEGORY_ORDER:
            row = f"| {cat} |"
            for f in target.files:
                row += f" {target.stats[f].category_counts.get(cat, 0)} |"
            if target.diffs:
                for f in target.diffs.keys():
                    val = target.category_deltas[f][cat].absolute
                    sign = "+" if val > 0 else ""
                    row += f" {sign}{val} |"
            lines.append(row)

        tot_row = "| **TOTAL** |"
        for f in target.files:
            tot_row += f" **{target.stats[f].total}** |"
        if target.diffs:
            base_tot = target.stats[target.baseline_label].total
            for f in target.diffs.keys():
                diff = target.stats[f].total - base_tot
                sign = "+" if diff > 0 else ""
                tot_row += f" **{sign}{diff}** |"
        lines.append(tot_row)
        lines.append("")

        has_unk = any(s.unknown_opcodes for s in target.stats.values())
        if has_unk:
            lines.append("> [!WARNING]")
            lines.append("> Unknown opcodes detected in one or more files:")
            for f in target.files:
                unks = target.stats[f].unknown_opcodes
                if unks:
                    lines.append(f"> - **{_esc(f)}**: " + ", ".join(f"`{op}` ({cnt})" for op, cnt in unks.items()))
            lines.append("")

        return "\n".join(lines).rstrip()

    raise TypeError(f"Unsupported target type: {type(target)}")


_HTML_BASE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         max-width: 960px; margin: 2em auto; padding: 0 1em; color: #24292f; background: #fff; }}
  h1 {{ color: #1f2328; border-bottom: 2px solid #0969da; padding-bottom: 0.3em; }}
  h2 {{ color: #1f2328; margin-top: 1.5em; border-bottom: 1px solid #d0d7de; padding-bottom: 0.2em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #d0d7de; }}
  th {{ background: #f6f8fa; font-weight: 600; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .bar {{ display: inline-block; height: 1em; background: #0969da; border-radius: 2px; vertical-align: middle; }}
  .section {{ margin: 2em 0; }}
  .summary {{ font-size: 1.1em; margin: 1em 0; background: #f6f8fa; padding: 12px; border-radius: 6px; }}
  .warning {{ background: #fff8c5; border: 1px solid #d4a72c; padding: 12px; border-radius: 6px; margin: 1em 0; }}
  code {{ font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace; background: #afb8c133; padding: 0.2em 0.4em; border-radius: 4px; }}
</style>
</head>
<body>
<h1>{title}</h1>
{content}
</body>
</html>
"""


def render_html(
    target: InstructionStats | ComparisonResult | dict[str, Any],
    *,
    title: str | None = None,
) -> str:
    """Render single stats or comparison as standalone HTML string."""
    if isinstance(target, dict):
        cat_counts = {c: target.get(c, 0) for c in _CATEGORY_ORDER}
        detailed = target.get("_detailed", {})
        total = sum(v for k, v in cat_counts.items() if isinstance(v, int))
        target = InstructionStats(
            category_counts=cat_counts,
            opcode_counts=detailed,
            unknown_opcodes={k: v for k, v in detailed.items() if classify_opcode(k).category == "MISC"},
            total=total,
        )

    if isinstance(target, InstructionStats):
        page_title = html.escape(title or "RISC-V Instruction Statistics")
        content_parts = [
            f'<div class="summary">',
            f'<strong>Total instructions:</strong> {target.total} &nbsp;|&nbsp; ',
            f'<strong>Source:</strong> <code>{html.escape(target.source or "inline")}</code> &nbsp;|&nbsp; ',
            f'<strong>ISA:</strong> <code>{html.escape(target.isa)}</code>',
            f'</div>',
        ]

        if target.unknown_opcodes:
            content_parts.append('<div class="warning"><strong>Warning: Unknown instructions detected:</strong><ul>')
            for op, cnt in sorted(target.unknown_opcodes.items()):
                content_parts.append(f'<li><code>{html.escape(op)}</code>: {cnt}</li>')
            content_parts.append('</ul></div>')

        content_parts.append('<div class="section"><table>')
        content_parts.append('<tr><th>Category</th><th class="num">Count</th><th class="num">Percent</th><th>Distribution</th></tr>')
        for cat in _CATEGORY_ORDER:
            cnt = target.category_counts.get(cat, 0)
            pct = (cnt / target.total * 100) if target.total > 0 else 0.0
            bar_w = int(pct * 3)
            content_parts.append(
                f'<tr><td>{html.escape(cat)}</td><td class="num">{cnt}</td><td class="num">{pct:.1f}%</td>'
                f'<td><span class="bar" style="width:{bar_w}px"></span></td></tr>'
            )
        content_parts.append(f'<tr><th>TOTAL</th><th class="num">{target.total}</th><th class="num">100.0%</th><th></th></tr>')
        content_parts.append('</table></div>')

        if target.opcode_counts:
            content_parts.append('<div class="section"><h2>Per-instruction Breakdown</h2><table>')
            content_parts.append('<tr><th>Instruction</th><th>Category</th><th class="num">Count</th></tr>')
            for op, cnt in sorted(target.opcode_counts.items(), key=lambda x: (-x[1], x[0])):
                cat = classify_opcode(op).category
                content_parts.append(f'<tr><td><code>{html.escape(op)}</code></td><td>{html.escape(cat)}</td><td class="num">{cnt}</td></tr>')
            content_parts.append('</table></div>')

        return _HTML_BASE_TEMPLATE.format(title=page_title, content="\n".join(content_parts))

    elif isinstance(target, ComparisonResult):
        page_title = html.escape(title or "RISC-V Instruction Count Comparison")
        content_parts = [
            f'<div class="summary">',
            f'<strong>Baseline:</strong> <code>{html.escape(target.baseline_label)}</code> &nbsp;|&nbsp; ',
            f'<strong>Compared files:</strong> {len(target.files)}',
            f'</div>',
        ]

        has_unk = any(s.unknown_opcodes for s in target.stats.values())
        if has_unk:
            content_parts.append('<div class="warning"><strong>Warning: Unknown instructions detected:</strong><ul>')
            for f in target.files:
                unks = target.stats[f].unknown_opcodes
                if unks:
                    unk_str = ", ".join(f"<code>{html.escape(op)}</code> ({cnt})" for op, cnt in unks.items())
                    content_parts.append(f'<li><strong>{html.escape(f)}</strong>: {unk_str}</li>')
            content_parts.append('</ul></div>')

        content_parts.append('<div class="section"><table>')
        hdr = '<tr><th>Category</th>' + "".join(f'<th class="num">{html.escape(f)}</th>' for f in target.files)
        if target.diffs:
            hdr += "".join(f'<th class="num">diff {html.escape(f)}</th>' for f in target.diffs.keys())
        hdr += '</tr>'
        content_parts.append(hdr)

        for cat in _CATEGORY_ORDER:
            row = f'<tr><td>{html.escape(cat)}</td>'
            for f in target.files:
                row += f'<td class="num">{target.stats[f].category_counts.get(cat, 0)}</td>'
            if target.diffs:
                for f in target.diffs.keys():
                    val = target.category_deltas[f][cat].absolute
                    sign = "+" if val > 0 else ""
                    row += f'<td class="num">{sign}{val}</td>'
            row += '</tr>'
            content_parts.append(row)

        tot_row = '<tr><th>TOTAL</th>'
        for f in target.files:
            tot_row += f'<th class="num">{target.stats[f].total}</th>'
        if target.diffs:
            base_tot = target.stats[target.baseline_label].total
            for f in target.diffs.keys():
                diff = target.stats[f].total - base_tot
                sign = "+" if diff > 0 else ""
                tot_row += f'<th class="num">{sign}{diff}</th>'
        tot_row += '</tr>'
        content_parts.append(tot_row)
        content_parts.append('</table></div>')

        return _HTML_BASE_TEMPLATE.format(title=page_title, content="\n".join(content_parts))

    raise TypeError(f"Unsupported target type: {type(target)}")


def generate_html_report(
    counts: dict[str, Any] | InstructionStats,
    output_path: str,
    title: str = "RISC-V Instruction Statistics",
) -> None:
    """Legacy API: Generate an HTML report file."""
    html_text = render_html(counts, title=title)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_text)


def render_chart(
    target: InstructionStats | ComparisonResult | dict[str, Any],
    output_path: str,
    *,
    title: str | None = None,
) -> None:
    """Generate matplotlib chart from stats or comparison."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for chart generation. "
            "Install it with: pip install matplotlib"
        )

    if isinstance(target, dict):
        cat_counts = {c: target.get(c, 0) for c in _CATEGORY_ORDER}
        detailed = target.get("_detailed", {})
        total = sum(v for k, v in cat_counts.items() if isinstance(v, int))
        target = InstructionStats(
            category_counts=cat_counts,
            opcode_counts=detailed,
            unknown_opcodes={k: v for k, v in detailed.items() if classify_opcode(k).category == "MISC"},
            total=total,
        )

    chart_title = title or ("RISC-V Instruction Distribution" if isinstance(target, InstructionStats) else "Multi-file Instruction Comparison")

    if isinstance(target, InstructionStats):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        categories = [c for c in _CATEGORY_ORDER if target.category_counts.get(c, 0) > 0]
        values = [target.category_counts.get(c, 0) for c in categories]

        colors = [
            "#4C72B0", "#55A868", "#C44E52", "#8172B2",
            "#CCB974", "#64B5CD", "#DDAA33", "#BB5566", "#004488",
        ]

        if not values or target.total == 0:
            ax1.text(0.5, 0.5, "No instructions found", ha="center", va="center", fontsize=12)
            ax2.text(0.5, 0.5, "No instructions found", ha="center", va="center", fontsize=12)
            ax1.set_title(chart_title)
            ax2.set_title("Instruction Distribution")
        else:
            bar_colors = [colors[i % len(colors)] for i in range(len(categories))]
            ax1.bar(categories, values, color=bar_colors, edgecolor="white", linewidth=0.8)
            ax1.set_title(chart_title)
            ax1.set_ylabel("Instruction Count")
            for i, v in enumerate(values):
                ax1.text(i, v + max(values) * 0.01, str(v), ha="center", fontsize=9)

            wedges, texts, autotexts = ax2.pie(
                values, labels=categories, autopct="%1.1f%%",
                colors=bar_colors, startangle=90,
            )
            for at in autotexts:
                at.set_fontsize(9)
            ax2.set_title("Instruction Distribution")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    elif isinstance(target, ComparisonResult):
        files = target.files
        categories = [
            c for c in _CATEGORY_ORDER
            if any(target.stats[f].category_counts.get(c, 0) > 0 for f in files)
        ]

        fig, ax = plt.subplots(figsize=(12, 5))
        colors = [
            "#4C72B0", "#55A868", "#C44E52", "#8172B2",
            "#CCB974", "#64B5CD", "#DDAA33", "#BB5566", "#004488",
        ]

        if not categories:
            ax.text(0.5, 0.5, "No instructions found", ha="center", va="center", fontsize=12)
            ax.set_title(chart_title)
        else:
            x = range(len(categories))
            width = 0.8 / len(files)
            for i, fname in enumerate(files):
                values = [target.stats[fname].category_counts.get(c, 0) for c in categories]
                offset = [xi + i * width for xi in x]
                ax.bar(
                    offset, values, width, label=fname,
                    color=colors[i % len(colors)],
                    edgecolor="white", linewidth=0.5,
                )
            ax.set_title(chart_title)
            ax.set_ylabel("Instruction Count")
            ax.set_xticks([xi + width * (len(files) - 1) / 2 for xi in x])
            ax.set_xticklabels(categories)
            ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()


def generate_chart(
    counts: dict[str, Any] | InstructionStats,
    output_path: str,
    title: str = "RISC-V Instruction Distribution",
) -> None:
    """Legacy API: Generate bar/pie chart for single file."""
    render_chart(counts, output_path, title=title)


def generate_multi_chart(
    multi_counts: dict[str, dict[str, int]] | ComparisonResult,
    output_path: str,
    title: str = "Multi-file Instruction Comparison",
) -> None:
    """Legacy API: Generate chart comparing multiple files."""
    if isinstance(multi_counts, ComparisonResult):
        render_chart(multi_counts, output_path, title=title)
        return

    # Convert legacy dict
    items = []
    for label, counts in multi_counts.items():
        cat_counts = {c: counts.get(c, 0) for c in _CATEGORY_ORDER}
        detailed = counts.get("_detailed", {})
        total = sum(v for k, v in cat_counts.items() if isinstance(v, int))
        s = InstructionStats(
            category_counts=cat_counts,
            opcode_counts=detailed,
            unknown_opcodes={k: v for k, v in detailed.items() if classify_opcode(k).category == "MISC"},
            total=total,
        )
        items.append((label, s))
    comp = compare_stats(items)
    render_chart(comp, output_path, title=title)


def print_comparison(result: ComparisonResult) -> None:
    """Legacy API: Pretty-print comparison to stdout."""
    print(render_text(result))


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point following Topic 12 contract. Returns exit code."""
    parser = argparse.ArgumentParser(
        description=(
            "RISC-V Instruction Counter - count and visualize "
            "instructions in assembly files"
        ),
    )
    parser.add_argument(
        "files", nargs="+",
        help="Assembly file(s) to analyze (.s)",
    )
    parser.add_argument(
        "--chart", type=str, default=None,
        help="Generate bar/pie chart and save to this path",
    )
    parser.add_argument(
        "--html", type=str, default=None,
        help="Generate HTML report and save to this path",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare multiple files side-by-side",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-instruction breakdown",
    )
    # Additional topic 12 options
    parser.add_argument(
        "--json", action="store_true",
        help="Print machine-readable report JSON to stdout",
    )
    parser.add_argument(
        "--json-output", type=str, default=None,
        help="Path to output structured JSON report",
    )
    parser.add_argument(
        "--markdown", type=str, default=None,
        help="Path to output GitHub Flavored Markdown report",
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Baseline label or file path for comparison (only with --compare)",
    )
    parser.add_argument(
        "--label", dest="labels", action="append", default=None,
        help="Custom label for input file (repeatable, must match file count)",
    )
    parser.add_argument(
        "--isa", type=str, default="unknown",
        help="Target ISA string for metadata tracking (default: unknown)",
    )
    parser.add_argument(
        "--fail-on-unknown", action="store_true",
        help="Exit with code 2 if any unknown opcodes are encountered",
    )

    args = parser.parse_args(argv)

    num_files = len(args.files)

    # 1. Validate argument combinations
    if num_files == 1 and args.compare:
        print("Error: --compare requires at least two files.", file=sys.stderr)
        return 1

    if args.baseline and not args.compare:
        print("Error: --baseline can only be used with --compare.", file=sys.stderr)
        return 1

    if args.labels is not None:
        if len(args.labels) != num_files:
            print(
                f"Error: Number of --label ({len(args.labels)}) does not match file count ({num_files}).",
                file=sys.stderr,
            )
            return 1
        if len(set(args.labels)) != len(args.labels):
            print("Error: Labels specified via --label must be unique.", file=sys.stderr)
            return 1

    # Check file existence
    for fpath in args.files:
        if not os.path.exists(fpath):
            print(f"Error: file not found: {fpath}", file=sys.stderr)
            return 1

    # 2. Execution
    has_unknown = False

    try:
        if args.compare and num_files > 1:
            comp = compare_files(args.files, labels=args.labels, baseline=args.baseline, isa=args.isa)
            for st in comp.stats.values():
                if st.unknown_opcodes:
                    has_unknown = True

            # Print comparison to stdout (suppress when machine-readable --json is requested)
            if not args.json:
                print(render_text(comp, verbose=args.verbose))

            # Charts
            if args.chart:
                render_chart(comp, args.chart)
                print(f"Chart saved to {args.chart}")

            # HTML report
            if args.html:
                with open(args.html, "w", encoding="utf-8") as f:
                    f.write(render_html(comp))
                print(f"HTML report saved to {args.html}")

            # Markdown report
            if args.markdown:
                with open(args.markdown, "w", encoding="utf-8") as f:
                    f.write(render_markdown(comp, verbose=args.verbose))
                print(f"Markdown report written to {args.markdown}")

            # JSON reports
            if args.json_output:
                with open(args.json_output, "w", encoding="utf-8") as f:
                    f.write(render_json(comp))
                print(f"JSON report written to {args.json_output}")

            if args.json:
                print(render_json(comp))

        else:
            # Single-file or sequential mode (original default behavior)
            multi: dict[str, Any] = {}
            multi_stats: list[InstructionStats] = []
            for fpath in args.files:
                st = analyze_file(fpath, isa=args.isa)
                label = os.path.basename(fpath)
                multi[label] = st
                multi_stats.append(st)
                if st.unknown_opcodes:
                    has_unknown = True
                if not args.json:
                    print(render_text(st, verbose=args.verbose))

            # Charts
            if args.chart and len(multi_stats) == 1:
                render_chart(multi_stats[0], args.chart)
                print(f"Chart saved to {args.chart}")
            elif args.chart and len(multi_stats) > 1:
                comp = compare_stats([(os.path.basename(f), s) for f, s in zip(args.files, multi_stats)])
                render_chart(comp, args.chart)
                print(f"Comparison chart saved to {args.chart}")

            # HTML report
            if args.html:
                target_html = multi_stats[0] if len(multi_stats) == 1 else compare_stats([(os.path.basename(f), s) for f, s in zip(args.files, multi_stats)])
                with open(args.html, "w", encoding="utf-8") as f:
                    f.write(render_html(target_html))
                print(f"HTML report saved to {args.html}")

            # Markdown report
            if args.markdown:
                target_md = multi_stats[0] if len(multi_stats) == 1 else compare_stats([(os.path.basename(f), s) for f, s in zip(args.files, multi_stats)])
                with open(args.markdown, "w", encoding="utf-8") as f:
                    f.write(render_markdown(target_md, verbose=args.verbose))
                print(f"Markdown report written to {args.markdown}")

            # JSON reports
            if args.json_output:
                target_json = multi_stats[0] if len(multi_stats) == 1 else compare_stats([(os.path.basename(f), s) for f, s in zip(args.files, multi_stats)])
                with open(args.json_output, "w", encoding="utf-8") as f:
                    f.write(render_json(target_json))
                print(f"JSON report written to {args.json_output}")

            if args.json:
                target_json = multi_stats[0] if len(multi_stats) == 1 else compare_stats([(os.path.basename(f), s) for f, s in zip(args.files, multi_stats)])
                print(render_json(target_json))

    except ImportError as e:
        print(f"Rendering dependency error: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"Error during instruction analysis: {e}", file=sys.stderr)
        return 1

    # 4. Strict unknown check
    if args.fail_on_unknown and has_unknown:
        print("Error: Unknown opcodes encountered with --fail-on-unknown enabled.", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
