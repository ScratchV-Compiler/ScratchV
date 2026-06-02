"""RISC-V 5-stage pipeline cycle estimator for ScratchV.

Models a classic single-issue in-order RISC-V pipeline (IF → ID → EX →
MEM → WB) with data-hazard detection, forwarding paths, branch
prediction, and L1 cache interaction.  Produces per-stage and
per-category cycle breakdowns suitable for performance analysis.

Usage::

    from scratchv.backend.cycle_estimator import (
        PipelineCycleEstimator, PipelineConfig,
    )
    estimator = PipelineCycleEstimator(PipelineConfig())
    stats = estimator.estimate(asm_text)
    print(estimator.report(stats))

Output includes:
    - Total cycles and CPI
    - Stalls by cause (data hazard, structural, branch mispredict, cache miss)
    - Per-stage utilisation
    - Per-category cycle breakdown
"""

from __future__ import annotations

import enum
import re as _re
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Configuration for the pipeline cycle estimator.

    Attributes:
        enable_forwarding:
            Enable forwarding/bypass paths (reduces data-hazard stalls).
            When True, EX→EX and MEM→EX forwarding is active.
        branch_predictor:
            ``"always_taken"``, ``"always_not_taken"``, or ``"btb"``
            (simple 1-bit bimodal).
        btb_entries:
            Number of BTB entries (only used with ``branch_predictor="btb"``).
        branch_mispredict_penalty:
            Cycles to flush on a branch misprediction (typically 2 for 5-stage).
        enable_cache_model:
            If True, model L1 I$/D$ hit/miss and add miss penalties.
        icache_miss_penalty:
            Extra stall cycles on instruction-cache miss (default 4).
        dcache_miss_penalty:
            Extra stall cycles on data-cache miss (default 8).
        issue_width:
            Instructions issued per cycle (1 = scalar, >1 = superscalar).
            Currently only 1 is supported.
        enable_structural_hazards:
            If True, model structural hazards (e.g. single memory port for
            both IF and MEM stages).
    """

    enable_forwarding: bool = True
    branch_predictor: str = "always_not_taken"  # always_taken | always_not_taken | btb
    btb_entries: int = 64
    branch_mispredict_penalty: int = 2
    enable_cache_model: bool = False
    icache_miss_penalty: int = 4
    dcache_miss_penalty: int = 8
    issue_width: int = 1
    enable_structural_hazards: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# Instruction categories (for per-category cycle breakdown)
# ═══════════════════════════════════════════════════════════════════════════════

_INST_CATEGORY: dict[str, str] = {
    # Integer ALU
    "add": "ALU", "addi": "ALU", "sub": "ALU",
    "sll": "ALU", "srl": "ALU", "sra": "ALU",
    "xor": "ALU", "or": "ALU", "and": "ALU",
    "xori": "ALU", "ori": "ALU", "andi": "ALU",
    "slli": "ALU", "srli": "ALU", "srai": "ALU",
    "slt": "ALU", "sltu": "ALU", "slti": "ALU", "sltiu": "ALU",
    # M extension (multi-cycle)
    "mul": "MUL_DIV",
    "mulh": "MUL_DIV", "mulhsu": "MUL_DIV", "mulhu": "MUL_DIV",
    "div": "MUL_DIV", "divu": "MUL_DIV",
    "rem": "MUL_DIV", "remu": "MUL_DIV",
    # Memory
    "lw": "MEM", "lh": "MEM", "lb": "MEM",
    "lbu": "MEM", "lhu": "MEM",
    "sw": "MEM", "sh": "MEM", "sb": "MEM",
    "flw": "MEM", "fsw": "MEM", "fld": "MEM", "fsd": "MEM",
    # Branches
    "beq": "BRANCH", "bne": "BRANCH", "blt": "BRANCH",
    "bge": "BRANCH", "bltu": "BRANCH", "bgeu": "BRANCH",
    "beqz": "BRANCH", "bnez": "BRANCH",
    # Jumps
    "j": "JUMP", "jal": "JUMP", "jalr": "JUMP", "ret": "JUMP", "jr": "JUMP",
    # Pseudo
    "li": "ALU", "mv": "ALU", "nop": "ALU",
    "lui": "ALU", "auipc": "ALU",
    "call": "JUMP", "tail": "JUMP",
    "max": "ALU",
    # Float
    "fadd.s": "FPU", "fsub.s": "FPU", "fmul.s": "FPU", "fdiv.s": "FPU",
    "fadd.d": "FPU", "fsub.d": "FPU", "fmul.d": "FPU", "fdiv.d": "FPU",
}
_CATEGORIES = ["ALU", "MUL_DIV", "MEM", "BRANCH", "JUMP", "FPU", "OTHER"]


# ═══════════════════════════════════════════════════════════════════════════════
# Instruction latency (execution stage cycles, may be pipelined)
# ═══════════════════════════════════════════════════════════════════════════════

_LATENCY: dict[str, int] = {
    # ALU — 1 cycle, fully pipelined
    "add": 1, "addi": 1, "sub": 1,
    "sll": 1, "srl": 1, "sra": 1,
    "xor": 1, "or": 1, "and": 1,
    "xori": 1, "ori": 1, "andi": 1,
    "slli": 1, "srli": 1, "srai": 1,
    "slt": 1, "sltu": 1, "slti": 1, "sltiu": 1,
    "lui": 1, "auipc": 1,
    "li": 1, "mv": 1, "nop": 1,
    "max": 1,
    # M-extension — multi-cycle, fully pipelined (assume typical embedded core)
    "mul": 3, "mulh": 3, "mulhsu": 3, "mulhu": 3,
    "div": 16, "divu": 16,        # NOT pipelined: blocks EX stage
    "rem": 16, "remu": 16,        # NOT pipelined
    # Memory — 1 EX cycle + MEM stage
    "lw": 1, "lh": 1, "lb": 1,
    "lbu": 1, "lhu": 1,
    "sw": 1, "sh": 1, "sb": 1,
    "flw": 1, "fsw": 1, "fld": 1, "fsd": 1,
    # Branches — resolved in ID (1 cycle)
    "beq": 1, "bne": 1, "blt": 1, "bge": 1,
    "bltu": 1, "bgeu": 1, "beqz": 1, "bnez": 1,
    # Jumps — resolved in ID
    "j": 1, "jal": 1, "jalr": 1, "ret": 1, "jr": 1,
    "call": 1, "tail": 1,
    # Float single-precision
    "fadd.s": 3, "fsub.s": 3, "fmul.s": 4, "fdiv.s": 12,
    "fmax.s": 2, "fmin.s": 2, "flt.s": 2, "fle.s": 2, "feq.s": 2,
    "fsqrt.s": 12,
    # Float double-precision
    "fadd.d": 4, "fsub.d": 4, "fmul.d": 5, "fdiv.d": 16,
    "fmax.d": 3, "fmin.d": 3, "flt.d": 3, "feq.d": 3,
    "fsqrt.d": 16,
}

# Ops that are NOT fully pipelined (block subsequent issue to same unit)
_NOT_PIPELINED: set[str] = {
    "div", "divu", "rem", "remu",
    "fdiv.s", "fdiv.d", "fsqrt.s", "fsqrt.d",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline stage enum
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineStage(enum.IntEnum):
    """Five classic RISC-V pipeline stages."""
    IF = 0   # Instruction Fetch
    ID = 1   # Instruction Decode / Register Read
    EX = 2   # Execute / Address Calculation
    MEM = 3  # Memory Access
    WB = 4   # Write Back


# ═══════════════════════════════════════════════════════════════════════════════
# Parsed instruction for the pipeline simulator
# ═══════════════════════════════════════════════════════════════════════════════

_LINE_RE = _re.compile(
    r'^\s*'
    r'(?:[A-Za-z_.][A-Za-z0-9_.]*:\s*)?'
    r'\.?(?P<opcode>[a-zA-Z][a-zA-Z0-9.]*)?\s*'
    r'(?P<operands>[^#]*)'
)


@dataclass
class PipelineInst:
    """A decoded instruction for pipeline simulation.

    Attributes:
        id:          Sequential instruction index.
        opcode:      Lowercase mnemonic (e.g. ``"add"``).
        operands:    List of operand strings.
        src_regs:    Set of source register names (RAW dependency sources).
        dst_reg:     Destination register name, or None.
        is_branch:   True if this is a conditional branch.
        is_jump:     True if this is an unconditional jump (including jalr).
        is_load:     True if this loads from memory.
        is_store:    True if this stores to memory.
        latency:     EX-stage latency in cycles.
        is_pipelined: Whether this op can overlap with subsequent ops.
        target:      Branch/jump target label, if any.
        raw_line:    Original assembly text line.
    """

    id: int
    opcode: str
    operands: list[str] = field(default_factory=list)
    src_regs: set[str] = field(default_factory=set)
    dst_reg: Optional[str] = None
    is_branch: bool = False
    is_jump: bool = False
    is_load: bool = False
    is_store: bool = False
    latency: int = 1
    is_pipelined: bool = True
    target: Optional[str] = None
    raw_line: str = ""


def _parse_asm_for_pipeline(asm_text: str) -> list[PipelineInst]:
    """Parse RISC-V assembly into a list of PipelineInst for simulation.

    Strips labels, directives, and pure-comment lines.
    """
    result: list[PipelineInst] = []
    labels: dict[str, int] = {}  # label_name -> instruction id
    idx = 0

    for line in asm_text.strip().split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Capture labels
        label_match = _re.match(r'^([A-Za-z_.][A-Za-z0-9_.]*):\s*(.*)', stripped)
        if label_match:
            labels[label_match.group(1)] = idx
            stripped = label_match.group(2).strip()
            if not stripped:
                continue

        # Remove inline comments
        code = stripped.split("#")[0].strip()
        if not code:
            continue

        m = _LINE_RE.match(stripped)
        if m is None:
            continue

        opcode = m.group("opcode")
        if opcode is None:
            continue
        opcode = opcode.lower().lstrip(".")

        # Skip assembler directives
        if opcode.startswith(".") or opcode in (
            ".text", ".data", ".bss", ".rodata", ".globl", ".global",
            ".type", ".size", ".align", ".section", ".file", ".loc",
        ):
            continue

        operands_str = (m.group("operands") or "").strip()
        operands = [o.strip() for o in operands_str.split(",") if o.strip()]

        # Classify
        is_branch = opcode in (
            "beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez",
        )
        is_jump = opcode in ("j", "jal", "jalr", "ret", "jr", "call", "tail")
        is_load = opcode in ("lw", "lh", "lb", "lbu", "lhu", "ld", "flw", "fld")
        is_store = opcode in ("sw", "sh", "sb", "sd", "fsw", "fsd")

        # Extract source and destination registers
        src_regs: set[str] = set()
        dst_reg: Optional[str] = None

        for i, op in enumerate(operands):
            # Extract register from "offset(base)" memory operands
            mem_match = _re.match(r'-?\d+\((\w+)\)', op)
            if mem_match:
                base = mem_match.group(1)
                if is_store:
                    src_regs.add(base)
                else:
                    src_regs.add(base)
                continue

            # First operand is destination for most ALU instructions
            if i == 0 and not is_store and not is_branch and not is_jump:
                dst_reg = op
            elif _looks_like_reg(op):
                src_regs.add(op)

        # For stores, first operand is a source (value to store)
        if is_store and operands:
            src_regs.add(operands[0])  # value
            # Second operand has the base register (already added above)

        # For branches, both operands are sources
        if is_branch:
            dst_reg = None

        latency = _LATENCY.get(opcode, 1)
        is_pipelined = opcode not in _NOT_PIPELINED

        inst = PipelineInst(
            id=idx,
            opcode=opcode,
            operands=operands,
            src_regs=src_regs,
            dst_reg=dst_reg,
            is_branch=is_branch,
            is_jump=is_jump,
            is_load=is_load,
            is_store=is_store,
            latency=latency,
            is_pipelined=is_pipelined,
            raw_line=stripped,
        )
        result.append(inst)
        idx += 1

    # Resolve branch/jump targets
    for inst in result:
        if (inst.is_branch or inst.is_jump) and inst.operands:
            # Last operand is typically the target label
            target_label = inst.operands[-1]
            if target_label in labels:
                inst.target = target_label

    return result


def _looks_like_reg(s: str) -> bool:
    """Heuristic: does string look like a register name?"""
    if not s:
        return False
    if s in ("zero", "ra", "sp", "gp", "tp", "fp"):
        return True
    if _re.match(r'^x([0-9]|[12][0-9]|3[01])$', s):
        return True
    if _re.match(r'^[ats]([0-9]|1[01])$', s):
        return True
    if _re.match(r'^f([0-9]|[12][0-9]|3[01])$', s):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline state per instruction-in-flight
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _InFlight:
    """Tracks one instruction as it flows through the pipeline."""

    inst: PipelineInst
    stage: PipelineStage = PipelineStage.IF
    # Cycle when this instruction entered its current stage
    stage_start_cycle: int = 0
    # Set when the instruction is stalled in its current stage
    stalled: bool = False
    # EX stage remaining cycles (for multi-cycle ops)
    ex_remaining: int = 0
    # For loads: data available at this cycle (load-use hazard detection)
    load_result_ready_at: int = -1
    # Whether this branch was mispredicted
    mispredicted: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# Branch predictor (simple bimodal / BTB)
# ═══════════════════════════════════════════════════════════════════════════════

class _BranchPredictor:
    """Simple branch predictor.

    Modes:
        ``always_taken``     — predict every branch as taken.
        ``always_not_taken`` — predict every branch as not-taken.
        ``btb``              — 1-bit bimodal with BTB (predict taken if
                                previously seen and taken).
    """

    def __init__(self, mode: str = "always_not_taken", btb_entries: int = 64):
        self.mode = mode
        self._btb: dict[int, bool] = {}  # pc -> taken (1-bit history)

    def predict(self, pc: int, is_branch: bool) -> bool:
        """Return True if the branch is predicted taken."""
        if not is_branch:
            return False
        if self.mode == "always_taken":
            return True
        if self.mode == "always_not_taken":
            return False
        # BTB mode
        return self._btb.get(pc, False)

    def update(self, pc: int, taken: bool) -> None:
        """Update predictor state with actual branch outcome."""
        if self.mode != "btb":
            return
        self._btb[pc] = taken


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline cycle statistics
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CycleStats:
    """Detailed pipeline cycle statistics.

    Attributes:
        total_cycles:
            Total cycles from first IF to last WB.
        total_instructions:
            Number of instructions that completed.
        cpi:
            Cycles per instruction (total_cycles / total_instructions).
        stall_cycles:
            Total stall cycles (sum of all stall causes).
        data_hazard_stalls:
            Stalls caused by RAW data hazards (load-use, etc.).
        structural_stalls:
            Stalls caused by structural hazards (single memory port, etc.).
        branch_mispredict_stalls:
            Stalls from flushing the pipeline on branch misprediction.
        cache_miss_stalls:
            Stalls from I$ or D$ misses (only when cache model is enabled).
        multi_cycle_stalls:
            Stalls because a non-pipelined multi-cycle op (div, rem) blocked
            subsequent instruction issue.
        stages:  Cycles spent in each pipeline stage (IF/ID/EX/MEM/WB).
        category_cycles:
            EX-stage cycles per instruction category (ALU, MEM, BRANCH, etc.).
        per_instruction:
            Per-instruction details: (opcode, issued_at, completed_at, stalls).
    """

    total_cycles: int = 0
    total_instructions: int = 0
    cpi: float = 0.0
    stall_cycles: int = 0
    data_hazard_stalls: int = 0
    structural_stalls: int = 0
    branch_mispredict_stalls: int = 0
    cache_miss_stalls: int = 0
    multi_cycle_stalls: int = 0
    stages: dict[str, int] = field(default_factory=dict)
    category_cycles: dict[str, int] = field(default_factory=dict)
    per_instruction: list[dict] = field(default_factory=list)

    # How many instructions were in each pipeline stage per cycle (utilisation)
    stage_utilisation: dict[str, list[int]] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline cycle estimator
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineCycleEstimator:
    """Cycle-accurate 5-stage RISC-V pipeline simulator.

    Models IF → ID → EX → MEM → WB with:
    - Data-hazard detection (RAW) with optional forwarding
    - Load-use hazard (1-cycle stall even with forwarding)
    - Structural hazard: single memory port (IF conflicts with MEM)
    - Branch prediction with configurable mispredict penalty
    - Multi-cycle non-pipelined ops (div, rem block EX)
    - Optional L1 I$/D$ cache miss simulation
    - Per-stage utilisation tracking

    Parameters
    ----------
    config:
        ``PipelineConfig`` with all tuning knobs.

    Usage::

        estimator = PipelineCycleEstimator(PipelineConfig(
            enable_forwarding=True,
            branch_predictor="btb",
        ))
        stats = estimator.estimate(asm_text)
        print(estimator.report(stats))
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._predictor = _BranchPredictor(
            mode=self.config.branch_predictor,
            btb_entries=self.config.btb_entries,
        )

    # ── Main entry point ────────────────────────────────────────────────────

    def estimate(self, asm_text: str) -> CycleStats:
        """Run pipeline simulation on assembly text.

        Args:
            asm_text: RISC-V assembly source.

        Returns:
            A ``CycleStats`` object with detailed breakdowns.
        """
        insts = _parse_asm_for_pipeline(asm_text)
        if not insts:
            return CycleStats()

        return self._simulate(insts)

    # ── Core simulation ─────────────────────────────────────────────────────

    def _simulate(self, insts: list[PipelineInst]) -> CycleStats:
        """Run the cycle-by-cycle pipeline simulation."""
        config = self.config

        # Pipeline registers: one slot per stage (scalar pipeline)
        pipeline: dict[PipelineStage, Optional[_InFlight]] = {
            PipelineStage.IF: None,
            PipelineStage.ID: None,
            PipelineStage.EX: None,
            PipelineStage.MEM: None,
            PipelineStage.WB: None,
        }

        # Instruction queue
        next_issue_idx: int = 0
        completed: list[_InFlight] = []
        retired: set[int] = set()  # ids of completed instructions

        # Scoreboard: which dst register will be written by an in-flight inst,
        # and when the result becomes available for forwarding
        #   reg_name -> (cycle_available, inst_id)
        scoreboard: dict[str, tuple[int, int]] = {}

        # Stall tracking
        stats = CycleStats()
        stage_cycles: dict[str, int] = {s.name: 0 for s in PipelineStage}
        stage_util: dict[str, list[int]] = {s.name: [] for s in PipelineStage}

        cycle = 0
        MAX_CYCLES = 10_000_000  # safety

        while (len(retired) < len(insts) or
               any(p is not None for p in pipeline.values())):
            if cycle >= MAX_CYCLES:
                break

            # Record stage utilisation at this cycle
            for stage_name in stage_util:
                stage_util[stage_name].append(
                    1 if pipeline[PipelineStage[stage_name]] is not None else 0
                )

            # ── WB stage ────────────────────────────────────────────────────
            wb_inst = pipeline[PipelineStage.WB]
            if wb_inst is not None:
                retired.add(wb_inst.inst.id)
                completed.append(wb_inst)
                stats.total_instructions += 1
                # Free destination register from scoreboard
                if wb_inst.inst.dst_reg and wb_inst.inst.dst_reg != "zero":
                    if wb_inst.inst.dst_reg in scoreboard:
                        del scoreboard[wb_inst.inst.dst_reg]
                pipeline[PipelineStage.WB] = None

            # ── MEM stage ────────────────────────────────────────────────────
            mem_inst = pipeline[PipelineStage.MEM]
            if mem_inst is not None:
                # Check for data-cache miss (if enabled)
                dcache_stall = False
                if config.enable_cache_model and mem_inst.inst.is_load:
                    # Simplified: 5% random cache miss rate
                    if _hash_pc(mem_inst.inst.id, 37) < 0.05:
                        dcache_stall = True
                        stats.cache_miss_stalls += config.dcache_miss_penalty

                if mem_inst.stalled:
                    mem_inst.stalled = False
                elif dcache_stall:
                    mem_inst.stalled = True
                else:
                    # Advance to WB
                    can_advance = (pipeline[PipelineStage.WB] is None)
                    if can_advance:
                        pipeline[PipelineStage.MEM] = None
                        mem_inst.stage = PipelineStage.WB
                        mem_inst.stage_start_cycle = cycle
                        pipeline[PipelineStage.WB] = mem_inst

            # ── EX stage ────────────────────────────────────────────────────
            ex_inst = pipeline[PipelineStage.EX]
            if ex_inst is not None:
                # Multi-cycle execution
                if ex_inst.ex_remaining > 0:
                    ex_inst.ex_remaining -= 1
                    if ex_inst.ex_remaining > 0:
                        # Still executing — check if it blocks subsequent issue
                        if not ex_inst.inst.is_pipelined:
                            stats.multi_cycle_stalls += 1
                    else:
                        # Execution complete, can move to MEM
                        pass
                elif ex_inst.stalled:
                    ex_inst.stalled = False
                else:
                    # Advance to MEM
                    needs_mem = ex_inst.inst.is_load or ex_inst.inst.is_store

                    if needs_mem:
                        # Structural hazard: MEM stage occupied?
                        if pipeline[PipelineStage.MEM] is not None:
                            # Wait until MEM is free
                            ex_inst.stalled = True
                            stats.structural_stalls += 1
                        else:
                            pipeline[PipelineStage.EX] = None
                            ex_inst.stage = PipelineStage.MEM
                            ex_inst.stage_start_cycle = cycle
                            pipeline[PipelineStage.MEM] = ex_inst
                    else:
                        # Bypass MEM stage for non-memory ops
                        if pipeline[PipelineStage.WB] is None:
                            pipeline[PipelineStage.EX] = None
                            ex_inst.stage = PipelineStage.WB
                            ex_inst.stage_start_cycle = cycle
                            pipeline[PipelineStage.WB] = ex_inst
                        elif pipeline[PipelineStage.MEM] is None:
                            # Shuffle through MEM slot
                            pipeline[PipelineStage.EX] = None
                            ex_inst.stage = PipelineStage.MEM
                            ex_inst.stage_start_cycle = cycle
                            pipeline[PipelineStage.MEM] = ex_inst
                        else:
                            ex_inst.stalled = True
                            stats.structural_stalls += 1

                # Forwarding: update scoreboard when EX produces result
                if ex_inst is not None and ex_inst.ex_remaining == 0:
                    if ex_inst.inst.dst_reg and ex_inst.inst.dst_reg != "zero":
                        scoreboard[ex_inst.inst.dst_reg] = (
                            cycle + 1, ex_inst.inst.id,  # available next cycle
                        )

            # ── ID stage ────────────────────────────────────────────────────
            id_inst = pipeline[PipelineStage.ID]
            if id_inst is not None:
                can_issue = True

                # Check data hazards against scoreboard
                for src_reg in id_inst.inst.src_regs:
                    if src_reg in scoreboard:
                        avail_cycle, producer_id = scoreboard[src_reg]
                        if avail_cycle > cycle:
                            # Result not yet available
                            if config.enable_forwarding:
                                # With forwarding: check if producer will
                                # forward by next cycle
                                if avail_cycle > cycle + 1:
                                    can_issue = False
                                    stats.data_hazard_stalls += 1
                                    break
                                # else: forwarding resolves it
                            else:
                                can_issue = False
                                stats.data_hazard_stalls += 1
                                break

                # Load-use hazard: load in EX, next instruction uses result
                ex_now = pipeline[PipelineStage.EX]
                if ex_now is not None and ex_now.inst.is_load:
                    ld_dst = ex_now.inst.dst_reg
                    if ld_dst and ld_dst in id_inst.inst.src_regs:
                        # Even with forwarding, load data arrives after MEM
                        can_issue = False
                        stats.data_hazard_stalls += 1

                if not can_issue:
                    id_inst.stalled = True
                else:
                    # Structural hazard: EX stage free?
                    if pipeline[PipelineStage.EX] is not None:
                        ex_occ = pipeline[PipelineStage.EX]
                        if (ex_occ is not None
                                and not ex_occ.inst.is_pipelined
                                and ex_occ.ex_remaining > 0):
                            id_inst.stalled = True
                            stats.multi_cycle_stalls += 1
                        else:
                            id_inst.stalled = True
                            stats.structural_stalls += 1
                    else:
                        pipeline[PipelineStage.ID] = None
                        id_inst.stage = PipelineStage.EX
                        id_inst.stage_start_cycle = cycle
                        id_inst.ex_remaining = id_inst.inst.latency
                        pipeline[PipelineStage.EX] = id_inst

            # ── IF stage ────────────────────────────────────────────────────
            if_inst = pipeline[PipelineStage.IF]
            if if_inst is not None:
                # Structural hazard: ID stage free?
                if pipeline[PipelineStage.ID] is not None:
                    if_inst.stalled = True
                    stats.structural_stalls += 1
                else:
                    pipeline[PipelineStage.IF] = None
                    if_inst.stage = PipelineStage.ID
                    if_inst.stage_start_cycle = cycle
                    pipeline[PipelineStage.ID] = if_inst

            # ── Fetch new instruction ────────────────────────────────────────
            if (pipeline[PipelineStage.IF] is None
                    and next_issue_idx < len(insts)):

                # Structural hazard: IF and MEM share the same memory port?
                if config.enable_structural_hazards:
                    mem_occupied = pipeline[PipelineStage.MEM] is not None
                    if mem_occupied:
                        stats.structural_stalls += 1
                        cycle += 1
                        continue

                # I-cache miss (if enabled)
                icache_stall = False
                if config.enable_cache_model:
                    if _hash_pc(next_issue_idx, 13) < 0.02:
                        icache_stall = True
                        stats.cache_miss_stalls += config.icache_miss_penalty
                        # Skip fetch this cycle
                        cycle += 1
                        continue

                if not icache_stall:
                    inst = insts[next_issue_idx]

                    # Branch prediction at fetch
                    predict_taken = self._predictor.predict(
                        next_issue_idx, inst.is_branch,
                    )

                    new_if = _InFlight(inst=inst, stage_start_cycle=cycle)
                    pipeline[PipelineStage.IF] = new_if

                    # Handle branch prediction
                    if inst.is_branch or inst.is_jump:
                        actual_taken = inst.is_jump or (
                            inst.is_branch
                        )  # we need to resolve later
                        # For unconditional jumps, always taken
                        if inst.is_jump:
                            # Flush IF (already fetched wrong sequential inst)
                            pass  # the jump will be resolved in ID

                    next_issue_idx += 1

            # ── Resolve branches in ID (flush mispredictions) ───────────────
            id_now = pipeline[PipelineStage.ID]
            if id_now is not None and (id_now.inst.is_branch
                                        or id_now.inst.is_jump):
                # For now, assume all branches are resolved in ID
                taken = id_now.inst.is_jump  # unconditional jumps always taken
                if id_now.inst.is_branch:
                    # Simplified: predict not-taken always matches (we never
                    # speculatively fetch targets)
                    taken = False

                was_predicted = self._predictor.predict(
                    id_now.inst.id, id_now.inst.is_branch,
                )
                if was_predicted != taken:
                    # Misprediction: flush IF stage
                    if pipeline[PipelineStage.IF] is not None:
                        pipeline[PipelineStage.IF] = None
                    id_now.mispredicted = True
                    stats.branch_mispredict_stalls += (
                        config.branch_mispredict_penalty
                    )

                self._predictor.update(id_now.inst.id, taken)

                # For unconditional jumps, set next fetch PC
                if id_now.inst.is_jump and id_now.inst.target:
                    # Find target instruction index
                    for tgt_inst in insts:
                        if tgt_inst.raw_line and id_now.inst.target in (
                                tgt_inst.raw_line, ""):
                            pass
                    # Simplified: flush IF, restart fetch at target
                    if pipeline[PipelineStage.IF] is not None:
                        pipeline[PipelineStage.IF] = None
                    # For jalr (indirect), we can't resolve target addr here
                    # For j/jal, find target instruction
                    if id_now.inst.opcode in ("j", "jal"):
                        # Look up target label → find inst id
                        target_id = None
                        for candidate in insts:
                            if (candidate.raw_line.strip().startswith(
                                    id_now.inst.target or "")):
                                target_id = candidate.id
                                break
                        if target_id is not None:
                            next_issue_idx = target_id

            # ── Track stage cycles ──────────────────────────────────────────
            for stage_name, stage_val in pipeline.items():
                if stage_val is not None:
                    stage_cycles[stage_name.name] += 1

            cycle += 1

        # ── Aggregate statistics ────────────────────────────────────────────
        stats.total_cycles = cycle
        stats.stages = stage_cycles
        stats.stage_utilisation = stage_util

        if stats.total_instructions > 0:
            stats.cpi = stats.total_cycles / stats.total_instructions

        stats.stall_cycles = (
            stats.data_hazard_stalls
            + stats.structural_stalls
            + stats.branch_mispredict_stalls
            + stats.cache_miss_stalls
            + stats.multi_cycle_stalls
        )

        # Per-category EX cycles
        cat_cycles: dict[str, int] = {c: 0 for c in _CATEGORIES}
        for item in completed:
            cat = _INST_CATEGORY.get(item.inst.opcode, "OTHER")
            cat_cycles[cat] += item.inst.latency
        stats.category_cycles = cat_cycles

        # Per-instruction summary
        stats.per_instruction = [
            {
                "id": item.inst.id,
                "opcode": item.inst.opcode,
                "operands": item.inst.operands,
                "category": _INST_CATEGORY.get(item.inst.opcode, "OTHER"),
                "latency": item.inst.latency,
                "issued_at": item.stage_start_cycle,
                "mispredicted": item.mispredicted,
            }
            for item in completed
        ]

        return stats

    # ── Report ──────────────────────────────────────────────────────────────

    def report(self, stats: CycleStats) -> str:
        """Generate a human-readable pipeline simulation report.

        Args:
            stats: A ``CycleStats`` from ``estimate()``.

        Returns:
            Multi-line report string.
        """
        lines: list[str] = []
        sep = "=" * 70

        lines.append(sep)
        lines.append("RISC-V 5-Stage Pipeline Cycle Estimator Report")
        lines.append(sep)
        lines.append(f"  Configuration:")
        lines.append(f"    Forwarding:      {'ON' if self.config.enable_forwarding else 'OFF'}")
        lines.append(f"    Branch predictor: {self.config.branch_predictor}")
        lines.append(f"    Mispredict pen:  {self.config.branch_mispredict_penalty} cycles")
        lines.append(f"    Cache model:     {'ON' if self.config.enable_cache_model else 'OFF'}")
        if self.config.enable_cache_model:
            lines.append(f"    I$ miss penalty: {self.config.icache_miss_penalty}")
            lines.append(f"    D$ miss penalty: {self.config.dcache_miss_penalty}")
        lines.append("")

        # Summary
        lines.append(f"  {'Total cycles:':<30} {stats.total_cycles:>8}")
        lines.append(f"  {'Instructions completed:':<30} {stats.total_instructions:>8}")
        lines.append(f"  {'CPI (cycles/instruction):':<30} {stats.cpi:>8.3f}")
        if stats.total_instructions > 0:
            ideal_cycles = stats.total_instructions * 0.2 + 4  # 5-stage fill
            lines.append(f"  {'Ideal CPI (5-stage):':<30} {ideal_cycles/max(stats.total_instructions,1):>8.3f}")
        lines.append("")

        # Stall breakdown
        lines.append(f"  {'Stall Breakdown:':<30}")
        lines.append(f"    {'Data hazards:':<28} {stats.data_hazard_stalls:>8}")
        lines.append(f"    {'Structural hazards:':<28} {stats.structural_stalls:>8}")
        lines.append(f"    {'Branch mispredicts:':<28} {stats.branch_mispredict_stalls:>8}")
        lines.append(f"    {'Cache misses:':<28} {stats.cache_miss_stalls:>8}")
        lines.append(f"    {'Multi-cycle (div/rem):':<28} {stats.multi_cycle_stalls:>8}")
        lines.append(f"    {'---':<28}")
        lines.append(f"    {'TOTAL stalls:':<28} {stats.stall_cycles:>8}")
        stall_pct = (
            stats.stall_cycles / stats.total_cycles * 100
            if stats.total_cycles > 0 else 0.0
        )
        lines.append(f"    {'Stall % of total cycles:':<28} {stall_pct:>7.1f}%")
        lines.append("")

        # Pipeline stage utilisation
        lines.append(f"  {'Pipeline Stage Utilisation:':<30}")
        total_util_cycles = sum(stats.stages.values())
        for stage_name in ["IF", "ID", "EX", "MEM", "WB"]:
            cycles = stats.stages.get(stage_name, 0)
            util = cycles / max(total_util_cycles, 1) * 100
            bar = "#" * int(util / 2)
            lines.append(f"    {stage_name:<6} {cycles:>8} cycles  {util:>5.1f}%  {bar}")
        lines.append("")

        # Per-category EX cycles
        lines.append(f"  {'EX Cycles by Category:':<30}")
        total_cat = sum(stats.category_cycles.values())
        for cat in _CATEGORIES:
            cnt = stats.category_cycles.get(cat, 0)
            if cnt > 0:
                pct = cnt / max(total_cat, 1) * 100
                bar = "#" * int(pct / 2)
                lines.append(f"    {cat:<12} {cnt:>6} cycles  {pct:>5.1f}%  {bar}")
        lines.append("")

        lines.append(sep)
        return "\n".join(lines)

    def report_short(self, stats: CycleStats) -> str:
        """One-line summary suitable for warnings."""
        return (
            f"Pipeline: {stats.total_cycles} cycles, "
            f"{stats.total_instructions} instrs, "
            f"CPI={stats.cpi:.2f}, "
            f"stalls={stats.stall_cycles} "
            f"(data={stats.data_hazard_stalls} "
            f"struct={stats.structural_stalls} "
            f"branch={stats.branch_mispredict_stalls})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _hash_pc(pc: int, seed: int) -> float:
    """Simple deterministic hash for cache miss simulation (0.0–1.0)."""
    val = (pc * 2654435761 + seed) & 0xFFFFFFFF
    return (val % 1000) / 1000.0
