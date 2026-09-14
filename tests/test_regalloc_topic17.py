"""Topic 17 acceptance tests: linear-scan convergence, reload alias fix, W9.

Coverage
--------
* Reload alias P0 counterexamples: 3-instruction impossible block (fail
  loudly), 6-instruction pressure block (must compute 3/6/7), and a
  7-instruction block that exercises runtime eviction.
* Spill + reload execution differential via a test-local mini interpreter.
* Assembly hygiene: no bare vregs, no ``SPILL_`` markers, no negative
  ``sp`` offsets, all branch targets defined.
* W9: 16-byte aligned frames, callee-saved save/restore across execution.
* Pipeline wiring: stage-1 default stays ``greedy``; ``linear`` is opt-in
  and its products go through ``AsmEmitter``.
"""

from __future__ import annotations

import re
import warnings

import pytest

from scratchv.backend import machine_types as mt
from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.frame_layout import FunctionFrameAllocator
from scratchv.backend.machine_types import (
    MachineInstr,
    MachineOp,
    MachineOperand,
)
from scratchv.backend.regalloc_linear import (
    LinearScanAllocator,
    LiveInterval,
    LsInstruction,
    RegAllocError,
    RegisterAliasError,
    SpillFallbackError,
    block_from_machine_instrs,
    machine_instrs_from_block,
)

_BARE_VREG = re.compile(r"(?<![A-Za-z0-9_.])v[0-9]+(?![A-Za-z0-9_])")
_NEG_OFFSET = re.compile(r"-\d+\(sp\)")
_INSERTED_MARKERS = ("reload ", "evict ", "store redefined ")
_JUMP_OPS = {"j", "jal", "call", "beq", "bne", "blt", "bge", "bnez"}


# ---------------------------------------------------------------------------
# Input blocks from the design document (3/6/7 instruction counterexamples)
# ---------------------------------------------------------------------------

def _three_block() -> list[LsInstruction]:
    """Two live inputs, one two-source add, a single-register pool."""
    return [
        LsInstruction(0, "li", ["v0", "1"], defines={"v0"}),
        LsInstruction(1, "li", ["v1", "2"], defines={"v1"}),
        LsInstruction(2, "add", ["v2", "v0", "v1"],
                      defines={"v2"}, uses={"v0", "v1"}),
    ]


def _six_block() -> list[LsInstruction]:
    """The T1 pressure block: v3=3, v4=6, v5=7 with a two-register pool."""
    return [
        LsInstruction(0, "li", ["v0", "1"], defines={"v0"}),
        LsInstruction(1, "li", ["v1", "2"], defines={"v1"}),
        LsInstruction(2, "li", ["v2", "3"], defines={"v2"}),
        LsInstruction(3, "add", ["v3", "v0", "v1"],
                      defines={"v3"}, uses={"v0", "v1"}),
        LsInstruction(4, "add", ["v4", "v2", "v3"],
                      defines={"v4"}, uses={"v2", "v3"}),
        LsInstruction(5, "add", ["v5", "v4", "v0"],
                      defines={"v5"}, uses={"v4", "v0"}),
    ]


def _seven_block() -> list[LsInstruction]:
    """T1 plus v6 = v5 + v1; triggers runtime eviction of v2."""
    return _six_block() + [
        LsInstruction(6, "add", ["v6", "v5", "v1"],
                      defines={"v6"}, uses={"v5", "v1"}),
    ]


def _redefine_block() -> list[LsInstruction]:
    """v0 is redefined after being spilled; v4 = 9 + 3 = 12."""
    return [
        LsInstruction(0, "li", ["v0", "1"], defines={"v0"}),
        LsInstruction(1, "li", ["v1", "2"], defines={"v1"}),
        LsInstruction(2, "li", ["v2", "3"], defines={"v2"}),
        LsInstruction(3, "add", ["v3", "v0", "v1"],
                      defines={"v3"}, uses={"v0", "v1"}),
        LsInstruction(4, "li", ["v0", "9"], defines={"v0"}),
        LsInstruction(5, "add", ["v4", "v0", "v2"],
                      defines={"v4"}, uses={"v0", "v2"}),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lines(allocated: list[LsInstruction]) -> list[str]:
    return [inst.to_asm() for inst in allocated]


def _execute_allocated(orig_block, allocated):
    """Mini-interpreter for allocated blocks (li/add/lw/sw).

    Returns ``(observed, regs, mem)`` where ``observed[vreg]`` is the value
    produced by the vreg's last (corresponding) original definition.
    """
    regs: dict[str, int] = {}
    mem: dict[str, int] = {}
    observed: dict[str, int] = {}
    ptr = 0

    for inst in allocated:
        ops = inst.operands
        if inst.opcode == "li":
            regs[ops[0]] = int(ops[1])
        elif inst.opcode == "add":
            regs[ops[0]] = regs[ops[1]] + regs[ops[2]]
        elif inst.opcode == "lw":
            regs[ops[0]] = mem.get(ops[1], 0)
        elif inst.opcode == "sw":
            mem[ops[1]] = regs[ops[0]]
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected opcode {inst.opcode!r}")

        inserted = any(
            inst.comment.startswith(marker) for marker in _INSERTED_MARKERS)
        if not inserted:
            orig = orig_block[ptr]
            ptr += 1
            if len(orig.defines) == 1 and ops:
                observed[next(iter(orig.defines))] = regs[ops[0]]

    assert ptr == len(orig_block), "allocated stream lost original instructions"
    return observed, regs, mem


def _assert_hygiene(text: str) -> None:
    body = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    assert "SPILL_" not in body
    assert not _BARE_VREG.search(body), body
    assert not _NEG_OFFSET.search(body), body


def _branch_targets(text: str) -> set[str]:
    targets: set[str] = set()
    for line in text.splitlines():
        parts = line.split("#", 1)[0].replace(",", " ").split()
        if parts and parts[0] in _JUMP_OPS and len(parts) > 1:
            targets.add(parts[-1])
    return targets


def _label_defs(text: str) -> set[str]:
    return {
        line.strip()[:-1]
        for line in text.splitlines()
        if line.strip().endswith(":")
    }


# ---------------------------------------------------------------------------
# 1. Alias counterexamples
# ---------------------------------------------------------------------------

class TestReloadAliasCounterexamples:
    def test_six_instruction_pressure_block_semantics(self):
        block = _six_block()
        alloc = LinearScanAllocator(phys_regs=["t0", "t1"])
        allocated = alloc.allocate_block(block)
        observed, _, _ = _execute_allocated(block, allocated)

        assert observed["v3"] == 3
        assert observed["v4"] == 6
        assert observed["v5"] == 7

        text = "\n".join(_lines(allocated))
        _assert_hygiene(text)
        assert "  sw " in text and "  lw " in text

    def test_three_instruction_block_fails_loudly(self):
        """A one-register pool cannot host a two-source add: raise, not alias."""
        alloc = LinearScanAllocator(phys_regs=["t0"])
        with pytest.raises(SpillFallbackError):
            alloc.emit(_three_block())
        assert issubclass(SpillFallbackError, RegAllocError)

    def test_empty_pool_raises_regalloc_error(self):
        alloc = LinearScanAllocator(phys_regs=[])
        with pytest.raises(RegAllocError):
            alloc.emit(_three_block())

    def test_alias_detection_raises_register_alias_error(self):
        alloc = LinearScanAllocator(phys_regs=["t0"])
        alloc._vreg_interval = {
            "x": LiveInterval("x", 0, 5, {4}),
            "y": LiveInterval("y", 0, 5, {4}),
        }
        alloc.alloc_map = {"x": "t0", "y": "t0"}
        inst = LsInstruction(2, "add", ["v9", "x", "y"],
                             defines={"v9"}, uses={"x", "y"})
        with pytest.raises(RegisterAliasError):
            alloc._occupied_at(2, inst)

    def test_seven_instruction_block_semantics(self):
        block = _seven_block()
        alloc = LinearScanAllocator(phys_regs=["t0", "t1"])
        allocated = alloc.allocate_block(block)
        observed, _, _ = _execute_allocated(block, allocated)

        assert observed["v3"] == 3
        assert observed["v4"] == 6
        assert observed["v5"] == 7
        assert observed["v6"] == 9

        text = "\n".join(_lines(allocated))
        _assert_hygiene(text)
        # Every reload's slot was written earlier in the stream.
        assert text.count("reload ") == 5
        assert re.search(r"store redefined v[0-9]", text)

    def test_spilled_redefinition_writeback_before_reload(self):
        block = _redefine_block()
        alloc = LinearScanAllocator(phys_regs=["t0", "t1"])
        allocated = alloc.allocate_block(block)
        observed, _, _ = _execute_allocated(block, allocated)
        assert observed["v4"] == 12

        lines = _lines(allocated)
        redefine = next(i for i, line in enumerate(lines) if ", 9" in line)
        writeback = next(
            i for i in range(redefine + 1, len(lines))
            if "store redefined v0" in lines[i]
        )
        reload_after = next(
            i for i in range(writeback + 1, len(lines))
            if "reload v0" in lines[i]
        )
        assert redefine < writeback < reload_after

    def test_reload_dedup_same_vreg_per_position(self):
        block = _six_block()
        alloc = LinearScanAllocator(phys_regs=["t0", "t1"])
        alloc.allocate(alloc.compute_live_intervals(block))
        # Duplicate an existing registration: dedup must keep one lw.
        alloc._reloads[5].append(alloc._reloads[5][0])
        lines = _lines(alloc._build_allocated_block(block))
        assert sum("reload v0" in line for line in lines) == 2  # pos 3 and 5

    def test_pure_definition_does_not_occupy_register(self):
        block = _six_block()
        alloc = LinearScanAllocator(phys_regs=["t0", "t1"])
        alloc.allocate(alloc.compute_live_intervals(block))
        owners = alloc._occupied_at(3, block[3])
        assert "v3" not in owners.values()  # pure def at its start position
        assert "v0" not in owners.values()  # already spilled at allocation
        assert "v1" in owners.values()

    def test_live_in_eviction_stores_incoming_value(self):
        block = [
            LsInstruction(0, "li", ["v0", "1"], defines={"v0"}),
            LsInstruction(1, "add", ["v3", "a", "b"],
                          defines={"v3"}, uses={"a", "b"}),
            LsInstruction(2, "add", ["v4", "v0", "a"],
                          defines={"v4"}, uses={"v0", "a"}),
        ]
        alloc = LinearScanAllocator(phys_regs=["t0", "t1"])
        lines = _lines(alloc.allocate_block(block))
        text = "\n".join(lines)
        _assert_hygiene(text)

        # The live-in victim is captured at block entry, before its reloads.
        store = next(i for i, l in enumerate(lines) if "evict a" in l)
        assert "  sw " in lines[store]
        reload_after = next(
            i for i in range(store + 1, len(lines)) if "reload a" in lines[i])
        assert store < reload_after

    def test_runtime_eviction_store_precedes_reload_and_skips_protected(self):
        """Directly drive the runtime (reload-time) eviction path."""
        alloc = LinearScanAllocator(phys_regs=["t0", "t1"])
        alloc._vreg_interval = {
            "x": LiveInterval("x", 0, 10, {9}),
            "y": LiveInterval("y", 0, 10, {9}),
            "z": LiveInterval("z", 0, 10, {5}),
        }
        alloc.alloc_map = {"x": "t0", "y": "t1", "z": "SPILL_z"}
        alloc._spilled = {"z"}
        alloc._spill_slots = {"z": 0}
        alloc.stack_slot = 4
        inst = LsInstruction(5, "add", ["v9", "z", "y"],
                             defines={"v9"}, uses={"z", "y"})

        reg, stores = alloc._pick_reload_reg(
            inst, "z", 0, dict(alloc.alloc_map),
            owners={"t0": "x", "t1": "y"}, loaded={},
        )
        # `y` is a protected operand, so `x` (t0) must be the victim and its
        # store must be emitted inline, before the caller's lw.
        assert reg == "t0"
        assert stores == ["  sw t0, 4(sp)  # evict x for reload"]
        assert "x" in alloc._spilled
        assert (("x", 4) in alloc._reloads.get(9, []))


# ---------------------------------------------------------------------------
# 2. Assembly hygiene / labels / machine_types consistency
# ---------------------------------------------------------------------------

class TestAssemblyHygiene:
    def test_no_vreg_spill_or_negative_offsets(self):
        for block in (_six_block(), _seven_block(), _redefine_block()):
            text = LinearScanAllocator(phys_regs=["t0", "t1"]).emit(block)
            _assert_hygiene(text)
            assert not re.search(r"#\s*\d+\(", text)

    def test_machine_instrs_from_block_has_no_vregs(self):
        block = _seven_block()
        alloc = LinearScanAllocator(phys_regs=["t0", "t1"])
        mi = machine_instrs_from_block(alloc.allocate_block(block))
        for instr in mi:
            for op in (instr.dst, instr.src1, instr.src2):
                if op is not None:
                    assert op.kind != "vreg", instr

    def test_mem_operand_round_trip(self):
        inst = LsInstruction(0, "lw", ["t0", "8(sp)"], comment="reload v")
        mi = machine_instrs_from_block([inst])[0]
        assert mi.src1.kind == "mem"
        assert mi.src1.value == "8(sp)"
        assert "lw t0, 8(sp)" in AsmEmitter([mi]).emit()

        back = block_from_machine_instrs([mi])[0]
        assert back.operands == ["t0", "8(sp)"]

    def test_labels_and_branch_targets_survive_linear_pipeline(self):
        prog = [
            MachineInstr(MachineOp.LABEL, comment="main"),
            MachineInstr(MachineOp.LI, MachineOperand.vreg("v0"),
                         MachineOperand.immediate(1)),
            MachineInstr(MachineOp.BNEZ, MachineOperand.vreg("v0"),
                         comment=".Lend"),
            MachineInstr(MachineOp.LI, MachineOperand.vreg("v1"),
                         MachineOperand.immediate(2)),
            MachineInstr(MachineOp.J, comment=".Done"),
            MachineInstr(MachineOp.LABEL, comment=".Lend"),
            MachineInstr(MachineOp.LI, MachineOperand.vreg("v1"),
                         MachineOperand.immediate(3)),
            MachineInstr(MachineOp.LABEL, comment=".Done"),
            MachineInstr(MachineOp.MV, MachineOperand.reg("a0"),
                         MachineOperand.vreg("v1")),
            MachineInstr(MachineOp.JALR, MachineOperand.reg("zero"),
                         MachineOperand.reg("ra"), comment="ret"),
        ]
        allocated = FunctionFrameAllocator().allocate_program(prog)
        text = AsmEmitter(allocated).emit()
        _assert_hygiene(text)
        assert "main:" in text
        assert ".Lend:" in text and ".Done:" in text
        assert re.search(r"bnez \w+, \.Lend", text), text
        assert re.search(r"j \.Done", text), text

        targets = _branch_targets(text)
        assert targets == {".Lend", ".Done"}
        assert targets <= _label_defs(text)

        from scratchv.backend.riscv_encoder import assemble_to_binary
        assemble_to_binary(text)  # must not raise

    def test_machine_types_allocatable_sets_consistent(self):
        assert len(mt.ALL_REGS) == 27
        assert mt.ALL_REGS == mt.ARG_REGS + mt.TEMP_REGS + mt.CALLEE_SAVED
        assert mt.CALLER_SAVED == mt.ARG_REGS + mt.TEMP_REGS
        assert mt.GREEDY_REGS == mt.TEMP_REGS + mt.CALLEE_SAVED
        assert len(mt.GREEDY_REGS) == 19
        assert set(mt.ALL_REGS).isdisjoint(
            {"zero", "ra", "sp", "gp", "tp", "fp"})
        assert mt.REG_NUMS["s0"] == 8 and mt.REG_NUMS["sp"] == 2

        from scratchv.backend import regalloc_linear as rl
        assert rl._DEFAULT_PHYS_REGS == mt.ALL_REGS
        assert rl._INT_REGS == mt.ALL_REGS


# ---------------------------------------------------------------------------
# 3. W9 frame allocation
# ---------------------------------------------------------------------------

def _callee_saved_factory(*, stack_base, pre_spilled, slot_hints):
    return LinearScanAllocator(
        phys_regs=["t0", "t1", "s0"],
        stack_base=stack_base,
        pre_spilled=pre_spilled,
        slot_hints=slot_hints,
        strict=True,
    )


def _foo_program() -> list[MachineInstr]:
    def li(dst, imm):
        return MachineInstr(MachineOp.LI, dst, MachineOperand.immediate(imm))

    def add(dst, a, b):
        return MachineInstr(MachineOp.ADD, dst, a, b)

    return [
        MachineInstr(MachineOp.LABEL, comment="foo"),
        li(MachineOperand.vreg("v0"), 5),
        li(MachineOperand.vreg("v1"), 7),
        li(MachineOperand.vreg("v2"), 9),
        add(MachineOperand.vreg("v3"),
            MachineOperand.vreg("v0"), MachineOperand.vreg("v1")),
        add(MachineOperand.vreg("v4"),
            MachineOperand.vreg("v3"), MachineOperand.vreg("v2")),
        MachineInstr(MachineOp.MV, MachineOperand.reg("a0"),
                     MachineOperand.vreg("v4")),
        MachineInstr(MachineOp.JALR, MachineOperand.reg("zero"),
                     MachineOperand.reg("ra"), comment="ret"),
    ]


class TestFunctionFrameAllocator:
    def test_frame_alignment_and_callee_saved_code(self):
        fa = FunctionFrameAllocator(
            alloc_factory=_callee_saved_factory)
        allocated = fa.allocate_program(_foo_program())
        info = fa.last_frame_info["foo"]

        assert info.frame_size % 16 == 0
        assert info.saved_offsets == {"s0": info.frame_size - 4}
        assert info.ra_offset is None  # foo contains no call

        text = AsmEmitter(allocated).emit()
        assert re.search(r"addi sp, sp, -%d" % info.frame_size, text)
        assert re.search(r"sw s0, %d\(sp\)" % info.saved_offsets["s0"], text)
        assert re.search(r"lw s0, %d\(sp\)" % info.saved_offsets["s0"], text)
        # epilogue loads s0 before the ret
        assert text.index("lw s0,") < text.index("jalr zero, ra")
        _assert_hygiene(text)

    def test_callee_saved_preserved_across_execution(self):
        from scratchv.backend.riscv_encoder import assemble_to_binary
        from scratchv.simulator.rv32_emulator import RV32Emulator

        fa = FunctionFrameAllocator(
            alloc_factory=_callee_saved_factory)
        text = AsmEmitter(fa.allocate_program(_foo_program())).emit()

        emu = RV32Emulator()
        emu.load_code(bytes(assemble_to_binary(text)))
        emu.regs[8] = 0x5A5A  # s0 sentinel
        emu.run(max_instr=1000)

        assert emu.regs[10] == 21       # a0 = (5 + 7) + 9
        assert emu.regs[8] == 0x5A5A    # s0 restored by the epilogue
        assert emu.regs[2] == RV32Emulator.STACK_TOP  # sp balanced


# ---------------------------------------------------------------------------
# 4. Pipeline wiring / defaults
# ---------------------------------------------------------------------------

class TestPipelineWiring:
    def test_stage_one_default_is_greedy(self):
        from scratchv.compiler import CompilerConfig
        from scratchv.main import args_to_config, build_arg_parser

        assert CompilerConfig().reg_alloc == "greedy"
        args = build_arg_parser().parse_args(["input.dsl"])
        assert args_to_config(args).reg_alloc == "greedy"

    def test_linear_v1_5_alias_normalizes_to_linear(self):
        from scratchv.main import args_to_config, build_arg_parser

        args = build_arg_parser().parse_args(
            ["input.dsl", "--reg-alloc", "linear-v1.5"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = args_to_config(args)
        assert config.reg_alloc == "linear"
        assert any(
            issubclass(w.category, DeprecationWarning) for w in caught)

    def test_unknown_mode_raises(self):
        from scratchv.compiler import CompilerConfig, CompilerDriver

        class _EmptyProgram:
            functions: list = []

        driver = CompilerDriver(CompilerConfig(reg_alloc="bogus"))
        with pytest.raises(ValueError):
            driver._generate_riscv_linear(_EmptyProgram())

    def test_linear_opt_in_end_to_end_forced_spill(self, tmp_path):
        from scratchv.backend.riscv_encoder import assemble_to_binary
        from scratchv.compiler import CompilerConfig, CompilerDriver
        from scratchv.simulator.rv32_emulator import RV32Emulator

        src = "for j = 0, 5\n    k = add(j, j)\nendfor\nreturn k\n"

        def compile_and_run(mode: str):
            result = CompilerDriver(
                CompilerConfig(reg_alloc=mode)).compile(
                    "topic17.dsl",
                    output_path=str(tmp_path / f"{mode}.s"),
                    dsl_source=src,
                )
            assert result.success, result.errors
            emu = RV32Emulator()
            emu.load_code(bytes(assemble_to_binary(result.output_text)))
            emu.run(max_instr=10000)
            return result.output_text, emu.regs[10]

        linear_text, linear_result = compile_and_run("linear")
        greedy_text, greedy_result = compile_and_run("greedy")

        assert linear_result == greedy_result == 8
        _assert_hygiene(linear_text)

        # The linear path forces the cross-block loop variables to memory.
        body = "\n".join(
            line.split("#", 1)[0] for line in linear_text.splitlines())
        assert re.search(r"lw \w+, \d+\(sp\)", body)
        assert re.search(r"sw \w+, \d+\(sp\)", body)

        # Every branch/jump target is defined in the emitted text.
        assert _branch_targets(linear_text) <= _label_defs(linear_text)
        assert _branch_targets(greedy_text) <= _label_defs(greedy_text)
