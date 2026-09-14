"""Tests for IR loop unrolling (Topic 10).

Structure / SSA / idempotency assertions follow the Topic 10 design
document; dynamic instruction counts are measured end-to-end through
``InstructionSelector -> RegisterAllocator(greedy) -> AsmEmitter ->
assemble_to_binary -> RV32Emulator``.

Note on measured numbers: the numbers asserted here are the values actually
produced by the current backend.  The design document's theoretical counts
(e.g. 21->12, 31->27, 36->30) exclude the branch-immediate expansion the
encoder inserts for ``bge iv, end`` and assume constant operands are not
inlined (the current selector inlines them).  The actual measurements show
strictly larger savings, so the design-document numbers are treated as
directional acceptance criteria here.
"""

from __future__ import annotations

from pathlib import Path

from scratchv.analysis.ir_verifier import ErrorLevel, IRVerifier
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, OpCode, Program
from scratchv.optimizer.loop_unroll import LoopUnroll, UnrollPlan


# ═══════════════════════════════════════════════════════════════════════════
# Program builders
# ═══════════════════════════════════════════════════════════════════════════

def _make_const(builder: IRBuilder, name: str, value: int):
    value_obj = builder.make_value(
        name=name, dtype=DataType.INT32, is_constant=False)
    builder._emit(OpCode.LOAD_CONST, value_obj, value=value)
    return value_obj


def build_case_program(end: int, use_acc: bool = True):
    """Design-document case-1 IR: ``one`` const, body uses the iv.

    ``use_acc=True`` is the exact design-document program (``v3 = acc + v2``
    with ``acc`` a function parameter).  ``use_acc=False`` avoids the free
    parameter to keep register pressure low for emulator runs.
    """
    builder = IRBuilder()
    params = []
    acc = None
    if use_acc:
        acc = builder.make_value(
            name="acc", dtype=DataType.INT32, is_constant=False)
        params = [acc]
    builder.new_function("main", params=params)
    builder.new_block("entry")
    one = _make_const(builder, "one", 1)
    iv = builder.for_loop(0, end)
    v2 = builder.add(iv, one)
    if use_acc:
        v3 = builder.add(acc, v2)
    else:
        v3 = builder.add(v2, v2)
    builder.endfor()
    builder.ret(v3)
    return builder.program


def build_low_pressure_program(end: int, start: int = 0):
    """Single-instruction loop body: ``v2 = iv + one``; return ``v2``."""
    builder = IRBuilder()
    builder.new_function("main")
    builder.new_block("entry")
    one = _make_const(builder, "one", 1)
    iv = builder.for_loop(start, end)
    v2 = builder.add(iv, one)
    builder.endfor()
    builder.ret(v2)
    return builder.program


def build_carried_param_simple_program(end: int):
    """Carried accumulator without iv use: ``acc = acc + one``.

    ``acc`` is a function parameter, so it has no definition outside the
    loop and the body self-reference (``dest == operand``) is its only
    definition; the low vreg count keeps partial unrolls runnable through
    today's backend.
    """
    builder = IRBuilder()
    acc = builder.make_value(
        name="acc", dtype=DataType.INT32, is_constant=False)
    builder.new_function("main", params=[acc])
    builder.new_block("entry")
    one = _make_const(builder, "one", 1)
    builder.for_loop(0, end)
    builder._emit(OpCode.ADD, acc, [acc, one])
    builder.endfor()
    builder.ret(acc)
    return builder.program


def build_carried_param_program(end: int, start: int = 0):
    """Probe-style carried accumulator: ``acc = acc + (iv + one)``.

    ``acc`` is a function parameter, so it is never defined before the loop
    and the body self-reference is the only definition of ``acc``.
    """
    builder = IRBuilder()
    acc = builder.make_value(
        name="acc", dtype=DataType.INT32, is_constant=False)
    builder.new_function("main", params=[acc])
    builder.new_block("entry")
    one = _make_const(builder, "one", 1)
    iv = builder.for_loop(start, end)
    t = builder.add(iv, one)
    builder._emit(OpCode.ADD, acc, [acc, t])
    builder.endfor()
    builder.ret(acc)
    return builder.program


def build_forward_ref_program(end: int):
    """Body reads ``u`` before defining it, then defines ``u`` (forward)."""
    builder = IRBuilder()
    builder.new_function("main")
    builder.new_block("entry")
    u = builder.make_value(name="u", dtype=DataType.INT32, is_constant=False)
    one = _make_const(builder, "one", 1)
    iv = builder.for_loop(0, end)
    t = builder.add(u, one)
    builder._emit(OpCode.ADD, u, [iv, one])
    builder.endfor()
    builder.ret(t)
    return builder.program


# ═══════════════════════════════════════════════════════════════════════════
# Helpers (documented as TestCaseHelpers in the design document)
# ═══════════════════════════════════════════════════════════════════════════

def count_ir(program: Program) -> int:
    """Total number of IR instructions across all functions/blocks."""
    return sum(
        len(block.instructions)
        for func in program.functions
        for block in func.blocks
    )


def count_for_markers(program: Program) -> int:
    return sum(
        1
        for func in program.functions
        for block in func.blocks
        for ins in block.instructions
        if ins.opcode in (OpCode.FOR, OpCode.ENDFOR)
    )


def emit_asm(program: Program) -> str:
    from scratchv.backend.asm_emit import AsmEmitter
    from scratchv.backend.instruction_select import InstructionSelector
    from scratchv.backend.register_alloc import RegisterAllocator

    machine = InstructionSelector(program).run()
    allocated = RegisterAllocator(machine, mode="greedy").run()
    return AsmEmitter(allocated).emit()


def run_asm_text(asm: str) -> tuple[int, int]:
    """Assemble and run *asm*; return ``(a0, dynamic_instruction_count)``."""
    from scratchv.backend.riscv_encoder import assemble_to_binary
    from scratchv.simulator.rv32_emulator import REG_ID, RV32Emulator

    binary = assemble_to_binary(asm)
    emulator = RV32Emulator()
    emulator.load_code(bytes(binary))
    dynamic = emulator.run()
    return emulator.regs[REG_ID["a0"]], dynamic


def run_rv32(program: Program) -> tuple[int, int]:
    """Run the compiled program; return ``(a0, dynamic_instruction_count)``."""
    return run_asm_text(emit_asm(program))


def verify_errors(program: Program):
    return [
        issue for issue in IRVerifier(program).verify()
        if issue.level == ErrorLevel.ERROR
    ]


def fingerprint(program: Program):
    return [
        (
            id(ins),
            ins.opcode,
            ins.dest.name if ins.dest is not None else None,
            tuple(op.name for op in ins.operands),
            tuple(sorted(ins.attrs.items(), key=lambda kv: kv[0])),
            ins.target,
        )
        for func in program.functions
        for block in func.blocks
        for ins in block.instructions
    ]


def block_instructions(program: Program):
    return program.functions[0].blocks[0].instructions


def find_for_indices(program: Program):
    return [
        (i, ins) for i, ins in enumerate(block_instructions(program))
        if ins.opcode == OpCode.FOR
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Case 1: full unrolling
# ═══════════════════════════════════════════════════════════════════════════

class TestLoopUnrollFull:
    def test_full_unroll_iv_used(self):
        program = build_case_program(4, use_acc=True)
        iv_name = find_for_indices(program)[0][1].dest.name
        assert count_ir(program) == 6

        unroll = LoopUnroll(program)
        assert unroll.run() == 1

        assert count_for_markers(program) == 0
        instructions = block_instructions(program)
        # 12 instructions replace FOR..ENDFOR (4 copies x (bind + 2 body)).
        assert count_ir(program) == 14
        assert len(instructions) == 14

        bindings = [
            ins for ins in instructions if ins.opcode == OpCode.LOAD_CONST
        ][1:]
        assert [ins.attrs["value"] for ins in bindings] == [0, 1, 2, 3]
        for ins in bindings:
            assert ins.dest.dtype == DataType.INT32
            assert ins.dest.is_constant is False
        # last copy reuses the original induction-variable name
        assert bindings[-1].dest.name == iv_name
        # body values reuse their original names on the last copy
        dests = [ins.dest.name for ins in instructions if ins.dest is not None]
        assert dests[-3] == iv_name
        assert instructions[-1].opcode == OpCode.RETURN
        assert emit_asm(program)
        assert verify_errors(program) == []

        stats = unroll.stats
        assert stats["full_unrolls"] == 1
        assert stats["partial_unrolls"] == 0
        assert stats["loops_seen"] == 1
        assert stats["instructions_added"] > 0

        # idempotent: second run changes nothing
        before = fingerprint(program)
        assert unroll.run() == 0
        assert fingerprint(program) == before

    def test_full_unroll_dynamic_equivalence(self):
        baseline = build_case_program(4, use_acc=True)
        a0_before, dynamic_before = run_rv32(baseline)

        optimized = build_case_program(4, use_acc=True)
        LoopUnroll(optimized).run()
        a0_after, dynamic_after = run_rv32(optimized)

        assert a0_before == a0_after == 4
        assert dynamic_before == 30
        assert dynamic_after == 15
        assert dynamic_after < dynamic_before

    def test_full_unroll_iv_used_after_loop(self):
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        one = _make_const(builder, "one", 1)
        iv = builder.for_loop(0, 3)
        v2 = builder.add(iv, one)
        builder.endfor()
        v4 = builder.add(iv, v2)
        builder.ret(v4)

        unroll = LoopUnroll(builder.program)
        assert unroll.run() == 1

        instructions = block_instructions(builder.program)
        finals = [
            ins for ins in instructions
            if ins.opcode == OpCode.LOAD_CONST
            and ins.dest.name.startswith(iv.name + "_final")
        ]
        assert len(finals) == 1
        assert finals[0].attrs["value"] == 3
        assert instructions[-2].operands[0].name == finals[0].dest.name
        assert verify_errors(builder.program) == []


# ═══════════════════════════════════════════════════════════════════════════
# Case 2: partial exact unrolling
# ═══════════════════════════════════════════════════════════════════════════

class TestLoopUnrollPartial:
    def test_partial_exact_divisor(self):
        program = build_case_program(6, use_acc=True)
        iv_name = find_for_indices(program)[0][1].dest.name

        unroll = LoopUnroll(program, full_threshold=2)
        assert unroll.run() == 1

        for_indices = find_for_indices(program)
        assert len(for_indices) == 1
        for_ins = for_indices[0][1]
        assert for_ins.attrs["start"] == 0
        assert for_ins.attrs["end"] == 2
        assert for_ins.attrs["step"] == 1
        assert for_ins.attrs["unrolled"] == 3

        instructions = block_instructions(program)
        # setup 2 consts + FOR + 9 group instrs + ENDFOR
        assert count_ir(program) == 15
        assert instructions[1].opcode == OpCode.LOAD_CONST
        assert instructions[1].attrs["value"] == 3
        assert instructions[2].opcode == OpCode.LOAD_CONST
        assert instructions[2].attrs["value"] == 1

        for_idx = for_indices[0][0]
        first_bind = instructions[for_idx + 1]
        assert first_bind.opcode == OpCode.MUL
        assert first_bind.dest.name.startswith(iv_name + "__")
        assert first_bind.operands[0].name == iv_name
        assert first_bind.operands[1].name == instructions[1].dest.name

        chain = [
            ins for ins in instructions[for_idx + 2:]
            if ins.opcode == OpCode.ADD
            and ins.dest.name.startswith(iv_name + "__")
            and ins.operands[0].name.startswith(iv_name + "__")
        ]
        assert len(chain) == 2
        assert chain[0].operands[0].name == first_bind.dest.name
        assert chain[1].operands[0].name == chain[0].dest.name

        assert verify_errors(program) == []
        assert unroll.stats["partial_unrolls"] == 1
        assert unroll.stats["partial_epilogues"] == 0

        before = fingerprint(program)
        assert unroll.run() == 0
        assert fingerprint(program) == before

    def test_partial_exact_dynamic_equivalence(self):
        baseline = build_low_pressure_program(6)
        a0_before, dynamic_before = run_rv32(baseline)

        optimized = build_low_pressure_program(6)
        unroll = LoopUnroll(optimized, full_threshold=2)
        assert unroll.run() == 1
        a0_after, dynamic_after = run_rv32(optimized)

        assert a0_before == a0_after == 6
        assert dynamic_before == 36
        assert dynamic_after == 28
        assert dynamic_after < dynamic_before

    def test_partial_exact_nonzero_start(self):
        baseline = build_low_pressure_program(14, start=5)
        a0_before, dynamic_before = run_rv32(baseline)

        optimized = build_low_pressure_program(14, start=5)
        iv_name = find_for_indices(optimized)[0][1].dest.name
        unroll = LoopUnroll(optimized, full_threshold=2)
        assert unroll.run() == 1
        a0_after, dynamic_after = run_rv32(optimized)

        assert a0_before == a0_after == 14
        assert dynamic_before == 51
        assert dynamic_after == 42
        assert dynamic_after < dynamic_before

        instructions = block_instructions(optimized)
        for_idx = find_for_indices(optimized)[0][0]
        mul = instructions[for_idx + 1]
        add_start = instructions[for_idx + 2]
        assert mul.opcode == OpCode.MUL
        assert mul.operands[0].name == iv_name
        assert add_start.opcode == OpCode.ADD
        assert add_start.operands[0].name == mul.dest.name
        start_const = next(
            ins for ins in instructions[:for_idx]
            if ins.opcode == OpCode.LOAD_CONST
            and ins.attrs["value"] == 5)
        assert add_start.operands[1].name == start_const.dest.name
        body = instructions[for_idx + 3]
        assert body.opcode == OpCode.ADD
        assert body.operands[0].name == add_start.dest.name


# ═══════════════════════════════════════════════════════════════════════════
# Case 3: epilogue (remainder loop) partial unrolling
# ═══════════════════════════════════════════════════════════════════════════

class TestLoopUnrollEpilogue:
    def test_partial_epilogue(self):
        program = build_case_program(7, use_acc=True)
        return_ins = block_instructions(program)[-1]
        canonical_v3 = return_ins.operands[0].name

        unroll = LoopUnroll(program, full_threshold=2, epilogue=True)
        assert unroll.run() == 1
        assert unroll.stats["partial_epilogues"] == 1

        for_indices = find_for_indices(program)
        assert len(for_indices) == 2
        main_for = for_indices[0][1]
        epilogue_for = for_indices[1][1]
        assert main_for.attrs["end"] == 1
        assert main_for.attrs["unrolled"] == 6
        assert epilogue_for.attrs["start"] == 6
        assert epilogue_for.attrs["end"] == 7
        assert epilogue_for.attrs["step"] == 1
        # the remainder loop carries the idempotency marker too (F5)
        assert epilogue_for.attrs["unrolled"] == 1

        # setup 2 + main (FOR + 18 + ENDFOR) + epilogue (FOR + 2 + ENDFOR)
        assert count_ir(program) == 28
        assert return_ins.operands[0].name == canonical_v3 + "__ep"
        assert verify_errors(program) == []
        names = [ins.dest.name for ins in block_instructions(program)
                 if ins.dest is not None]
        assert len(names) == len(set(names))

    def test_epilogue_not_emitted_when_exact(self):
        program = build_case_program(6, use_acc=True)
        unroll = LoopUnroll(program, full_threshold=2, epilogue=True)
        assert unroll.run() == 1
        assert unroll.stats["partial_epilogues"] == 0
        assert len(find_for_indices(program)) == 1

    def test_partial_epilogue_dynamic_equivalence(self):
        baseline = build_low_pressure_program(7)
        a0_before, dynamic_before = run_rv32(baseline)

        optimized = build_low_pressure_program(7)
        unroll = LoopUnroll(optimized, full_threshold=2, epilogue=True)
        assert unroll.run() == 1
        a0_after, dynamic_after = run_rv32(optimized)

        assert a0_before == a0_after == 7
        assert dynamic_before == 41
        assert dynamic_after == 32
        assert dynamic_after < dynamic_before


# ═══════════════════════════════════════════════════════════════════════════
# Case 3b: true loop-carried values (dest == operand) — F1/F3
# ═══════════════════════════════════════════════════════════════════════════

class TestLoopUnrollCarriedValues:
    """Simulation coverage for genuine loop-carried values.

    ``build_carried_param_simple_program`` keeps the vreg count below the point
    where the known encoder temp-register fallback corrupts partial
    unrolls (an existing backend defect, not fixed in this topic), so the
    partial/exact and epilogue numbers below are checked end-to-end.
    """

    def test_full_unroll_carried_value_simulation(self):
        baseline = build_carried_param_program(4)
        a0_before, dynamic_before = run_rv32(baseline)

        optimized = build_carried_param_program(4)
        unroll = LoopUnroll(optimized)
        assert unroll.run() == 1
        assert unroll.stats["full_unrolls"] == 1
        a0_after, dynamic_after = run_rv32(optimized)

        assert a0_before == a0_after == 10  # sum(1..4)
        assert dynamic_after < dynamic_before
        assert verify_errors(optimized) == []

    def test_partial_exact_carried_value_simulation(self):
        baseline = build_carried_param_simple_program(6)
        a0_before, dynamic_before = run_rv32(baseline)

        optimized = build_carried_param_simple_program(6)
        unroll = LoopUnroll(
            optimized, full_threshold=2, max_factor=2)
        assert unroll.run() == 1
        assert unroll.stats["partial_unrolls"] == 1
        assert unroll.stats["partial_epilogues"] == 0
        a0_after, dynamic_after = run_rv32(optimized)

        assert a0_before == a0_after == 6  # one increment per iteration
        assert dynamic_after < dynamic_before
        assert verify_errors(optimized) == []

    def test_epilogue_r1_carried_value_simulation(self):
        baseline = build_carried_param_simple_program(5)
        a0_before, dynamic_before = run_rv32(baseline)

        optimized = build_carried_param_simple_program(5)
        unroll = LoopUnroll(
            optimized, full_threshold=2, max_factor=2, epilogue=True)
        assert unroll.run() == 1
        assert unroll.stats["partial_epilogues"] == 1
        a0_after, dynamic_after = run_rv32(optimized)

        assert a0_before == a0_after == 5  # one increment per iteration
        assert dynamic_after < dynamic_before
        assert verify_errors(optimized) == []

    def test_epilogue_r_gt1_carried_value_is_skipped(self):
        """F1: never emit the stale-value epilogue; leave the loop alone."""
        program = build_carried_param_program(11)
        before = fingerprint(program)

        unroll = LoopUnroll(program, full_threshold=2, epilogue=True)
        assert unroll.run() == 0
        assert fingerprint(program) == before
        assert unroll.stats["skipped"]["carried_value"] == 1
        assert unroll.stats["loops_unrolled"] == 0

        a0, _ = run_rv32(program)
        assert a0 == 66  # sum(1..11), the pre-pass semantics

    def test_epilogue_r_gt1_forward_reference_is_skipped(self):
        program = build_forward_ref_program(11)
        before = fingerprint(program)

        unroll = LoopUnroll(program, full_threshold=2, epilogue=True)
        assert unroll.run() == 0
        assert fingerprint(program) == before
        assert unroll.stats["skipped"]["carried_value"] == 1

    def test_epilogue_skips_only_carried_shapes(self):
        """The new guard must not reject carried-free epilogue loops."""
        program = build_low_pressure_program(11)
        unroll = LoopUnroll(program, full_threshold=2, epilogue=True)
        assert unroll.run() == 1
        assert unroll.stats["partial_epilogues"] == 1
        assert unroll.stats["skipped"]["carried_value"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Case 4: negative scenarios (IR must stay untouched)
# ═══════════════════════════════════════════════════════════════════════════

class TestLoopUnrollNegative:
    def test_skip_step_not_one(self):
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        one = _make_const(builder, "one", 1)
        iv = builder.for_loop(0, 4, step=2)
        builder.add(iv, one)
        builder.endfor()
        builder.ret()

        unroll = LoopUnroll(builder.program)
        before = fingerprint(builder.program)
        assert unroll.run() == 0
        assert fingerprint(builder.program) == before
        assert unroll.stats["skipped"]["step_not_one"] == 1

    def test_skip_unpaired(self):
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        one = _make_const(builder, "one", 1)
        outer = builder.for_loop(0, 2)
        inner = builder.for_loop(0, 2)
        builder.add(inner, one)
        builder.endfor()
        builder.add(outer, one)
        builder.ret()

        unroll = LoopUnroll(builder.program)
        before = fingerprint(builder.program)
        assert unroll.run() == 0
        assert fingerprint(builder.program) == before
        assert unroll.stats["skipped"]["unpaired"] >= 1

    def test_skip_body_too_large(self):
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        one = _make_const(builder, "one", 1)
        iv = builder.for_loop(0, 4)
        for i in range(100):
            builder.add(one, one)
        builder.endfor()
        builder.ret(iv)

        unroll = LoopUnroll(builder.program, body_limit=64)
        before = fingerprint(builder.program)
        assert unroll.run() == 0
        assert fingerprint(builder.program) == before
        assert unroll.stats["skipped"]["body_too_large"] == 1

    def test_skip_iv_redefined(self):
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        one = _make_const(builder, "one", 1)
        iv = builder.for_loop(0, 4)
        # `iv` redefined inside the body: group-counter semantics break.
        builder._emit(OpCode.ADD, iv, [iv, one])
        builder.endfor()
        builder.ret()

        unroll = LoopUnroll(builder.program)
        before = fingerprint(builder.program)
        assert unroll.run() == 0
        assert fingerprint(builder.program) == before
        assert unroll.stats["skipped"]["multi_def"] == 1

    def test_skip_counts_stable_across_rescans(self):
        """F4: rescanning after each unroll must not inflate skip counters."""
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        one = _make_const(builder, "one", 1)
        for _ in range(3):
            iv = builder.for_loop(0, 4, step=2)
            builder.add(iv, one)
            builder.endfor()
        iv4 = builder.for_loop(0, 4)
        builder.add(iv4, one)
        builder.endfor()
        builder.ret()

        unroll = LoopUnroll(builder.program)
        assert unroll.run() == 1
        assert unroll.stats["skipped"]["step_not_one"] == 3
        assert unroll.stats["loops_seen"] == 4


class TestLoopUnrollIdempotency:
    def test_epilogue_remainder_loop_is_idempotent(self):
        """F5: the generated remainder loop carries the marker too."""
        program = build_low_pressure_program(23)
        unroll = LoopUnroll(program, full_threshold=2, epilogue=True)
        assert unroll.run() == 1
        assert unroll.stats["partial_epilogues"] == 1

        markers = [
            ins.attrs.get("unrolled") for _, ins in find_for_indices(program)
        ]
        assert markers == [8, 1]

        before = fingerprint(program)
        assert unroll.run() == 0
        assert fingerprint(program) == before
        assert unroll.stats["loops_unrolled"] == 0


class _ExplodingLoopUnroll(LoopUnroll):
    """Applies a plan, then raises to exercise the rollback path."""

    def _apply_unroll(self, func, block, for_idx, endfor_idx, plan):
        super()._apply_unroll(func, block, for_idx, endfor_idx, plan)
        raise RuntimeError("injected failure")


class TestLoopUnrollRollback:
    def test_failed_apply_restores_operands(self):
        """F6: a mid-rewrite exception must restore operands as well."""
        program = build_case_program(7, use_acc=True)
        return_ins = block_instructions(program)[-1]
        canonical = return_ins.operands[0]
        before = fingerprint(program)

        unroll = _ExplodingLoopUnroll(
            program, full_threshold=2, epilogue=True)
        assert unroll.run() == 0
        assert unroll.stats["skipped"]["internal_error"] == 1
        assert fingerprint(program) == before
        # _redirect_uses rewrote this operand to ``v3__ep`` before the
        # failure; the snapshot must bring the original value back.
        restored = block_instructions(program)[-1].operands[0]
        assert restored is canonical
        assert restored.name == "v_3"


# ═══════════════════════════════════════════════════════════════════════════
# Case 5: nested loops, inner first
# ═══════════════════════════════════════════════════════════════════════════

class _RecordingLoopUnroll(LoopUnroll):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan_sequence: list[tuple[str, int]] = []

    def _apply_unroll(self, func, block, for_idx, endfor_idx, plan):
        self.plan_sequence.append((plan.mode, plan.U))
        return super()._apply_unroll(func, block, for_idx, endfor_idx, plan)


class TestLoopUnrollNested:
    def _build_nested(self):
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        x = builder.make_value(name="x", dtype=DataType.INT32)
        y = builder.make_value(name="y", dtype=DataType.INT32)
        acc = builder.make_value(name="acc", dtype=DataType.INT32)
        builder.for_loop(0, 2)
        builder.for_loop(0, 3)
        t = builder.add(x, y)
        a2 = builder.add(acc, t)
        builder.endfor()
        builder.endfor()
        builder.ret(a2)
        return builder.program

    def test_inner_first(self):
        program = self._build_nested()
        unroll = _RecordingLoopUnroll(program)
        assert unroll.run() == 2
        assert unroll.stats["full_unrolls"] == 2
        assert unroll.plan_sequence == [("full", 3), ("full", 2)]

        assert count_for_markers(program) == 0
        body = [
            ins for ins in block_instructions(program)
            if ins.opcode != OpCode.RETURN
        ]
        assert len(body) == 12
        assert verify_errors(program) == []


# ═══════════════════════════════════════════════════════════════════════════
# Helper tests + integration
# ═══════════════════════════════════════════════════════════════════════════

class TestCaseHelpers:
    def test_count_ir_and_run_rv32(self):
        program = build_low_pressure_program(4)
        assert count_ir(program) == 5
        a0, dynamic = run_rv32(program)
        assert a0 == 4
        assert dynamic > 0

    def test_skip_unprofitable_partial(self):
        program = build_low_pressure_program(5)
        unroll = LoopUnroll(program, full_threshold=2)
        assert unroll.run() == 0
        assert unroll.stats["skipped"]["no_factor"] == 1


class TestLoopUnrollIntegration:
    DSL_SOURCE = (
        "for i = 0, 4\n"
        "  t = add(i, one)\n"
        "  acc = add(acc, t)\n"
        "endfor\n"
        "return acc\n"
    )

    def test_compiler_driver_runs_unroll(self, tmp_path):
        from scratchv.compiler import CompilerConfig, CompilerDriver

        driver = CompilerDriver(CompilerConfig(
            optimize_level="all", dump_ir=True, reg_alloc="greedy",
            loop_unroll=True))
        result = driver.compile(
            "", str(tmp_path / "on.s"), dsl_source=self.DSL_SOURCE)
        assert result.success
        after = result.ir_dump.split("--- IR Dump (after")[1]
        assert "endfor" not in after
        stats = result.stats["passes"]["loop-unroll"]
        assert stats["loops_unrolled"] == 1
        assert stats["full_unrolls"] == 1

    def test_no_loop_unroll_keeps_markers(self, tmp_path):
        from scratchv.compiler import CompilerConfig, CompilerDriver

        driver = CompilerDriver(CompilerConfig(
            optimize_level="all", dump_ir=True, reg_alloc="greedy",
            loop_unroll=False))
        result = driver.compile(
            "", str(tmp_path / "off.s"), dsl_source=self.DSL_SOURCE)
        assert result.success
        after = result.ir_dump.split("--- IR Dump (after")[1]
        assert "endfor" in after
        assert "loop-unroll" not in result.stats["passes"]

    def test_cli_flag_mapping(self):
        from scratchv.main import args_to_config, build_arg_parser

        parser = build_arg_parser()
        defaults = args_to_config(parser.parse_args(["input.dsl"]))
        # opt-in by default: the greedy allocator reload defect (B2) makes
        # unrolling unsafe as a default at optimize_level "all"
        assert defaults.loop_unroll is False
        assert defaults.unroll_max_factor == 8
        assert defaults.unroll_full_threshold == 8
        assert defaults.unroll_body_limit == 64
        assert defaults.unroll_max_growth == 512
        assert defaults.unroll_epilogue is False

        opt_in = args_to_config(
            parser.parse_args(["input.dsl", "--loop-unroll"]))
        assert opt_in.loop_unroll is True

        args = args_to_config(parser.parse_args([
            "input.dsl", "--no-loop-unroll", "--unroll-factor", "4",
            "--unroll-full-threshold", "2", "--unroll-body-limit", "32",
            "--unroll-max-growth", "100", "--unroll-epilogue",
        ]))
        assert args.loop_unroll is False
        assert args.unroll_max_factor == 4
        assert args.unroll_full_threshold == 2
        assert args.unroll_body_limit == 32
        assert args.unroll_max_growth == 100
        assert args.unroll_epilogue is True

        # last flag wins when both are given
        both_off = args_to_config(parser.parse_args(
            ["input.dsl", "--loop-unroll", "--no-loop-unroll"]))
        assert both_off.loop_unroll is False
        both_on = args_to_config(parser.parse_args(
            ["input.dsl", "--no-loop-unroll", "--loop-unroll"]))
        assert both_on.loop_unroll is True

    def test_no_loop_unroll_matches_default_on_loop_free_program(
            self, tmp_path):
        from scratchv.compiler import CompilerConfig, CompilerDriver

        source = "a = add(x, y)\nreturn a\n"
        common = dict(optimize_level="all", reg_alloc="greedy")
        on = CompilerDriver(CompilerConfig(**common)).compile(
            "", str(tmp_path / "on.s"), dsl_source=source)
        off = CompilerDriver(CompilerConfig(
            loop_unroll=False, **common)).compile(
            "", str(tmp_path / "off.s"), dsl_source=source)
        assert on.success and off.success
        assert on.output_text == off.output_text


class TestLoopUnrollDefaultOff:
    """F2: unrolling is opt-in until the allocator reload defect is fixed.

    The default ``--optimize all`` pipeline must keep compiling the
    pre-topic way; ``--loop-unroll`` / ``loop_unroll=True`` opts in and is
    exercised on a low-pressure loop where the backend is still correct.
    """

    HIGH_PRESSURE_DSL = (
        "for i = 0, 12\n"
        "  t = add(i, one)\n"
        "  u = add(t, one)\n"
        "  v = add(u, one)\n"
        "endfor\n"
        "return v\n"
    )

    LOW_PRESSURE_DSL = (
        "for i = 0, 6\n"
        "  t = add(i, one)\n"
        "endfor\n"
        "return t\n"
    )

    def test_config_default_is_off(self):
        from scratchv.compiler import CompilerConfig

        assert CompilerConfig().loop_unroll is False

    def test_default_flags_compile_high_pressure_loop_correctly(
            self, tmp_path):
        from scratchv.compiler import CompilerConfig, CompilerDriver

        driver = CompilerDriver(CompilerConfig(
            optimize_level="all", dump_ir=True, reg_alloc="greedy"))
        result = driver.compile(
            "", str(tmp_path / "default.s"),
            dsl_source=self.HIGH_PRESSURE_DSL)
        assert result.success
        assert "loop-unroll" not in result.stats["passes"]
        assert "endfor" in result.ir_dump.split("--- IR Dump (after")[1]

        a0, _ = run_asm_text(result.output_text)
        # ``one`` is a free (never defined) value read as 0, so the loop
        # yields i + 3*0 = 11; re-enabling unroll by default before the
        # allocator is fixed made the greedy backend return 10 here.
        assert a0 == 11

    def test_opt_in_unroll_is_correct_and_faster_on_low_pressure(
            self, tmp_path):
        from scratchv.compiler import CompilerConfig, CompilerDriver

        common = dict(optimize_level="all", reg_alloc="greedy")
        off = CompilerDriver(CompilerConfig(**common)).compile(
            "", str(tmp_path / "off.s"), dsl_source=self.LOW_PRESSURE_DSL)
        on = CompilerDriver(CompilerConfig(
            loop_unroll=True, **common)).compile(
            "", str(tmp_path / "on.s"), dsl_source=self.LOW_PRESSURE_DSL)
        assert off.success and on.success
        assert "loop-unroll" not in off.stats["passes"]
        assert "loop-unroll" in on.stats["passes"]

        a0_off, dyn_off = run_asm_text(off.output_text)
        a0_on, dyn_on = run_asm_text(on.output_text)
        assert a0_off == a0_on == 5
        assert dyn_on < dyn_off


class TestNoLoopUnrollGolden:
    """F3: byte-level pre-topic compatibility on real benchmark cases."""

    CASES = ("013_for_sum", "014_for_dot", "019_nested_loop")

    def test_cli_matches_baseline_golden(self, tmp_path):
        from scratchv.main import main

        root = Path(__file__).parents[1]
        golden_dir = Path(__file__).parent / "golden"
        for case in self.CASES:
            source = root / "benchmarks" / "cases" / f"{case}.dsl"
            # ``.golden`` suffix: plain ``*.s`` is gitignored as build output
            expected = (golden_dir / f"{case}.s.golden").read_text()

            off = tmp_path / f"{case}_off.s"
            assert main([
                str(source), "--optimize", "all", "--no-loop-unroll",
                "--count-instr", "-o", str(off),
            ]) == 0
            assert off.read_text() == expected

            # default flags (opt-in off) must reproduce the same bytes
            default = tmp_path / f"{case}_default.s"
            assert main([
                str(source), "--optimize", "all",
                "--count-instr", "-o", str(default),
            ]) == 0
            assert default.read_text() == expected


# ═══════════════════════════════════════════════════════════════════════════
# Plan selection unit checks
# ═══════════════════════════════════════════════════════════════════════════

class TestUnrollPlanSelection:
    def test_growth_limit_skips(self):
        program = build_low_pressure_program(7)
        unroll = LoopUnroll(
            program, full_threshold=2, epilogue=True, max_growth=2)
        assert unroll.run() == 0
        assert unroll.stats["skipped"]["growth_limit"] >= 1

    def test_bad_attrs_skips(self):
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        iv = builder.for_loop(0, 4)
        builder.endfor()
        builder.ret(iv)
        builder.program.functions[0].blocks[0].instructions[0].attrs = {
            "start": "0", "end": 4, "step": 1}

        unroll = LoopUnroll(builder.program)
        assert unroll.run() == 0
        assert unroll.stats["skipped"]["bad_attrs"] == 1

    def test_select_plan_returns_dataclass(self):
        program = build_low_pressure_program(4)
        unroll = LoopUnroll(program)
        pairs = unroll._find_pairs(block_instructions(program))
        plan = unroll._select_plan(block_instructions(program), *pairs[0])
        assert isinstance(plan, UnrollPlan)
        assert plan.mode == "full"
        assert plan.U == 4
