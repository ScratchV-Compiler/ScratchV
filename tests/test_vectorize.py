"""Tests for the phase-1 SIMD vectorizer (Topic 29).

Covers the vector builder APIs, the strip-mining decision tree
(C1-C7 of the design document) and the structured rejection report.
"""

from __future__ import annotations

import pytest

from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Instruction, OpCode, Program
from scratchv.optimizer.vectorize import (
    REASON_ALIASING_STORE,
    REASON_NESTED_CONTROL_FLOW,
    REASON_NO_ELEMENT_PATTERN,
    REASON_NON_CONSTANT_BOUNDS,
    REASON_NON_ELEMENTWISE_IV,
    REASON_REGION_VALUE_ESCAPES,
    REASON_TRIP_TOO_SMALL,
    REASON_UNSUPPORTED_START,
    REASON_UNSUPPORTED_STEP,
    Vectorizer,
)

VECTOR_OPS = (
    OpCode.VLOAD,
    OpCode.VSTORE,
    OpCode.VBCAST,
    OpCode.VADD,
    OpCode.VSUB,
    OpCode.VMUL,
    OpCode.VDIV,
    OpCode.VRELU,
)


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_map_loop(n: int) -> Program:
    """out[i] = relu(a[i] * b[i]) for i in [0, n), hand-written IR."""
    b = IRBuilder()
    b.new_function("main")
    b.new_block("entry")
    a = b.load_const(0x400000, dtype=DataType.INT32)
    bb = b.load_const(0x410000, dtype=DataType.INT32)
    o = b.load_const(0x420000, dtype=DataType.INT32)
    iv = b.for_loop(0, n)
    c4 = b.load_const(4, dtype=DataType.INT32)
    off = b.mul(iv, c4)
    va = b.load(b.add(a, off))
    vb = b.load(b.add(bb, off))
    r = b.relu(b.mul(va, vb))
    b.store(b.add(o, off), r)
    b.endfor()
    b.ret()
    return b.program


def _make_chain_loop(n: int) -> Program:
    """Exercises all eight vector ops in one vectorized region."""
    b = IRBuilder()
    b.new_function("main")
    b.new_block("entry")
    a = b.load_const(0x400000, dtype=DataType.INT32)
    bb = b.load_const(0x410000, dtype=DataType.INT32)
    o = b.load_const(0x420000, dtype=DataType.INT32)
    iv = b.for_loop(0, n)
    c4 = b.load_const(4, dtype=DataType.INT32)
    k = b.load_const(4, dtype=DataType.INT32)
    off = b.mul(iv, c4)
    va = b.load(b.add(a, off))
    vb = b.load(b.add(bb, off))
    s = b.add(va, vb)
    d = b.sub(va, vb)
    m = b.mul(s, d)
    q = b.div(m, k)
    r = b.relu(q)
    b.store(b.add(o, off), r)
    b.endfor()
    b.ret()
    return b.program


def _make_inplace_relu_loop(n: int) -> Program:
    """a[i] = relu(a[i]) with a separate address ADD per access.

    The vector rewrite deduplicates the two ``ADD`` instructions into one
    (same base), so the rewritten body is *shorter* than the original
    body: the F1 N < B case.
    """
    b = IRBuilder()
    b.new_function("main")
    b.new_block("entry")
    a = b.load_const(0x400000, dtype=DataType.INT32)
    iv = b.for_loop(0, n)
    c4 = b.load_const(4, dtype=DataType.INT32)
    off = b.mul(iv, c4)
    va = b.load(b.add(a, off))
    r = b.relu(va)
    b.store(b.add(a, off), r)
    b.endfor()
    b.ret()
    return b.program


def _make_div_two_base_loop(n: int) -> Program:
    """out[i] = a[i] / k with both input and output element chains.

    The rewritten body gains a ``VBCAST``, so it is *longer* than the
    original body: the F1 N > B case.
    """
    b = IRBuilder()
    b.new_function("main")
    b.new_block("entry")
    a = b.load_const(0x400000, dtype=DataType.INT32)
    o = b.load_const(0x420000, dtype=DataType.INT32)
    k = b.load_const(3, dtype=DataType.INT32)
    iv = b.for_loop(0, n)
    c4 = b.load_const(4, dtype=DataType.INT32)
    off = b.mul(iv, c4)
    va = b.load(b.add(a, off))
    r = b.div(va, k)
    b.store(b.add(o, off), r)
    b.endfor()
    b.ret()
    return b.program


def _instructions(program: Program) -> list[Instruction]:
    return program.functions[0].blocks[0].instructions


def _for_indices(program: Program) -> list[int]:
    return [i for i, instr in enumerate(_instructions(program))
            if instr.opcode is OpCode.FOR]


def _body(program: Program, for_index: int) -> list[Instruction]:
    instrs = _instructions(program)
    body = []
    for instr in instrs[for_index + 1:]:
        if instr.opcode is OpCode.ENDFOR:
            break
        body.append(instr)
    return body


def _find(instrs: list[Instruction], opcode: OpCode) -> list[Instruction]:
    return [instr for instr in instrs if instr.opcode is opcode]


def _matching_endfor(instrs: list[Instruction], for_index: int) -> int:
    depth = 0
    for j in range(for_index, len(instrs)):
        op = instrs[j].opcode
        if op is OpCode.FOR:
            depth += 1
        elif op is OpCode.ENDFOR:
            depth -= 1
            if depth == 0:
                return j
    raise AssertionError("FOR without matching ENDFOR")


# ── Builder API ─────────────────────────────────────────────────────────

class TestVectorIrBuilders:
    def _new_builder(self):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        return b

    def test_builder_emits_vector_ops(self):
        b = self._new_builder()
        addr = b.make_value(name="addr", dtype=DataType.INT32)
        scalar = b.make_value(name="s", dtype=DataType.INT32)
        va = b.vload(addr, width=4)
        vb = b.vload(addr, width=4)
        bc = b.vbcast(scalar, width=4)
        results = [
            (b.vadd(va, vb), OpCode.VADD),
            (b.vsub(va, vb), OpCode.VSUB),
            (b.vmul(va, vb), OpCode.VMUL),
            (b.vdiv(va, vb), OpCode.VDIV),
            (b.vrelu(va), OpCode.VRELU),
        ]
        instrs = _instructions(b.program)

        assert va.shape == (4,) and vb.shape == (4,) and bc.shape == (4,)
        assert _find(instrs, OpCode.VLOAD)[0].attrs["width"] == 4
        assert _find(instrs, OpCode.VLOAD)[0].attrs["elem_bytes"] == 4
        assert _find(instrs, OpCode.VLOAD)[0].attrs["align"] == 4
        assert _find(instrs, OpCode.VBCAST)[0].attrs == {"width": 4}
        for value, opcode in results:
            assert value.shape == (4,)
            instr = _find(instrs, opcode)[0]
            assert instr.attrs == {"width": 4}

    def test_vstore_returns_instruction(self):
        b = self._new_builder()
        addr = b.make_value(name="addr", dtype=DataType.INT32)
        vec = b.vbcast(b.make_value(name="s", dtype=DataType.INT32), width=4)
        instr = b.vstore(addr, vec)
        assert isinstance(instr, Instruction)
        assert instr.opcode is OpCode.VSTORE
        assert instr.dest is None
        assert instr.attrs["width"] == 4

    def test_builder_width_inference(self):
        b = self._new_builder()
        addr = b.make_value(name="addr", dtype=DataType.INT32)
        va = b.vload(addr, width=2)
        vb = b.vload(addr, width=2)
        assert b.vadd(va, vb).shape == (2,)
        assert b.vrelu(va).shape == (2,)

    def test_is_vector_covers_all_eight_ops(self):
        assert all(op.is_vector() for op in VECTOR_OPS)
        for opcode in (OpCode.ADD, OpCode.LOAD, OpCode.FOR, OpCode.RELU):
            assert not opcode.is_vector()


# ── Strip-mining structure ──────────────────────────────────────────────

class TestVectorizerStructure:
    def test_strip_mining_no_remainder(self):
        program = _make_map_loop(16)
        vec = Vectorizer(program, width=4)
        result = vec.run(program)

        assert result.changes == 1
        for_indices = _for_indices(program)
        assert len(for_indices) == 1

        for_instr = _instructions(program)[for_indices[0]]
        assert for_instr.attrs == {
            "start": 0, "end": 4, "step": 1,
            "vector_width": 4, "elem_bytes": 4, "orig_trip": 16,
        }

        body = _body(program, for_indices[0])
        counts = {
            OpCode.VLOAD: len(_find(body, OpCode.VLOAD)),
            OpCode.VMUL: len(_find(body, OpCode.VMUL)),
            OpCode.VRELU: len(_find(body, OpCode.VRELU)),
            OpCode.VSTORE: len(_find(body, OpCode.VSTORE)),
        }
        assert counts == {
            OpCode.VLOAD: 2,
            OpCode.VMUL: 1,
            OpCode.VRELU: 1,
            OpCode.VSTORE: 1,
        }
        for instr in body:
            if instr.opcode.is_vector():
                assert instr.attrs["width"] == 4
                if instr.dest is not None:
                    assert instr.dest.shape == (4,)

        assert vec.last_report[0].status == "vectorized"
        assert vec.last_report[0].strips == 4
        assert vec.last_report[0].remainder == 0
        assert vec.last_report[0].vector_ops == 5

    def test_all_eight_vector_ops_are_generated(self):
        program = _make_chain_loop(16)
        vec = Vectorizer(program, width=4)
        result = vec.run(program)

        assert result.changes == 1
        body = _body(program, _for_indices(program)[0])
        generated = {instr.opcode for instr in body
                     if instr.opcode.is_vector()}
        assert generated == set(VECTOR_OPS)

    def test_strip_mining_with_remainder(self):
        program = _make_map_loop(17)
        vec = Vectorizer(program, width=4)
        vec.run(program)

        for_indices = _for_indices(program)
        assert len(for_indices) == 2

        strip_for = _instructions(program)[for_indices[0]]
        assert strip_for.attrs["end"] == 4
        remainder_for = _instructions(program)[for_indices[1]]
        assert remainder_for.attrs == {"start": 16, "end": 17, "step": 1}

        remainder_body = _body(program, for_indices[1])
        assert remainder_body
        assert not any(instr.opcode.is_vector() for instr in remainder_body)
        assert all(instr.dest.name.endswith("__rem")
                   for instr in remainder_body if instr.dest)

        vector_body = _body(program, for_indices[0])
        vector_names = {instr.dest.name for instr in vector_body if instr.dest}
        rem_names = {instr.dest.name for instr in remainder_body if instr.dest}
        assert not (vector_names & rem_names)

        assert vec.last_report[0].remainder == 1

    def test_width_2(self):
        program = _make_map_loop(16)
        vec = Vectorizer(program, width=2)
        vec.run(program)

        for_indices = _for_indices(program)
        assert len(for_indices) == 1
        for_instr = _instructions(program)[for_indices[0]]
        assert for_instr.attrs["vector_width"] == 2
        assert for_instr.attrs["end"] == 8

        body = _body(program, for_indices[0])
        for instr in body:
            if instr.opcode.is_vector():
                assert instr.attrs["width"] == 2
                if instr.dest is not None:
                    assert instr.dest.shape == (2,)

    def test_trip_too_small_ir_unchanged(self):
        program = _make_map_loop(3)
        before = program.dump()
        vec = Vectorizer(program, width=4)
        result = vec.run(program)

        assert result.changes == 0
        assert program.dump() == before
        assert vec.last_report[0].status == "rejected"
        assert vec.last_report[0].reason == REASON_TRIP_TOO_SMALL

    def test_consecutive_remainder_loops_both_vectorized(self):
        """After a rewritten region the scan must resume past its remainder.

        Guards ``_skip_rewritten``: a stale index would re-enter the
        remainder loop of the first region and/or skip the second one.
        """
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        a = b.load_const(0x400000, dtype=DataType.INT32)
        for _ in range(2):
            iv = b.for_loop(0, 17)
            c4 = b.load_const(4, dtype=DataType.INT32)
            off = b.mul(iv, c4)
            va = b.load(b.add(a, off))
            r = b.relu(va)
            b.store(b.add(a, off), r)
            b.endfor()
        b.ret()

        vec = Vectorizer(b.program, width=2)
        result = vec.run(b.program)

        assert result.changes == 2
        assert [rec.status for rec in vec.last_report] == [
            "vectorized", "vectorized"]
        assert len(_for_indices(b.program)) == 4


# ── Remainder loop placement (review F1) ────────────────────────────────
#
# The remainder block must be inserted *after* the rewritten strip ENDFOR
# (its position shifts with the length of the new body) and before the
# trailing `return`.  The original bug inserted it at the stale pre-rewrite
# index: a longer vector body nested the remainder inside the strip loop,
# a shorter one left it after `return` as dead code.

class TestRemainderPlacement:
    @pytest.mark.parametrize("width", [2, 4])
    @pytest.mark.parametrize(
        "builder, relation",
        [
            pytest.param(_make_inplace_relu_loop, "N<B",
                         id="shorter-vector-body"),
            pytest.param(_make_div_two_base_loop, "N>B",
                         id="longer-vector-body"),
        ],
    )
    def test_remainder_is_sibling_after_strip_loop(self, builder, relation,
                                                   width):
        n = 17
        program = builder(n)
        vec = Vectorizer(program, width=width)
        result = vec.run(program)

        assert result.changes == 1
        instrs = _instructions(program)
        for_indices = _for_indices(program)
        assert len(for_indices) == 2

        strip_for, rem_for = for_indices
        strip_endfor = _matching_endfor(instrs, strip_for)

        # The strip loop must not contain the remainder FOR (no nesting).
        assert not any(instr.opcode is OpCode.FOR
                       for instr in instrs[strip_for + 1:strip_endfor])
        # The remainder FOR is the immediate sibling of the strip ENDFOR.
        assert rem_for == strip_endfor + 1

        strips = n // width
        assert instrs[rem_for].attrs == {
            "start": strips * width, "end": n, "step": 1}
        rem_endfor = _matching_endfor(instrs, rem_for)
        # The remainder loop is before the trailing `return`, never dead.
        assert rem_endfor < len(instrs) - 1
        assert instrs[-1].opcode is OpCode.RETURN
        assert not any(instr.opcode is OpCode.RETURN
                       for instr in instrs[:rem_endfor])

        remainder_body = _body(program, rem_for)
        assert remainder_body
        assert not any(instr.opcode.is_vector() for instr in remainder_body)
        assert all(instr.dest.name.endswith("__rem")
                   for instr in remainder_body if instr.dest)


# ── Rejections (design doc 2.5 I1-I4) ───────────────────────────────────

class TestVectorizerRejects:
    def _reject_reason(self, program: Program, width: int = 4) -> str:
        vec = Vectorizer(program, width=width)
        result = vec.run(program)
        assert result.changes == 0
        assert vec.last_report[0].status == "rejected"
        return vec.last_report[0].reason

    def test_i1_reduction_without_memory(self):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        s = b.load_const(0, dtype=DataType.INT32)
        iv = b.for_loop(0, 10)
        acc = b.add(s, iv)
        b.endfor()
        b.ret(acc)
        assert self._reject_reason(b.program) == REASON_NO_ELEMENT_PATTERN

    def test_i1b_iv_in_element_expression(self):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        a = b.load_const(0x400000, dtype=DataType.INT32)
        o = b.load_const(0x420000, dtype=DataType.INT32)
        iv = b.for_loop(0, 16)
        c4 = b.load_const(4, dtype=DataType.INT32)
        off = b.mul(iv, c4)
        va = b.load(b.add(a, off))
        s = b.add(va, iv)
        b.store(b.add(o, off), s)
        b.endfor()
        b.ret()
        assert self._reject_reason(b.program) == REASON_NON_ELEMENTWISE_IV

    def test_i2_aliasing_store(self):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        a = b.load_const(0x400000, dtype=DataType.INT32)
        iv = b.for_loop(0, 16)
        c4 = b.load_const(4, dtype=DataType.INT32)
        c1 = b.load_const(1, dtype=DataType.INT32)
        off = b.mul(iv, c4)
        va = b.load(b.add(a, off))
        im1 = b.sub(iv, c1)
        off2 = b.mul(im1, c4)
        b.store(b.add(a, off2), va)
        b.endfor()
        b.ret()
        assert self._reject_reason(b.program) == REASON_ALIASING_STORE

    @pytest.mark.parametrize("n", [16, 17])
    def test_iv_live_out_rejected(self, n):
        """Review F2: the original IV is redefined as the strip index."""
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        a = b.load_const(0x400000, dtype=DataType.INT32)
        o = b.load_const(0x420000, dtype=DataType.INT32)
        iv = b.for_loop(0, n)
        c4 = b.load_const(4, dtype=DataType.INT32)
        off = b.mul(iv, c4)
        va = b.load(b.add(a, off))
        b.store(b.add(o, off), va)
        b.endfor()
        c4b = b.load_const(4, dtype=DataType.INT32)
        b.store(b.add(o, c4b), iv)  # IV used after ENDFOR
        b.ret()
        assert self._reject_reason(b.program) == REASON_REGION_VALUE_ESCAPES

    def test_region_local_value_live_out_rejected(self):
        """Review F2: a region-local LOAD result escapes the region."""
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        a = b.load_const(0x400000, dtype=DataType.INT32)
        o = b.load_const(0x420000, dtype=DataType.INT32)
        iv = b.for_loop(0, 16)
        c4 = b.load_const(4, dtype=DataType.INT32)
        off = b.mul(iv, c4)
        va = b.load(b.add(a, off))
        r = b.relu(va)
        b.store(b.add(o, off), r)
        b.endfor()
        c4b = b.load_const(4, dtype=DataType.INT32)
        b.store(b.add(o, c4b), va)  # region-local LOAD result live-out
        b.ret()
        assert self._reject_reason(b.program) == REASON_REGION_VALUE_ESCAPES

    def test_cross_base_overlapping_rejected(self):
        """Review F3: different base names may still overlap in memory."""
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        out = b.load_const(0x420000, dtype=DataType.INT32)
        c4 = b.load_const(4, dtype=DataType.INT32)
        src = b.sub(out, c4)  # element-wise alias: src[i] == out[i - 1]
        iv = b.for_loop(0, 16)
        c4b = b.load_const(4, dtype=DataType.INT32)
        off = b.mul(iv, c4b)
        va = b.load(b.add(src, off))
        b.store(b.add(out, off), va)
        b.endfor()
        b.ret()
        assert self._reject_reason(b.program) == REASON_ALIASING_STORE

    def test_cross_base_unknown_pointer_rejected(self):
        """Review F3: non-constant bases cannot be proven disjoint."""
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        src = b.make_value(name="src", dtype=DataType.INT32)
        out = b.load_const(0x420000, dtype=DataType.INT32)
        iv = b.for_loop(0, 16)
        c4 = b.load_const(4, dtype=DataType.INT32)
        off = b.mul(iv, c4)
        va = b.load(b.add(src, off))
        b.store(b.add(out, off), va)
        b.endfor()
        b.ret()
        assert self._reject_reason(b.program) == REASON_ALIASING_STORE

    def test_cross_base_distinct_constants_allowed(self):
        """Distinct constant bases with disjoint ranges stay vectorizable."""
        program = _make_map_loop(16)
        vec = Vectorizer(program, width=4)
        result = vec.run(program)
        assert result.changes == 1
        assert vec.last_report[0].status == "vectorized"

    def test_i3_dynamic_bounds(self):
        program = _make_map_loop(16)
        for_instr = _instructions(program)[_for_indices(program)[0]]
        for_instr.attrs.pop("end")
        assert self._reject_reason(program) == REASON_NON_CONSTANT_BOUNDS

    def test_i4_nested_control_flow(self):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        a = b.load_const(0x400000, dtype=DataType.INT32)
        o = b.load_const(0x420000, dtype=DataType.INT32)
        cond = b.load_const(1, dtype=DataType.INT32)
        iv = b.for_loop(0, 16)
        c4 = b.load_const(4, dtype=DataType.INT32)
        off = b.mul(iv, c4)
        va = b.load(b.add(a, off))
        b.br_if(cond, "then", "else")
        b.store(b.add(o, off), va)
        b.endfor()
        b.ret()
        assert self._reject_reason(b.program) == REASON_NESTED_CONTROL_FLOW

    def test_unsupported_step(self):
        program = _make_map_loop(16)
        for_instr = _instructions(program)[_for_indices(program)[0]]
        for_instr.attrs["step"] = 2
        assert self._reject_reason(program) == REASON_UNSUPPORTED_STEP

    def test_unsupported_start(self):
        program = _make_map_loop(16)
        for_instr = _instructions(program)[_for_indices(program)[0]]
        for_instr.attrs["start"] = 4
        assert self._reject_reason(program) == REASON_UNSUPPORTED_START

    def test_rejections_do_not_touch_ir(self):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        a = b.load_const(0x400000, dtype=DataType.INT32)
        o = b.load_const(0x420000, dtype=DataType.INT32)
        iv = b.for_loop(0, 16)
        c4 = b.load_const(4, dtype=DataType.INT32)
        off = b.mul(iv, c4)
        va = b.load(b.add(a, off))
        r = b.relu(b.add(va, va))
        n = b.neg(r)
        b.store(b.add(o, off), n)
        b.endfor()
        b.ret()

        before = b.program.dump()
        vec = Vectorizer(b.program, width=4)
        vec.run(b.program)
        # NEG has no phase-1 vector op → rejected as unsupported.
        assert vec.last_report[0].reason == "unsupported-op"
        assert b.program.dump() == before


# ── Report ──────────────────────────────────────────────────────────────

class TestVectorizeReport:
    def test_report_counts(self):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        a = b.load_const(0x400000, dtype=DataType.INT32)
        o = b.load_const(0x420000, dtype=DataType.INT32)
        s = b.load_const(0, dtype=DataType.INT32)
        # Rejected loop first: no element address pattern.
        iv = b.for_loop(0, 16)
        b.add(s, iv)
        b.endfor()
        # Vectorizable loop second.
        iv2 = b.for_loop(0, 16)
        c4 = b.load_const(4, dtype=DataType.INT32)
        off = b.mul(iv2, c4)
        va = b.load(b.add(a, off))
        r = b.relu(va)
        b.store(b.add(o, off), r)
        b.endfor()
        b.ret()

        vec = Vectorizer(b.program, width=4)
        result = vec.run(b.program)

        assert result.changes == 1
        assert result.message == "vectorized 1/2 loop(s), width=4"
        assert len(vec.last_report) == 2
        assert vec.last_report[0].status == "rejected"
        assert vec.last_report[1].status == "vectorized"
        assert len(result.warnings) == 1
        assert "rejected: no-memory-element-pattern" in result.warnings[0]


# ── Width validation (review F6) ────────────────────────────────────────

class TestVectorWidthValidation:
    @pytest.mark.parametrize("width", [0, 1, -2, True, "4"])
    def test_invalid_width_raises_value_error(self, width):
        program = _make_map_loop(16)
        with pytest.raises(ValueError, match="vector width"):
            Vectorizer(program, width=width)

    @pytest.mark.parametrize("width", [2, 4])
    def test_valid_width_is_accepted(self, width):
        program = _make_map_loop(16)
        assert Vectorizer(program, width=width).width == width
