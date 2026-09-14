"""Tests for phase-1 scalar lowering of vector ops (Topic 29).

The executable differential tests run the same chain for the scalar
baseline and the vectorized program:

    InstructionSelector -> RegisterAllocator(greedy) -> AsmEmitter
    -> assemble_to_binary -> RV32Emulator

Note on widths: the pre-existing greedy allocator assigns one physical
register per distinct virtual register and does not reload evicted
registers, so executables with more than the 19 allocatable registers are
miscompiled regardless of vectorization.  The differential tests are
therefore built to stay within that budget (which is why the binary
multiply case runs at W=2 while the unary/broadcast cases run at W=4).
"""

import re

import pytest

from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.instruction_select import InstructionSelector
from scratchv.backend.machine_types import MachineOp
from scratchv.backend.register_alloc import RegisterAllocator
from scratchv.backend.riscv_encoder import assemble_to_binary
from scratchv.backend.vector_scalar import (
    VectorLoweringError,
    VectorScalarExpander,
)
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Instruction, OpCode, Value
from scratchv.optimizer.vectorize import Vectorizer
from scratchv.simulator.rv32_emulator import RV32Emulator

BASE_A = 0x400000
BASE_B = 0x410000
BASE_OUT = 0x420000

_VECTOR_MNEMONIC_RE = re.compile(r"^\s*v[a-z]", re.MULTILINE)
_RTYPE_IMMEDIATE_RE = re.compile(
    r"^\s*(add|sub|mul|div)\s+\S+,\s*\S+,\s*-?\d+\s*$", re.MULTILINE)


# ── Program builders ────────────────────────────────────────────────────

def _make_relu_loop(n: int):
    b = IRBuilder()
    b.new_function("main")
    b.new_block("entry")
    a = b.load_const(BASE_A, dtype=DataType.INT32)
    o = b.load_const(BASE_OUT, dtype=DataType.INT32)
    iv = b.for_loop(0, n)
    c4 = b.load_const(4, dtype=DataType.INT32)
    off = b.mul(iv, c4)
    va = b.load(b.add(a, off))
    r = b.relu(va)
    b.store(b.add(o, off), r)
    b.endfor()
    b.ret()
    return b.program


def _make_inplace_relu_loop(n: int):
    """a[i] = relu(a[i]); a separate address ADD per access (F1 N<B).

    The vector rewrite deduplicates the two ADD instructions, so the new
    body is shorter than the original and the pre-fix remainder insertion
    landed after the trailing ``return`` (dead code, review F1).
    """
    b = IRBuilder()
    b.new_function("main")
    b.new_block("entry")
    a = b.load_const(BASE_A, dtype=DataType.INT32)
    iv = b.for_loop(0, n)
    c4 = b.load_const(4, dtype=DataType.INT32)
    off = b.mul(iv, c4)
    va = b.load(b.add(a, off))
    r = b.relu(va)
    b.store(b.add(a, off), r)
    b.endfor()
    b.ret()
    return b.program


def _make_mul_loop(n: int):
    b = IRBuilder()
    b.new_function("main")
    b.new_block("entry")
    a = b.load_const(BASE_A, dtype=DataType.INT32)
    bb = b.load_const(BASE_B, dtype=DataType.INT32)
    o = b.load_const(BASE_OUT, dtype=DataType.INT32)
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


def _make_div_loop(n: int, divisor: int = 3):
    b = IRBuilder()
    b.new_function("main")
    b.new_block("entry")
    a = b.load_const(BASE_A, dtype=DataType.INT32)
    o = b.load_const(BASE_OUT, dtype=DataType.INT32)
    k = b.load_const(divisor, dtype=DataType.INT32)
    iv = b.for_loop(0, n)
    c4 = b.load_const(4, dtype=DataType.INT32)
    off = b.mul(iv, c4)
    va = b.load(b.add(a, off))
    r = b.div(va, k)
    b.store(b.add(o, off), r)
    b.endfor()
    b.ret()
    return b.program


# ── Execution helper ────────────────────────────────────────────────────

def _compile(program):
    machine = InstructionSelector(program).run()
    asm = AsmEmitter(RegisterAllocator(machine, mode="greedy").run()).emit()
    return machine, asm, assemble_to_binary(asm)


def _compile_and_run(program, n, inputs_a, inputs_b=None,
                     out_base: int = BASE_OUT):
    _, asm, binary = _compile(program)
    emu = RV32Emulator()
    emu.load_code(bytes(binary))
    for i, x in enumerate(inputs_a):
        emu.write_i32(BASE_A + 4 * i, int(x))
    if inputs_b is not None:
        for i, x in enumerate(inputs_b):
            emu.write_i32(BASE_B + 4 * i, int(x))
    emu.run(max_instr=200000)
    return asm, [emu.read_i32(out_base + 4 * i) for i in range(n)]


# ── Sanity: no vector instructions in the lowered output ────────────────

class TestLoweringSanity:
    def test_no_vector_mnemonic_in_asm(self):
        program = _make_mul_loop(16)
        Vectorizer(program, width=4).run(program)
        machine, asm, _ = _compile(program)

        assert not _VECTOR_MNEMONIC_RE.search(asm)
        vector_names = {
            instr.dest.name
            for block in program.functions[0].blocks
            for instr in block.instructions
            if instr.opcode.is_vector() and instr.dest is not None
        }
        for instr in machine:
            for op in (instr.dst, instr.src1, instr.src2):
                if op is not None and op.kind == "vreg":
                    assert op.value not in vector_names

    def test_lane_addressing_uses_addi(self):
        program = _make_relu_loop(16)
        Vectorizer(program, width=4).run(program)
        _, asm, _ = _compile(program)

        assert "addi a1," in asm
        assert "lw" in asm and "a1(0)" in asm
        assert not _RTYPE_IMMEDIATE_RE.search(asm)

    def test_lowering_emits_expected_machine_ops(self):
        program = _make_mul_loop(16)
        Vectorizer(program, width=2).run(program)
        machine, _, _ = _compile(program)
        ops = {instr.op for instr in machine if instr.op != MachineOp.LABEL}
        assert MachineOp.LW in ops
        assert MachineOp.SW in ops
        assert MachineOp.MUL in ops
        assert MachineOp.MAX in ops
        assert MachineOp.ADDI in ops


# ── Executable differential ─────────────────────────────────────────────

_EXECUTABLE_MATRIX = [
    pytest.param(w, n, pattern, id=f"w{w}-n{n}-{pattern}")
    for w in (2, 4)
    for n in (16, 17)
    for pattern in ("map", "broadcast", "inplace")
    # W=4 with a remainder needs more than the greedy allocator's 19
    # virtual registers for map/broadcast; the pre-existing allocator
    # spills without reloading there (review section 4, out of scope).
    # Those two cells are covered structurally in tests/test_vectorize.py.
    if (w, n, pattern) not in ((4, 17, "map"), (4, 17, "broadcast"))
]


class TestEmulatorDifferential:
    A_NEG = [7, -3, 0, 100000, -1, 2, -2048, 2047,
             123456, -654321, 0, -5, 9, -9, 42, -42]
    B_NEG = [3, 5, -7, 3, -8, -2, 15, 16,
             2, 11, 0, -12, 13, 13, -3, 4]

    def _run_pair(self, make, n, width, inputs_a, inputs_b, reference,
                  out_base: int = BASE_OUT):
        baseline = make(n)
        _, out_scalar = _compile_and_run(baseline, n, inputs_a, inputs_b,
                                         out_base)

        vectorized = make(n)
        result = Vectorizer(vectorized, width=width).run(vectorized)
        assert result.changes == 1
        _, out_vector = _compile_and_run(vectorized, n, inputs_a, inputs_b,
                                         out_base)

        assert out_scalar == reference
        assert out_vector == out_scalar
        assert out_vector == reference

    @pytest.mark.parametrize("width,n,pattern", _EXECUTABLE_MATRIX)
    def test_matrix_matches_scalar(self, width, n, pattern):
        inputs = (self.A_NEG + [11])[:n]
        if pattern == "broadcast":
            inputs = [abs(x) for x in inputs]
            self._run_pair(_make_div_loop, n, width, inputs, None,
                           [x // 3 for x in inputs])
        elif pattern == "inplace":
            self._run_pair(_make_inplace_relu_loop, n, width, inputs, None,
                           [max(x, 0) for x in inputs], out_base=BASE_A)
        else:
            self._run_pair(_make_relu_loop, n, width, inputs, None,
                           [max(x, 0) for x in inputs])

    def test_inplace_remainder_w2_matches_scalar(self):
        """Review F1 repro: in-place relu, n=17, W=2 → a[16] must be 0."""
        inputs = self.A_NEG + [-9]
        reference = [max(x, 0) for x in inputs]
        assert len(reference) == 17 and reference[16] == 0
        self._run_pair(_make_inplace_relu_loop, 17, 2, inputs, None,
                       reference, out_base=BASE_A)

    def test_div_two_base_remainder_w2_matches_scalar(self):
        """Review F1 repro: two address chains, n=17, W=2 (N>B insertion)."""
        inputs = [abs(x) + 1 for x in (self.A_NEG + [11])]
        reference = [x // 3 for x in inputs]
        self._run_pair(_make_div_loop, 17, 2, inputs, None, reference)

    def test_relu_map_w4_matches_scalar(self):
        reference = [max(x, 0) for x in self.A_NEG]
        self._run_pair(_make_relu_loop, 16, 4, self.A_NEG, None, reference)

    def test_mul_map_w2_matches_scalar(self):
        reference = [max(x * y, 0) for x, y in zip(self.A_NEG, self.B_NEG)]
        self._run_pair(_make_mul_loop, 16, 2,
                       self.A_NEG, self.B_NEG, reference)

    def test_div_broadcast_w4_matches_scalar(self):
        inputs = [12, 3, 0, 99, 6, 27, 2049, 2043,
                  123, 654, 0, 51, 9, 33, 42, 45]
        reference = [x // 3 for x in inputs]
        self._run_pair(_make_div_loop, 16, 4, inputs, None, reference)

    def test_remainder_loop_w2_matches_scalar(self):
        inputs = self.A_NEG + [11]
        reference = [max(x, 0) for x in inputs]
        self._run_pair(_make_relu_loop, 17, 2, inputs, None, reference)


# ── Lowering errors ─────────────────────────────────────────────────────

class TestLoweringErrors:
    def test_missing_lane_binding(self):
        expander = VectorScalarExpander()
        expander.begin_function("main")
        va = Value(name="va", dtype=DataType.INT32, shape=(4,))
        vb = Value(name="vb", dtype=DataType.INT32, shape=(4,))
        dest = Value(name="vd", dtype=DataType.INT32, shape=(4,))
        instr = Instruction(
            opcode=OpCode.VADD, dest=dest, operands=[va, vb],
            attrs={"width": 4})
        with pytest.raises(VectorLoweringError) as exc:
            expander.expand(instr)
        assert "has no lane binding" in str(exc.value)

    def test_width_mismatch(self):
        expander = VectorScalarExpander()
        expander.begin_function("main")
        va = Value(name="va", dtype=DataType.INT32, shape=(2,))
        dest = Value(name="vd", dtype=DataType.INT32, shape=(4,))
        instr = Instruction(
            opcode=OpCode.VRELU, dest=dest, operands=[va],
            attrs={"width": 4})
        with pytest.raises(VectorLoweringError):
            expander.expand(instr)

    def test_non_vector_op_is_refused(self):
        expander = VectorScalarExpander()
        expander.begin_function("main")
        instr = Instruction(opcode=OpCode.RELU)
        with pytest.raises(VectorLoweringError):
            expander.expand(instr)
