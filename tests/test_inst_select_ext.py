"""Tests for Extended Instruction Selector."""

import pytest
from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.inst_select_ext import ExtendedInstructionSelector
from scratchv.backend.register_alloc import (
    MachineOp, RegisterAllocator,
)
from scratchv.backend.riscv_encoder import REG_MAP, assemble_to_binary
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import (  # noqa: F401
    OpCode, Value, DataType,
)
from scratchv.simulator.tinyfive import ProfiledMachine


class TestExtendedSelectorBasic:
    """Tests for the extended instruction selector."""

    def test_creation(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a")
        b = builder.make_value(name="b")
        c = builder.add(a, b)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        assert selector.enable_fp64 is True
        assert selector.use_hardware_sqrt is False

    def test_creation_fp64_disabled(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        selector = ExtendedInstructionSelector(
            builder.program, enable_fp64=False)
        assert selector.enable_fp64 is False

    def test_run_basic(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a")
        b = builder.make_value(name="b")
        c = builder.add(a, b)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        assert len(instrs) > 0
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.ADD in ops

    def test_run_relu(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a")
        c = builder.relu(a)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.MAX in ops

    def test_run_sub(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a")
        b = builder.make_value(name="b")
        c = builder.sub(a, b)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.SUB in ops

    def test_supported_ops(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        selector = ExtendedInstructionSelector(builder.program)
        ops = selector.supported_ops
        assert "add" in ops
        assert "sqrt" in ops
        assert "min" in ops
        assert "max" in ops
        assert "abs" in ops

    def test_supported_ops_no_fp64(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        selector = ExtendedInstructionSelector(
            builder.program, enable_fp64=False)
        ops = selector.supported_ops
        assert "add" in ops
        # fp64 ops should still be in unsupported list
        # (they're always defined, just not enabled)


class TestExtendedSelectorNeg:
    """Tests for neg instruction handling."""

    def test_neg(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a")
        c = builder.neg(a)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.SUB in ops


class TestTypeDetection:
    """Tests for float64 type detection."""

    def test_is_fp64_int32(self):
        from scratchv.ir.types import Instruction, OpCode

        v = Value(name="x", dtype=DataType.INT32)
        instr = Instruction(opcode=OpCode.ADD, operands=[v],
                            dest=Value(name="y", dtype=DataType.INT32))

        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        selector = ExtendedInstructionSelector(builder.program)
        assert not selector._is_fp64(instr)

    def test_is_fp64_float64(self):
        from scratchv.ir.types import Instruction, OpCode

        v = Value(name="x", dtype=DataType.FLOAT64)
        instr = Instruction(opcode=OpCode.ADD, operands=[v],
                            dest=Value(name="y", dtype=DataType.FLOAT64))

        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        selector = ExtendedInstructionSelector(builder.program)
        assert selector._is_fp64(instr)

    def test_is_fp64_fp64_disabled(self):
        from scratchv.ir.types import Instruction, OpCode

        v = Value(name="x", dtype=DataType.FLOAT64)
        instr = Instruction(opcode=OpCode.ADD, operands=[v],
                            dest=Value(name="y", dtype=DataType.FLOAT64))

        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        selector = ExtendedInstructionSelector(
            builder.program, enable_fp64=False)
        assert not selector._is_fp64(instr)


class TestLoadConst:
    """Tests for load_const selection."""

    def test_load_const_int(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        c = builder.load_const(42)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.LI in ops


class TestAllBaseOps:
    """Verify all base ops from the parent selector still work."""

    def test_load_store(self):

        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        ptr = builder.make_value(name="ptr")
        loaded = builder.load(ptr)
        builder.store(loaded, ptr)
        builder.ret(loaded)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.LW in ops
        assert MachineOp.SW in ops


# ═══════════════════════════════════════════════════════════════════════
# [Topic 28] Extended instruction selection
# ═══════════════════════════════════════════════════════════════════════

NEW_OPCODES = [
    "sqrt", "min", "max", "abs", "idiv", "rem", "mod",
    "load_f64", "store_f64", "load_const_f64",
    "fadd_d", "fsub_d", "fmul_d", "fdiv_d",
    "fcmp_l_d", "fcmp_eq_d", "fcvt_s_d", "fcvt_d_s",
]

DISPATCH_EXPECTED_OP = {
    "sqrt": MachineOp.CALL,
    "min": MachineOp.SLT,
    "max": MachineOp.SLT,  # branchless sequence (see Topic 28 review F4)
    "abs": MachineOp.SRAI,
    "idiv": MachineOp.DIV,
    "rem": MachineOp.REM,
    "mod": MachineOp.REM,
    "load_f64": MachineOp.FLD,
    "store_f64": MachineOp.FSD,
    "load_const_f64": MachineOp.LI_D,
    "fadd_d": MachineOp.FADD_D,
    "fsub_d": MachineOp.FSUB_D,
    "fmul_d": MachineOp.FMUL_D,
    "fdiv_d": MachineOp.FDIV_D,
    "fcmp_l_d": MachineOp.FLT_D,
    "fcmp_eq_d": MachineOp.FEQ_D,
    "fcvt_s_d": MachineOp.FCVT_S_D,
    "fcvt_d_s": MachineOp.FCVT_D_S,
}


def _make_single_op_program(op):
    """Build a minimal valid program containing one new-op instruction."""
    builder = IRBuilder()
    builder.new_function("test")
    builder.new_block("entry")
    i32 = DataType.INT32
    f32 = DataType.FLOAT32
    f64 = DataType.FLOAT64
    a = builder.make_value(name="a", dtype=i32)
    b = builder.make_value(name="b", dtype=i32)
    x = builder.make_value(name="x", dtype=f32)
    dx = builder.make_value(name="dx", dtype=f64)
    dy = builder.make_value(name="dy", dtype=f64)

    if op == "sqrt":
        builder.sqrt(x)
    elif op == "min":
        builder.min(a, b)
    elif op == "max":
        builder.max(a, b)
    elif op == "abs":
        builder.abs(a)
    elif op == "idiv":
        builder.idiv(a, b)
    elif op == "rem":
        builder.rem(a, b)
    elif op == "mod":
        builder.mod(a, b)
    elif op == "load_f64":
        builder.load_f64(a)
    elif op == "store_f64":
        builder.store_f64(a, dx)
    elif op == "load_const_f64":
        builder.load_const_f64(1.5)
    elif op == "fadd_d":
        builder.fadd_d(dx, dy)
    elif op == "fsub_d":
        builder.fsub_d(dx, dy)
    elif op == "fmul_d":
        builder.fmul_d(dx, dy)
    elif op == "fdiv_d":
        builder.fdiv_d(dx, dy)
    elif op == "fcmp_l_d":
        builder.fcmp_l_d(dx, dy)
    elif op == "fcmp_eq_d":
        builder.fcmp_eq_d(dx, dy)
    elif op == "fcvt_s_d":
        builder.fcvt_s_d(dx)
    elif op == "fcvt_d_s":
        builder.fcvt_d_s(x)
    else:
        raise AssertionError(f"unhandled opcode {op}")
    return builder


def _clean_asm_lines(asm: str) -> list:
    """Strip comments and blank lines from emitted assembly."""
    return [
        line.split("#")[0].strip()
        for line in asm.splitlines()
        if line.split("#")[0].strip()
    ]


def _is_immediate_token(token: str) -> bool:
    try:
        int(token, 0)
        return True
    except ValueError:
        return False


def _eval_int_sequence(asm: str, regs: dict) -> dict:
    """Evaluate the integer sequences emitted by the extended selector.

    Test-local mini evaluator over the emitted assembly (vreg names act as
    registers).  It exists so literal materialization and min/max/abs
    semantics can be checked without an external simulator.
    """
    values = {"x0": 0}
    values.update(regs)

    def val(token: str) -> int:
        if _is_immediate_token(token):
            return int(token, 0)
        return values.get(token, 0)

    for line in _clean_asm_lines(asm):
        tokens = line.replace(",", " ").split()
        if not tokens or line.startswith(".") or line.endswith(":"):
            continue
        op, args = tokens[0], tokens[1:]
        if op in ("li", "mv"):
            values[args[0]] = val(args[1])
        elif op == "slt":
            values[args[0]] = 1 if val(args[1]) < val(args[2]) else 0
        elif op == "sub":
            values[args[0]] = val(args[1]) - val(args[2])
        elif op == "and":
            values[args[0]] = val(args[1]) & val(args[2])
        elif op == "add":
            values[args[0]] = val(args[1]) + val(args[2])
        elif op == "srai":
            values[args[0]] = val(args[1]) >> val(args[2])
        elif op == "xor":
            values[args[0]] = val(args[1]) ^ val(args[2])
        else:
            raise AssertionError(f"unexpected integer sequence op: {line}")
    return values


def _allocated_pipeline(builder, **selector_kwargs):
    """Select, greedily allocate and emit one builder program."""
    instrs = ExtendedInstructionSelector(
        builder.program, **selector_kwargs).run()
    allocated = RegisterAllocator(instrs, mode="greedy").run()
    return allocated, AsmEmitter(allocated).emit()


class TestDispatchCoverage:
    """Every new opcode must dispatch to its handler (Topic 28)."""

    def test_new_opcode_handler_exists(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        selector = ExtendedInstructionSelector(builder.program)
        for value in NEW_OPCODES:
            assert hasattr(selector, f"_select_{value}"), value

    def test_handler_and_opcode_one_to_one(self):
        enum_values = {op.value for op in OpCode}
        handlers = {
            name[len("_select_"):]
            for name in dir(ExtendedInstructionSelector)
            if name.startswith("_select_")
        }
        for value in NEW_OPCODES:
            assert value in enum_values
            assert value in handlers
        # No [Topic 28] handler without a matching OpCode member.
        new_handlers = {h for h in handlers if h in set(NEW_OPCODES)}
        assert new_handlers == set(NEW_OPCODES)

    @pytest.mark.parametrize("op", NEW_OPCODES)
    def test_dispatch_table(self, op):
        builder = _make_single_op_program(op)
        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = {i.op for i in instrs if i.op != MachineOp.LABEL}
        assert DISPATCH_EXPECTED_OP[op] in ops

    def test_unique_temps_two_mins(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        b = builder.make_value(name="b", dtype=DataType.INT32)
        c = builder.make_value(name="c", dtype=DataType.INT32)
        d = builder.make_value(name="d", dtype=DataType.INT32)
        builder.min(a, b)
        builder.min(c, d)

        instrs = ExtendedInstructionSelector(builder.program).run()
        names = [
            i.dst.value for i in instrs
            if i.dst is not None and i.dst.value.startswith("__min")
        ]
        # ``and`` writes its result in place, so the sub temp repeats.
        unique = list(dict.fromkeys(names))
        assert len(unique) == 6
        assert sorted(
            unique, key=lambda n: int(n.rsplit("_", 1)[1])) == unique


class TestAsmText:
    """Assembly text for MIN/ABS/SQRT/f64 sequences (Topic 28)."""

    def test_min_branchless_asm(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        b = builder.make_value(name="b", dtype=DataType.INT32)
        dest = builder.min(a, b)

        instrs = ExtendedInstructionSelector(builder.program).run()
        lines = _clean_asm_lines(AsmEmitter(instrs).emit())
        seq = [ln for ln in lines if "__min" in ln]
        assert seq == [
            "slt __min_slt_1, a, b",
            "sub __min_mask_2, x0, __min_slt_1",
            "sub __min_sub_3, a, b",
            "and __min_sub_3, __min_sub_3, __min_mask_2",
            f"add {dest.name}, b, __min_sub_3",
        ]

    def test_max_branchless_asm(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        b = builder.make_value(name="b", dtype=DataType.INT32)
        dest = builder.max(a, b)

        instrs = ExtendedInstructionSelector(builder.program).run()
        lines = _clean_asm_lines(AsmEmitter(instrs).emit())
        seq = [ln for ln in lines if "__max" in ln]
        assert seq == [
            "slt __max_slt_1, a, b",
            "sub __max_mask_2, x0, __max_slt_1",
            "sub __max_sub_3, b, a",
            "and __max_sub_3, __max_sub_3, __max_mask_2",
            f"add {dest.name}, a, __max_sub_3",
        ]
        ops = {i.op for i in instrs if i.op != MachineOp.LABEL}
        assert MachineOp.MAX not in ops

    def test_abs_branchless_asm(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        x = builder.make_value(name="x", dtype=DataType.INT32)
        dest = builder.abs(x)

        instrs = ExtendedInstructionSelector(builder.program).run()
        lines = _clean_asm_lines(AsmEmitter(instrs).emit())
        seq = [ln for ln in lines if "__abs" in ln]
        assert seq == [
            "srai __abs_srai_1, x, 31",
            "xor __abs_xor_2, x, __abs_srai_1",
            f"sub {dest.name}, __abs_xor_2, __abs_srai_1",
        ]

    def test_sqrt_software_uses_a0_and_call(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        x = builder.make_value(name="x", dtype=DataType.FLOAT32)
        dest = builder.sqrt(x)

        instrs = ExtendedInstructionSelector(builder.program).run()
        lines = _clean_asm_lines(AsmEmitter(instrs).emit())
        idx = lines.index("mv a0, x")
        assert lines[idx + 1] == "call sqrtf"
        assert lines[idx + 2] == f"mv {dest.name}, a0"

    def test_sqrt_software_f64_calls_sqrt(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        dx = builder.make_value(name="dx", dtype=DataType.FLOAT64)
        builder.sqrt(dx, dtype=DataType.FLOAT64)

        instrs = ExtendedInstructionSelector(builder.program).run()
        lines = _clean_asm_lines(AsmEmitter(instrs).emit())
        assert "call sqrt" in lines

    @pytest.mark.parametrize("value,bits", [
        (2.5, 1075838976),   # 0x40200000
        (4.0, 1082130432),   # 0x40800000
    ])
    def test_sqrt_immediate_uses_bit_pattern(self, value, bits):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        imm = builder.make_const(value, dtype=DataType.FLOAT32)
        dest = builder.sqrt(imm)

        instrs = ExtendedInstructionSelector(builder.program).run()
        lines = _clean_asm_lines(AsmEmitter(instrs).emit())
        assert f"li a0, {bits}" in lines
        assert "call sqrtf" in lines
        assert f"mv {dest.name}, a0" in lines

    def test_sqrt_immediate_is_not_truncated(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        imm = builder.make_const(2.5, dtype=DataType.FLOAT32)
        builder.sqrt(imm)

        instrs = ExtendedInstructionSelector(builder.program).run()
        lines = _clean_asm_lines(AsmEmitter(instrs).emit())
        # Regression: the old path emitted ``li a0, 2`` for 2.5.
        assert "li a0, 2" not in lines

    def test_sqrt_hardware_immediate_is_materialized(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        imm = builder.make_const(2.0, dtype=DataType.FLOAT32)
        dest = builder.sqrt(imm)

        instrs = ExtendedInstructionSelector(
            builder.program, use_hardware_sqrt=True).run()
        lines = _clean_asm_lines(AsmEmitter(instrs).emit())
        assert "li __sqrt_imm_1, 1073741824" in lines
        assert f"fsqrt.s {dest.name}, __sqrt_imm_1" in lines
        # No hardware instruction may carry an immediate source operand.
        assert not any(
            ln.startswith("fsqrt.s") and "__sqrt_imm" not in ln
            for ln in lines)

    def test_sqrt_hardware_f32_f64(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        x = builder.make_value(name="x", dtype=DataType.FLOAT32)
        dx = builder.make_value(name="dx", dtype=DataType.FLOAT64)
        d32 = builder.sqrt(x)
        d64 = builder.sqrt(dx, dtype=DataType.FLOAT64)

        instrs = ExtendedInstructionSelector(
            builder.program, use_hardware_sqrt=True).run()
        lines = _clean_asm_lines(AsmEmitter(instrs).emit())
        assert f"fsqrt.s {d32.name}, x" in lines
        assert f"fsqrt.d {d64.name}, dx" in lines

    def test_fp64_dtype_driven_add(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        dx = builder.make_value(name="dx", dtype=DataType.FLOAT64)
        dy = builder.make_value(name="dy", dtype=DataType.FLOAT64)
        dest = builder.make_value(name="dd", dtype=DataType.FLOAT64)
        builder._emit(OpCode.ADD, dest, [dx, dy])

        instrs = ExtendedInstructionSelector(builder.program).run()
        lines = _clean_asm_lines(AsmEmitter(instrs).emit())
        assert f"fadd.d {dest.name}, dx, dy" in lines

    def test_store_f64_operand_order(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        ptr = builder.make_value(name="p", dtype=DataType.INT32)
        val = builder.make_value(name="v", dtype=DataType.FLOAT64)
        builder.store_f64(ptr, val)

        instrs = ExtendedInstructionSelector(builder.program).run()
        lines = _clean_asm_lines(AsmEmitter(instrs).emit())
        assert "fsd v, p" in lines


class TestFp64Gate:
    """fail-loud when enable_fp64=False (Topic 28)."""

    def test_fadd_d_without_fp64_raises(self):
        builder = _make_single_op_program("fadd_d")
        selector = ExtendedInstructionSelector(
            builder.program, enable_fp64=False)
        with pytest.raises(ValueError, match="enable_fp64"):
            selector.run()

    def test_f64_add_without_fp64_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        dx = builder.make_value(name="dx", dtype=DataType.FLOAT64)
        dy = builder.make_value(name="dy", dtype=DataType.FLOAT64)
        dest = builder.make_value(name="dd", dtype=DataType.FLOAT64)
        builder._emit(OpCode.ADD, dest, [dx, dy])

        selector = ExtendedInstructionSelector(
            builder.program, enable_fp64=False)
        with pytest.raises(ValueError, match="enable_fp64"):
            selector.run()

    def test_f64_min_without_fp64_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        dx = builder.make_value(name="dx", dtype=DataType.FLOAT64)
        dy = builder.make_value(name="dy", dtype=DataType.FLOAT64)
        builder.min(dx, dy, dtype=DataType.FLOAT64)

        selector = ExtendedInstructionSelector(
            builder.program, enable_fp64=False)
        with pytest.raises(ValueError, match="enable_fp64"):
            selector.run()

    def test_f64_sqrt_without_fp64_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        dx = builder.make_value(name="dx", dtype=DataType.FLOAT64)
        builder.sqrt(dx, dtype=DataType.FLOAT64)

        selector = ExtendedInstructionSelector(
            builder.program, enable_fp64=False)
        with pytest.raises(ValueError, match="enable_fp64"):
            selector.run()

    def test_integer_ops_without_fp64_still_work(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        b = builder.make_value(name="b", dtype=DataType.INT32)
        builder.add(a, b)

        selector = ExtendedInstructionSelector(
            builder.program, enable_fp64=False)
        instrs = selector.run()
        lines = _clean_asm_lines(AsmEmitter(instrs).emit())
        assert any(ln.startswith("add ") for ln in lines)


class TestIllegalDtype:
    """dtype guards raise ValueError with the opcode name (Topic 28)."""

    def test_sqrt_int_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        builder.sqrt(a, dtype=DataType.INT32)
        with pytest.raises(ValueError, match="sqrt"):
            ExtendedInstructionSelector(builder.program).run()

    def test_min_f32_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        x = builder.make_value(name="x", dtype=DataType.FLOAT32)
        y = builder.make_value(name="y", dtype=DataType.FLOAT32)
        builder.min(x, y, dtype=DataType.FLOAT32)
        with pytest.raises(ValueError, match="min"):
            ExtendedInstructionSelector(builder.program).run()

    def test_idiv_f64_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        dx = builder.make_value(name="dx", dtype=DataType.FLOAT64)
        dy = builder.make_value(name="dy", dtype=DataType.FLOAT64)
        builder.idiv(dx, dy)
        with pytest.raises(ValueError, match="idiv"):
            ExtendedInstructionSelector(builder.program).run()


class TestLoadConstF64:
    """Exact IEEE-754 bit patterns for f64 constants (Topic 28)."""

    @pytest.mark.parametrize("value,bits", [
        (1.5, 4609434218613702656),
        (2.0, 4611686018427387904),
        (-0.0, 9223372036854775808),
    ])
    def test_exact_bits(self, value, bits):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        builder.load_const_f64(value)

        instrs = ExtendedInstructionSelector(builder.program).run()
        li_d = [i for i in instrs if i.op == MachineOp.LI_D]
        assert len(li_d) == 1
        assert li_d[0].src1.value == bits

    def test_missing_value_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        dest = builder.make_value(dtype=DataType.FLOAT64)
        builder._emit(OpCode.LOAD_CONST_F64, dest, attrs={})
        with pytest.raises(ValueError, match="load_const_f64"):
            ExtendedInstructionSelector(builder.program).run()

    def test_non_numeric_value_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        dest = builder.make_value(dtype=DataType.FLOAT64)
        builder._emit(OpCode.LOAD_CONST_F64, dest, value="nope")
        with pytest.raises(ValueError, match="load_const_f64"):
            ExtendedInstructionSelector(builder.program).run()


class TestAsmEmitterFailLoud:
    """AsmEmitter must not silently drop unmapped MachineOps (Topic 28)."""

    def test_unknown_machine_op_raises(self):
        from scratchv.backend.machine_types import (
            MachineInstr, MachineOp as MOp,
        )
        instrs = [MachineInstr(MOp.GLOBL, comment="foo")]
        with pytest.raises(ValueError, match="GLOBL"):
            AsmEmitter(instrs).emit()


class TestSupportedOpsNew:
    """supported_ops reflects the 18 new opcodes (Topic 28)."""

    def test_new_ops_present(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        ops = ExtendedInstructionSelector(builder.program).supported_ops
        for value in NEW_OPCODES:
            assert value in ops

    def test_fp64_ops_gated(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        ops = ExtendedInstructionSelector(
            builder.program, enable_fp64=False).supported_ops
        for value in ("sqrt", "min", "max", "abs",
                      "idiv", "rem", "mod"):
            assert value in ops
        for value in ("load_f64", "fadd_d", "fcvt_d_s"):
            assert value not in ops


class TestFloatLiteralMaterialization:
    """F1: float literals keep their exact value (Topic 28 review)."""

    @pytest.mark.parametrize("value", [2.5, -2.5, 1.5])
    def test_sqrt_f64_immediate_raises(self, value):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        imm = builder.make_const(value, dtype=DataType.FLOAT64)
        builder.sqrt(imm, dtype=DataType.FLOAT64)
        with pytest.raises(ValueError, match="FLOAT64 literal"):
            ExtendedInstructionSelector(builder.program).run()

    def test_sqrt_hardware_f64_immediate_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        imm = builder.make_const(2.5, dtype=DataType.FLOAT64)
        builder.sqrt(imm, dtype=DataType.FLOAT64)
        with pytest.raises(ValueError, match="FLOAT64 literal"):
            ExtendedInstructionSelector(
                builder.program, use_hardware_sqrt=True).run()

    def test_fadd_d_f64_immediate_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        dx = builder.make_value(name="dx", dtype=DataType.FLOAT64)
        imm = builder.make_const(1.5, dtype=DataType.FLOAT64)
        builder.fadd_d(imm, dx)
        with pytest.raises(ValueError, match="FLOAT64 literal"):
            ExtendedInstructionSelector(builder.program).run()

    def test_store_f64_f64_immediate_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        ptr = builder.make_value(name="p", dtype=DataType.INT32)
        imm = builder.make_const(1.5, dtype=DataType.FLOAT64)
        builder.store_f64(ptr, imm)
        with pytest.raises(ValueError, match="FLOAT64 literal"):
            ExtendedInstructionSelector(builder.program).run()

    def test_load_const_f64_result_is_an_operand(self):
        """A value materialized by load_const_f64 is a vreg, not a literal."""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        dx = builder.make_value(name="dx", dtype=DataType.FLOAT64)
        const = builder.load_const_f64(1.5)
        dest = builder.fadd_d(const, dx)

        instrs = ExtendedInstructionSelector(builder.program).run()
        lines = _clean_asm_lines(AsmEmitter(instrs).emit())
        assert f"fadd.d {dest.name}, {const.name}, dx" in lines


class TestDtypeSignatures:
    """F2/F5: dest and operands must match the opcode dtype signature."""

    def test_min_mixed_i32_f64_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        dx = builder.make_value(name="dx", dtype=DataType.FLOAT64)
        dest = builder.make_value(name="r", dtype=DataType.INT32)
        builder._emit(OpCode.MIN, dest, [a, dx])
        with pytest.raises(ValueError, match="min"):
            ExtendedInstructionSelector(builder.program).run()

    def test_max_mixed_i64_f64_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT64)
        dx = builder.make_value(name="dx", dtype=DataType.FLOAT64)
        dest = builder.make_value(name="r", dtype=DataType.INT64)
        builder._emit(OpCode.MAX, dest, [a, dx])
        with pytest.raises(ValueError, match="max"):
            ExtendedInstructionSelector(builder.program).run()

    def test_sqrt_f64_operand_f32_dest_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        dx = builder.make_value(name="dx", dtype=DataType.FLOAT64)
        builder.sqrt(dx)  # dest defaults to FLOAT32
        with pytest.raises(ValueError, match="sqrt"):
            ExtendedInstructionSelector(builder.program).run()

    def test_abs_f64_operand_i64_dest_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        dx = builder.make_value(name="dx", dtype=DataType.FLOAT64)
        builder.abs(dx, dtype=DataType.INT64)
        with pytest.raises(ValueError, match="abs"):
            ExtendedInstructionSelector(builder.program).run()

    def test_fadd_d_i32_operands_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        c = builder.make_value(name="c", dtype=DataType.INT32)
        dest = builder.make_value(name="r", dtype=DataType.FLOAT64)
        builder._emit(OpCode.FADD_D, dest, [a, c])
        with pytest.raises(ValueError, match="fadd_d"):
            ExtendedInstructionSelector(builder.program).run()

    def test_fcmp_l_d_f32_operands_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        x = builder.make_value(name="x", dtype=DataType.FLOAT32)
        y = builder.make_value(name="y", dtype=DataType.FLOAT32)
        dest = builder.make_value(name="r", dtype=DataType.INT32)
        builder._emit(OpCode.FCMP_L_D, dest, [x, y])
        with pytest.raises(ValueError, match="fcmp_l_d"):
            ExtendedInstructionSelector(builder.program).run()

    def test_fcmp_l_d_f64_dest_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        dx = builder.make_value(name="dx", dtype=DataType.FLOAT64)
        dy = builder.make_value(name="dy", dtype=DataType.FLOAT64)
        dest = builder.make_value(name="r", dtype=DataType.FLOAT64)
        builder._emit(OpCode.FCMP_L_D, dest, [dx, dy])
        with pytest.raises(ValueError, match="fcmp_l_d"):
            ExtendedInstructionSelector(builder.program).run()

    def test_fcvt_s_d_f32_operand_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        x = builder.make_value(name="x", dtype=DataType.FLOAT32)
        dest = builder.make_value(name="r", dtype=DataType.FLOAT32)
        builder._emit(OpCode.FCVT_S_D, dest, [x])
        with pytest.raises(ValueError, match="fcvt_s_d"):
            ExtendedInstructionSelector(builder.program).run()

    def test_fcvt_d_s_f64_operand_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        dx = builder.make_value(name="dx", dtype=DataType.FLOAT64)
        dest = builder.make_value(name="r", dtype=DataType.FLOAT64)
        builder._emit(OpCode.FCVT_D_S, dest, [dx])
        with pytest.raises(ValueError, match="fcvt_d_s"):
            ExtendedInstructionSelector(builder.program).run()

    def test_load_f64_i32_dest_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        ptr = builder.make_value(name="p", dtype=DataType.INT32)
        dest = builder.make_value(name="r", dtype=DataType.INT32)
        builder._emit(OpCode.LOAD_F64, dest, [ptr])
        with pytest.raises(ValueError, match="load_f64"):
            ExtendedInstructionSelector(builder.program).run()

    def test_store_f64_f64_addr_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        addr = builder.make_value(name="p", dtype=DataType.FLOAT64)
        val = builder.make_value(name="v", dtype=DataType.FLOAT64)
        builder.store_f64(addr, val)
        with pytest.raises(ValueError, match="address"):
            ExtendedInstructionSelector(builder.program).run()

    def test_store_f64_i32_value_raises(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        ptr = builder.make_value(name="p", dtype=DataType.INT32)
        val = builder.make_value(name="v", dtype=DataType.INT32)
        builder.store_f64(ptr, val)
        with pytest.raises(ValueError, match="FLOAT64 value"):
            ExtendedInstructionSelector(builder.program).run()


FP64_MISSING_DEST_CASES = [
    ("fadd_d", OpCode.FADD_D,
     [DataType.FLOAT64, DataType.FLOAT64], {}),
    ("fsub_d", OpCode.FSUB_D,
     [DataType.FLOAT64, DataType.FLOAT64], {}),
    ("fmul_d", OpCode.FMUL_D,
     [DataType.FLOAT64, DataType.FLOAT64], {}),
    ("fdiv_d", OpCode.FDIV_D,
     [DataType.FLOAT64, DataType.FLOAT64], {}),
    ("fcmp_l_d", OpCode.FCMP_L_D,
     [DataType.FLOAT64, DataType.FLOAT64], {}),
    ("fcmp_eq_d", OpCode.FCMP_EQ_D,
     [DataType.FLOAT64, DataType.FLOAT64], {}),
    ("fcvt_s_d", OpCode.FCVT_S_D, [DataType.FLOAT64], {}),
    ("fcvt_d_s", OpCode.FCVT_D_S, [DataType.FLOAT32], {}),
    ("load_f64", OpCode.LOAD_F64, [DataType.INT32], {}),
    ("load_const_f64", OpCode.LOAD_CONST_F64, [], {"value": 1.5}),
]


class TestFp64MissingDest:
    """F5: fp64 handlers must reject instructions without a dest."""

    @pytest.mark.parametrize(
        "opname,opcode,dtypes,attrs", FP64_MISSING_DEST_CASES,
        ids=[case[0] for case in FP64_MISSING_DEST_CASES])
    def test_missing_dest_raises(self, opname, opcode, dtypes, attrs):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        operands = [builder.make_value(dtype=d) for d in dtypes]
        builder._emit(opcode, None, operands, **attrs)
        with pytest.raises(ValueError, match=opname):
            ExtendedInstructionSelector(builder.program).run()


class TestIntegerMinMaxSemantics:
    """F3/F4: integer MIN/MAX/ABS sequences are semantically correct."""

    @pytest.mark.parametrize("a,b", [
        (5, 2), (2, 5), (0, 0), (-3, 2), (2, -3), (-7, -2),
    ])
    def test_min_registers(self, a, b):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        av = builder.make_value(name="a", dtype=DataType.INT32)
        bv = builder.make_value(name="b", dtype=DataType.INT32)
        dest = builder.min(av, bv)

        asm = AsmEmitter(ExtendedInstructionSelector(
            builder.program).run()).emit()
        values = _eval_int_sequence(asm, {av.name: a, bv.name: b})
        assert values[dest.name] == min(a, b)

    @pytest.mark.parametrize("a,b", [
        (5, 2), (2, 5), (0, 0), (-3, 2), (2, -3), (-7, -2),
    ])
    def test_max_registers(self, a, b):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        av = builder.make_value(name="a", dtype=DataType.INT32)
        bv = builder.make_value(name="b", dtype=DataType.INT32)
        dest = builder.max(av, bv)

        asm = AsmEmitter(ExtendedInstructionSelector(
            builder.program).run()).emit()
        values = _eval_int_sequence(asm, {av.name: a, bv.name: b})
        assert values[dest.name] == max(a, b)

    @pytest.mark.parametrize("a,const", [
        (5, 2), (1, 2), (-3, 2), (7, 2),
    ])
    def test_min_immediate_operand(self, a, const):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        av = builder.make_value(name="a", dtype=DataType.INT32)
        dest = builder.min(av, builder.make_const(const, dtype=DataType.INT32))

        asm = AsmEmitter(ExtendedInstructionSelector(
            builder.program).run()).emit()
        lines = _clean_asm_lines(asm)
        # Regression: the constant must not appear as a register operand.
        for line in lines:
            tokens = line.replace(",", " ").split()
            if tokens[0] in ("sub", "and", "add", "slt"):
                assert all(
                    not _is_immediate_token(tok) for tok in tokens[2:]
                ), line
        assert any(ln.startswith("li ") for ln in lines)
        values = _eval_int_sequence(asm, {av.name: a})
        assert values[dest.name] == min(a, const)

    @pytest.mark.parametrize("a,const", [
        (5, 2), (1, 2), (-3, 2), (7, 2),
    ])
    def test_max_immediate_operand(self, a, const):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        av = builder.make_value(name="a", dtype=DataType.INT32)
        dest = builder.max(av, builder.make_const(const, dtype=DataType.INT32))

        asm = AsmEmitter(ExtendedInstructionSelector(
            builder.program).run()).emit()
        lines = _clean_asm_lines(asm)
        for line in lines:
            tokens = line.replace(",", " ").split()
            if tokens[0] in ("sub", "and", "add", "slt"):
                assert all(
                    not _is_immediate_token(tok) for tok in tokens[2:]
                ), line
        assert any(ln.startswith("li ") for ln in lines)
        values = _eval_int_sequence(asm, {av.name: a})
        assert values[dest.name] == max(a, const)

    @pytest.mark.parametrize("a", [-7, -1, 0, 1, 7])
    def test_abs(self, a):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        av = builder.make_value(name="a", dtype=DataType.INT32)
        dest = builder.abs(av)

        asm = AsmEmitter(ExtendedInstructionSelector(
            builder.program).run()).emit()
        values = _eval_int_sequence(asm, {av.name: a})
        assert values[dest.name] == abs(a)


class TestMinMaxExecution:
    """F3/F4 end-to-end: allocated MIN/MAX/ABS executes correctly."""

    def setup_method(self):
        pytest.importorskip("tinyfive")

    @staticmethod
    def _execute(allocated, asm, inputs, result_comment):
        result = next(i for i in allocated if i.comment == result_comment)
        binary = assemble_to_binary(asm)
        words = [
            int.from_bytes(binary[i:i + 4], "little")
            for i in range(0, len(binary), 4)
        ]
        machine = ProfiledMachine(mem_size=4096)
        machine.load_binary(words, origin=0)
        for name, value in inputs.items():
            machine.set_reg(REG_MAP[name], value)
        machine.run(instructions=len(words), start=0, strict=True)
        assert machine.last_error is None
        return machine.get_reg(REG_MAP[result.dst.value])

    def test_min_immediate_executes(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        av = builder.make_value(name="a", dtype=DataType.INT32)
        builder.min(av, builder.make_const(2, dtype=DataType.INT32))
        allocated, asm = _allocated_pipeline(builder)

        slt = next(i for i in allocated if i.comment == "min: slt")
        for value, expected in [(5, 2), (1, 1), (2, 2), (-4, -4)]:
            got = self._execute(
                allocated, asm, {slt.src1.value: value}, "min result")
            assert got == expected

    def test_max_registers_executes(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        av = builder.make_value(name="a", dtype=DataType.INT32)
        bv = builder.make_value(name="b", dtype=DataType.INT32)
        builder.max(av, bv)
        allocated, asm = _allocated_pipeline(builder)

        slt = next(i for i in allocated if i.comment == "max: slt")
        for a, b in [(3, 7), (7, 3), (-2, 5), (5, -2)]:
            got = self._execute(
                allocated, asm,
                {slt.src1.value: a, slt.src2.value: b}, "max result")
            assert got == max(a, b)

    def test_abs_executes(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        av = builder.make_value(name="a", dtype=DataType.INT32)
        builder.abs(av)
        allocated, asm = _allocated_pipeline(builder)

        srai = next(
            i for i in allocated if i.comment == "abs: srai 31")
        for value in (-7, -1, 0, 5):
            got = self._execute(
                allocated, asm, {srai.src1.value: value}, "abs: sub")
            assert got == abs(value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
