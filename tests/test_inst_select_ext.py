"""Tests for Extended Instruction Selector."""

import pytest
from scratchv.backend.inst_select_ext import ExtendedInstructionSelector
from scratchv.backend.target_features import TargetFeatures
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Instruction, OpCode
from scratchv.backend.register_alloc import MachineOp


class TestExtendedSelectorBasic:
    """Tests for the extended instruction selector."""

    def test_creation(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        b = builder.make_value(name="b", dtype=DataType.INT32)
        c = builder.add(a, b)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        assert selector.features == TargetFeatures("rv32imfd", "ilp32d")

    def test_f_target_rejects_double_precision(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        builder.ret(builder.load_const(1.0, DataType.FLOAT64))
        selector = ExtendedInstructionSelector(
            builder.program, features=TargetFeatures("rv32imf", "ilp32f"))
        with pytest.raises(ValueError, match="requires the RISC-V D extension"):
            selector.run()

    def test_run_basic(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        b = builder.make_value(name="b", dtype=DataType.INT32)
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
        a = builder.make_value(name="a", dtype=DataType.INT32)
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
        a = builder.make_value(name="a", dtype=DataType.INT32)
        b = builder.make_value(name="b", dtype=DataType.INT32)
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


class TestExtendedSelectorNeg:
    """Tests for neg instruction handling."""

    def test_neg(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        c = builder.neg(a)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.SUB in ops


class TestExtendedSelectorNumericOps:
    @pytest.mark.parametrize(
        ("method", "dtype", "expected"),
        [
            ("min", DataType.FLOAT32, MachineOp.FMIN_S),
            ("min", DataType.FLOAT64, MachineOp.FMIN_D),
            ("max", DataType.FLOAT32, MachineOp.FMAX_S),
            ("max", DataType.FLOAT64, MachineOp.FMAX_D),
            ("abs", DataType.FLOAT32, MachineOp.FABS_S),
            ("abs", DataType.FLOAT64, MachineOp.FABS_D),
        ],
    )
    def test_float_min_max_abs_select_typed_instruction(
        self, method, dtype, expected,
    ):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        lhs = builder.load_const(4.0, dtype)
        rhs = builder.load_const(2.0, dtype)
        args = (lhs,) if method == "abs" else (lhs, rhs)
        result = getattr(builder, method)(*args)
        builder.ret(result)

        instructions = ExtendedInstructionSelector(builder.program).run()

        selected = next(item for item in instructions if item.op is expected)
        assert selected.dst.value == result.name
        assert selected.src1.value == lhs.name
        if method != "abs":
            assert selected.src2.value == rhs.name

    def test_rem_rejects_non_i32_ir_even_without_verifier(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        lhs = builder.make_value(name="lhs", dtype=DataType.FLOAT32)
        rhs = builder.make_value(name="rhs", dtype=DataType.FLOAT32)
        builder.current_block.add(
            Instruction(
                OpCode.REM,
                builder.make_value(name="result", dtype=DataType.FLOAT32),
                [lhs, rhs],
            )
        )

        with pytest.raises(ValueError, match="rem does not support f32"):
            ExtendedInstructionSelector(builder.program).run()


class TestLoadConst:
    """Tests for load_const selection."""

    def test_load_const_int(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        c = builder.load_const(42, dtype=DataType.INT32)
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
        ptr = builder.make_value(name="ptr", dtype=DataType.FLOAT32)
        loaded = builder.load(ptr)
        builder.store(ptr, loaded)
        builder.ret(loaded)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.FLW in ops
        assert MachineOp.FSW in ops


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
