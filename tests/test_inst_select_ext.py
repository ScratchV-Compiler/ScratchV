"""Tests for Extended Instruction Selector."""

import pytest
from scratchv.backend.inst_select_ext import ExtendedInstructionSelector
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value, DataType  # noqa: F401
from scratchv.backend.register_alloc import MachineOp


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
