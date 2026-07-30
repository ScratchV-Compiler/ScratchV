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

    def test_neg_f32(self):
        """f32 neg → FNEG_S (default dtype is f32)."""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a")
        c = builder.neg(a)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.FNEG_S in ops

    def test_neg_i32(self):
        """i32 neg → SUB x0, rs."""
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


class TestDivRem:
    """Tests for div, rem, mod instruction selection."""

    def test_div_int32(self):
        """整数除法 → MachineOp.DIV"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        b = builder.make_value(name="b", dtype=DataType.INT32)
        c = builder.div(a, b)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.DIV in ops

    def test_div_float32(self):
        """f32 除法 → MachineOp.FDIV_S"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT32)
        b = builder.make_value(name="b", dtype=DataType.FLOAT32)
        c = builder.div(a, b)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.FDIV_S in ops

    def test_div_float64(self):
        """f64 除法 → MachineOp.FDIV_D"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT64)
        b = builder.make_value(name="b", dtype=DataType.FLOAT64)
        c = builder.div(a, b)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.FDIV_D in ops

    def test_rem_int32(self):
        """整数取余 → MachineOp.REM"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        b = builder.make_value(name="b", dtype=DataType.INT32)
        c = builder.rem(a, b)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.REM in ops

    def test_div_by_constant(self):
        """常量除法 → 常量折叠"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_const(10, dtype=DataType.INT32)
        b = builder.make_const(2, dtype=DataType.INT32)
        c = builder.div(a, b)
        builder.ret(c)

        from scratchv.optimizer.constant_folding import ConstantFolder
        folder = ConstantFolder(builder.program)
        num_folded = folder.run()
        assert num_folded == 1

    def test_rem_by_constant(self):
        """常量取余 → 常量折叠"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_const(10, dtype=DataType.INT32)
        b = builder.make_const(3, dtype=DataType.INT32)
        c = builder.rem(a, b)
        builder.ret(c)

        from scratchv.optimizer.constant_folding import ConstantFolder
        folder = ConstantFolder(builder.program)
        num_folded = folder.run()
        assert num_folded == 1

    def test_div_by_zero_constant(self):
        """除零常量 → 不折叠（安全跳过）"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_const(10, dtype=DataType.INT32)
        b = builder.make_const(0, dtype=DataType.INT32)
        c = builder.div(a, b)
        builder.ret(c)

        from scratchv.optimizer.constant_folding import ConstantFolder
        folder = ConstantFolder(builder.program)
        num_folded = folder.run()
        assert num_folded == 0  # 除零不折叠

    def test_supported_ops_has_div_rem(self):
        """supported_ops 包含 div/rem/mod"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        selector = ExtendedInstructionSelector(builder.program)
        ops = selector.supported_ops
        assert "div" in ops
        assert "idiv" in ops
        assert "rem" in ops
        assert "mod" in ops


class TestSqrtMinMaxAbs:
    """Tests for sqrt, min, max, abs instruction selection."""

    # --- sqrt ---

    def test_sqrt_f32(self):
        """f32 sqrt → 库调用 sqrtf"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT32)
        c = builder.sqrt(a)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.CALL in ops, "sqrt f32 should emit CALL sqrtf"

    def test_sqrt_f64(self):
        """f64 sqrt → 库调用 sqrt"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT64)
        c = builder.sqrt(a)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.CALL in ops, "sqrt f64 should emit CALL sqrt"

    def test_sqrt_hardware_f32(self):
        """f32 sqrt (硬件) → FSQRT_S"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT32)
        c = builder.sqrt(a)
        builder.ret(c)

        selector = ExtendedInstructionSelector(
            builder.program, use_hardware_sqrt=True)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.SQRT_S in ops, "hardware f32 sqrt should emit FSQRT_S"

    def test_sqrt_hardware_f64(self):
        """f64 sqrt (硬件) → SQRT_D"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT64)
        c = builder.sqrt(a)
        builder.ret(c)

        selector = ExtendedInstructionSelector(
            builder.program, use_hardware_sqrt=True)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.SQRT_D in ops, "hardware f64 sqrt should emit SQRT_D"

    # --- abs ---

    def test_abs_f32(self):
        """f32 abs → FABS_S"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT32)
        c = builder.abs(a)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.FABS_S in ops, "f32 abs should emit FABS_S"

    def test_abs_f64(self):
        """f64 abs → FABS_D"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT64)
        c = builder.abs(a)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.FABS_D in ops, "f64 abs should emit FABS_D"

    def test_abs_i32(self):
        """i32 abs → SRAI/XOR/SUB branchless 序列"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        c = builder.abs(a)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.SRAI in ops, "i32 abs should emit SRAI"
        assert MachineOp.XOR in ops, "i32 abs should emit XOR"
        assert MachineOp.SUB in ops, "i32 abs should emit SUB"

    # --- min ---

    def test_min_f32(self):
        """f32 min → FMIN_S"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT32)
        b = builder.make_value(name="b", dtype=DataType.FLOAT32)
        c = builder.min(a, b)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.FMIN_S in ops, "f32 min should emit FMIN_S"

    def test_min_f64(self):
        """f64 min → FMIN_D"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT64)
        b = builder.make_value(name="b", dtype=DataType.FLOAT64)
        c = builder.min(a, b)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.FMIN_D in ops, "f64 min should emit FMIN_D"

    def test_min_i32(self):
        """i32 min → SLT/SUB/AND/ADD branchless 序列"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        b = builder.make_value(name="b", dtype=DataType.INT32)
        c = builder.min(a, b)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.SLT in ops, "i32 min should emit SLT"
        assert MachineOp.AND in ops, "i32 min should emit AND"
        assert MachineOp.ADD in ops, "i32 min should emit ADD"

    # --- max ---

    def test_max_f32(self):
        """f32 max → FMAX_S"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT32)
        b = builder.make_value(name="b", dtype=DataType.FLOAT32)
        c = builder.max(a, b)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.FMAX_S in ops, "f32 max should emit FMAX_S"

    def test_max_f64(self):
        """f64 max → FMAX_D"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT64)
        b = builder.make_value(name="b", dtype=DataType.FLOAT64)
        c = builder.max(a, b)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.FMAX_D in ops, "f64 max should emit FMAX_D"

    def test_max_i32(self):
        """i32 max → MAX pseudo"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        b = builder.make_value(name="b", dtype=DataType.INT32)
        c = builder.max(a, b)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.MAX in ops, "i32 max should emit MAX pseudo"

    # --- neg (f32) ---

    def test_neg_f32(self):
        """f32 neg → FNEG_S"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT32)
        c = builder.neg(a)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.FNEG_S in ops, "f32 neg should emit FNEG_S"


class TestTransposeConcat:
    """Tests for transpose and concat instruction selection."""

    def test_transpose(self):
        """transpose → MV (no-op)"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT32)
        c = builder.transpose(a)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.MV in ops, "transpose should emit MV"

    def test_concat(self):
        """concat → MV (no-op)"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT32)
        c = builder.concat(a)
        builder.ret(c)

        selector = ExtendedInstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.MV in ops, "concat should emit MV"

    def test_transpose_base_selector(self):
        """基类 InstructionSelector 也能 dispatch transpose"""
        from scratchv.backend.instruction_select import InstructionSelector

        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT32)
        c = builder.transpose(a)
        builder.ret(c)

        selector = InstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.MV in ops, "base selector should dispatch transpose"

    def test_concat_base_selector(self):
        """基类 InstructionSelector 也能 dispatch concat"""
        from scratchv.backend.instruction_select import InstructionSelector

        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.FLOAT32)
        c = builder.concat(a)
        builder.ret(c)

        selector = InstructionSelector(builder.program)
        instrs = selector.run()
        ops = [i.op for i in instrs if i.op != MachineOp.LABEL]
        assert MachineOp.MV in ops, "base selector should dispatch concat"


class TestSupportedOpsExtended:
    """Verify supported_ops covers all new operators."""

    def test_supported_ops_has_sqrt_min_max_abs(self):
        """supported_ops 包含 sqrt/min/max/abs"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        selector = ExtendedInstructionSelector(builder.program)
        ops = selector.supported_ops
        assert "sqrt" in ops
        assert "min" in ops
        assert "max" in ops
        assert "abs" in ops

    def test_supported_ops_has_transpose_concat(self):
        """supported_ops 包含 transpose/concat"""
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        selector = ExtendedInstructionSelector(builder.program)
        ops = selector.supported_ops
        assert "transpose" in ops
        assert "concat" in ops


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
