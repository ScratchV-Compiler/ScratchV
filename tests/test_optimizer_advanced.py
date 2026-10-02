"""Tests for advanced optimizer passes: peephole, muladd_fusion, LICM."""

import numpy as np

from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, OpCode, Value
from scratchv.optimizer.peephole import IRPeepholeOptimizer
from scratchv.optimizer.muladd_fusion import MulAddFusion
from scratchv.optimizer.licm import LICM
from scratchv.verification.ir_interpreter import IRInterpreter


class TestIRPeepholeOptimizer:
    def test_eliminate_integer_add_zero_redirects_return(self):
        builder = IRBuilder()
        a = Value("a", DataType.INT32, shape=(2,))
        builder.new_function("test", [a])
        builder.new_block("entry")
        result = builder.add(a, builder.make_const(0, DataType.INT32))
        builder.ret(result)
        assert IRPeepholeOptimizer().optimize(builder.program) == 1
        assert builder.current_block.instructions[0].operands == [a]
        np.testing.assert_array_equal(
            IRInterpreter(builder.program).run({"a": np.array([2, -3], np.int32)}).return_value,
            [2, -3],
        )

    def test_eliminate_mul_one_after_checked_float_operation(self):
        builder = IRBuilder()
        a = Value("a", DataType.FLOAT32, shape=(2,))
        builder.new_function("test", [a])
        builder.new_block("entry")
        finite = builder.abs(a)
        result = builder.mul(finite, builder.make_const(1))
        builder.ret(result)
        assert IRPeepholeOptimizer().optimize(builder.program) == 1
        assert all(i.opcode != OpCode.MUL for i in builder.current_block.instructions)
        np.testing.assert_array_equal(
            IRInterpreter(builder.program).run({"a": np.array([2, -3], np.float32)}).return_value,
            [2, 3],
        )

    def test_preserve_tensor_mul_zero_shape_and_signed_zero(self):
        builder = IRBuilder()
        a = Value("a", DataType.FLOAT32, shape=(2,))
        builder.new_function("test", [a])
        builder.new_block("entry")
        result = builder.mul(a, builder.make_const(0))
        result.shape = a.shape
        builder.ret(result)
        assert IRPeepholeOptimizer().optimize(builder.program) == 0
        actual = IRInterpreter(builder.program).run({"a": np.array([2, -3], np.float32)}).return_value
        np.testing.assert_array_equal(actual, [0, 0])
        np.testing.assert_array_equal(np.signbit(actual), [False, True])


class TestMulAddFusion:
    def test_retain_binary_mul_add_contract(self):
        builder = IRBuilder()
        values = [Value(name, DataType.FLOAT32, shape=(2,)) for name in ("a", "b", "acc")]
        builder.new_function("test", values)
        builder.new_block("entry")
        builder.ret(builder.add(builder.mul(*values[:2]), values[2]))
        before = builder.program.dump()
        assert MulAddFusion().optimize(builder.program) == 0
        assert builder.program.dump() == before
        feed = {"a": np.array([2, 3], np.float32), "b": np.array([4, 5], np.float32),
                "acc": np.array([6, 7], np.float32)}
        np.testing.assert_array_equal(IRInterpreter(builder.program).run(feed).return_value, [14, 22])

    def test_no_fuse_without_mul(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a")
        b = builder.make_value(name="b")
        builder.ret(builder.add(a, b))
        assert MulAddFusion().optimize(builder.program) == 0


class TestLICM:
    def test_hoist_invariant(self):
        builder = IRBuilder()
        a = Value("a", DataType.INT32, shape=(2,))
        b = Value("b", DataType.INT32, shape=(2,))
        builder.new_function("test", [a, b])
        builder.new_block("entry")

        # Loop with invariant mul inside
        iv = builder.for_loop(0, 10)  # FOR
        # This mul depends on a, b defined outside the loop -> invariant
        c = builder.mul(a, b)
        builder.add(c, iv)  # this depends on iv → variant, keep
        builder.endfor()
        builder.ret()

        opt = LICM()
        count = opt.optimize(builder.program)
        assert count == 1  # mul should be hoisted

        block = builder.program.functions[0].blocks[0]
        # Find the FOR and check if mul is before it
        for_idx = next(i for i, instr in enumerate(block.instructions)
                       if instr.opcode == OpCode.FOR)
        mul_idx = next(i for i, instr in enumerate(block.instructions)
                       if instr.opcode == OpCode.MUL)
        assert mul_idx < for_idx, "MUL should be hoisted before FOR"

    def test_no_hoist_variant(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")

        iv = builder.for_loop(0, 10)
        # This add depends on iv (loop variant) → should not be hoisted
        builder.add(iv, builder.load_const(1, DataType.INT32))
        builder.endfor()
        builder.ret()

        opt = LICM()
        opt.optimize(builder.program)
        # The load_const IS invariant and gets hoisted.
        # But the add depending on iv stays in the loop.
        # Verify the add remains after the FOR.
        block = builder.program.functions[0].blocks[0]
        for_idx = next(i for i, instr in enumerate(block.instructions)
                       if instr.opcode == OpCode.FOR)
        add_instrs = [i for i in block.instructions if i.opcode == OpCode.ADD]
        # The ADD (variant) should still be inside the loop (after FOR)
        assert all(block.instructions.index(i) > for_idx for i in add_instrs)
