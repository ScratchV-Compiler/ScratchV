"""Tests for advanced optimizer passes: peephole, muladd_fusion, LICM."""

from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import OpCode
from scratchv.optimizer.peephole import IRPeepholeOptimizer
from scratchv.optimizer.muladd_fusion import MulAddFusion
from scratchv.optimizer.licm import LICM


class TestIRPeepholeOptimizer:
    def test_eliminate_addi_zero(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a")
        c = builder.make_value(name="c")
        # add a, 0  (no-op)
        zero = builder.make_value(name="_z", is_constant=True, const_value=0)
        builder._emit(OpCode.ADD, c, [a, zero])
        builder.ret(c)

        opt = IRPeepholeOptimizer(builder.program)
        opt.run()
        # The addi 0 should have been removed
        block = builder.program.functions[0].blocks[0]
        assert all(i.opcode != OpCode.ADD for i in block.instructions)

    def test_eliminate_mul_one(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a")
        c = builder.make_value(name="c")
        one = builder.make_value(name="_o", is_constant=True, const_value=1)
        builder._emit(OpCode.MUL, c, [a, one])
        builder.ret(c)

        opt = IRPeepholeOptimizer(builder.program)
        count = opt.run()
        assert count >= 1
        # MUL with 1 should be replaced
        block = builder.program.functions[0].blocks[0]
        has_mul = any(i.opcode == OpCode.MUL for i in block.instructions)
        assert not has_mul

    def test_eliminate_mul_zero(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a")
        c = builder.make_value(name="c")
        zero = builder.make_value(name="_z", is_constant=True, const_value=0)
        builder._emit(OpCode.MUL, c, [a, zero])
        builder.ret(c)

        opt = IRPeepholeOptimizer(builder.program)
        opt.run()
        block = builder.program.functions[0].blocks[0]
        mul_instrs = [i for i in block.instructions if i.opcode == OpCode.MUL]
        assert len(mul_instrs) == 0
        # Should be replaced with load_const 0
        lc_instrs = [i for i in block.instructions
                     if i.opcode == OpCode.LOAD_CONST]
        assert any(i.attrs.get("value") == 0 for i in lc_instrs)


class TestMulAddFusion:
    def test_fuse_mul_add(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a")
        b = builder.make_value(name="b")
        acc = builder.make_value(name="acc")

        tmp = builder.mul(a, b)
        result = builder.add(tmp, acc)
        builder.ret(result)

        opt = MulAddFusion(builder.program)
        count = opt.run()
        assert count == 1

        block = builder.program.functions[0].blocks[0]
        # Should have one ADD (the fused one) and no separate MUL+ADD
        adds = [i for i in block.instructions if i.opcode == OpCode.ADD]
        muls = [i for i in block.instructions if i.opcode == OpCode.MUL]
        assert len(adds) == 1
        assert len(muls) == 0

    def test_no_fuse_without_mul(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a")
        b = builder.make_value(name="b")
        c = builder.add(a, b)
        builder.ret(c)

        opt = MulAddFusion(builder.program)
        count = opt.run()
        assert count == 0


class TestLICM:
    def test_hoist_invariant(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")

        a = builder.make_value(name="a")
        b = builder.make_value(name="b")

        # Loop with invariant mul inside
        iv = builder.for_loop(0, 10)  # FOR
        # This mul depends on a, b defined outside the loop -> invariant
        c = builder.mul(a, b)
        builder.add(c, iv)  # this depends on iv → variant, keep
        builder.endfor()
        builder.ret()

        opt = LICM(builder.program)
        count = opt.run()
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
        builder.add(iv, builder.load_const(1.0))
        builder.endfor()
        builder.ret()

        opt = LICM(builder.program)
        opt.run()
        # The load_const IS invariant and gets hoisted.
        # But the add depending on iv stays in the loop.
        # Verify the add remains after the FOR.
        block = builder.program.functions[0].blocks[0]
        for_idx = next(i for i, instr in enumerate(block.instructions)
                       if instr.opcode == OpCode.FOR)
        add_instrs = [i for i in block.instructions if i.opcode == OpCode.ADD]
        # The ADD (variant) should still be inside the loop (after FOR)
        assert all(block.instructions.index(i) > for_idx for i in add_instrs)
