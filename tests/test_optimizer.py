"""Tests for optimizer passes."""

from scratchv.ir.builder import IRBuilder
from scratchv.optimizer.constant_folding import ConstantFolder
from scratchv.optimizer.dead_code import DeadCodeEliminator
from scratchv.ir.types import OpCode


class TestConstantFolder:
    def test_fold_add_constants(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")

        c1 = builder.load_const(3.0)
        c2 = builder.load_const(4.0)
        r = builder.add(c1, c2)
        builder.ret(r)

        folder = ConstantFolder(builder.program)
        count = folder.run()
        assert count == 1

        block = builder.program.functions[0].blocks[0]
        # The add (index 2) should be replaced by load_const 7.0
        assert block.instructions[2].opcode == OpCode.LOAD_CONST
        assert block.instructions[2].attrs["value"] == 7.0

    def test_fold_mul_constants(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")

        c1 = builder.load_const(2.0)
        c2 = builder.load_const(5.0)
        r = builder.mul(c1, c2)
        builder.ret(r)

        folder = ConstantFolder(builder.program)
        count = folder.run()
        assert count == 1
        # The mul (index 2) was replaced by load_const 10.0
        block = builder.program.functions[0].blocks[0]
        assert block.instructions[2].attrs["value"] == 10.0

    def test_no_fold_with_variable(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")

        a = builder.make_value(name="a")
        c = builder.load_const(5.0)
        r = builder.add(a, c)
        builder.ret(r)

        folder = ConstantFolder(builder.program)
        count = folder.run()
        assert count == 0  # cannot fold because 'a' is not constant


class TestDeadCodeEliminator:
    def test_eliminate_unused(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")

        a = builder.load_const(1.0)
        b = builder.load_const(2.0)
        builder.add(a, b)  # unused!
        d = builder.load_const(3.0)
        builder.ret(d)

        elim = DeadCodeEliminator(builder.program)
        count = elim.run()
        assert count == 1  # the add should be eliminated

        block = builder.program.functions[0].blocks[0]
        instrs = block.instructions
        assert all(i.opcode != OpCode.ADD for i in instrs)

    def test_keep_used_value(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")

        a = builder.load_const(1.0)
        b = builder.load_const(2.0)
        c = builder.add(a, b)  # used by ret
        builder.ret(c)

        elim = DeadCodeEliminator(builder.program)
        count = elim.run()
        assert count == 0  # nothing eliminated
