"""Behavior tests for dead-code elimination."""

from __future__ import annotations

import pytest

from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import BasicBlock, Function, Instruction, OpCode, Program, Value
from scratchv.optimizer.dead_code import DeadCodeEliminator


class TestDeadCodeUseDef:
    def test_removes_complete_dead_dependency_chain(self):
        builder = IRBuilder()
        builder.new_function("test")
        block = builder.new_block("entry")

        lhs = builder.load_const(1.0)
        rhs = builder.load_const(2.0)
        builder.add(lhs, rhs)
        live = builder.load_const(3.0)
        builder.ret(live)
        expected_survivors = tuple(block.instructions[-2:])

        changes = DeadCodeEliminator().optimize(builder.program)

        assert changes == 3
        assert tuple(map(id, block.instructions)) == tuple(
            map(id, expected_survivors)
        )
        assert [instr.opcode for instr in block.instructions] == [
            OpCode.LOAD_CONST,
            OpCode.RETURN,
        ]

    def test_leaves_multi_block_function_unchanged(self):
        builder = IRBuilder()
        builder.new_function("test")
        entry = builder.new_block("entry")
        value = builder.load_const(1.0)
        exit_block = builder.new_block("exit")
        builder.ret(value)
        entry_before = tuple(entry.instructions)
        exit_before = tuple(exit_block.instructions)

        changes = DeadCodeEliminator().optimize(builder.program)

        assert changes == 0
        assert tuple(map(id, entry.instructions)) == tuple(map(id, entry_before))
        assert tuple(map(id, exit_block.instructions)) == tuple(
            map(id, exit_before)
        )

    def test_leaves_duplicate_definitions_unchanged(self):
        builder = IRBuilder()
        builder.new_function("test")
        block = builder.new_block("entry")
        first = Instruction(OpCode.LOAD_CONST, dest=Value("x"))
        second = Instruction(OpCode.LOAD_CONST, dest=Value("x"))
        ret = Instruction(OpCode.RETURN, operands=[second.dest])
        block.add(first)
        block.add(second)
        block.add(ret)
        instructions_before = tuple(block.instructions)

        changes = DeadCodeEliminator().optimize(builder.program)

        assert changes == 0
        assert tuple(map(id, block.instructions)) == tuple(
            map(id, instructions_before)
        )

    def test_leaves_function_with_phi_nodes_unchanged(self):
        builder = IRBuilder()
        builder.new_function("test")
        block = builder.new_block("entry")
        builder.load_const(1.0)
        block.phi_nodes.append(Instruction(OpCode.ADD, dest=Value("phi")))
        instructions_before = tuple(block.instructions)

        changes = DeadCodeEliminator().optimize(builder.program)

        assert changes == 0
        assert tuple(map(id, block.instructions)) == tuple(
            map(id, instructions_before)
        )

    def test_preserves_complete_return_dependency_chain(self):
        builder = IRBuilder()
        builder.new_function("test")
        block = builder.new_block("entry")
        lhs = builder.load_const(2.0)
        rhs = builder.load_const(3.0)
        added = builder.add(lhs, rhs)
        factor = builder.load_const(4.0)
        result = builder.mul(added, factor)
        builder.ret(result)
        instructions_before = tuple(block.instructions)

        changes = DeadCodeEliminator().optimize(builder.program)

        assert changes == 0
        assert tuple(map(id, block.instructions)) == tuple(
            map(id, instructions_before)
        )

    def test_removes_interleaved_dead_chains_without_reordering_live_code(self):
        builder = IRBuilder()
        builder.new_function("test")
        block = builder.new_block("entry")
        live_lhs = builder.load_const(2.0)
        first_dead_source = builder.load_const(10.0)
        live_rhs = builder.load_const(3.0)
        builder.neg(first_dead_source)
        second_dead_source = builder.load_const(20.0)
        live_result = builder.add(live_lhs, live_rhs)
        builder.exp(second_dead_source)
        builder.ret(live_result)
        expected_survivors = tuple(
            block.instructions[index] for index in (0, 2, 5, 7)
        )

        changes = DeadCodeEliminator().optimize(builder.program)

        assert changes == 4
        assert tuple(map(id, block.instructions)) == tuple(
            map(id, expected_survivors)
        )

    def test_preserves_store_pointer_and_value_dependencies(self):
        builder = IRBuilder()
        builder.new_function("test")
        block = builder.new_block("entry")
        pointer = builder.alloca(1)
        lhs = builder.load_const(2.0)
        rhs = builder.load_const(3.0)
        stored = builder.add(lhs, rhs)
        builder.store(pointer, stored)
        builder.load_const(99.0)
        expected_survivors = tuple(block.instructions[:-1])

        changes = DeadCodeEliminator().optimize(builder.program)

        assert changes == 1
        assert tuple(map(id, block.instructions)) == tuple(
            map(id, expected_survivors)
        )

    def test_preserves_branch_condition_dependency_chain(self):
        builder = IRBuilder()
        builder.new_function("test")
        block = builder.new_block("entry")
        lhs = builder.load_const(2.0)
        rhs = builder.load_const(3.0)
        condition = builder.sub(lhs, rhs)
        builder.br_if(condition, "then", "else")
        instructions_before = tuple(block.instructions)

        changes = DeadCodeEliminator().optimize(builder.program)

        assert changes == 0
        assert tuple(map(id, block.instructions)) == tuple(
            map(id, instructions_before)
        )

    @pytest.mark.parametrize(
        "opcode",
        [
            OpCode.STORE,
            OpCode.RETURN,
            OpCode.BR,
            OpCode.BR_IF,
            OpCode.FOR,
            OpCode.ENDFOR,
            OpCode.ALLOCA,
        ],
    )
    def test_preserves_each_effect_root(self, opcode):
        program = Program()
        function = Function("test")
        block = BasicBlock("entry")
        root = Instruction(opcode, dest=Value("result"))
        block.add(root)
        function.add_block(block)
        program.add_function(function)

        changes = DeadCodeEliminator().optimize(program)

        assert changes == 0
        assert len(block.instructions) == 1
        assert block.instructions[0] is root

    @pytest.mark.parametrize("opcode", [OpCode.LABEL, OpCode.ADD])
    def test_preserves_destless_instruction_conservatively(self, opcode):
        program = Program()
        function = Function("test")
        block = BasicBlock("entry")
        destless_pure_instruction = Instruction(opcode)
        block.add(destless_pure_instruction)
        function.add_block(block)
        program.add_function(function)

        changes = DeadCodeEliminator().optimize(program)

        assert changes == 0
        assert len(block.instructions) == 1
        assert block.instructions[0] is destless_pure_instruction

    def test_treats_implicit_inputs_as_live_leaves(self):
        builder = IRBuilder()
        builder.new_function("test")
        block = builder.new_block("entry")
        result = builder.add(Value("left_input"), Value("right_input"))
        builder.ret(result)
        instructions_before = tuple(block.instructions)

        changes = DeadCodeEliminator().optimize(builder.program)

        assert changes == 0
        assert tuple(map(id, block.instructions)) == tuple(
            map(id, instructions_before)
        )

    def test_repeated_invocations_report_only_current_changes(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        lhs = builder.load_const(1.0)
        rhs = builder.load_const(2.0)
        builder.add(lhs, rhs)
        live = builder.load_const(3.0)
        builder.ret(live)
        eliminator = DeadCodeEliminator()

        first_changes = eliminator.optimize(builder.program)
        second_changes = eliminator.optimize(builder.program)

        assert (first_changes, second_changes) == (3, 0)

    def test_handles_empty_program(self):
        empty_program = Program()
        zero_block_program = Program()
        zero_block_program.add_function(Function("empty"))

        assert (
            DeadCodeEliminator().optimize(empty_program),
            DeadCodeEliminator().optimize(zero_block_program),
        ) == (0, 0)

    def test_analyzes_same_value_name_independently_per_function(self):
        program = Program()
        live_function = Function("live")
        live_block = live_function.new_block("entry")
        live_value = Value("x")
        live_definition = Instruction(OpCode.LOAD_CONST, dest=live_value)
        live_return = Instruction(OpCode.RETURN, operands=[live_value])
        live_block.add(live_definition)
        live_block.add(live_return)
        program.add_function(live_function)

        dead_function = Function("dead")
        dead_block = dead_function.new_block("entry")
        dead_definition = Instruction(OpCode.LOAD_CONST, dest=Value("x"))
        dead_return = Instruction(OpCode.RETURN)
        dead_block.add(dead_definition)
        dead_block.add(dead_return)
        program.add_function(dead_function)

        changes = DeadCodeEliminator().optimize(program)

        assert changes == 1
        assert live_block.instructions == [live_definition, live_return]
        assert dead_block.instructions == [dead_return]

    def test_terminates_on_cyclic_malformed_dependencies(self):
        program = Program()
        function = Function("test")
        block = function.new_block("entry")
        first_value = Value("first")
        second_value = Value("second")
        first = Instruction(
            OpCode.ADD,
            dest=first_value,
            operands=[second_value],
        )
        second = Instruction(
            OpCode.ADD,
            dest=second_value,
            operands=[first_value],
        )
        ret = Instruction(OpCode.RETURN, operands=[first_value])
        block.add(first)
        block.add(second)
        block.add(ret)
        program.add_function(function)

        changes = DeadCodeEliminator().optimize(program)

        assert changes == 0
        assert block.instructions == [first, second, ret]
