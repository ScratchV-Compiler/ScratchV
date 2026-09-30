"""Branches and loops must execute actual paths, with strict local storage."""

import numpy as np
import pytest

from benchmarks.ir_interpreter_cases import builder, make_case
from scratchv.analysis.adapters import IRCFGAdapter
from scratchv.analysis.ir_verifier import verify_ir
from scratchv.ir.types import DataType as D
from scratchv.ir.types import Value
from scratchv.verification.ir_interpreter import IRExecutionError, IRInterpreter


@pytest.mark.parametrize("flag,expected", [(0, [-2, 3]), (1, [3, -2]), (-1, [3, -2])])
def test_only_selected_branch(flag, expected):
    case = make_case("branch")
    case.inputs["condition"][()] = flag
    actual = IRInterpreter(case.program).run(case.inputs).return_value
    np.testing.assert_array_equal(actual, expected)


def test_dead_branch_does_not_divide_and_preflight_rejects_unknown_attributes():
    condition = Value("condition", D.INT32)
    b = builder(condition)
    b.br_if(condition, "good", "bad")
    b.new_block("good")
    b.ret(b.make_const(7.0))
    b.new_block("bad")
    b.ret(b.div(b.make_const(1.0), b.make_const(0.0)))
    interpreter = IRInterpreter(b.program)
    assert interpreter.run({"condition": np.array(1, dtype="int32")}).return_value == 7
    with pytest.raises(IRExecutionError, match="NumericError"):
        interpreter.run({"condition": np.array(0, dtype="int32")})
    b.current_block.instructions[0].attrs["unknown"] = 1
    with pytest.raises(IRExecutionError, match="UnsupportedAttribute"):
        interpreter.run({"condition": np.array(1, dtype="int32")})
    b.current_block.instructions[0].attrs.clear()
    with pytest.raises(IRExecutionError, match="ShapeError"):
        interpreter.run({"condition": np.array([1], dtype="int32")})


@pytest.mark.parametrize(
    "comparison,a,b_value,expected",
    [
        ("==", 3, 3, 1),
        ("!=", 3, 3, 0),
        ("<", 2, 3, 1),
        ("<=", 3, 3, 1),
        (">", 2, 3, 0),
        (">=", 3, 3, 1),
    ],
)
def test_two_operand_scalar_branch(comparison, a, b_value, expected):
    x, y = Value("x", D.INT64), Value("y", D.INT64)
    b = builder(x, y)
    b.br_compare(x, comparison, y, "yes", "no")
    b.new_block("yes")
    b.ret(b.make_const(1, D.INT32))
    b.new_block("no")
    b.ret(b.make_const(0, D.INT32))
    actual = IRInterpreter(b.program).run(
        {"x": np.array(a, dtype="int64"), "y": np.array(b_value, dtype="int64")}
    )
    assert actual.return_value == expected


def test_step_count_and_limit_map_to_original_instruction():
    case = make_case("loop_sum")
    interpreter = IRInterpreter(case.program)
    result = interpreter.run({}, max_steps=37)
    assert result.return_value == 10
    assert result.executed_steps == 37
    with pytest.raises(IRExecutionError) as found:
        interpreter.run({}, max_steps=4)
    assert found.value.code == "StepLimitExceeded"
    assert (
        found.value.block_name,
        found.value.instruction_index,
        found.value.stage,
    ) == ("entry", 2, "for-test")


def test_cross_block_loop_and_original_locations():
    b = builder()
    slot = b.alloca(4, D.INT32)
    b.store(slot, b.make_const(0, D.INT32))
    i = b.for_loop(0, 3)
    b.new_block("body")
    b.store(slot, b.add(b.load(slot), i))
    b.new_block("loop_end")
    b.endfor()
    b.new_block("after")
    b.ret(b.load(slot))
    assert verify_ir(b.program)[0]
    result = IRInterpreter(b.program).run({})
    assert result.return_value == 3
    origins = IRCFGAdapter(b.program.functions[0]).execution_plan.origins.values()
    assert any(
        p.block_name == "loop_end"
        and p.instruction_index == 0
        and p.stage == "for-step"
        for p in origins
    )


def test_collision_free_loop_names():
    x = Value("for_step_1", D.INT32)
    b = builder(x)
    slot = b.alloca(4, D.INT32)
    b.store(slot, x)
    b.for_loop(0, 2)
    b.store(slot, b.add(b.load(slot), b.make_const(1, D.INT32)))
    b.endfor()
    b.ret(b.load(slot))
    b.new_block("for_hdr1")
    b.ret()
    result = IRInterpreter(b.program).run({"for_step_1": np.array(5, dtype="int32")})
    assert result.return_value == 7
    names = [block.name for block in IRCFGAdapter(b.program.functions[0]).blocks()]
    assert len(names) == len(set(names))
    assert "for_hdr1_1" in names


def test_loop_integer_boundary_reports_overflow_instead_of_wraparound_loop():
    b = builder()
    b.for_loop(2**31 - 2, 2**31 - 1, 2)
    b.endfor()
    b.ret()
    with pytest.raises(IRExecutionError) as found:
        IRInterpreter(b.program).run({})
    assert found.value.code == "NumericError"
    assert found.value.stage == "for-step"


def test_memory_branch_merge_and_i64_values():
    condition = Value("condition", D.INT32)
    b = builder(condition)
    slot = b.alloca(8, D.INT64)
    b.br_if(condition, "left", "right")
    b.new_block("left")
    b.store(slot, b.make_const(2**60 + 1, D.INT64))
    b.br("merge")
    b.new_block("right")
    b.store(slot, b.make_const(-3, D.INT64))
    b.br("merge")
    b.new_block("merge")
    b.ret(b.load(slot))
    interpreter = IRInterpreter(b.program)
    for flag, expected in [(1, 2**60 + 1), (0, -3)]:
        result = interpreter.run({"condition": np.array(flag, dtype="int32")})
        assert result.return_value.dtype == np.dtype("int64")
        assert int(result.return_value) == expected


@pytest.mark.parametrize(
    "mode",
    [
        "uninitialized",
        "size",
        "unaligned",
        "numeric_address",
        "vector_store",
        "numeric_pointer",
        "return_pointer",
    ],
)
def test_memory_errors(mode):
    b = builder()
    slot = b.alloca(4 if mode != "size" else 0, D.INT32)
    if mode == "unaligned":
        b.current_block.instructions[0].attrs["size"] = 5
    if mode in ("uninitialized", "size", "unaligned"):
        b.ret(b.load(slot))
    if mode == "numeric_address":
        b.ret(b.load(b.make_const(42, D.INT32)))
    if mode == "vector_store":
        value = Value("value", D.INT32)
        b.current_func.params = [value]
        b.store(slot, value)
        b.ret()
    if mode == "numeric_pointer":
        b.ret(b.add(slot, b.make_const(1, D.INT32)))
    if mode == "return_pointer":
        b.ret(slot)
    inputs = (
        {"value": np.array([1, 2], dtype="int32")} if mode == "vector_store" else {}
    )
    with pytest.raises(IRExecutionError, match="MemoryError"):
        IRInterpreter(b.program).run(inputs)


def test_shared_plan_does_not_make_new_ops_execute_outside_zero_trip_loop():
    from scratchv.optimizer.licm import LICM

    b = builder()
    b.for_loop(0, 0)
    b.sqrt(b.make_const(-1.0))
    b.endfor()
    b.ret(b.make_const(2.0))
    assert LICM().optimize(b.program) == 0
    assert IRInterpreter(b.program).run({}).return_value == 2
