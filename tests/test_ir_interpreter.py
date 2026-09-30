"""Execution contracts: direct Programs, independent outputs and failure paths."""

import copy
import pickle

import numpy as np
import pytest

from benchmarks.ir_interpreter_cases import CASE_FACTORIES, builder, compare, make_case
from scratchv.ir.types import DataType as D
from scratchv.ir.types import Function, Program, Value
from scratchv.ir.types import OpCode as O
from scratchv.verification.ir_interpreter import IRExecutionError, IRInterpreter


@pytest.mark.parametrize("name", list(CASE_FACTORIES))
def test_direct_cases(name):
    case = make_case(name)
    original = pickle.dumps(case.program)
    snapshots = {k: v.copy() for k, v in {**case.inputs, **case.initializers}.items()}
    interpreter = IRInterpreter(case.program)
    for _ in range(2):
        result = interpreter.run(case.inputs, initializers=case.initializers)
        compare(result.return_value, case)
        assert result.executed_steps > 0
    assert pickle.dumps(case.program) == original
    for key, data in {**case.inputs, **case.initializers}.items():
        np.testing.assert_array_equal(data, snapshots[key])


def test_run_returns_copy_and_resets_state():
    x = Value("x", shape=(2,))
    b = builder(x)
    b.ret(x)
    interpreter = IRInterpreter(b.program)
    first = np.array([1, 2], dtype="float32")
    result = interpreter.run({"x": first})
    result.return_value[0] = 99
    assert first[0] == 1
    with pytest.raises(IRExecutionError, match="DTypeError"):
        interpreter.run({"x": first.astype("float64")})
    np.testing.assert_array_equal(
        interpreter.run({"x": np.array([3, 4], dtype="float32")}).return_value, [3, 4]
    )


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("missing_input", "BindingError"),
        ("extra_input", "BindingError"),
        ("wrong_dtype", "DTypeError"),
        ("wrong_shape", "ShapeError"),
        ("missing_weight", "BindingError"),
        ("unknown_weight", "BindingError"),
        ("nan", "NumericError"),
        ("positive_inf", "NumericError"),
    ],
)
def test_binding_errors(mutation, code):
    case = make_case("matmul_add_softmax")
    if mutation == "missing_input":
        case.inputs.clear()
    if mutation == "extra_input":
        case.inputs["typo"] = case.inputs["x"]
    if mutation == "wrong_dtype":
        case.inputs["x"] = case.inputs["x"].astype("float64")
    if mutation == "wrong_shape":
        case.inputs["x"] = case.inputs["x"].reshape(3, 2)
    if mutation == "missing_weight":
        case.initializers.pop("W")
    if mutation == "unknown_weight":
        case.initializers["typo"] = case.initializers["W"]
    if mutation == "nan":
        case.inputs["x"][0, 0] = np.nan
    if mutation == "positive_inf":
        case.inputs["x"][0, 0] = np.inf
    with pytest.raises(IRExecutionError) as found:
        IRInterpreter(case.program).run(case.inputs, initializers=case.initializers)
    assert found.value.code == code


def test_select_one_function_and_preserve_warning():
    b = builder()
    b.ret(b.make_const(3.0))
    b.new_block("unused")
    b.ret(b.make_const(99.0))
    other = Function("other")
    b.program.add_function(other)
    other.new_block("entry").add(copy.deepcopy(b.current_block.instructions[-1]))
    interpreter = IRInterpreter(b.program)
    with pytest.raises(IRExecutionError, match="EntryError"):
        interpreter.run({})
    with pytest.raises(IRExecutionError, match="EntryError"):
        interpreter.run({}, function_name="missing")
    result = interpreter.run({}, function_name="main")
    assert result.return_value == 3
    assert result.executed_steps == 1
    assert any(i.level.value == "warning" for i in result.diagnostics)
    assert interpreter.run({}, function_name="other").return_value == 99


def test_empty_invalid_and_void_returns():
    with pytest.raises(IRExecutionError, match="EntryError"):
        IRInterpreter(Program()).run({})
    b = builder()
    with pytest.raises(IRExecutionError, match="InvalidProgram"):
        IRInterpreter(b.program).run({})
    b.ret()
    assert IRInterpreter(b.program).run({}).return_value is None


def test_error_position_and_cause():
    x = Value("x")
    b = builder(x)
    b.ret(b.reshape(x, (3,)))
    with pytest.raises(IRExecutionError) as found:
        IRInterpreter(b.program).run({"x": np.ones(2, dtype="float32")})
    issue = found.value
    assert (
        issue.function_name,
        issue.block_name,
        issue.instruction_index,
        issue.opcode,
    ) == ("main", "entry", 0, O.RESHAPE)
    assert isinstance(issue.__cause__, ValueError)


def test_missing_value_not_zero_and_shape_on_result():
    b = builder()
    b.ret(Value("missing"))
    with pytest.raises(IRExecutionError, match="InvalidProgram"):
        IRInterpreter(b.program).run({})
    x = Value("x")
    b = builder(x)
    result = b.neg(x)
    result.shape = (3,)
    b.ret(result)
    with pytest.raises(IRExecutionError, match="ShapeError"):
        IRInterpreter(b.program).run({"x": np.ones(2, dtype="float32")})


def test_global_constants_and_conflicts():
    literal = Value("literal", D.INT64, True, 2**60 + 1)
    b = builder()
    b.program.global_values.append(literal)
    b.ret(literal)
    interpreter = IRInterpreter(b.program)
    assert interpreter.run({}).return_value == 2**60 + 1
    with pytest.raises(IRExecutionError, match="BindingError"):
        interpreter.run({}, initializers={"literal": np.array(3, dtype="int64")})


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_bad_step_limits(limit):
    b = builder()
    b.ret()
    with pytest.raises(IRExecutionError, match="InvalidOptions"):
        IRInterpreter(b.program).run({}, max_steps=limit)
