"""LICM must preserve checked execution, including zero-trip loops."""

import copy
import math

import numpy as np
import pytest

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as D
from scratchv.ir.types import OpCode, Value
from scratchv.optimizer.licm import LICM
from scratchv.verification.ir_interpreter import IRExecutionError, IRInterpreter


def builder(*params):
    b = IRBuilder()
    b.new_function("main", list(params))
    b.new_block("entry")
    return b


def compare(b, inputs=None, *, count):
    original = copy.deepcopy(b.program)
    assert verify_ir(original)[0]
    assert LICM().optimize(b.program) == count
    assert verify_ir(b.program)[0]
    before = IRInterpreter(original).run(inputs or {}).return_value
    after = IRInterpreter(b.program).run(inputs or {}).return_value
    np.testing.assert_array_equal(after, before)
    return b.current_block.instructions


def before_for(b, value):
    instructions = b.current_block.instructions
    definition = next(i for i, inst in enumerate(instructions) if inst.dest is value)
    loop = next(i for i, inst in enumerate(instructions) if inst.opcode == OpCode.FOR)
    return definition < loop


@pytest.mark.parametrize("trips", [0, 3])
def test_dependencies_move_in_order_and_preserve_memory_result(trips):
    b = builder()
    slot = b.alloca(4, D.INT32)
    b.store(slot, b.make_const(0, D.INT32))
    b.for_loop(0, trips)
    a = b.load_const(7, D.INT32)
    c = b.add(a, b.make_const(2, D.INT32))
    d = b.mul(c, b.make_const(3, D.INT32))
    b.store(slot, d)
    b.endfor()
    b.ret(b.load(slot))
    instructions = compare(b, count=3)
    assert all(before_for(b, v) for v in (a, c, d))
    assert [i.dest for i in instructions if i.dest in (a, c, d)] == [a, c, d]
    assert LICM().optimize(b.program) == 0


@pytest.mark.parametrize("op", ["add", "sub", "mul", "neg"])
def test_integer_wrap_operations_with_known_shapes_are_safe(op):
    x = Value("x", D.INT32, shape=(2,))
    b = builder(x)
    b.for_loop(0, 0)
    result = (
        getattr(b, op)(x)
        if op == "neg"
        else getattr(b, op)(x, b.make_const(1, D.INT32))
    )
    b.endfor()
    b.ret(b.make_const(1, D.INT32))
    compare(b, {"x": np.array([2**31 - 1, -(2**31)], dtype="int32")}, count=1)
    assert before_for(b, result)


@pytest.mark.parametrize(
    "op",
    ["div_zero", "div_overflow", "sqrt", "exp", "float_add", "float_mul", "float_neg"],
)
def test_numeric_errors_are_not_speculated(op):
    b = builder()
    b.for_loop(0, 0)
    if op == "div_zero":
        b.div(b.make_const(2, D.INT32), b.make_const(0, D.INT32))
    elif op == "div_overflow":
        b.div(b.make_const(-(2**31), D.INT32), b.make_const(-1, D.INT32))
    elif op == "sqrt":
        b.sqrt(b.make_const(-1.0))
    elif op == "exp":
        b.exp(b.make_const(1000.0))
    elif op == "float_neg":
        b.neg(b.make_const(float("-inf")))
    else:
        getattr(b, op.removeprefix("float_"))(
            b.make_const(np.finfo("float32").max.item()),
            b.make_const(2.0 if op == "float_mul" else np.finfo("float32").max.item()),
        )
    b.endfor()
    b.ret(b.make_const(3.0))
    compare(b, count=0)


def test_division_proves_nonzero_but_retains_possible_signed_overflow():
    x = Value("x", D.INT64, shape=(2,))
    b = builder(x)
    b.for_loop(0, 0)
    safe = b.div(x, b.make_const(2, D.INT64))
    unsafe = b.div(x, b.make_const(-1, D.INT64))
    b.endfor()
    b.ret(b.make_const(1, D.INT64))
    compare(b, {"x": np.array([-(2**63), 6], dtype="int64")}, count=1)
    assert before_for(b, safe) and not before_for(b, unsafe)


def test_sqrt_after_relu_has_a_nonnegative_finite_range():
    x = Value("x", shape=(3,))
    b = builder(x)
    b.for_loop(0, 2)
    positive = b.relu(x)
    result = b.sqrt(positive)
    b.endfor()
    b.ret(b.make_const(1.0))
    compare(b, {"x": np.array([-np.inf, -2, 4], dtype="float32")}, count=2)
    assert before_for(b, result)


def test_new_tensor_operations_move_as_a_chain_and_preserve_return_value():
    x = Value("x", shape=(2, 3))
    b = builder(x)
    slot = b.alloca(4)
    b.store(slot, b.make_const(0.0))
    b.for_loop(0, 2)
    finite = b.relu(x)
    root = b.sqrt(finite)
    bounded = b.sigmoid(root)
    row = b.slice(bounded, (0,), (1,), axes=(0,))
    batch = b.unsqueeze(row, (0,))
    expanded = b.expand(batch, (2, 1, 3))
    selected = b.gather(expanded, b.make_const(1, D.INT64))
    mean = b.reduce_mean(selected, keepdims=False)
    b.store(slot, mean)
    b.endfor()
    b.ret(b.load(slot))
    inputs = {"x": np.array([[4, 9, 16], [-np.inf, -1, 0]], dtype="float32")}
    compare(b, inputs, count=8)
    assert before_for(b, mean)
    expected = sum(1 / (1 + math.exp(-v)) for v in (2, 3, 4)) / 3
    np.testing.assert_allclose(
        IRInterpreter(b.program).run(inputs).return_value, expected, rtol=1e-6
    )


@pytest.mark.parametrize(
    "op", ["gather", "slice", "unsqueeze", "expand", "reshape", "transpose"]
)
@pytest.mark.parametrize("valid", [False, True])
def test_shape_and_index_proofs(op, valid):
    x = Value("x", D.INT32, shape=(2, 3))
    b = builder(x)
    b.for_loop(0, 0)
    if op == "gather":
        result = b.gather(x, b.make_const(-2 if valid else 2, D.INT64))
    elif op == "slice":
        result = b.slice(x, (0,), (2,), axes=(0 if valid else 2,), steps=(-1,))
    elif op == "unsqueeze":
        result = b.unsqueeze(x, (-1,) if valid else (3,))
    elif op == "expand":
        result = b.expand(x, (4, 2, 3) if valid else (4, 2, 2))
    elif op == "reshape":
        result = b.reshape(x, (3, 2) if valid else (0, 0, 0))
    else:
        result = b.transpose(x, (1, 0) if valid else (0, 0))
    b.endfor()
    b.ret(b.make_const(1, D.INT32))
    compare(b, {"x": np.arange(6, dtype="int32").reshape(2, 3)}, count=int(valid))
    assert before_for(b, result) == valid


def test_mean_checks_empty_dimensions_and_intermediate_sum_overflow():
    x = Value("x", shape=(2,))
    empty = Value("empty", shape=(0,))
    b = builder(x, empty)
    b.for_loop(0, 0)
    bounded = b.sigmoid(x)
    mean = b.reduce_mean(bounded)
    unbounded = b.reduce_mean(x)
    empty_mean = b.reduce_mean(empty)
    b.endfor()
    b.ret(b.make_const(1.0))
    compare(
        b,
        {
            "x": np.full(2, np.finfo("float32").max, dtype="float32"),
            "empty": np.empty(0, dtype="float32"),
        },
        count=2,
    )
    assert before_for(b, mean)
    assert not before_for(b, unbounded) and not before_for(b, empty_mean)


def test_unknown_shapes_and_runtime_indices_do_not_establish_proofs():
    x = Value("x", D.INT32)
    y = Value("y", D.INT32)
    data = Value("data", D.INT32, shape=(2,))
    index = Value("index", D.INT64)
    b = builder(x, y, data, index)
    b.for_loop(0, 0)
    b.add(x, y)
    b.gather(data, index)
    b.endfor()
    b.ret(b.make_const(1, D.INT32))
    compare(
        b,
        {
            "x": np.ones(2, dtype="int32"),
            "y": np.ones(3, dtype="int32"),
            "data": np.ones(2, dtype="int32"),
            "index": np.array(5, dtype="int64"),
        },
        count=0,
    )


def test_float_shape_operation_must_account_for_negative_infinity_masks():
    x = Value("x", shape=(2,))
    b = builder(x)
    b.for_loop(0, 0)
    b.unsqueeze(x, (0,))
    b.endfor()
    b.ret(b.make_const(1.0))
    compare(b, {"x": np.array([-np.inf, 1], dtype="float32")}, count=0)


def test_float_arithmetic_needs_range_proof_even_with_compatible_shapes():
    x = Value("x", shape=(2,))
    b = builder(x)
    b.for_loop(0, 0)
    b.add(x, x)
    b.mul(x, x)
    b.sqrt(x)
    b.endfor()
    b.ret(b.make_const(1.0))
    compare(b, {"x": np.array([-1, np.finfo("float32").max], dtype="float32")}, count=0)


def test_declared_result_shape_is_part_of_the_safety_proof():
    x = Value("x", D.INT32, shape=(2,))
    b = builder(x)
    b.for_loop(0, 0)
    result = b.neg(x)
    result.shape = (3,)
    b.endfor()
    b.ret(b.make_const(1, D.INT32))
    compare(b, {"x": np.ones(2, dtype="int32")}, count=0)


def test_unsupported_attribute_never_passes_safety_check():
    b = builder()
    b.for_loop(0, 0)
    b.sqrt(b.make_const(4.0))
    b.current_block.instructions[-1].attrs["unrecognized"] = True
    b.endfor()
    b.ret()
    assert verify_ir(b.program)[0]
    assert LICM().optimize(b.program) == 0


def test_memory_allocation_and_loads_stay_inside_loop():
    b = builder()
    external = b.alloca(4, D.INT32)
    b.store(external, b.make_const(0, D.INT32))
    iv = b.for_loop(0, 3)
    local = b.alloca(4, D.INT32)
    b.store(local, iv)
    b.store(external, b.add(b.load(external), b.load(local)))
    b.endfor()
    b.ret(b.load(external))
    compare(b, count=0)
    assert not before_for(b, local)
    assert IRInterpreter(b.program).run({}).return_value == 3


def test_uninitialized_load_in_zero_trip_loop_is_not_executed():
    b = builder()
    slot = b.alloca(4, D.INT32)
    b.for_loop(0, 0)
    b.load(slot)
    b.endfor()
    b.ret(b.make_const(1, D.INT32))
    compare(b, count=0)


@pytest.mark.parametrize(
    "op,data",
    [("neg", -np.inf), ("relu", np.inf), ("sigmoid", np.nan), ("unsqueeze", -np.inf)],
)
def test_prior_float_load_does_not_prove_finite_data(op, data):
    b = builder()
    slot = b.alloca(4)
    b.store(slot, b.make_const(data))
    value = b.load(slot)
    b.for_loop(0, 0)
    result = b.unsqueeze(value, (0,)) if op == "unsqueeze" else getattr(b, op)(value)
    b.endfor()
    b.ret(b.make_const(1.0))
    compare(b, count=0)
    assert not before_for(b, result)


def test_adjacent_and_nested_loops_are_processed_without_stale_indices():
    b = builder()
    for _ in range(2):
        outer = b.for_loop(0, 2)
        b.for_loop(0, 0)
        b.load_const(2, D.INT32)
        b.add(outer, b.make_const(1, D.INT32))
        b.sqrt(b.make_const(-1.0))
        b.endfor()
        b.endfor()
    b.ret(b.make_const(1, D.INT32))
    # Constants move from inner to outer to preheader; outer-dependent ADDs
    # move out of the inner loop only. Each physical move counts once.
    compare(b, count=6)
    assert sum(i.opcode == OpCode.FOR for i in b.current_block.instructions) == 4


def test_definition_in_predecessor_is_available_at_insertion():
    b = builder()
    x = b.load_const(9.0)
    b.br("loop")
    b.new_block("loop")
    b.for_loop(0, 0)
    result = b.sqrt(x)
    b.endfor()
    b.ret(b.make_const(1.0))
    compare(b, count=1)
    assert before_for(b, result)


def test_loop_defined_constant_flag_cannot_bypass_definition_availability():
    b = builder()
    slot = b.alloca(4, D.INT32)
    b.store(slot, b.make_const(3, D.INT32))
    b.for_loop(0, 2)
    value = b.load(slot)
    # Resolution uses the named SSA definition before a literal flag.
    value.is_constant = True
    value.const_value = 3
    result = b.add(value, b.make_const(1, D.INT32))
    b.endfor()
    b.ret(b.make_const(1, D.INT32))
    compare(b, count=0)
    assert not before_for(b, result)


def test_invalid_program_does_not_supply_an_availability_proof():
    b = builder()
    b.for_loop(0, 0)
    b.neg(Value("undefined", D.INT32))
    b.endfor()
    b.ret()
    snapshot = b.program.dump()
    assert not verify_ir(b.program)[0]
    assert LICM().optimize(b.program) == 0
    assert b.program.dump() == snapshot


def test_parameter_literal_flag_does_not_replace_runtime_binding():
    x = Value("x", is_constant=True, const_value=4.0)
    b = builder(x)
    b.for_loop(0, 0)
    b.sqrt(x)
    b.endfor()
    b.ret(b.make_const(1.0))
    compare(b, {"x": np.array(-1.0, dtype="float32")}, count=0)


def test_reachable_error_remains_at_original_instruction():
    b = builder()
    b.for_loop(0, 1)
    b.sqrt(b.make_const(-1.0))
    b.endfor()
    b.ret()
    assert LICM().optimize(b.program) == 0
    with pytest.raises(IRExecutionError) as exc:
        IRInterpreter(b.program).run({})
    assert exc.value.code == "NumericError"
    assert exc.value.instruction_index == 1
