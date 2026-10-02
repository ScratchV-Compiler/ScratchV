"""Optimization must preserve executable SSA, tensors and checked arithmetic."""

import copy
from pathlib import Path

import numpy as np
import pytest

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as D, OpCode as O, Value
from scratchv.optimizer.constant_folding import ConstantFolder
from scratchv.optimizer.muladd_fusion import MulAddFusion
from scratchv.optimizer.peephole import IRPeepholeOptimizer
from scratchv.pass_manager import create_optimization_pass_manager
from scratchv.verification.ir_interpreter import IRExecutionError, IRInterpreter


DTYPES = [(D.FLOAT32, np.float32), (D.FLOAT64, np.float64),
          (D.INT32, np.int32), (D.INT64, np.int64)]


def builder(*params):
    result = IRBuilder()
    result.new_function("test", list(params))
    result.new_block("entry")
    return result


def equivalent(program, feed, optimizer):
    assert verify_ir(program) == (True, [])
    expected = IRInterpreter(program).run(feed).return_value
    changed = copy.deepcopy(program)
    count = optimizer.optimize(changed)
    assert verify_ir(changed) == (True, [])
    actual = IRInterpreter(changed).run(feed).return_value
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    if actual.dtype.kind == "f":
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    return changed, count, actual


@pytest.mark.parametrize("dtype,numpy_dtype", DTYPES)
@pytest.mark.parametrize("opcode,constant", [(O.ADD, 0), (O.MUL, 1), (O.MUL, 0)])
@pytest.mark.parametrize("shape", [(), (2, 3)])
def test_peephole_preserves_scalar_and_tensor_contract(dtype, numpy_dtype, opcode, constant, shape):
    x = Value("x", dtype, shape=shape)
    b = builder(x)
    result = b.make_value(dtype=dtype)
    result.shape = shape
    b._emit(opcode, result, [x, b.make_const(constant, dtype)])
    b.ret(result)
    data = np.full(shape, -3, dtype=numpy_dtype)
    changed, count, _ = equivalent(b.program, {"x": data}, IRPeepholeOptimizer())
    removable = dtype in (D.INT32, D.INT64) and (opcode == O.ADD or constant == 1)
    assert count == int(removable)
    assert IRPeepholeOptimizer().optimize(changed) == 0


def test_integer_identity_uses_are_rewritten_across_out_of_order_blocks():
    x = Value("x", D.INT64, shape=(2,))
    b = builder(x)
    entry = b.current_block
    b.br("middle")
    middle = b.new_block("middle")
    a = b.add(x, b.make_const(0, D.INT64))
    a.shape = x.shape
    z = b.mul(a, b.make_const(1, D.INT64))
    z.shape = x.shape
    b.br("exit")
    exit_block = b.new_block("exit")
    b.ret(z)
    b.current_func.returns = [z]
    b.current_func.blocks = [entry, exit_block, middle]
    changed, count, _ = equivalent(
        b.program, {"x": np.array([-7, 2**60], np.int64)}, IRPeepholeOptimizer())
    assert count == 2
    assert changed.functions[0].returns[0].name == "x"
    assert changed.functions[0].blocks[1].instructions[0].operands[0].name == "x"


@pytest.mark.parametrize("opcode,constant", [(O.ADD, 0), (O.MUL, 1), (O.MUL, 0)])
def test_float_identity_does_not_hide_nonfinite_error(opcode, constant):
    x = Value("x", D.FLOAT32, shape=(1,))
    b = builder(x)
    y = b.make_value()
    b._emit(opcode, y, [x, b.make_const(constant)])
    b.ret(b.relu(y))
    assert IRPeepholeOptimizer().optimize(b.program) == 0
    with pytest.raises(IRExecutionError, match="NumericError"):
        IRInterpreter(b.program).run({"x": np.array([-np.inf], np.float32)})


def test_float_add_zero_keeps_signed_zero_semantics():
    x = Value("x", D.FLOAT32, shape=(2,))
    b = builder(x)
    b.ret(b.add(x, b.make_const(0.0)))
    _, count, result = equivalent(
        b.program, {"x": np.array([-0.0, 0.0], np.float32)}, IRPeepholeOptimizer())
    assert count == 0
    assert not np.signbit(result).any()


def test_float_mul_one_removal_preserves_negative_zero_after_finite_producer():
    x = Value("x", D.FLOAT32, shape=(2,))
    b = builder(x)
    source = b.neg(x)
    source.shape = x.shape
    product = b.mul(source, b.make_const(1.0))
    product.shape = x.shape
    b.ret(product)
    _, count, result = equivalent(
        b.program, {"x": np.array([0.0, -0.0], np.float32)}, IRPeepholeOptimizer())
    assert count == 1
    np.testing.assert_array_equal(np.signbit(result), [True, False])


def test_peephole_keeps_shape_failure_and_unsupported_attributes():
    for attrs, shape, error in [({}, (3,), "ShapeError"), ({"invalid": 1}, (2,), "UnsupportedAttribute")]:
        x = Value("x", D.INT32, shape=(2,))
        b = builder(x)
        dest = Value("y", D.INT32, shape=shape)
        b._emit(O.ADD, dest, [x, b.make_const(0, D.INT32)], **attrs)
        b.ret(dest)
        assert IRPeepholeOptimizer().optimize(b.program) == 0
        with pytest.raises(IRExecutionError, match=error):
            IRInterpreter(b.program).run({"x": np.ones(2, np.int32)})


def test_peephole_does_not_turn_memory_arithmetic_into_valid_pointer_use():
    b = builder()
    ptr = b.alloca(4, D.INT32)
    invalid = b.add(ptr, b.make_const(0, D.INT32))
    b.store(invalid, b.make_const(1, D.INT32))
    b.ret(b.load(invalid))
    assert IRPeepholeOptimizer().optimize(b.program) == 0
    with pytest.raises(IRExecutionError, match="MemoryError"):
        IRInterpreter(b.program).run({})


def test_peephole_preserves_value_captured_before_for_counter_increment():
    b = builder()
    counter = b.for_loop(0, 2)
    captured = b.add(counter, b.make_const(0, D.INT32))
    b.endfor()
    b.ret(captured)
    _, count, actual = equivalent(b.program, {}, IRPeepholeOptimizer())
    assert count == 0 and actual == 1


@pytest.mark.parametrize("pass_type", [ConstantFolder, IRPeepholeOptimizer])
def test_parameter_constant_hint_never_overrides_runtime_value(pass_type):
    x = Value("x", D.INT32, is_constant=True, const_value=0)
    b = builder(x)
    if pass_type is ConstantFolder:
        result = b.add(x, b.make_const(2, D.INT32))
    else:
        result = b.add(b.make_const(2, D.INT32), x)
    b.ret(result)
    _, count, actual = equivalent(b.program, {"x": np.array(7, np.int32)}, pass_type())
    assert count == 0 and actual == 9


@pytest.mark.parametrize("dtype,numpy_dtype,power", [(D.FLOAT32, np.float32, 24), (D.FLOAT64, np.float64, 53)])
def test_constant_folder_rounds_each_float_operation(dtype, numpy_dtype, power):
    b = builder()
    a = b.make_const(2**power, dtype)
    b.ret(b.sub(b.add(a, b.make_const(1, dtype)), a))
    changed, count, actual = equivalent(b.program, {}, ConstantFolder())
    assert count == 2 and actual == numpy_dtype(0)
    assert ConstantFolder().optimize(changed) == 0


@pytest.mark.parametrize("dtype,numpy_dtype", DTYPES[2:])
@pytest.mark.parametrize("opcode,a,b", [
    (O.ADD, "maximum", 1), (O.SUB, "minimum", 1), (O.MUL, "maximum", 2),
    (O.DIV, -7, 3), (O.DIV, 7, -3), (O.ADD, 2**30 + 1, 0),
])
def test_constant_folder_keeps_integer_width_and_truncated_division(dtype, numpy_dtype, opcode, a, b):
    limits = np.iinfo(numpy_dtype)
    a = limits.max if a == "maximum" else limits.min if a == "minimum" else a
    ir = builder()
    dest = ir.make_value(dtype=dtype)
    ir._emit(opcode, dest, [ir.make_const(int(a), dtype), ir.make_const(b, dtype)])
    ir.ret(dest)
    changed, count, _ = equivalent(ir.program, {}, ConstantFolder())
    assert count == 1
    assert isinstance(changed.functions[0].blocks[0].instructions[0].attrs["value"], int)


def test_constant_folder_keeps_all_int64_bits():
    b = builder()
    a = b.load_const(2**60, D.INT64)
    b.ret(b.sub(b.add(a, b.make_const(1, D.INT64)), a))
    _, count, actual = equivalent(b.program, {}, ConstantFolder())
    assert count == 2 and actual == 1


@pytest.mark.parametrize("dtype,opcode,a,b", [
    (D.FLOAT32, O.MUL, float(np.finfo(np.float32).max), 2.0),
    (D.FLOAT64, O.DIV, 1.0, 0.0),
    (D.INT32, O.DIV, -(2**31), -1),
    (D.INT64, O.DIV, -(2**63), -1),
    (D.INT64, O.DIV, 1, 0),
])
def test_constant_folder_preserves_numeric_errors(dtype, opcode, a, b):
    ir = builder()
    result = ir.make_value(dtype=dtype)
    ir._emit(opcode, result, [ir.make_const(a, dtype), ir.make_const(b, dtype)])
    ir.ret(result)
    assert ConstantFolder().optimize(ir.program) == 0
    with pytest.raises(IRExecutionError, match="NumericError"):
        IRInterpreter(ir.program).run({})


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype,numpy_dtype", DTYPES)
def test_mul_add_keeps_shared_products_and_binary_tensor_semantics(shared, dtype, numpy_dtype):
    x, y, z = [Value(name, dtype, shape=shape)
               for name, shape in [("x", (2, 1)), ("y", (1, 3)), ("z", (3,))]]
    b = builder(x, y, z)
    product = b.mul(x, y)
    total = b.add(product, z)
    b.ret(b.add(total, product) if shared else total)
    feed = {"x": np.array([[2], [-3]], numpy_dtype), "y": np.array([[1, 2, 4]], numpy_dtype),
            "z": np.array([3, 5, 7], numpy_dtype)}
    changed, count, _ = equivalent(b.program, feed, MulAddFusion())
    assert count == 0 and changed.dump() == b.program.dump()


@pytest.mark.parametrize("level", ["none", "basic", "all"])
def test_real_qwen_artifact_optimization_matches_ort(level):
    """Optional local integration; ordinary unit tests need no downloaded model."""
    directory = Path(__file__).resolve().parents[1] / "output/qwen3-small-pr91"
    if not (directory / "model.onnx").is_file():
        pytest.skip("Local Qwen3 probe artifacts have not been generated")
    from scratchv.frontend.onnx_parser import ONNXParser

    parser = ONNXParser()
    program = parser.parse(str(directory / "model.onnx"))
    manager = create_optimization_pass_manager(level)

    def check(pass_, current):
        assert verify_ir(current) == (True, []), pass_.name

    manager.before_pass = manager.after_pass = check
    manager.run(program)
    cases = sorted(directory.glob("inputs_*.npz"))
    assert len(cases) == 7
    for path in cases:
        name = path.stem.removeprefix("inputs_")
        with np.load(path) as inputs:
            feed = dict(inputs)
        actual = IRInterpreter(program).run(feed, initializers=parser.initializers).return_value
        reference = np.load(directory / f"logits_{name}_ort.npy")
        baseline = np.load(directory / f"logits_{name}_ir.npy")
        assert actual.shape == reference.shape == (1, 256, 128)
        assert actual.dtype == reference.dtype == np.float32
        assert np.isfinite(actual).all()
        assert np.max(np.abs(actual - reference)) < 1e-5
        np.testing.assert_array_equal(actual, baseline)
