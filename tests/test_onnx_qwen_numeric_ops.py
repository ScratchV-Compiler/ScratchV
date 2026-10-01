"""Qwen3 ONNX numeric operators and exported static-parameter chains vs ORT."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto as T, helper, numpy_helper

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.frontend.onnx_parser import ONNXParser, ONNXParseError
from scratchv.verification.ir_interpreter import IRExecutionError, IRInterpreter

ort = pytest.importorskip("onnxruntime")
DTYPES = {"float32": T.FLOAT, "float64": T.DOUBLE, "int32": T.INT32, "int64": T.INT64}


def save(tmp_path, nodes, feed, output_dtype, output_shape, initializers=None):
    model = helper.make_model(helper.make_graph(
        nodes, "qwen_numeric", [helper.make_tensor_value_info(n, DTYPES[str(a.dtype)], a.shape)
                                for n, a in feed.items()],
        [helper.make_tensor_value_info("y", output_dtype, output_shape)],
        [numpy_helper.from_array(a, n) for n, a in (initializers or {}).items()]),
        opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    return path


def compare(path, feed, *, exact=False):
    parser = ONNXParser()
    program = parser.parse(str(path))
    assert verify_ir(program) == (True, [])
    actual = IRInterpreter(program).run(feed, initializers=parser.initializers).return_value
    reference = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feed)[0]
    assert actual.shape == reference.shape
    assert actual.dtype == reference.dtype
    if exact or actual.dtype.kind in "iu":
        np.testing.assert_array_equal(actual, reference)
    else:
        np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-6)
    return parser, program, actual


@pytest.mark.parametrize("source", DTYPES)
@pytest.mark.parametrize("target", DTYPES)
def test_cast_all_supported_numeric_pairs(tmp_path, source, target):
    values = [[-3.75, -0.75, 0.0], [0.75, 3.75, 123.0]]
    feed = {"x": np.array(values, dtype=source)}
    path = save(tmp_path, [helper.make_node("Cast", ["x"], ["y"], to=DTYPES[target])],
                feed, DTYPES[target], [2, 3])
    compare(path, feed, exact=True)


def test_cast_integer_narrowing_keeps_onnx_low_bits(tmp_path):
    feed = {"x": np.array([2**32 + 3, -(2**32 + 3), 2**31], dtype="int64")}
    path = save(tmp_path, [helper.make_node("Cast", ["x"], ["y"], to=T.INT32)], feed, T.INT32, [3])
    compare(path, feed, exact=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(), (1,), (2, 3)])
def test_identity_preserves_dtype_and_rank(tmp_path, dtype, shape):
    feed = {"x": np.full(shape, -3, dtype=dtype)}
    path = save(tmp_path, [helper.make_node("Identity", ["x"], ["y"])], feed, DTYPES[dtype], shape)
    _, _, actual = compare(path, feed, exact=True)
    actual[...] = 42
    np.testing.assert_array_equal(feed["x"], np.full(shape, -3, dtype=dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_abs_preserves_numeric_type(tmp_path, dtype):
    feed = {"x": np.array([[-17, 0, 12], [4, -2, 99]], dtype=dtype)}
    path = save(tmp_path, [helper.make_node("Abs", ["x"], ["y"])], feed, DTYPES[dtype], [2, 3])
    compare(path, feed, exact=True)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("op,values", [("Sin", [-6.1, -1.25, 0, 1.2, 6.1]),
                                     ("Cos", [-6.1, -1.25, 0, 1.2, 6.1]),
                                     ("Reciprocal", [-10, -0.25, 0.5, 2, 1000])])
def test_float_unary_ops(tmp_path, dtype, op, values):
    feed = {"x": np.array(values, dtype=dtype)}
    path = save(tmp_path, [helper.make_node(op, ["x"], ["y"])], feed, DTYPES[dtype], [5])
    compare(path, feed)


@pytest.mark.parametrize("base_dtype,exponent_dtype", [
    ("float32", "int64"), ("float32", "float32"), ("float64", "int32"),
    ("float64", "float64"), ("int32", "int32"), ("int64", "int64"),
])
def test_pow_broadcast_and_independent_exponent_type(tmp_path, base_dtype, exponent_dtype):
    feed = {"base": np.array([[1], [2], [3]], dtype=base_dtype),
            "exponent": np.array([1, 2, 3], dtype=exponent_dtype)}
    path = save(tmp_path, [helper.make_node("Pow", ["base", "exponent"], ["y"])],
                feed, DTYPES[base_dtype], [3, 3])
    compare(path, feed)


def test_pow_qwen_float32_with_int64_scalar_two(tmp_path):
    feed = {"x": np.array([[-3.125, 0.5, 1.25]], dtype="float32")}
    nodes = [helper.make_node("Constant", [], ["exponent"], value_int=2),
             helper.make_node("Pow", ["x", "exponent"], ["y"])]
    path = save(tmp_path, nodes, feed, T.FLOAT, [1, 3])
    compare(path, feed)


@pytest.mark.parametrize("attribute,value,dtype,shape", [
    ("value_int", 2**63 - 1, T.INT64, ()),
    ("value_ints", [-(2**63), 2**63 - 1], T.INT64, (2,)),
    ("value_ints", [7], T.INT64, (1,)),
    ("value_ints", [], T.INT64, (0,)),
    ("value_float", 1.25, T.FLOAT, ()),
    ("value_floats", [1.25], T.FLOAT, (1,)),
    ("value_floats", [-1.25, 2.5], T.FLOAT, (2,)),
])
def test_constant_attributes_preserve_value_and_rank(tmp_path, attribute, value, dtype, shape):
    # Explicit AttributeProto types permit an empty repeated field.
    node = helper.make_node("Constant", [], ["y"])
    attr_type = onnx.AttributeProto.INTS if attribute == "value_ints" else None
    node.attribute.append(helper.make_attribute(attribute, value, attr_type=attr_type))
    path = save(tmp_path, [node], {}, dtype, shape)
    compare(path, {}, exact=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(), (1,), (2, 2)])
def test_constant_tensor_payload(tmp_path, dtype, shape):
    data = np.full(shape, 7, dtype=dtype)
    node = helper.make_node("Constant", [], ["y"], value=numpy_helper.from_array(data))
    path = save(tmp_path, [node], {}, DTYPES[dtype], shape)
    compare(path, {}, exact=True)


def test_constant_external_tensor_payload(tmp_path):
    node = helper.make_node("Constant", [], ["y"],
                            value=numpy_helper.from_array(np.arange(6, dtype="float32")))
    path = save(tmp_path, [node], {}, T.FLOAT, [6])
    model = onnx.load(path)
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="constant.data", size_threshold=0, convert_attribute=True)
    compare(path, {}, exact=True)


def test_constant_cast_abs_expand_chain(tmp_path):
    nodes = [helper.make_node("Constant", [], ["shape"], value_ints=[-2, 3]),
             helper.make_node("Cast", ["shape"], ["cast_shape"], to=T.INT64),
             helper.make_node("Abs", ["cast_shape"], ["positive_shape"]),
             helper.make_node("Expand", ["x", "positive_shape"], ["y"])]
    feed = {"x": np.array([[1, 2, 3]], dtype="float32")}
    compare(save(tmp_path, nodes, feed, T.FLOAT, [2, 3]), feed)


def test_constant_cast_reshape_slice_chain_preserves_int64_max(tmp_path):
    nodes = [helper.make_node("Constant", [], ["start"], value_int=1),
             helper.make_node("Constant", [], ["end"], value_int=2**63 - 1),
             helper.make_node("Constant", [], ["shape"], value_ints=[-1]),
             helper.make_node("Cast", ["start"], ["start_cast"], to=T.INT64),
             helper.make_node("Reshape", ["start_cast", "shape"], ["starts"]),
             helper.make_node("Reshape", ["end", "shape"], ["ends"]),
             helper.make_node("Slice", ["x", "starts", "ends"], ["y"])]
    feed = {"x": np.arange(5, dtype="float32")}
    compare(save(tmp_path, nodes, feed, T.FLOAT, [4]), feed)


def test_identity_concat_shape_chain(tmp_path):
    nodes = [helper.make_node("Constant", [], ["a"], value_ints=[3]),
             helper.make_node("Constant", [], ["b"], value_ints=[2]),
             helper.make_node("Identity", ["a"], ["alias"]),
             helper.make_node("Cast", ["b"], ["cast_b"], to=T.INT64),
             helper.make_node("Concat", ["alias", "cast_b"], ["shape"], axis=0),
             helper.make_node("Reshape", ["x", "shape"], ["y"])]
    feed = {"x": np.arange(6, dtype="float32").reshape(2, 3)}
    compare(save(tmp_path, nodes, feed, T.FLOAT, [3, 2]), feed)


def test_qwen_rms_norm_numeric_chain(tmp_path):
    nodes = [helper.make_node("Constant", [], ["two"], value_int=2),
             helper.make_node("Constant", [], ["axis"], value_int=-1),
             helper.make_node("Constant", [], ["shape"], value_ints=[-1]),
             helper.make_node("Reshape", ["axis", "shape"], ["axes"]),
             helper.make_node("Pow", ["x", "two"], ["squared"]),
             helper.make_node("ReduceMean", ["squared", "axes"], ["mean"], keepdims=1),
             helper.make_node("Constant", [], ["eps"], value_float=1e-6),
             helper.make_node("Add", ["mean", "eps"], ["sum"]),
             helper.make_node("Sqrt", ["sum"], ["root"]),
             helper.make_node("Reciprocal", ["root"], ["inverse"]),
             helper.make_node("Mul", ["x", "inverse"], ["y"])]
    feed = {"x": np.array([[[1, -2, 3, -4], [0.25, 0.5, 0.75, 1]]], dtype="float32")}
    compare(save(tmp_path, nodes, feed, T.FLOAT, [1, 2, 4]), feed)


@pytest.mark.parametrize("to", [T.FLOAT16, T.BOOL, T.STRING, T.UINT8])
def test_unsupported_cast_target_is_not_silently_float32(tmp_path, to):
    feed = {"x": np.array([1, 2], dtype="float32")}
    path = save(tmp_path, [helper.make_node("Cast", ["x"], ["y"], to=to)], feed, to, [2])
    with pytest.raises(ONNXParseError, match="Unsupported ONNX element type"):
        ONNXParser().parse(str(path))


@pytest.mark.parametrize("to,data", [(T.FLOAT16, np.array([1], dtype="float16")),
                                   (T.BOOL, np.array([True], dtype="bool"))])
def test_unsupported_constant_type_fails_explicitly(tmp_path, to, data):
    path = save(tmp_path, [helper.make_node("Constant", [], ["y"], value=numpy_helper.from_array(data))],
                {}, to, [1])
    with pytest.raises(ONNXParseError, match="Unsupported ONNX element type"):
        ONNXParser().parse(str(path))


@pytest.mark.parametrize("op,values", [("Reciprocal", [0.0]), ("Cast", [float(2**63)])])
def test_undefined_or_nonfinite_numeric_result_is_rejected(tmp_path, op, values):
    feed = {"x": np.array(values, dtype="float64")}
    attrs = {"to": T.INT64} if op == "Cast" else {}
    path = save(tmp_path, [helper.make_node(op, ["x"], ["y"], **attrs)],
                feed, T.INT64 if op == "Cast" else T.DOUBLE, [1])
    parser = ONNXParser()
    program = parser.parse(str(path))
    with pytest.raises(IRExecutionError, match="NumericError"):
        IRInterpreter(program).run(feed, initializers=parser.initializers)
