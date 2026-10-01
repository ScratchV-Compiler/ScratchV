"""ORT-backed contracts for the eight shape/tensor ops used by the W1 probe.

The frontend currently freezes shape/axis tensors at parse time. These tests
cover that static contract, including failures, rather than every ONNX type.
"""

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper

from scratchv.frontend.onnx_parser import ONNXParseError, ONNXParser
from scratchv.verification.ir_interpreter import IRExecutionError, IRInterpreter

ort = pytest.importorskip("onnxruntime")
DTYPES = ("float32", "float64", "int32", "int64")
FLOATS = ("float32", "float64")
I64_MIN = np.iinfo(np.int64).min
I64_MAX = np.iinfo(np.int64).max


def _model(tmp_path, op, feed, *, constants=None, attrs=None, inputs=None,
           output_shape=None, opset=18, check=True):
    constants = constants or {}
    node = helper.make_node(op, inputs or [*feed, *constants], ["y"], **(attrs or {}))
    output_dtype = helper.np_dtype_to_tensor_dtype(next(iter(feed.values())).dtype)
    graph = helper.make_graph(
        [node], "shape_op",
        [helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(x.dtype), x.shape)
         for name, x in feed.items()],
        [helper.make_tensor_value_info("y", output_dtype, output_shape)],
        [numpy_helper.from_array(x, name) for name, x in constants.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 9
    if check:
        onnx.checker.check_model(model)
    path = tmp_path / "shape_op.onnx"
    onnx.save(model, path)
    return path


def _run_ir(path, feed):
    parser = ONNXParser()
    program = parser.parse(str(path))
    return IRInterpreter(program).run(feed, initializers=parser.initializers).return_value


def _compare(path, feed):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # Exercise the operator itself rather than relying on an ORT graph rewrite.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    expected = ort.InferenceSession(str(path), options,
                                    providers=["CPUExecutionProvider"]).run(None, feed)[0]
    actual = _run_ir(path, feed)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if actual.dtype.kind in "iu":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    return actual


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("axis,indices", [
    (None, np.array([0, -1], dtype="int64")),
    (-1, np.array(-1, dtype="int32")),
    (1, np.array([[0, -1], [-2, 1]], dtype="int64")),
    (0, np.array([], dtype="int64")),
], ids=["default-axis", "scalar-negative-index", "matrix-indices", "empty-indices"])
def test_gather(tmp_path, dtype, axis, indices):
    x = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    attrs = {} if axis is None else {"axis": axis}
    shape = np.take(x, indices, axis=axis or 0).shape
    feed = {"x": x, "indices": indices}
    path = _model(tmp_path, "Gather", feed, attrs=attrs, output_shape=shape)
    _compare(path, feed)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shapes,axis", [
    ([(1, 3), (2, 3), (1, 3)], 0),
    ([(2, 1), (2, 3)], -1),
    ([(2, 0), (2, 3)], 1),
    ([(2, 3)], -2),
], ids=["three-inputs", "negative-axis", "empty-input", "single-input"])
def test_concat(tmp_path, dtype, shapes, axis):
    feed = {f"x{i}": np.arange(np.prod(shape), dtype=dtype).reshape(shape) + i
            for i, shape in enumerate(shapes)}
    shape = np.concatenate(list(feed.values()), axis=axis).shape
    path = _model(tmp_path, "Concat", feed, attrs={"axis": axis}, output_shape=shape)
    _compare(path, feed)


@pytest.mark.parametrize("shape,target,dtype", [
    ((1, 3), [2, 3], "float32"),
    ((2, 3), [3], "int64"),
    ((2, 1), [1, 4], "float32"),
    ((3,), [2, 1, 3], "float32"),
    ((), [2, 3], "float64"),
    ((2, 3), [], "int32"),
    ((1, 3), [0, 3], "float32"),
], ids=["broadcast-axis", "shorter-target", "bidirectional", "leading-dimensions", "scalar",
        "empty-target", "zero-dimension"])
def test_expand(tmp_path, shape, target, dtype):
    x = np.arange(np.prod(shape, dtype=int), dtype=dtype).reshape(shape)
    feed = {"x": x}
    path = _model(tmp_path, "Expand", feed,
                  constants={"shape": np.array(target, dtype="int64")},
                  output_shape=np.broadcast_shapes(shape, tuple(target)))
    _compare(path, feed)


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("opset,axes,keepdims,noop", [
    (17, None, 1, None),
    (17, [-1], 0, None),
    (17, [0, -1], 1, None),
    (18, [1], 0, 0),
    (18, [-1], None, 0),
    (18, [-1, 0], 0, 0),
    (18, [], 1, 0),
    (18, [], 0, 1),
    (18, None, 0, 1),
], ids=["v17-all-default", "v17-negative", "v17-multiple", "v18-input-axes", "v18-default-keepdims",
        "v18-unsorted", "v18-empty-reduce", "v18-empty-noop", "v18-omitted-noop"])
def test_reduce_mean(tmp_path, dtype, opset, axes, keepdims, noop):
    x = np.arange(24, dtype=dtype).reshape(2, 3, 4) / 7
    attrs = {} if keepdims is None else {"keepdims": keepdims}
    constants = {}
    if opset == 17:
        if axes is not None:
            attrs["axes"] = axes
    else:
        attrs["noop_with_empty_axes"] = noop
        if axes is not None:
            constants["axes"] = np.array(axes, dtype="int64")
    expected_shape = (x.shape if not axes and noop else
                      np.mean(x, axis=tuple(axes) if axes else None,
                              keepdims=keepdims is None or bool(keepdims)).shape)
    feed = {"x": x}
    path = _model(tmp_path, "ReduceMean", feed, constants=constants, attrs=attrs,
                  output_shape=expected_shape, opset=opset)
    _compare(path, feed)


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("shape", [(), (1,), (2, 3)], ids=["scalar", "vector", "matrix"])
def test_sqrt(tmp_path, dtype, shape):
    x = np.arange(np.prod(shape, dtype=int), dtype=dtype).reshape(shape)
    if x.ndim == 0:
        x[...] = 1e-8
    feed = {"x": x}
    path = _model(tmp_path, "Sqrt", feed, output_shape=shape)
    _compare(path, feed)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("perm", [None, [0, 1, 2], [0, 2, 1]])
def test_transpose(tmp_path, dtype, perm):
    x = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    feed = {"x": x}
    path = _model(tmp_path, "Transpose", feed, attrs={} if perm is None else {"perm": perm},
                  output_shape=np.transpose(x, perm).shape)
    _compare(path, feed)


@pytest.mark.parametrize("shape,axes,dtype", [
    ((2, 3), [0], "float32"),
    ((2, 3), [-1], "int64"),
    ((2, 3), [3, 0], "float32"),
    ((2, 3), [-4, -1], "float64"),
    ((), [0, 1], "int32"),
], ids=["leading", "trailing-negative", "unsorted", "multiple-negative", "scalar"])
def test_unsqueeze(tmp_path, shape, axes, dtype):
    x = np.arange(np.prod(shape, dtype=int), dtype=dtype).reshape(shape)
    feed = {"x": x}
    path = _model(tmp_path, "Unsqueeze", feed,
                  constants={"axes": np.array(axes, dtype="int64")},
                  output_shape=np.expand_dims(x, tuple(axes)).shape)
    _compare(path, feed)


@pytest.mark.parametrize("shape,starts,ends,axes,steps,out_shape", [
    ((10,), [1], [8], None, None, (7,)),
    ((10,), [1], [8], None, [2], (4,)),
    ((2, 6), [1], [6], [-1], [2], (2, 3)),
    ((3, 4), [1, 0], [4, 3], [1, 0], [2, 2], (2, 2)),
    ((10,), [8], [1], [0], [-2], (4,)),
    ((5,), [I64_MAX], [I64_MIN], None, [-1], (5,)),
    ((5,), [-20], [I64_MIN], None, [-1], (1,)),
    ((5,), [4], [-1], None, [-1], (0,)),
    ((5,), [4], [-5], None, [-1], (4,)),
    ((5,), [-20], [20], None, None, (5,)),
    ((5,), [4], [1], None, None, (0,)),
    ((0, 3), [0], [I64_MAX], [0], [1], (0, 3)),
    ((0, 3), [I64_MAX], [I64_MIN], [0], [-1], (0, 3)),
], ids=["defaults", "steps-without-axes", "negative-axis", "multiple-axes",
        "reverse-stride", "reverse-sentinels", "reverse-clamp-start",
        "reverse-end-minus-one", "reverse-excludes-first",
        "forward-clamp", "empty-result", "empty-input", "reverse-empty-input"])
def test_slice(tmp_path, shape, starts, ends, axes, steps, out_shape):
    x = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    constants = {"starts": np.array(starts, dtype="int64"),
                 "ends": np.array(ends, dtype="int64")}
    inputs = ["x", "starts", "ends"]
    if axes is not None or steps is not None:
        inputs.append("axes" if axes is not None else "")
    if axes is not None:
        constants["axes"] = np.array(axes, dtype="int64")
    if steps is not None:
        inputs.append("steps")
        constants["steps"] = np.array(steps, dtype="int64")
    feed = {"x": x}
    path = _model(tmp_path, "Slice", feed, constants=constants, inputs=inputs,
                  output_shape=out_shape)
    _compare(path, feed)


@pytest.mark.parametrize("op,attrs,constants,extra_feed", [
    ("Gather", {"axis": 2}, {}, {"indices": np.array([0], dtype="int64")}),
    ("Gather", {}, {}, {"indices": np.array([2], dtype="int64")}),
    ("Gather", {}, {}, {"indices": np.array([-3], dtype="int32")}),
    ("Concat", {"axis": 2}, {}, {"other": np.ones((2, 3), dtype="float32")}),
    ("Concat", {"axis": 0}, {}, {"other": np.ones((2, 4), dtype="float32")}),
    ("Expand", {}, {"shape": np.array([2, 4], dtype="int64")}, {}),
    ("Expand", {}, {"shape": np.array([-1, 3], dtype="int64")}, {}),
    ("ReduceMean", {}, {"axes": np.array([2], dtype="int64")}, {}),
    ("ReduceMean", {}, {"axes": np.array([0, 0], dtype="int64")}, {}),
    ("Transpose", {"perm": [0, 0]}, {}, {}),
    ("Transpose", {"perm": [1]}, {}, {}),
    ("Unsqueeze", {}, {"axes": np.array([3], dtype="int64")}, {}),
    ("Unsqueeze", {}, {"axes": np.array([0, 0], dtype="int64")}, {}),
    ("Unsqueeze", {}, {"axes": np.array([0, -4], dtype="int64")}, {}),
], ids=["gather-axis", "gather-index-high", "gather-index-low", "concat-axis",
        "concat-shape", "expand-broadcast", "expand-negative", "mean-axis",
        "mean-duplicates", "transpose-duplicates", "transpose-rank",
        "unsqueeze-axis", "unsqueeze-duplicates", "unsqueeze-normalized-duplicates"])
def test_invalid_shapes_axes_and_indices_fail_explicitly(tmp_path, op, attrs,
                                                       constants, extra_feed):
    feed = {"x": np.arange(6, dtype="float32").reshape(2, 3), **extra_feed}
    path = _model(tmp_path, op, feed, attrs=attrs, constants=constants, check=False)
    with pytest.raises((ONNXParseError, IRExecutionError)):
        _run_ir(path, feed)


@pytest.mark.parametrize("starts,ends,axes,steps", [
    ([0], [2], [0], [0]),
    ([0, 0], [2], [0, 1], [1, 1]),
    ([0], [2], [2], [1]),
    ([0, 0], [2, 2], [0, 0], [1, 1]),
], ids=["zero-step", "unequal-lengths", "axis-out-of-range", "duplicate-axes"])
def test_invalid_slice_parameters_fail_explicitly(tmp_path, starts, ends, axes, steps):
    feed = {"x": np.arange(6, dtype="float32").reshape(2, 3)}
    constants = {name: np.array(values, dtype="int64")
                 for name, values in zip(("starts", "ends", "axes", "steps"),
                                         (starts, ends, axes, steps))}
    path = _model(tmp_path, "Slice", feed, constants=constants, check=False)
    with pytest.raises((ONNXParseError, IRExecutionError)):
        _run_ir(path, feed)


@pytest.mark.parametrize("op,parameter,value", [
    ("Expand", "shape", [2, 3]),
    ("Unsqueeze", "axes", [0]),
    ("ReduceMean", "axes", [1]),
    ("Slice", "starts", [0]),
])
def test_runtime_shape_parameters_are_rejected_by_static_frontend(tmp_path, op,
                                                               parameter, value):
    feed = {"x": np.arange(6, dtype="float32").reshape(2, 3),
            parameter: np.array(value, dtype="int64")}
    constants = {"ends": np.array([1], dtype="int64")} if op == "Slice" else {}
    path = _model(tmp_path, op, feed, constants=constants, check=False)
    with pytest.raises(ONNXParseError, match="constant integer vector"):
        ONNXParser().parse(str(path))


def test_sqrt_negative_data_is_an_explicit_numeric_error(tmp_path):
    # IR execution intentionally rejects nonfinite arithmetic rather than
    # silently returning the NaN produced by ONNX Sqrt for a negative argument.
    feed = {"x": np.array([-1.0, 4.0], dtype="float32")}
    path = _model(tmp_path, "Sqrt", feed, output_shape=[2])
    with pytest.raises(IRExecutionError, match="NumericError"):
        _run_ir(path, feed)
