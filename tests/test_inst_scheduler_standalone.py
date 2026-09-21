"""Execute generated RV32 CNN operators against independent integer references."""

from contextlib import redirect_stdout
import io
import os
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from benchmarks.cnn_schedule_execution import assemble_listing, execute_pair
from scratchv.standalone.onnx_to_riscv_standalone import (
    RISCVEmitter, convert_onnx_to_riscv, rv_bne,
)


@pytest.fixture(autouse=True)
def execution_tools():
    missing = [name for name in ("clang", "ld.lld", "qemu-riscv32") if not shutil.which(name)]
    if missing:
        if os.environ.get("SCRATCHV_REQUIRE_RISCV_EXECUTION") == "1":
            pytest.fail(f"Execution tools missing: {missing}")
        pytest.skip(f"Execution tools missing: {missing}")


def run_graph(tmp_path, shape, out_shape, nodes, tensors, *, compact=True):
    graph = helper.make_graph(
        nodes, "standalone-execution",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, out_shape)],
        tensors,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    path = tmp_path / "case.onnx"
    onnx.save(model, path)
    metadata = {}
    binaries = []
    for enabled, name in ((False, "before"), (True, "after")):
        binary, assembly = tmp_path / f"{name}.bin", tmp_path / f"{name}.s"
        with redirect_stdout(io.StringIO()):
            assert convert_onnx_to_riscv(str(path), str(binary), str(assembly),
                                       const_merge=compact, schedule=enabled, metadata=metadata) == 0
        assert assemble_listing(assembly, tmp_path) == binary.read_bytes()[:metadata["code_bytes"]]
        binaries.append(binary)
    result = execute_pair(*binaries, metadata, tmp_path)
    assert result["status"] == "passed", result
    return result["samples"]


def input_values(shape, seed):
    n = int(np.prod(shape))
    return (np.zeros(n, dtype=np.int64) if seed == 0 else
            (np.arange(n, dtype=np.int64) * 37 + seed * 17) % 65536 - 32768).reshape(shape)


def fixed_tensor(name, values):
    # Powers-of-two scaling is exact; small products avoid int32 overflow.
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32) / 65536, name)


def conv_reference(x, weights, bias, stride, pad):
    _, channels, height, width = x.shape
    outputs, _, kh, kw = weights.shape
    sh, sw = stride
    oh, ow = (height + 2 * pad - kh) // sh + 1, (width + 2 * pad - kw) // sw + 1
    padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    result = np.empty((1, outputs, oh, ow), dtype=np.int64)
    for oc in range(outputs):
        for row in range(oh):
            for col in range(ow):
                window = padded[0, :channels, row * sh:row * sh + kh, col * sw:col * sw + kw]
                result[0, oc, row, col] = bias[oc] + np.sum((window * weights[oc]) >> 16)
    return result


@pytest.mark.parametrize("kernel,pad", [(3, 0), (2, 0), (3, 1)])
@pytest.mark.parametrize("compact", [False, True])
def test_repeated_convolutions_preserve_coordinates_bias_and_channel_stride(tmp_path, kernel, pad, compact):
    shape = (1, 2, 8, 9)
    weights = ((np.arange(2 * 2 * kernel * kernel).reshape(2, 2, kernel, kernel) % 11) - 5) * 256
    bias = np.array([512, -128], dtype=np.int64)
    first_shape = conv_reference(np.zeros(shape, dtype=np.int64), weights, bias, (1, 2), pad).shape
    out_shape = conv_reference(np.zeros(first_shape, dtype=np.int64), weights, bias, (2, 1), pad).shape
    nodes = [helper.make_node("Conv", [source, "weights", "bias"], [target], name="same_name",
                              kernel_shape=[kernel, kernel], strides=list(stride), pads=[pad] * 4)
             for source, target, stride in (("input", "middle", (1, 2)), ("middle", "output", (2, 1)))]
    samples = run_graph(tmp_path, shape, out_shape, nodes,
                        [fixed_tensor("weights", weights), fixed_tensor("bias", bias)], compact=compact)
    for sample in samples:
        x = input_values(shape, sample["seed"])
        expected = conv_reference(conv_reference(x, weights, bias, (1, 2), pad), weights, bias, (2, 1), pad)
        assert sample["before"]["output_q16"] == expected.ravel().tolist()


def test_repeated_relu_pool_reshape_and_gemm_use_their_own_loop_targets(tmp_path):
    shape = (1, 1, 64, 64)  # Fixed inputs include both positive and negative values.
    weights = (np.arange(512).reshape(2, 256) % 8 - 3) * 256
    weights2 = np.array([[512, -256]], dtype=np.int64)
    nodes = [
        helper.make_node("Relu", ["input"], ["relu1"]),
        helper.make_node("MaxPool", ["relu1"], ["pool1"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Relu", ["pool1"], ["relu2"]),
        helper.make_node("MaxPool", ["relu2"], ["pool2"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Reshape", ["pool2", "shape1"], ["flat"]),
        helper.make_node("Gemm", ["flat", "weights1", "bias1"], ["fc1"], transB=1),
        helper.make_node("Gemm", ["fc1", "weights2", "bias2"], ["fc2"], transB=1),
        helper.make_node("Reshape", ["fc2", "shape2"], ["output"]),
    ]
    tensors = [fixed_tensor("weights1", weights), fixed_tensor("bias1", [1024, 512]),
               fixed_tensor("weights2", weights2), fixed_tensor("bias2", [128]),
               numpy_helper.from_array(np.array([1, 256], dtype=np.int64), "shape1"),
               numpy_helper.from_array(np.array([1, 1], dtype=np.int64), "shape2")]
    for sample in run_graph(tmp_path, shape, (1, 1), nodes, tensors):
        x = np.maximum(input_values(shape, sample["seed"]), 0)
        for _ in range(2):
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] // 2, 2, x.shape[3] // 2, 2).max(axis=(3, 5))
        fc1 = np.sum((x.ravel() * weights) >> 16, axis=1) + [1024, 512]
        expected = int(np.sum((fc1 * weights2[0]) >> 16)) + 128
        assert sample["before"]["output_q16"] == [expected]


def test_sigmoid_emits_the_half_constant_without_immediate_truncation(tmp_path):
    shape = (1, 4)
    nodes = [helper.make_node("Sigmoid", ["input"], ["output"])]
    for sample in run_graph(tmp_path, shape, shape, nodes, []):
        expected = 32768 + (input_values(shape, sample["seed"]) >> 3)
        assert sample["before"]["output_q16"] == expected.ravel().tolist()


def test_duplicate_labels_and_unencodable_branch_offsets_fail():
    emitter = RISCVEmitter()
    emitter.label("loop")
    with pytest.raises(ValueError, match="Duplicate label"):
        emitter.label("loop")
    emitter.emit_branch(rv_bne, 5, 0, "too_far")
    emitter.code.extend([0] * 1024)
    emitter.label("too_far")
    with pytest.raises(ValueError, match="Out-of-range"):
        emitter.resolve_fixups()
