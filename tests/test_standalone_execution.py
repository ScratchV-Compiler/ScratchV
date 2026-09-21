"""Execute standalone RV32 code against integer references and the tracked CNN."""

from contextlib import redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import re
import resource
import shutil
import struct
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from scratchv.standalone.onnx_to_riscv_standalone import (
    RISCVEmitter, convert_onnx_to_riscv, rv_bne, rv_j,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 7)


@pytest.fixture(scope="module")
def execution_tools():
    missing = [name for name in ("clang", "ld.lld", "qemu-riscv32") if not shutil.which(name)]
    if missing:
        if os.environ.get("SCRATCHV_REQUIRE_RISCV_EXECUTION") == "1":
            pytest.fail(f"Execution tools missing: {missing}")
        pytest.skip(f"Execution tools missing: {missing}")


def execute_binary(binary, workspace_bytes, values, output_elements, directory):
    """Call the unmodified flat binary using its bare-metal input/output ABI."""
    input_path = directory / "input.bin"
    input_path.write_bytes(np.asarray(values, dtype="<i4").tobytes())
    assembly, elf = directory / "execute.s", directory / "execute.elf"
    guard_bytes, output_bytes = 256, output_elements * 4
    total = workspace_bytes + output_bytes + 3 * guard_bytes
    assembly.write_text(f'''.option norvc
.option norelax
.text
.globl _start
_start:
 la sp, workspace
 la a0, input_tensor
 la a1, output_tensor
 call cnn_entry
 li a0, 1
 la a1, dump_start
 li a2, {total}
 li a7, 64
 ecall
 li t0, {total}
 bne a0, t0, failed
 li a0, 0
 li a7, 93
 ecall
failed:
 li a0, 1
 li a7, 93
 ecall
.balign 4
cnn_entry:
 .incbin "{binary.as_posix()}"
.data
.balign 4
input_tensor:
 .incbin "{input_path.as_posix()}"
.bss
.balign 16
dump_start:
 .space {guard_bytes}
workspace:
 .space {workspace_bytes}
 .space {guard_bytes}
output_tensor:
 .space {output_bytes}
 .space {guard_bytes}
''')
    subprocess.run([
        "clang", "--target=riscv32-linux-gnu", "-march=rv32im", "-mabi=ilp32",
        "-nostdlib", "-static", "-fuse-ld=lld", "-Wl,--no-relax",
        str(assembly), "-o", str(elf),
    ], capture_output=True, check=True, timeout=30)
    result = subprocess.run(
        ["qemu-riscv32", str(elf)], capture_output=True, timeout=60,
        preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)),
    )
    assert result.returncode == 0, (result.returncode, result.stderr.decode(errors="replace"))
    assert len(result.stdout) == total
    data = result.stdout
    end_workspace = guard_bytes + workspace_bytes
    output_start = end_workspace + guard_bytes
    assert not any(data[:guard_bytes]), "Write before workspace"
    assert not any(data[end_workspace:output_start]), "Write beyond workspace"
    assert not any(data[output_start + output_bytes:]), "Write beyond output"
    return {
        "output_q16": list(struct.unpack(f"<{output_elements}i", data[output_start:output_start + output_bytes])),
        "workspace_sha256": hashlib.sha256(data[guard_bytes:end_workspace]).hexdigest(),
    }


def run_model(directory, path, input_elements, output_elements, *, compact):
    directory.mkdir(exist_ok=True)
    binary, assembly = directory / "cnn.bin", directory / "cnn.s"
    log = io.StringIO()
    with redirect_stdout(log):
        assert convert_onnx_to_riscv(str(path), str(binary), str(assembly), const_merge=compact) == 0
    # The public compiler reports its allocated workspace; do not duplicate its planner.
    match = re.search(r"Workspace: ([\d,]+) bytes", log.getvalue())
    assert match, log.getvalue()
    workspace_bytes = int(match[1].replace(",", ""))
    return [{"seed": seed, **execute_binary(binary, workspace_bytes,
                input_values((input_elements,), seed), output_elements, directory)} for seed in SEEDS]


def run_graph(tmp_path, shape, out_shape, nodes, tensors, *, compact=True):
    graph = helper.make_graph(
        nodes, "standalone-execution",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, out_shape)], tensors,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    path = tmp_path / "case.onnx"
    onnx.save(model, path)
    return run_model(tmp_path, path, int(np.prod(shape)), int(np.prod(out_shape)), compact=compact)


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
def test_repeated_convolutions_preserve_coordinates_bias_and_channel_stride(tmp_path, execution_tools, kernel, pad, compact):
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
        assert sample["output_q16"] == expected.ravel().tolist()


def test_repeated_relu_pool_reshape_and_gemm_use_their_own_loop_targets(tmp_path, execution_tools):
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
        assert sample["output_q16"] == [expected]


def test_sigmoid_emits_the_half_constant_without_immediate_truncation(tmp_path, execution_tools):
    shape = (1, 4)
    nodes = [helper.make_node("Sigmoid", ["input"], ["output"])]
    for sample in run_graph(tmp_path, shape, shape, nodes, []):
        expected = 32768 + (input_values(shape, sample["seed"]) >> 3)
        assert sample["output_q16"] == expected.ravel().tolist()


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


def test_undefined_loop_target_fails_instead_of_emitting_a_zero_offset_jump():
    emitter = RISCVEmitter()
    emitter.emit_jump(rv_j, "missing")
    with pytest.raises(ValueError, match="Undefined label"):
        emitter.resolve_fixups()


def test_tracked_cnn_runs_with_and_without_constant_merge(tmp_path, execution_tools):
    path = ROOT / "models/graph/cnn.onnx"
    assert path.is_file(), "Regression requires the tracked CNN, not a generated substitute"
    model = onnx.load(path)
    elements = lambda value: int(np.prod([d.dim_value for d in value.type.tensor_type.shape.dim]))
    inputs, outputs = elements(model.graph.input[0]), elements(model.graph.output[0])
    baseline = run_model(tmp_path / "default", path, inputs, outputs, compact=False)
    compact = run_model(tmp_path / "const-merge", path, inputs, outputs, compact=True)
    # Normal return and unchanged guards are checked by execute_binary.
    assert baseline == compact  # Compare all workspace bytes as well as final outputs.
    assert all(0 <= value <= 65536 for row in baseline for value in row["output_q16"])
