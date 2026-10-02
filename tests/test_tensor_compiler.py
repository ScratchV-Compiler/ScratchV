"""Exercise the shared compiler entry, real weights and fail-closed backends."""
import json

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import pytest

from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.main import args_to_config, build_arg_parser


def model_file(tmp_path):
    weights = np.array([[1.25, -2.5], [3.0, 4.5]], dtype=np.float32)
    model = helper.make_model(helper.make_graph(
        [helper.make_node("MatMul", ["x", "weights"], ["y"])], "projection",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])],
        [numpy_helper.from_array(weights, "weights")],
    ), opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    return path, weights


@pytest.mark.parametrize("level", ["none", "basic", "all"])
def test_tensor_compiler_preserves_initializer_payload_and_abi(tmp_path, level):
    path, weights = model_file(tmp_path)
    driver = CompilerDriver(CompilerConfig(backend="tensor-c", optimize_level=level))
    output = tmp_path / "model.c"
    result = driver.compile(str(path), str(output))
    assert result.success, result.errors
    np.testing.assert_array_equal(driver.initializers["weights"], weights)
    artifact = driver.tensor_artifact
    assert [spec.name for spec in artifact.inputs] == ["x"]
    assert artifact.output.shape == (3, 2)
    assert artifact.output.numpy_dtype == np.dtype("float32")
    assert artifact.constant_bytes >= weights.nbytes
    assert output.read_text() == artifact.source == result.output_text
    assert result.stats["optimization"]["level"] == level


@pytest.mark.parametrize("backend", ["riscv", "llvm"])
def test_legacy_backends_reject_tensor_program_before_writing_output(tmp_path, backend):
    path, _ = model_file(tmp_path)
    output = tmp_path / "output.txt"
    output.write_text("preserve existing evidence")
    result = CompilerDriver(CompilerConfig(backend=backend)).compile(str(path), str(output))
    assert not result.success
    assert "tensor-c" in result.errors[0]
    assert output.read_text() == "preserve existing evidence"


def test_tensor_compiler_rejects_insufficient_workspace(tmp_path):
    path, _ = model_file(tmp_path)
    driver = CompilerDriver(CompilerConfig(backend="tensor-c", max_tensor_workspace_bytes=1))
    result = driver.compile(str(path), str(tmp_path / "impossible.c"))
    assert not result.success
    assert not (tmp_path / "impossible.c").exists()


def test_reused_driver_clears_previous_model_payload(tmp_path):
    path, _ = model_file(tmp_path)
    driver = CompilerDriver(CompilerConfig(backend="tensor-c"))
    assert driver.compile(str(path), str(tmp_path / "first.c")).success
    result = driver.compile(str(tmp_path / "absent.onnx"), str(tmp_path / "second.c"))
    assert not result.success
    assert driver.initializers == {} and driver.tensor_artifact is None


def test_cli_exposes_tensor_workspace_budget():
    args = build_arg_parser().parse_args(["model.onnx", "--backend", "tensor-c", "--tensor-workspace-mib", "16"])
    config = args_to_config(args)
    assert config.backend == "tensor-c"
    assert config.max_tensor_workspace_bytes == 16 * 1024 * 1024


@pytest.mark.parametrize("backend", ["ir", "tensor-c"])
@pytest.mark.parametrize("flag", ["beautify_asm", "peephole_asm", "const_merge", "count_instr", "cycle_stats", "use_dag_isel"])
def test_text_and_tensor_backends_reject_assembly_only_options(tmp_path, backend, flag):
    path, _ = model_file(tmp_path)
    config = CompilerConfig(backend=backend, **{flag: True})
    result = CompilerDriver(config).compile(str(path), str(tmp_path / "output.txt"))
    assert not result.success
    assert "Assembly-only" in result.errors[0]


@pytest.mark.parametrize("backend", ["ir", "tensor-c"])
def test_legacy_verify_flag_cannot_claim_tensor_execution(tmp_path, backend):
    path, _ = model_file(tmp_path)
    result = CompilerDriver(CompilerConfig(backend=backend, verify=True)).compile(
        str(path), str(tmp_path / "output.txt"))
    assert not result.success
    assert "does not execute" in result.errors[0]


def test_qemu_gate_fails_and_preserves_report_when_tool_is_missing(tmp_path, monkeypatch):
    from probes.w2_qwen3_small import riscv

    def absent(**kwargs):
        raise FileNotFoundError("qemu-system-riscv64 missing")

    # Isolate the toolchain failure after the separately tested model evidence
    # check. Missing/invalid upstream artifacts now fail before tool discovery.
    monkeypatch.setattr(riscv, "validated_model_artifacts", lambda directory: ({}, [], {}))
    monkeypatch.setattr(riscv, "discover_toolchain", absent)
    assert riscv.main(["--model-dir", str(tmp_path), "--output-dir", str(tmp_path / "result")]) == 1
    report = json.loads((tmp_path / "result/report.json").read_text())
    assert not report["passed"]
    assert report["stage"] == "toolchain"
    assert "qemu-system-riscv64 missing" in report["error"]
    assert (tmp_path / "result/report.md").exists()
    assert (tmp_path / "result/report.html").exists()
