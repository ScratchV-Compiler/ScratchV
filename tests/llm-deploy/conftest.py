"""Shared local assets and execution fixtures for the W1/W2 pytest suite."""
from __future__ import annotations

from collections.abc import Callable
import copy
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.w1_qwen3_export import run as full_model
from probes.w1_tiny_transformer.tiny_transformer import Config, build
from probes.w2_qwen3_small.diagnostics import tensor_diff
from scratchv.analysis.ir_verifier import verify_ir
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.pass_manager import create_optimization_pass_manager
from scratchv.runtime.riscv_tensor import (
    build_riscv_tensor, discover_toolchain, run_riscv_tensor,
)
from scratchv.verification.ir_interpreter import IRInterpreter


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def qwen3_model_dir(repository_root: Path) -> Path:
    return repository_root / "assets/qwen3-0.6b/onnx"


@pytest.fixture(scope="session")
def qwen3_model_files(qwen3_model_dir: Path) -> list[dict]:
    # Also rejects unresolved LFS pointers and truncated weight shards.
    return full_model.verify_files(qwen3_model_dir, full_model.MANIFEST)


@pytest.fixture(scope="session")
def qwen3_model_path(qwen3_model_dir: Path, qwen3_model_files) -> Path:
    return qwen3_model_dir / "model.onnx"


@pytest.fixture(scope="session")
def tokenizer_dir(repository_root: Path) -> Path:
    from probes.w2_runtime.run import verify_assets

    directory = repository_root / "assets/qwen3-0.6b/tokenizer"
    verify_assets(directory)
    return directory


@pytest.fixture(scope="session")
def tokenizers(tokenizer_dir: Path) -> tuple:
    from probes.w2_runtime.run import require_environment
    from transformers import AutoTokenizer
    from scratchv.runtime.qwen3_tokenizer import Qwen3Tokenizer

    require_environment()
    adapter = Qwen3Tokenizer.from_directory(tokenizer_dir)
    references = tuple(
        AutoTokenizer.from_pretrained(
            tokenizer_dir, use_fast=fast, local_files_only=True, trust_remote_code=False,
        )
        for fast in (True, False)
    )
    assert tuple(type(value).__name__ for value in references) == (
        "Qwen2TokenizerFast", "Qwen2Tokenizer",
    )
    return adapter, *references


@pytest.fixture(scope="session")
def two_layer_transformer(tmp_path_factory) -> dict:
    config = Config()
    model, weights = build(config)
    path = tmp_path_factory.mktemp("tiny-transformer") / "model.onnx"
    onnx.save(model, path)
    return {"path": path, "config": config, "weights": weights}


@pytest.fixture(scope="session")
def two_layer_transformer_path(two_layer_transformer: dict) -> Path:
    return two_layer_transformer["path"]


@pytest.fixture(scope="session")
def torch_cpu():
    import torch
    from probes.w2_qwen3_small.run import environment

    environment()
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    yield torch
    torch.set_num_threads(previous_threads)
    torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)


@pytest.fixture(scope="session")
def ort_session() -> Callable:
    def create(path: Path, *, disable_optimizations: bool = False):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        if disable_optimizations:
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    return create


@pytest.fixture(scope="session")
def assert_tensor() -> Callable:
    def compare(actual: np.ndarray, expected: np.ndarray) -> None:
        result = tensor_diff(actual, expected, atol=1e-5)
        assert result["passed"], result

    return compare


@pytest.fixture(scope="session")
def assert_valid_ir() -> Callable:
    def check(program, stage: str = "pytest") -> None:
        passed, issues = verify_ir(program, stage=stage)
        assert passed, "\n".join(str(issue) for issue in issues)

    return check


@pytest.fixture(scope="session")
def ir_model(assert_valid_ir) -> Callable:
    cache = {}

    def load(path: Path, level: str = "none") -> tuple:
        key = path, level
        if key not in cache:
            parser = ONNXParser()
            original = parser.parse(str(path))
            assert_valid_ir(original, "after-parse")
            manager = create_optimization_pass_manager(level)
            manager.before_pass = lambda p, value: assert_valid_ir(value, f"before:{p.name}")
            manager.after_pass = lambda p, value: assert_valid_ir(value, f"after:{p.name}")
            program = manager.run_pipeline(copy.deepcopy(original)).data
            assert_valid_ir(program, "after-optimization")
            cache[key] = program, parser.initializers
        return cache[key]

    return load


@pytest.fixture(scope="session")
def run_ir(ir_model) -> Callable:
    def run(path: Path, feed: dict, level: str = "none") -> np.ndarray:
        program, bindings = ir_model(path, level)
        return IRInterpreter(program).run(feed, initializers=bindings).return_value

    return run


@pytest.fixture(scope="session")
def riscv_toolchain():
    # Missing tools must fail the requested tests, never silently skip them.
    return discover_toolchain()


@pytest.fixture(scope="session")
def riscv_executable(tmp_path_factory, riscv_toolchain) -> Callable:
    cache = {}

    def compile_model(path: Path, level: str = "all"):
        key = path, level
        if key not in cache:
            directory = tmp_path_factory.mktemp(f"rv64-{path.stem}-{level}")
            driver = CompilerDriver(CompilerConfig(
                backend="tensor-c", optimize_level=level, verify_ir=True,
            ))
            result = driver.compile(str(path), str(directory / "model.c"))
            assert result.success, result.errors
            cache[key] = build_riscv_tensor(driver.tensor_artifact, directory / "build", riscv_toolchain)
        return cache[key]

    return compile_model


@pytest.fixture(scope="session")
def run_riscv(tmp_path_factory, riscv_executable) -> Callable:
    def run(path: Path, feed: dict, level: str = "all"):
        executable = riscv_executable(path, level)
        directory = tmp_path_factory.mktemp(f"qemu-{path.stem}-{level}")
        return run_riscv_tensor(executable, feed, directory, timeout=180)

    return run
