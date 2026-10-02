"""Require real RV64 FP32 MatMul execution under QEMU; never skip missing tools."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

from probes.w2_qwen3_small.diagnostics import tensor_diff
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.runtime.riscv_tensor import (
    build_riscv_tensor, discover_toolchain, run_riscv_tensor,
)


def run_probe(out, report, *, cc=None, qemu=None, timeout=120):
    tools = discover_toolchain(cc=cc, qemu=qemu)
    rng = np.random.default_rng(314159)
    cases = [
        ("4x4_fractional", (4, 4), (4, 4)),
        ("projection", (1, 7, 32), (32, 64)),
        ("attention", (1, 4, 7, 16), (1, 4, 16, 7)),
        ("batch_broadcast", (2, 1, 3, 5), (1, 4, 5, 2)),
        ("vector_dot", (5,), (5,)),
        ("matrix_vector", (3, 5), (5,)),
        ("vector_matrix", (5,), (5, 4)),
    ]
    report["cases"] = []
    for name, left, right in cases:
        folder = out / name
        folder.mkdir()
        a = rng.normal(size=left).astype(np.float32) / np.float32(3)
        b = rng.normal(size=right).astype(np.float32) / np.float32(3)
        expected_shape = np.matmul(a, b).shape
        model = helper.make_model(helper.make_graph(
            [helper.make_node("MatMul", ["a", "b"], ["y"])], name,
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, left),
             helper.make_tensor_value_info("b", TensorProto.FLOAT, right)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, expected_shape)],
        ), opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
        path = folder / "model.onnx"
        onnx.save(model, path)
        feed = {"a": a, "b": b}
        expected = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feed)[0]
        driver = CompilerDriver(CompilerConfig(backend="tensor-c", optimize_level="all", verify_ir=True))
        result = driver.compile(str(path), str(folder / "model.c"))
        if not result.success:
            raise RuntimeError("; ".join(result.errors))
        executable = build_riscv_tensor(driver.tensor_artifact, folder / "build", tools)
        actual = run_riscv_tensor(executable, feed, folder / "run", timeout=timeout)
        comparison = tensor_diff(actual.output, expected, atol=1e-5)
        np.save(folder / "reference.npy", expected)
        np.save(folder / "qemu.npy", actual.output)
        report["cases"].append({"name": name, "left": left, "right": right,
                                "command": actual.command, **comparison})
        print(f"[{name}] {'PASS' if comparison['passed'] else 'FAIL'} max_abs={comparison.get('max_abs')}", flush=True)
    report["passed"] = all(case["passed"] for case in report["cases"])


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, default=ROOT / "output/qemu-matmul")
    cli.add_argument("--cc")
    cli.add_argument("--qemu")
    cli.add_argument("--timeout", type=float, default=120)
    args = cli.parse_args(argv)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        cli.error("--output-dir must be empty")
    out.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "backend": "ScratchV tensor-c -> RV64GC -> QEMU", "atol": 1e-5, "rtol": 0}
    start = time.perf_counter()
    try:
        run_probe(out, report, cc=args.cc, qemu=args.qemu, timeout=args.timeout)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    report["seconds"] = time.perf_counter() - start
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
