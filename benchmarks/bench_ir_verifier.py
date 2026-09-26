"""Compare shared-pipeline compilation with IR verification disabled/enabled.

Run with ``python -m benchmarks.bench_ir_verifier``. Timings include parsing,
optimizations, code generation and writing output; they are not model runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path

from scratchv.compiler import CompilerConfig, CompilerDriver


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models/graph/cnn.onnx"


def benchmark(model: Path, *, repeats: int = 5, warmup: int = 1,
              levels: tuple[str, ...] = ("none", "basic", "all")) -> dict:
    """Use fresh parses for both sides and fail on compilation/output mismatch."""
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup non-negative")
    if not levels or any(level not in ("none", "basic", "all") for level in levels):
        raise ValueError("levels must contain none, basic or all")
    model = Path(model).resolve()
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    rows = []
    with tempfile.TemporaryDirectory(prefix="scratchv-ir-bench-") as directory:
        output = str(Path(directory) / "output.s")
        for level in levels:
            samples = {False: [], True: []}
            reference = None
            for iteration in range(warmup + repeats):
                # Alternate order to reduce bias from consistently running one first.
                order = (False, True) if iteration % 2 == 0 else (True, False)
                for enabled in order:
                    driver = CompilerDriver(CompilerConfig(
                        backend="riscv", optimize_level=level, verify_ir=enabled,
                    ))
                    start = time.perf_counter()
                    result = driver.compile(str(model), output)
                    elapsed = time.perf_counter() - start
                    if not result.success:
                        raise RuntimeError(
                            f"optimize={level}, verify_ir={enabled}: "
                            + "; ".join(result.errors)
                        )
                    if not result.output_text:
                        raise RuntimeError("Compilation produced empty output")
                    digest = hashlib.sha256(result.output_text.encode()).hexdigest()
                    if reference is None:
                        reference = digest
                    elif digest != reference:
                        raise RuntimeError(
                            f"optimize={level}: generated assembly differs between runs"
                        )
                    if iteration >= warmup:
                        samples[enabled].append(elapsed)
            disabled = statistics.median(samples[False])
            enabled = statistics.median(samples[True])
            rows.append({
                "optimize": level,
                "disabled_samples_s": samples[False],
                "enabled_samples_s": samples[True],
                "disabled_median_s": disabled,
                "enabled_median_s": enabled,
                "overhead_s": enabled - disabled,
                "overhead_percent": (enabled / disabled - 1) * 100 if disabled else None,
                "output_equal": True,
                "output_sha256": reference,
            })
    return {
        "model": str(model), "model_sha256": model_hash,
        "python": platform.python_version(), "platform": platform.platform(),
        "backend": "riscv", "repeats": repeats, "warmup": warmup,
        "timing_scope": "parse + optimize + codegen + output write",
        "results": rows,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Topic 21 IR 验证器编译耗时对比", "",
        f"模型：`{Path(report['model']).name}`；后端：RISC-V；"
        f"预热 {report['warmup']} 次，采样 {report['repeats']} 次，取中位数。", "",
        "通过 CompilerConfig.verify_ir 开关，在正式编译流程的解析后、每个 IR pass "
        "前后和代码生成前验证。两组均重新解析模型。", "",
        "| 优化级别 | 关闭验证 (ms) | 开启验证 (ms) | 差值 (ms) | 差值 (%) | 汇编一致 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["results"]:
        percent = row["overhead_percent"]
        percent_text = f"{percent:.2f}" if percent is not None else "N/A"
        lines.append(
            f"| {row['optimize']} | {row['disabled_median_s'] * 1000:.3f} "
            f"| {row['enabled_median_s'] * 1000:.3f} "
            f"| {row['overhead_s'] * 1000:.3f} | {percent_text} | 是 |"
        )
    lines.extend([
        "", "计时包含解析、优化、代码生成和写文件，不是模型运行耗时，也不是验证器函数的独立耗时。",
        "差值可能受测量噪声影响而为负；不以耗时差值作为 CI 通过门槛。"
        "编译或验证失败、汇编内容不一致会使 benchmark 失败。",
        "汇编一致仅检查开关未改变生成结果，不代表与 ONNX Runtime 数值等价。", "",
    ])
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json-output", type=Path,
                        default=Path("benchmark_reports/ir_verifier.json"))
    parser.add_argument("--markdown", type=Path,
                        default=Path("benchmark_reports/ir_verifier.md"))
    args = parser.parse_args(argv)
    report = benchmark(args.model, repeats=args.repeats, warmup=args.warmup)
    markdown = render_markdown(report)
    for path, content in (
        (args.json_output, json.dumps(report, indent=2) + "\n"),
        (args.markdown, markdown),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
