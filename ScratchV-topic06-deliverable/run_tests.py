import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from scratchv.simulator.tinyfive import verify_assembly

SUITE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SUITE_DIR.parent
TEST_DIR = SUITE_DIR / "tests_main"
BUILD_DIR = SUITE_DIR / "build"
REPORT_DIR = SUITE_DIR / "reports"
REPORT_FILE = REPORT_DIR / "report.md"
HTML_REPORT_FILE = REPORT_DIR / "report.html"
JSON_REPORT_FILE = REPORT_DIR / "report.json"
CHART_FILE = REPORT_DIR / "course_report_instructions.png"
BASELINE_FILE = REPORT_DIR / "benchmark_baseline.json"
FAILURE_DIR = REPORT_DIR / "failures"
REGRESSION_THRESHOLD_PCT = 5.0
COMPILE_TIMEOUT_SEC = 30.0
SIMULATION_TIMEOUT_SEC = 5.0


def run_compile(dsl_file: Path, timeout: float = COMPILE_TIMEOUT_SEC):
    output_file = BUILD_DIR / (dsl_file.stem + ".s")
    register_map_file = BUILD_DIR / (dsl_file.stem + ".registers.json")
    register_map_file.unlink(missing_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "scratchv.main",
        str(dsl_file),
        "-o",
        str(output_file),
        "--optimize",
        "all",
        "--emit-register-map",
        str(register_map_file),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            cwd=PROJECT_ROOT,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        timeout_message = f"compile timeout after {timeout:.0f}s"
        stderr = f"{stderr.rstrip()}\n{timeout_message}" if stderr else timeout_message
        result = subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout=stdout,
            stderr=stderr,
        )

    return result, output_file, register_map_file


def _last_nonempty_line(text):
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else None


def write_compile_failure_log(dsl_file, result, output_file, compile_time_sec):
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        f"{dsl_file.parent.name}-{dsl_file.stem}",
    )
    log_file = FAILURE_DIR / f"{safe_name}.compile.log"
    args = result.args
    if isinstance(args, (list, tuple)):
        command = shlex.join(str(arg) for arg in args)
    else:
        command = str(args or "(compiler was not started)")
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    log_file.write_text(
        "\n".join([
            f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
            f"Case: {dsl_file}",
            f"Command: {command}",
            f"Return code: {result.returncode}",
            f"Timed out: {result.returncode == 124}",
            f"Compile time (s): {compile_time_sec:.4f}",
            f"Assembly output: {output_file}",
            f"Assembly exists: {output_file.exists()}",
            "",
            "--- stdout ---",
            stdout,
            "",
            "--- stderr ---",
            stderr,
            "",
        ]),
        encoding="utf-8",
    )
    return log_file


SUPPORTED_OUTPUT_DTYPES = {"bool", "int32", "int64", "float32", "float64"}


def _value_shape(value):
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    child_shapes = [_value_shape(item) for item in value]
    if any(shape != child_shapes[0] for shape in child_shapes[1:]):
        raise ValueError("expected output must be a rectangular tensor")
    return (len(value),) + child_shapes[0]


def _validate_output_dtype(value, dtype):
    values = value if isinstance(value, list) else [value]
    for item in values:
        if isinstance(item, list):
            _validate_output_dtype(item, dtype)
        elif dtype == "bool" and not isinstance(item, bool):
            raise ValueError(f"expected output contains non-bool value: {item!r}")
        elif dtype.startswith("int") and (isinstance(item, bool) or not isinstance(item, int)):
            raise ValueError(f"expected output contains non-integer value: {item!r}")
        elif dtype.startswith("float") and (isinstance(item, bool) or not isinstance(item, (int, float))):
            raise ValueError(f"expected output contains non-numeric value: {item!r}")


def _load_expected_output(metadata, meta_file):
    has_inline = "expected_return" in metadata
    has_file = "expected_output_file" in metadata
    if has_inline == has_file:
        raise ValueError("define exactly one of expected_return or expected_output_file")

    if has_inline:
        return metadata["expected_return"]

    expected_file = meta_file.parent / metadata["expected_output_file"]
    payload = json.loads(expected_file.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "expected_return" in payload:
        return payload["expected_return"]
    return payload


def _validate_metadata(metadata, meta_file):
    expected = _load_expected_output(metadata, meta_file)
    output_type = metadata.get("expected_output_type", "scalar")
    if output_type == "return_value":
        output_type = "scalar"
    if output_type not in {"scalar", "tensor"}:
        raise ValueError(f"unsupported expected_output_type: {output_type}")

    actual_shape = _value_shape(expected)
    if output_type == "scalar" and actual_shape:
        raise ValueError("scalar expected output cannot contain a list")
    if output_type == "tensor" and not actual_shape:
        raise ValueError("tensor expected output must contain a nested JSON array")

    declared_shape = metadata.get("output_shape")
    if output_type == "tensor" and declared_shape is None:
        raise ValueError("tensor expected output requires output_shape")
    if declared_shape is not None:
        if not isinstance(declared_shape, list) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in declared_shape
        ):
            raise ValueError("output_shape must be a list of non-negative integers")
        if tuple(declared_shape) != actual_shape:
            raise ValueError(
                f"output_shape {declared_shape} does not match expected output shape {list(actual_shape)}"
            )

    dtype = metadata.get("output_dtype")
    if output_type == "tensor" and dtype is None:
        raise ValueError("tensor expected output requires output_dtype")
    if dtype is not None:
        if dtype not in SUPPORTED_OUTPUT_DTYPES:
            raise ValueError(f"unsupported output_dtype: {dtype}")
        _validate_output_dtype(expected, dtype)

    metadata["expected_output_type"] = output_type
    metadata["expected_return"] = expected
    return metadata


def load_metadata(dsl_file: Path):
    meta_file = dsl_file.with_suffix(".meta.json")
    if not meta_file.exists():
        return {
            "description": "",
            "expected_output_type": "scalar",
            "expected_return": None,
            "_metadata_error": f"metadata file not found: {meta_file}",
        }
    try:
        metadata = json.loads(meta_file.read_text(encoding="utf-8"))
        return _validate_metadata(metadata, meta_file)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "description": "",
            "expected_output_type": "invalid",
            "expected_return": None,
            "_metadata_error": str(exc),
        }


def load_initial_registers(register_map_file: Path, inputs: dict) -> dict[str, int]:
    """Load compiler-emitted register assignments for scalar DSL inputs."""
    try:
        payload = json.loads(register_map_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid compiler register map: {exc}") from exc

    register_map = payload.get("register_map")
    if not isinstance(register_map, dict):
        raise ValueError("invalid compiler register map: missing register_map object")

    result: dict[str, int] = {}
    for name, value in inputs.items():
        register = register_map.get(name)
        if not isinstance(register, str):
            continue
        if isinstance(value, bool):
            result[register] = int(value)
        elif isinstance(value, int):
            result[register] = value
        elif isinstance(value, float) and value.is_integer():
            result[register] = int(value)
    return result


def run_simulation(
    asm_file: Path,
    initial_registers: dict[str, int] | None = None,
    timeout: float = SIMULATION_TIMEOUT_SEC,
):
    if not asm_file.exists():
        return {
            "success": False,
            "instr_count": 0,
            "return_value": None,
            "backend": "none",
            "error": "assembly file not found",
        }

    code = "\n".join([
        "import json, sys",
        "from pathlib import Path",
        "from scratchv.simulator.tinyfive import verify_assembly",
        "asm = Path(sys.argv[1]).read_text(encoding='utf-8')",
        "initial_registers = json.loads(sys.argv[2])",
        "try:",
        "    result = verify_assembly(asm, initial_registers=initial_registers)",
        "except Exception as exc:",
        "    result = {'success': False, 'instr_count': 0, "
        "'return_value': None, 'backend': 'tinyfive', 'error': str(exc)}",
        "print(json.dumps(result, ensure_ascii=False))",
    ])

    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(asm_file),
                json.dumps(initial_registers or {}),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            cwd=PROJECT_ROOT,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "instr_count": 0,
            "return_value": None,
            "backend": "timeout",
            "error": f"simulation timeout after {timeout:.0f}s",
        }

    if completed.returncode != 0:
        return {
            "success": False,
            "instr_count": 0,
            "return_value": None,
            "backend": "tinyfive",
            "error": (completed.stderr or completed.stdout or "simulation failed").strip(),
        }

    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {
            "success": False,
            "instr_count": 0,
            "return_value": None,
            "backend": "tinyfive",
            "error": f"invalid simulation output: {exc}",
        }


def values_equal(lhs, rhs):
    if isinstance(lhs, list) and isinstance(rhs, list):
        if len(lhs) != len(rhs):
            return False
        return all(values_equal(a, b) for a, b in zip(lhs, rhs))
    if isinstance(lhs, float) or isinstance(rhs, float):
        return math.isclose(lhs, rhs, rel_tol=1e-7, abs_tol=1e-7)
    return lhs == rhs


def summarize_benchmark_runs(instr_counts):
    if not instr_counts:
        return {
            "runs": 0,
            "avg_instr_count": None,
            "min_instr_count": None,
            "max_instr_count": None,
            "ci95_instr_count": None,
        }
    avg = sum(instr_counts) / len(instr_counts)
    if len(instr_counts) > 1:
        variance = sum((value - avg) ** 2 for value in instr_counts) / (len(instr_counts) - 1)
        ci95 = 1.96 * math.sqrt(variance) / math.sqrt(len(instr_counts))
    else:
        ci95 = 0.0
    return {
        "runs": len(instr_counts),
        "avg_instr_count": avg,
        "min_instr_count": min(instr_counts),
        "max_instr_count": max(instr_counts),
        "ci95_instr_count": ci95,
    }


def detect_regression(
    avg_instr_count,
    baseline_instr_count,
    threshold_pct=REGRESSION_THRESHOLD_PCT,
):
    delta = avg_instr_count - baseline_instr_count
    delta_pct = 0.0 if baseline_instr_count == 0 else (delta / baseline_instr_count) * 100.0
    return {
        "baseline_instr_count": baseline_instr_count,
        "delta": round(delta, 4),
        "delta_pct": round(delta_pct, 4),
        "threshold_pct": threshold_pct,
        "regressed": delta_pct > threshold_pct,
    }


def load_baseline():
    if not BASELINE_FILE.exists():
        return {}
    return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))


def save_baseline(results, preserve_existing=False):
    REPORT_DIR.mkdir(exist_ok=True)
    payload = load_baseline() if preserve_existing else {}
    for r in results:
        if r["avg_instr_count"] is None:
            continue
        payload[r["name"]] = {
            "category": r["category"],
            "avg_instr_count": r["avg_instr_count"],
            "runs": r["benchmark_runs"],
        }
    BASELINE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _markdown_cell(value):
    return str(value).replace("\n", " ").replace("|", "\\|")


def _report_path(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(SUITE_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def write_chart(results):
    try:
        mpl_config_dir = Path(tempfile.gettempdir()) / "scratchv-matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    names = [result["name"] for result in results]
    values = [_reported_instr_count(result) for result in results]
    width = max(10, len(names) * 0.45)
    fig, ax = plt.subplots(figsize=(width, 5))
    ax.bar(range(len(names)), values, color="#2563eb")
    ax.set_title("ScratchV Course Benchmark Instruction Counts")
    ax.set_ylabel("Instructions")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(CHART_FILE, dpi=160)
    plt.close(fig)
    return CHART_FILE


def _reported_instr_count(result):
    average = result.get("avg_instr_count")
    return result["instr_count"] if average is None else average


def _report_value(value, precision=None, prefix=""):
    if value is None:
        return "null"
    if precision is not None:
        return f"{prefix}{value:.{precision}f}"
    return f"{prefix}{value}"


def generate_unified_report_text_cn(
    results,
    passed,
    failed,
    regression_threshold_pct=REGRESSION_THRESHOLD_PCT,
    selection_category=None,
    selection_filter=None,
    include_chart=False,
):
    mode = results[0]["mode"] if results else "normal"
    pass_rate = 0.0 if not results else passed / len(results) * 100.0
    lines = [
        "# ScratchV DSL 编译器性能测试报告\n\n",
        "## 测试概览\n\n",
        f"- Schema 版本: 1\n",
        f"- 运行模式: {mode}\n",
        f"- 类别筛选: {_report_value(selection_category)}\n",
        f"- 名称筛选: {_report_value(selection_filter)}\n",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"- 用例总数: {len(results)}\n",
        f"- 通过数量: {passed}\n",
        f"- 失败数量: {failed}\n",
        f"- 通过率: {pass_rate:.1f}%\n",
        f"- 测试目录: `{_report_path(TEST_DIR)}`\n",
        f"- 汇编输出目录: `{_report_path(BUILD_DIR)}`\n",
        f"- 性能基线文件: `{_report_path(BASELINE_FILE)}`\n",
        f"- 性能退化阈值: {regression_threshold_pct:.2f}%\n",
        f"- 单次编译超时: {COMPILE_TIMEOUT_SEC:.0f}s\n",
        f"- 单次模拟超时: {SIMULATION_TIMEOUT_SEC:.0f}s\n\n",
        "## 测试结果\n\n",
        "| 用例 | 类别 | 状态 | 编译返回码 | 编译日志 | 模拟后端 | 指令数 | Benchmark 次数 | Benchmark 停止原因 | 平均指令数 | 95% 置信区间 | 最小 | 最大 | 编译耗时(s) | 模拟耗时(s) | 总耗时(s) | 基线 | 变化量 | 变化率(%) | 退化阈值(%) | 是否退化 | 预期输出 | TinyFive 输出 | 输出匹配 | 汇编文件 |\n",
        "|---|---|---|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|\n",
    ]

    for result in results:
        lines.append(
            f"| {result['name']} | {result['category']} | {result['status']} | "
            f"{result['compile_returncode']} | {_report_value(result['compile_log'])} | "
            f"{result['backend']} | {result['instr_count']} | "
            f"{_report_value(result['benchmark_runs'])} | "
            f"{_report_value(result['benchmark_stopped_reason'])} | "
            f"{_report_value(result['avg_instr_count'], 2)} | "
            f"{_report_value(result['ci95_instr_count'], 2, '±')} | "
            f"{_report_value(result['min_instr_count'])} | "
            f"{_report_value(result['max_instr_count'])} | "
            f"{result['compile_time_sec']:.4f} | {result['simulation_time_sec']:.4f} | "
            f"{result['total_time_sec']:.4f} | "
            f"{_report_value(result['baseline_instr_count'], 2)} | "
            f"{_report_value(result['delta'], 2)} | "
            f"{_report_value(result['delta_pct'], 2)} | "
            f"{_report_value(result['threshold_pct'], 2)} | "
            f"{_report_value(result['regressed'])} | {_markdown_cell(result['expected'])} | "
            f"{_markdown_cell(result['actual'])} | {result['matched']} | {result['asm']} |\n"
        )

    if include_chart:
        lines.extend([
            "\n## 性能图表\n\n",
            f"![课程版指令数图表]({CHART_FILE.name})\n\n",
        ])
    lines.append("\n## 用例详情\n\n")
    for result in results:
        lines.extend([
            f"### {result['name']}\n\n",
            f"- 类别: {result['category']}\n",
            f"- 描述: {result['description']}\n",
            f"- 预期输出 ({result['expected_type']}): {result['expected']}\n",
            f"- TinyFive 输出: {result['actual']}\n",
            f"- 输出是否匹配: {result['matched']}\n",
            f"- TinyFive 初始寄存器: {result['initial_registers']}\n",
            f"- 模拟后端: {result['backend']}\n",
            f"- 指令数: {result['instr_count']}\n",
            f"- 编译返回码: {result['compile_returncode']}\n",
            f"- 编译是否超时: {result['compile_timed_out']}\n",
            f"- 编译错误摘要: {_report_value(result['compile_error'])}\n",
            f"- 编译失败日志: {_report_value(result['compile_log'])}\n",
            f"- Benchmark 重复次数: {_report_value(result['benchmark_runs'])}\n",
            f"- Benchmark 停止原因: {_report_value(result['benchmark_stopped_reason'])}\n",
            f"- 平均指令数: {_report_value(result['avg_instr_count'], 2)}\n",
            f"- 95% 置信区间: {_report_value(result['ci95_instr_count'], 2, '±')}\n",
            f"- 最小指令数: {_report_value(result['min_instr_count'])}\n",
            f"- 最大指令数: {_report_value(result['max_instr_count'])}\n",
            f"- 编译耗时(s): {result['compile_time_sec']:.4f}\n",
            f"- 模拟耗时(s): {result['simulation_time_sec']:.4f}\n",
            f"- 总耗时(s): {result['total_time_sec']:.4f}\n",
            f"- 基线指令数: {_report_value(result['baseline_instr_count'], 2)}\n",
            f"- 性能变化量: {_report_value(result['delta'], 2)}\n",
            f"- 性能变化率(%): {_report_value(result['delta_pct'], 2)}\n",
            f"- 性能退化阈值(%): {_report_value(result['threshold_pct'], 2)}\n",
            f"- 是否性能退化: {_report_value(result['regressed'])}\n",
            f"- 汇编文件: {result['asm']}\n\n",
        ])
    return "".join(lines)


def write_unified_html_report_cn(
    results,
    passed,
    failed,
    regression_threshold_pct=REGRESSION_THRESHOLD_PCT,
    selection_category=None,
    selection_filter=None,
):
    try:
        from jinja2 import Template
    except ImportError:
        return None

    template = Template("""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>ScratchV 测试报告</title>
  <style>
    body { font-family: "Microsoft YaHei", sans-serif; margin: 32px; color: #1f2937; }
    .summary { background: #f8fafc; border: 1px solid #e5e7eb; padding: 14px 18px; }
    .table-wrap { overflow-x: auto; }
    table { border-collapse: collapse; width: 100%; margin-top: 18px; font-size: 12px; white-space: nowrap; }
    th, td { border: 1px solid #d1d5db; padding: 7px 9px; text-align: left; }
    th { background: #f3f4f6; }
    .pass { color: #047857; font-weight: 700; }
    .fail { color: #b91c1c; font-weight: 700; }
    img { max-width: 100%; margin-top: 18px; border: 1px solid #e5e7eb; }
  </style>
</head>
<body>
  <h1>ScratchV DSL 编译器性能测试报告</h1>
  <div class="summary">
    <p>Schema 版本：1，运行模式：{{ mode }}</p>
    <p>类别筛选：{{ fmt(selection_category) }}，名称筛选：{{ fmt(selection_filter) }}</p>
    <p>用例总数：{{ total }}，通过：{{ passed }}，失败：{{ failed }}</p>
    <p>性能退化阈值：{{ fmt(regression_threshold_pct, 2) }}%</p>
  </div>
  <img src="{{ chart_name }}" alt="课程版指令数图表">
  <div class="table-wrap"><table>
    <thead><tr>
      <th>用例</th><th>类别</th><th>状态</th><th>编译返回码</th><th>编译日志</th><th>模拟后端</th><th>指令数</th>
      <th>Benchmark 次数</th><th>Benchmark 停止原因</th><th>平均指令数</th><th>95% 置信区间</th><th>最小</th><th>最大</th>
      <th>编译耗时(s)</th><th>模拟耗时(s)</th><th>总耗时(s)</th>
      <th>基线</th><th>变化量</th><th>变化率(%)</th><th>退化阈值(%)</th><th>是否退化</th>
      <th>预期输出</th><th>TinyFive 输出</th><th>输出匹配</th><th>汇编文件</th>
    </tr></thead>
    <tbody>{% for r in results %}<tr>
      <td>{{ r.name }}</td><td>{{ r.category }}</td>
      <td class="{{ 'pass' if r.status == 'PASS' else 'fail' }}">{{ r.status }}</td>
      <td>{{ r.compile_returncode }}</td><td>{{ fmt(r.compile_log) }}</td>
      <td>{{ r.backend }}</td><td>{{ r.instr_count }}</td>
      <td>{{ fmt(r.benchmark_runs) }}</td><td>{{ fmt(r.benchmark_stopped_reason) }}</td><td>{{ fmt(r.avg_instr_count, 2) }}</td>
      <td>{{ fmt(r.ci95_instr_count, 2, '±') }}</td><td>{{ fmt(r.min_instr_count) }}</td><td>{{ fmt(r.max_instr_count) }}</td>
      <td>{{ fmt(r.compile_time_sec, 4) }}</td><td>{{ fmt(r.simulation_time_sec, 4) }}</td><td>{{ fmt(r.total_time_sec, 4) }}</td>
      <td>{{ fmt(r.baseline_instr_count, 2) }}</td><td>{{ fmt(r.delta, 2) }}</td><td>{{ fmt(r.delta_pct, 2) }}</td>
      <td>{{ fmt(r.threshold_pct, 2) }}</td><td>{{ fmt(r.regressed) }}</td>
      <td>{{ r.expected }}</td><td>{{ r.actual }}</td><td>{{ r.matched }}</td><td>{{ r.asm }}</td>
    </tr>{% endfor %}</tbody>
  </table></div>
</body>
</html>
""")
    mode = results[0]["mode"] if results else "normal"
    HTML_REPORT_FILE.write_text(
        template.render(
            mode=mode,
            total=len(results),
            passed=passed,
            failed=failed,
            regression_threshold_pct=regression_threshold_pct,
            selection_category=selection_category,
            selection_filter=selection_filter,
            chart_name=CHART_FILE.name,
            results=results,
            fmt=_report_value,
        ),
        encoding="utf-8",
    )
    return HTML_REPORT_FILE


def write_json_report(
    results,
    passed,
    failed,
    regression_threshold_pct=REGRESSION_THRESHOLD_PCT,
    selection_category=None,
    selection_filter=None,
    full_report=False,
):
    mode = results[0]["mode"] if results else "normal"
    payload = {
        "schema_version": 1,
        "mode": mode,
        "report_level": "full" if full_report else "light",
        "regression_threshold_pct": regression_threshold_pct,
        "selection": {
            "category": selection_category,
            "filter": selection_filter,
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
        },
        "results": results,
    }
    JSON_REPORT_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return JSON_REPORT_FILE


def write_report(
    results,
    passed,
    failed,
    regression_threshold_pct=REGRESSION_THRESHOLD_PCT,
    selection_category=None,
    selection_filter=None,
    full_report=False,
):
    REPORT_DIR.mkdir(exist_ok=True)
    chart_path = write_chart(results) if full_report else None
    REPORT_FILE.write_text(
        generate_unified_report_text_cn(
            results,
            passed,
            failed,
            regression_threshold_pct=regression_threshold_pct,
            selection_category=selection_category,
            selection_filter=selection_filter,
            include_chart=chart_path is not None,
        ),
        encoding="utf-8",
    )
    html_path = (
        write_unified_html_report_cn(
            results,
            passed,
            failed,
            regression_threshold_pct=regression_threshold_pct,
            selection_category=selection_category,
            selection_filter=selection_filter,
        )
        if full_report else None
    )
    json_path = write_json_report(
        results,
        passed,
        failed,
        regression_threshold_pct=regression_threshold_pct,
        selection_category=selection_category,
        selection_filter=selection_filter,
        full_report=full_report,
    )
    print(f"\nMarkdown report written to {REPORT_FILE}")
    print(f"JSON report written to {json_path}")
    if full_report:
        if html_path:
            print(f"HTML report written to {html_path}")
        else:
            print("HTML report skipped: jinja2 is not installed")
        if chart_path:
            print(f"Chart written to {chart_path}")
        else:
            print("Chart skipped: matplotlib is not installed")


def non_negative_float(value):
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def select_dsl_files(test_dir, category=None, name_filter=None):
    dsl_files = sorted(test_dir.rglob("*.dsl"), key=lambda path: str(path).lower())
    if category:
        expected_category = category.lower()
        dsl_files = [
            path for path in dsl_files
            if path.parent.name.lower() == expected_category
        ]
    if name_filter:
        expected_name = name_filter.lower()
        dsl_files = [
            path for path in dsl_files
            if expected_name in path.stem.lower()
        ]
    return dsl_files


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run ScratchV DSL benchmark suite.")
    parser.add_argument("--benchmark", type=int, default=0, metavar="N",
                        help="Run each case N times and report average instruction count.")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Write current benchmark averages to the baseline file.")
    parser.add_argument("--full-report", action="store_true",
                        help="Also generate the optional HTML report and PNG chart.")
    parser.add_argument(
        "--regression-threshold",
        type=non_negative_float,
        default=REGRESSION_THRESHOLD_PCT,
        metavar="PERCENT",
        help="Mark instruction-count increases above this percentage as regressions (default: 5).",
    )
    parser.add_argument(
        "--category",
        help="Only run cases in this test category (for example: activation or tensor).",
    )
    parser.add_argument(
        "--filter",
        dest="name_filter",
        help="Only run cases whose file name contains this text.",
    )
    parser.add_argument(
        "--fail-on-test-failure",
        action="store_true",
        help="Return exit code 1 when one or more selected cases fail.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    BUILD_DIR.mkdir(exist_ok=True)
    baseline = load_baseline() if args.benchmark else {}

    all_dsl_files = list(TEST_DIR.rglob("*.dsl"))

    if not all_dsl_files:
        print("No DSL test files found.")
        return 2

    dsl_files = select_dsl_files(
        TEST_DIR,
        category=args.category,
        name_filter=args.name_filter,
    )
    if not dsl_files:
        print("No DSL test cases matched the selected filters.")
        return 2

    passed = 0
    failed = 0
    results = []

    print("Running DSL compiler tests...")
    print("=" * 50)

    for dsl_file in dsl_files:
        print(f"\n[TEST] {dsl_file}")

        case_start = time.perf_counter()
        meta = load_metadata(dsl_file)
        compile_start = time.perf_counter()
        metadata_error = meta.get("_metadata_error")
        if metadata_error:
            output_file = BUILD_DIR / (dsl_file.stem + ".s")
            register_map_file = BUILD_DIR / (dsl_file.stem + ".registers.json")
            result = subprocess.CompletedProcess(
                args=[],
                returncode=2,
                stdout="",
                stderr=f"metadata error: {metadata_error}",
            )
        else:
            result, output_file, register_map_file = run_compile(dsl_file)
        compile_time_sec = time.perf_counter() - compile_start
        compile_log = None
        compile_error = None
        if result.returncode != 0:
            compile_log = write_compile_failure_log(
                dsl_file,
                result,
                output_file,
                compile_time_sec,
            )
            compile_error = _last_nonempty_line(result.stderr or result.stdout)
        register_map_error = None
        initial_registers = {}
        if result.returncode == 0:
            try:
                initial_registers = load_initial_registers(
                    register_map_file,
                    meta.get("inputs", {}),
                )
            except ValueError as exc:
                register_map_error = str(exc)
        simulation_start = time.perf_counter()
        if result.returncode != 0:
            sim_result = {
                "success": False,
                "instr_count": 0,
                "return_value": None,
                "backend": "none",
                "error": (result.stderr or result.stdout or "compile failed").strip(),
            }
        elif register_map_error:
            sim_result = {
                "success": False,
                "instr_count": 0,
                "return_value": None,
                "backend": "none",
                "error": register_map_error,
            }
        else:
            sim_result = run_simulation(output_file, initial_registers)
        simulation_time_sec = time.perf_counter() - simulation_start
        expected_value = meta.get("expected_return")
        actual_value = sim_result.get("return_value")
        matched = bool(sim_result.get("success")) and values_equal(actual_value, expected_value)
        benchmark_counts = []
        benchmark_summary = {
            "runs": None,
            "avg_instr_count": None,
            "min_instr_count": None,
            "max_instr_count": None,
            "ci95_instr_count": None,
        }
        regression = {
            "baseline_instr_count": None,
            "delta": None,
            "delta_pct": None,
            "threshold_pct": None,
            "regressed": None,
        }
        benchmark_stopped_reason = None

        if args.benchmark > 0:
            regression["threshold_pct"] = args.regression_threshold
            if result.returncode != 0 or not output_file.exists():
                benchmark_stopped_reason = "benchmark skipped: compile failed"
                benchmark_summary = summarize_benchmark_runs([])
            elif sim_result.get("backend") == "timeout":
                benchmark_stopped_reason = "benchmark skipped: initial simulation timeout"
                benchmark_summary = summarize_benchmark_runs([])
            elif not sim_result.get("success"):
                benchmark_stopped_reason = "benchmark skipped: initial simulation failed"
                benchmark_summary = summarize_benchmark_runs([])
            else:
                for run_index in range(1, args.benchmark + 1):
                    benchmark_result = run_simulation(output_file, initial_registers)
                    if benchmark_result.get("backend") == "timeout":
                        benchmark_stopped_reason = (
                            f"benchmark stopped: timeout on run {run_index}"
                        )
                        break
                    if not benchmark_result.get("success"):
                        benchmark_stopped_reason = (
                            f"benchmark stopped: simulation failed on run {run_index}"
                        )
                        break
                    benchmark_counts.append(benchmark_result.get("instr_count", 0))
                benchmark_summary = summarize_benchmark_runs(benchmark_counts)

            if benchmark_summary["avg_instr_count"] is not None and benchmark_stopped_reason is None:
                regression["regressed"] = False

            baseline_entry = baseline.get(dsl_file.stem)
            if (
                baseline_entry
                and benchmark_summary["avg_instr_count"] is not None
                and benchmark_stopped_reason is None
            ):
                regression = detect_regression(
                    avg_instr_count=benchmark_summary["avg_instr_count"],
                    baseline_instr_count=baseline_entry.get("avg_instr_count", 0.0),
                    threshold_pct=args.regression_threshold,
                )
        total_time_sec = time.perf_counter() - case_start

        ok = (
            result.returncode == 0
            and output_file.exists()
            and sim_result["success"]
            and matched
            and regression["regressed"] is not True
            and benchmark_stopped_reason is None
        )

        if ok:
            print("PASS")
            passed += 1
            status = "PASS"
        else:
            print("FAIL")
            failed += 1
            status = "FAIL"
            if sim_result.get("error"):
                print(sim_result["error"])
            else:
                print(
                    "output mismatch: "
                    f"expected={expected_value}, "
                    f"tinyfive={actual_value}"
                )

        results.append({
            "mode": "benchmark" if args.benchmark > 0 else "normal",
            "name": dsl_file.stem,
            "category": dsl_file.parent.name,
            "path": _report_path(dsl_file),
            "status": status,
            "description": meta.get("description", ""),
            "expected_type": meta.get("expected_output_type", "scalar"),
            "output_dtype": meta.get("output_dtype"),
            "output_shape": meta.get("output_shape"),
            "expected": expected_value,
            "actual": actual_value,
            "matched": matched,
            "initial_registers": initial_registers,
            "register_map": _report_path(register_map_file),
            "backend": sim_result.get("backend", "none"),
            "instr_count": sim_result.get("instr_count", 0),
            "compile_returncode": result.returncode,
            "compile_timed_out": result.returncode == 124,
            "compile_error": compile_error,
            "compile_log": _report_path(compile_log) if compile_log else None,
            "compile_time_sec": compile_time_sec,
            "simulation_time_sec": simulation_time_sec,
            "total_time_sec": total_time_sec,
            "benchmark_runs": benchmark_summary["runs"],
            "benchmark_stopped_reason": benchmark_stopped_reason,
            "avg_instr_count": benchmark_summary["avg_instr_count"],
            "min_instr_count": benchmark_summary["min_instr_count"],
            "max_instr_count": benchmark_summary["max_instr_count"],
            "ci95_instr_count": benchmark_summary["ci95_instr_count"],
            "baseline_instr_count": regression["baseline_instr_count"],
            "delta": regression["delta"],
            "delta_pct": regression["delta_pct"],
            "threshold_pct": regression["threshold_pct"],
            "regressed": regression["regressed"],
            "asm": _report_path(output_file),
        })

    print("\n" + "=" * 50)
    print(f"Total: {len(dsl_files)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if args.benchmark and args.update_baseline:
        save_baseline(
            results,
            preserve_existing=bool(args.category or args.name_filter),
        )
        print(f"Baseline written to {BASELINE_FILE}")

    write_report(
        results,
        passed,
        failed,
        regression_threshold_pct=args.regression_threshold,
        selection_category=args.category,
        selection_filter=args.name_filter,
        full_report=args.full_report,
    )
    return 1 if args.fail_on_test_failure and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
