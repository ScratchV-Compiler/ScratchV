import argparse
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from scratchv.simulator.tinyfive import verify_assembly

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_ROOT / "tests" / "topic06" / "cases"
BUILD_DIR = PROJECT_ROOT / "build" / "topic06"
REPORT_DIR = PROJECT_ROOT / "benchmark_reports" / "topic06"
METADATA_FILE = REPORT_DIR / "metadata.json"
REPORT_FILE = REPORT_DIR / "report.md"
HTML_REPORT_FILE = REPORT_DIR / "report.html"
JSON_REPORT_FILE = REPORT_DIR / "report.json"
CHART_FILE = REPORT_DIR / "course_report_instructions.png"
BASELINE_FILE = PROJECT_ROOT / "benchmarks" / "topic06" / "baseline.json"
FAILURE_DIR = REPORT_DIR / "failures"
CASE_REPORT_DIR = REPORT_DIR / "cases"
REGRESSION_THRESHOLD_PCT = 5.0
COMPILE_TIMEOUT_SEC = 30.0
SIMULATION_TIMEOUT_SEC = 5.0
INTERPRETER_TIMEOUT_SEC = 5.0

INTERPRETER_SUPPORTED_OPS = {
    "add", "sub", "mul", "div", "neg", "exp", "relu", "gelu",
    "matmul", "dot", "softmax", "maxpool",
}


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


def interpreter_support_for_source(source: str) -> tuple[bool, str | None]:
    """Reject syntax that DSLInterpreter currently accepts without executing correctly."""
    control_flow = re.search(
        r"^\s*(for|endfor|if|else|endif|while|endwhile)\b",
        source,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if control_flow:
        return False, f"unsupported control flow: {control_flow.group(1).lower()}"

    operations = {
        match.group(1).lower()
        for match in re.finditer(r"^\s*\w+\s*=\s*(\w+)\s*\(", source, re.MULTILINE)
    }
    unsupported = sorted(operations - INTERPRETER_SUPPORTED_OPS)
    if unsupported:
        return False, f"unsupported interpreter operation(s): {', '.join(unsupported)}"
    return True, None


def run_interpreter(
    dsl_file: Path,
    inputs: dict,
    timeout: float = INTERPRETER_TIMEOUT_SEC,
):
    """Run the DSL reference interpreter in an isolated process."""
    source = dsl_file.read_text(encoding="utf-8")
    supported, reason = interpreter_support_for_source(source)
    if not supported:
        return {
            "success": False,
            "return_value": None,
            "backend": "interpreter",
            "status": "UNSUPPORTED",
            "error": reason,
        }

    code = "\n".join([
        "import json, sys",
        "from pathlib import Path",
        "import numpy as np",
        "from scratchv.verification.verifier import DSLInterpreter",
        "source = Path(sys.argv[1]).read_text(encoding='utf-8')",
        "raw_inputs = json.loads(sys.argv[2])",
        "inputs = {name: np.asarray(value) for name, value in raw_inputs.items()}",
        "try:",
        "    value = DSLInterpreter().run(source, inputs)",
        "    if isinstance(value, np.ndarray):",
        "        value = value.item() if value.ndim == 0 else value.tolist()",
        "    elif isinstance(value, np.generic):",
        "        value = value.item()",
        "    result = {'success': True, 'return_value': value, "
        "'backend': 'interpreter', 'status': 'EXECUTED', 'error': None}",
        "except Exception as exc:",
        "    result = {'success': False, 'return_value': None, "
        "'backend': 'interpreter', 'status': 'ERROR', 'error': str(exc)}",
        "print(json.dumps(result, ensure_ascii=False))",
    ])

    try:
        completed = subprocess.run(
            [sys.executable, "-c", code, str(dsl_file), json.dumps(inputs)],
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
            "return_value": None,
            "backend": "interpreter",
            "status": "TIMEOUT",
            "error": f"interpreter timeout after {timeout:.0f}s",
        }

    if completed.returncode != 0:
        return {
            "success": False,
            "return_value": None,
            "backend": "interpreter",
            "status": "ERROR",
            "error": (completed.stderr or completed.stdout or "interpreter failed").strip(),
        }
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {
            "success": False,
            "return_value": None,
            "backend": "interpreter",
            "status": "ERROR",
            "error": f"invalid interpreter output: {exc}",
        }


def values_equal(lhs, rhs):
    if isinstance(lhs, list) and isinstance(rhs, list):
        if len(lhs) != len(rhs):
            return False
        return all(values_equal(a, b) for a, b in zip(lhs, rhs))
    if isinstance(lhs, float) or isinstance(rhs, float):
        return math.isclose(lhs, rhs, rel_tol=1e-7, abs_tol=1e-7)
    return lhs == rhs


def tinyfive_input_abi_supported(inputs: dict) -> tuple[bool, str | None]:
    """Return whether every input can be injected directly into an integer register."""
    unsupported = []
    for name, value in inputs.items():
        if isinstance(value, bool) or isinstance(value, int):
            continue
        if isinstance(value, float) and value.is_integer():
            continue
        unsupported.append(name)
    if unsupported:
        return False, "non-scalar TinyFive input ABI is unavailable: " + ", ".join(unsupported)
    return True, None


def classify_tinyfive_result(
    compile_result,
    sim_result,
    matched,
    input_abi_supported,
    metadata_error=None,
    register_map_error=None,
):
    if metadata_error:
        return "SKIPPED", "metadata_error"
    if compile_result.returncode == 124:
        return "SKIPPED", "compile_timeout"
    if compile_result.returncode != 0:
        return "SKIPPED", "compile_error"
    if register_map_error:
        return "SKIPPED", "register_map_error"
    if not input_abi_supported:
        return "UNSUPPORTED", "input_abi_unsupported"
    if sim_result.get("backend") == "skipped":
        return "SKIPPED", "backend_not_selected"
    if sim_result.get("backend") == "timeout":
        return "TIMEOUT", "simulation_timeout"
    if not sim_result.get("success"):
        error = str(sim_result.get("error") or "").lower()
        if "not installed" in error:
            return "UNAVAILABLE", "tinyfive_unavailable"
        if "unsupported" in error or "unknown" in error:
            return "ERROR", "unsupported_assembly_instruction"
        if "assembl" in error or "encode" in error:
            return "ERROR", "assembly_encoding_error"
        return "ERROR", "simulation_error"
    if not matched:
        return "MISMATCH", "result_mismatch"
    return "PASS", None


def count_assembly_instructions(asm_file: Path) -> int | None:
    if not asm_file.exists():
        return None
    try:
        from scratchv.backend._asm_parser import parse_asm

        return sum(
            line.opcode is not None and not line.is_directive
            for line in parse_asm(asm_file.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError):
        return None


def summarize_backend_matrix(results):
    def counts(field):
        summary = {}
        for result in results:
            value = result.get(field) or "UNKNOWN"
            summary[value] = summary.get(value, 0) + 1
        return summary

    comparable = [r for r in results if r.get("backend_outputs_match") is not None]
    return {
        "interpreter": counts("interpreter_status"),
        "tinyfive": counts("tinyfive_status"),
        "cross_backend": {
            "comparable": len(comparable),
            "matched": sum(r["backend_outputs_match"] is True for r in comparable),
            "mismatched": sum(r["backend_outputs_match"] is False for r in comparable),
        },
    }


def detect_regression(
    current_instr_count,
    baseline_instr_count,
    threshold_pct=REGRESSION_THRESHOLD_PCT,
):
    delta = current_instr_count - baseline_instr_count
    delta_pct = 0.0 if baseline_instr_count == 0 else (delta / baseline_instr_count) * 100.0
    return {
        "baseline_instr_count": baseline_instr_count,
        "delta": round(delta, 4),
        "delta_pct": round(delta_pct, 4),
        "threshold_pct": threshold_pct,
        "regressed": delta_pct > threshold_pct,
    }


def build_cost_model_metrics(
    static_asm_instruction_count,
    sim_result,
):
    dynamic_count = sim_result.get("instr_count") if sim_result.get("success") else None
    perf_counters = sim_result.get("perf_counters") or {}
    metrics = {
        "static_asm_instructions": static_asm_instruction_count,
        "machine_instructions": sim_result.get("machine_code_instructions"),
        "code_size_bytes": sim_result.get("code_size_bytes"),
        "dynamic_instructions": dynamic_count,
    }
    for name in ("load", "store", "mul", "add", "madd", "branch"):
        metrics[f"dynamic_{name}"] = perf_counters.get(name)
    return metrics


def detect_cost_model_regressions(current, baseline, threshold_pct):
    comparisons = {}
    for metric, current_value in current.items():
        baseline_value = baseline.get(metric)
        if current_value is None or baseline_value is None:
            comparisons[metric] = {
                "current": current_value,
                "baseline": baseline_value,
                "delta": None,
                "delta_pct": None,
                "regressed": None,
            }
            continue
        delta = current_value - baseline_value
        if baseline_value == 0:
            delta_pct = 0.0 if current_value == 0 else None
            regressed = current_value > 0
        else:
            delta_pct = delta / baseline_value * 100.0
            regressed = delta_pct > threshold_pct
        comparisons[metric] = {
            "current": current_value,
            "baseline": baseline_value,
            "delta": round(delta, 4),
            "delta_pct": None if delta_pct is None else round(delta_pct, 4),
            "regressed": regressed,
        }
    return comparisons


def load_baseline():
    if not BASELINE_FILE.exists():
        return {}
    payload = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("cases"), dict):
        return payload["cases"]
    return payload


def baseline_instruction_count(entry):
    if not entry:
        return None
    return entry.get("dynamic_instruction_count")


def save_baseline(results, preserve_existing=False):
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    cases = load_baseline() if preserve_existing else {}
    for r in results:
        if (
            r.get("tinyfive_status") != "PASS"
            or r.get("backend_outputs_match") is False
        ):
            continue
        cases[r["name"]] = {
            "category": r["category"],
            "dynamic_instruction_count": r["instr_count"],
            "cost_model": r["cost_model"],
        }
    payload = {
        "primary_metric": "dynamic_instructions",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "cases": cases,
    }
    BASELINE_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _markdown_cell(value):
    return str(value).replace("\n", " ").replace("|", "\\|")


def _report_path(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
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
    return result["instr_count"] or 0


def _report_value(value, precision=None, prefix=""):
    if value is None:
        return "null"
    if precision is not None:
        return f"{prefix}{value:.{precision}f}"
    return f"{prefix}{value}"


def _mermaid_instruction_chart(results):
    measured = [
        (result["name"], result["instr_count"])
        for result in results
        if result["instr_count"] is not None
    ]
    if not measured:
        return "本次运行没有可展示的 TinyFive 动态指令数据。\n"

    labels = ", ".join(
        f'"{name.replace(chr(34), chr(92) + chr(34))}"'
        for name, _ in measured
    )
    values = ", ".join(str(value) for _, value in measured)
    y_max = max(1, math.ceil(max(value for _, value in measured) * 1.1))
    return (
        "```mermaid\n"
        "---\n"
        "config:\n"
        "    xyChart:\n"
        "        width: 1200\n"
        "        xAxis:\n"
        "            labelRotation: -45\n"
        "---\n"
        "xychart-beta\n"
        "    title \"TinyFive Dynamic Instruction Counts\"\n"
        f"    x-axis [{labels}]\n"
        f"    y-axis \"Instructions\" 0 --> {y_max}\n"
        f"    bar [{values}]\n"
        "```\n"
    )


def generate_unified_report_text_cn(
    results,
    passed,
    failed,
    regression_threshold_pct=REGRESSION_THRESHOLD_PCT,
    selection_category=None,
    selection_filter=None,
):
    pass_rate = 0.0 if not results else passed / len(results) * 100.0
    lines = [
        "# ScratchV DSL 编译器性能测试报告\n\n",
        "## 测试概览\n\n",
        f"- 用例类别筛选: {_report_value(selection_category)}\n",
        f"- 用例名称筛选: {_report_value(selection_filter)}\n",
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
        "| 用例 | 类别 | 解释器状态 | TinyFive 状态 | 失败类型 | TinyFive 动态指令数 | 基线 | 变化率(%) | 是否退化 | 预期输出 | 解释器输出 | TinyFive 输出 |\n",
        "|---|---|---|---|---|---:|---:|---:|---|---|---|---|\n",
    ]

    for result in results:
        lines.append(
            f"| {result['name']} | {result['category']} | "
            f"{result['interpreter_status']} | {result['tinyfive_status']} | "
            f"{_report_value(result['tinyfive_failure_kind'])} | "
            f"{_report_value(result['instr_count'])} | "
            f"{_report_value(result['baseline_instr_count'], 2)} | "
            f"{_report_value(result['delta_pct'], 2)} | "
            f"{_report_value(result['regressed'])} | {_markdown_cell(_report_value(result['expected']))} | "
            f"{_markdown_cell(_report_value(result['interpreter_actual']))} | "
            f"{_markdown_cell(_report_value(result['actual']))} |\n"
        )

    lines.extend([
        "\n## 性能图表\n\n",
        _mermaid_instruction_chart(results),
        "\n## 单用例报告\n\n",
        "每个用例的后端结果、性能指标、耗时诊断和生成汇编位于 "
        "`benchmark_reports/topic06/cases/` 目录。\n",
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
    <p>用例类别筛选：{{ fmt(selection_category) }}，用例名称筛选：{{ fmt(selection_filter) }}</p>
    <p>用例总数：{{ total }}，通过：{{ passed }}，失败：{{ failed }}</p>
    <p>性能退化阈值：{{ fmt(regression_threshold_pct, 2) }}%</p>
  </div>
  <img src="{{ chart_name }}" alt="课程版指令数图表">
  <div class="table-wrap"><table>
    <thead><tr>
      <th>用例</th><th>类别</th><th>解释器状态</th><th>TinyFive 状态</th><th>失败类型</th><th>TinyFive 动态指令数</th>
      <th>基线</th><th>变化率(%)</th><th>是否退化</th>
      <th>预期输出</th><th>解释器输出</th><th>TinyFive 输出</th>
    </tr></thead>
    <tbody>{% for r in results %}<tr>
      <td>{{ r.name }}</td><td>{{ r.category }}</td>
      <td>{{ r.interpreter_status }}</td><td>{{ r.tinyfive_status }}</td><td>{{ fmt(r.tinyfive_failure_kind) }}</td>
      <td>{{ fmt(r.instr_count) }}</td>
      <td>{{ fmt(r.baseline_instr_count, 2) }}</td><td>{{ fmt(r.delta_pct, 2) }}</td><td>{{ fmt(r.regressed) }}</td>
      <td>{{ fmt(r.expected) }}</td><td>{{ fmt(r.interpreter_actual) }}</td><td>{{ fmt(r.actual) }}</td>
    </tr>{% endfor %}</tbody>
  </table></div>
</body>
</html>
""")
    HTML_REPORT_FILE.write_text(
        template.render(
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
    payload = {
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
        "backend_matrix": summarize_backend_matrix(results),
        "results": results,
    }
    JSON_REPORT_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return JSON_REPORT_FILE


def write_test_metadata(
    results,
    passed,
    failed,
    regression_threshold_pct=REGRESSION_THRESHOLD_PCT,
    selection_category=None,
    selection_filter=None,
):
    """Persist runner output without coupling test execution to report rendering."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "artifact_type": "topic06_test_metadata",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "regression_threshold_pct": regression_threshold_pct,
        "selection": {
            "category": selection_category,
            "filter": selection_filter,
        },
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
        },
        "backend_matrix": summarize_backend_matrix(results),
        "results": results,
    }
    METADATA_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return METADATA_FILE


def _case_performance_markdown(result):
    if result["tinyfive_status"] != "PASS":
        reason = result.get("simulation_error") or result.get("tinyfive_failure_kind")
        return "".join([
            "## 性能指标\n\n",
            "TinyFive 未有效执行，不生成性能指标。\n\n",
            f"- 原因: {_report_value(reason)}\n",
            f"- 编译耗时(s): {result['compile_time_sec']:.6f}\n",
            f"- 解释器耗时(s): {result['interpreter_time_sec']:.6f}\n\n",
        ])

    metric_labels = (
        ("static_asm_instructions", "静态汇编指令数"),
        ("machine_instructions", "编码后机器指令数"),
        ("code_size_bytes", "代码大小(bytes)"),
        ("dynamic_instructions", "TinyFive 动态指令数"),
        ("dynamic_load", "动态 load"),
        ("dynamic_store", "动态 store"),
        ("dynamic_mul", "动态 mul"),
        ("dynamic_add", "动态 add"),
        ("dynamic_madd", "动态 madd"),
        ("dynamic_branch", "动态 branch"),
    )
    lines = [
        "## 性能指标\n\n",
        "| 指标 | 当前值 | 基线 | 变化率(%) | 是否退化 |\n",
        "|---|---:|---:|---:|---|\n",
    ]
    for key, label in metric_labels:
        comparison = result["cost_model_comparison"].get(key, {})
        lines.append(
            f"| {label} | {_report_value(result['cost_model'].get(key))} | "
            f"{_report_value(comparison.get('baseline'))} | "
            f"{_report_value(comparison.get('delta_pct'))} | "
            f"{_report_value(comparison.get('regressed'))} |\n"
        )
    lines.extend([
        "\n## 耗时诊断\n\n",
        f"- 编译耗时(s): {result['compile_time_sec']:.6f}\n",
        f"- 解释器耗时(s): {result['interpreter_time_sec']:.6f}\n",
        f"- TinyFive 模拟耗时(s): {result['simulation_time_sec']:.6f}\n\n",
    ])
    return "".join(lines)


def write_case_reports(results):
    CASE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for result in results:
        report_stem = f"{result['category']}-{result['name']}"
        markdown_path = CASE_REPORT_DIR / f"{report_stem}.md"
        json_path = CASE_REPORT_DIR / f"{report_stem}.json"
        result["case_report_md"] = _report_path(markdown_path)
        result["case_report_json"] = _report_path(json_path)

        asm_text = ""
        asm_path = PROJECT_ROOT / result["asm"]
        if asm_path.exists():
            asm_text = asm_path.read_text(encoding="utf-8")

        interpreter_match = (
            result["interpreter_matched"]
            if result["interpreter_status"] in {"PASS", "MISMATCH"}
            else None
        )
        tinyfive_match = (
            result["matched"]
            if result["tinyfive_status"] in {"PASS", "MISMATCH"}
            else None
        )
        markdown_path.write_text("".join([
            f"# {result['name']} 测试详情\n\n",
            "## 基本信息\n\n",
            f"- 类别: {result['category']}\n",
            f"- DSL: `{result['path']}`\n",
            f"- 描述: {result['description']}\n",
            f"- 总体状态: {result['status']}\n\n",
            "## 后端结果\n\n",
            f"- 预期输出: {_report_value(result['expected'])}\n\n",
            "| 后端 | 状态 | 实际输出 | 与期望匹配 | 失败原因 |\n",
            "|---|---|---|---|---|\n",
            f"| DSLInterpreter | {result['interpreter_status']} | "
            f"{_markdown_cell(_report_value(result['interpreter_actual']))} | "
            f"{_report_value(interpreter_match)} | "
            f"{_markdown_cell(_report_value(result['interpreter_error']))} |\n",
            f"| TinyFive | {result['tinyfive_status']} | "
            f"{_markdown_cell(_report_value(result['actual']))} | "
            f"{_report_value(tinyfive_match)} | "
            f"{_markdown_cell(_report_value(result['simulation_error']))} |\n\n",
            _case_performance_markdown(result),
            "## 生成汇编\n\n",
            "```asm\n",
            asm_text.rstrip(),
            "\n```\n",
        ]), encoding="utf-8")
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def write_report(
    results,
    passed,
    failed,
    regression_threshold_pct=REGRESSION_THRESHOLD_PCT,
    selection_category=None,
    selection_filter=None,
    full_report=False,
):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_case_reports(results)
    chart_path = write_chart(results) if full_report else None
    REPORT_FILE.write_text(
        generate_unified_report_text_cn(
            results,
            passed,
            failed,
            regression_threshold_pct=regression_threshold_pct,
            selection_category=selection_category,
            selection_filter=selection_filter,
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
    parser.add_argument("--update-baseline", action="store_true",
                        help="Write current TinyFive instruction counts to the baseline file.")
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
        "--verification-backend",
        choices=("both", "tinyfive", "interpreter"),
        default="both",
        help="Select result verification backend(s); both keeps verdicts independent.",
    )
    parser.add_argument(
        "--fail-on-test-failure",
        action="store_true",
        help="Return exit code 1 when one or more selected cases fail.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.update_baseline and args.verification_backend == "interpreter":
        print("Cannot update a TinyFive cost-model baseline in interpreter-only mode.")
        return 2
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline()

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
        interpreter_start = time.perf_counter()
        if args.verification_backend in {"both", "interpreter"} and not meta.get("_metadata_error"):
            interpreter_result = run_interpreter(dsl_file, meta.get("inputs", {}))
        else:
            interpreter_result = {
                "success": False,
                "return_value": None,
                "backend": "interpreter",
                "status": "SKIPPED",
                "error": (
                    "interpreter backend not selected"
                    if not meta.get("_metadata_error")
                    else "invalid metadata"
                ),
            }
        interpreter_time_sec = time.perf_counter() - interpreter_start
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
        input_abi_supported, input_abi_reason = tinyfive_input_abi_supported(
            meta.get("inputs", {}),
        )
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
        elif not input_abi_supported:
            sim_result = {
                "success": False,
                "instr_count": None,
                "return_value": None,
                "backend": "skipped",
                "error": input_abi_reason,
            }
        elif args.verification_backend == "interpreter":
            sim_result = {
                "success": False,
                "instr_count": 0,
                "return_value": None,
                "backend": "skipped",
                "error": "TinyFive backend not selected",
            }
        else:
            sim_result = run_simulation(output_file, initial_registers)
        simulation_time_sec = time.perf_counter() - simulation_start
        expected_value = meta.get("expected_return")
        actual_value = sim_result.get("return_value")
        matched = bool(sim_result.get("success")) and values_equal(actual_value, expected_value)
        interpreter_actual = interpreter_result.get("return_value")
        interpreter_matched = bool(interpreter_result.get("success")) and values_equal(
            interpreter_actual,
            expected_value,
        )
        if interpreter_result.get("status") == "EXECUTED":
            interpreter_result["status"] = "PASS" if interpreter_matched else "MISMATCH"
        interpreter_failure_kind = None
        if interpreter_result.get("status") == "UNSUPPORTED":
            interpreter_failure_kind = "interpreter_control_flow_unsupported"
        elif interpreter_result.get("status") == "TIMEOUT":
            interpreter_failure_kind = "interpreter_timeout"
        elif interpreter_result.get("status") == "ERROR":
            interpreter_failure_kind = "interpreter_error"
        elif interpreter_result.get("status") == "MISMATCH":
            interpreter_failure_kind = "interpreter_result_mismatch"
        backend_outputs_match = (
            values_equal(interpreter_actual, actual_value)
            if (
                interpreter_result.get("success")
                and sim_result.get("success")
                and input_abi_supported
            )
            else None
        )
        static_asm_instruction_count = count_assembly_instructions(output_file)
        regression = {
            "baseline_instr_count": None,
            "delta": None,
            "delta_pct": None,
            "threshold_pct": args.regression_threshold,
            "regressed": None,
        }

        tinyfive_status, tinyfive_failure_kind = classify_tinyfive_result(
            result,
            sim_result,
            matched,
            input_abi_supported,
            metadata_error=metadata_error,
            register_map_error=register_map_error,
        )

        baseline_entry = baseline.get(dsl_file.stem)
        if tinyfive_status == "PASS":
            regression["regressed"] = False
            baseline_count = baseline_instruction_count(baseline_entry)
            if baseline_count is not None:
                regression = detect_regression(
                    current_instr_count=sim_result.get("instr_count", 0),
                    baseline_instr_count=baseline_count,
                    threshold_pct=args.regression_threshold,
                )

        cost_model = build_cost_model_metrics(
            static_asm_instruction_count,
            sim_result,
        )
        cost_model_comparison = {}
        cost_model_regressed = None
        baseline_entry = baseline.get(dsl_file.stem)
        if (
            baseline_entry
            and tinyfive_status == "PASS"
        ):
            baseline_cost_model = baseline_entry.get("cost_model") or {
                "dynamic_instructions": baseline_instruction_count(baseline_entry),
            }
            cost_model_comparison = detect_cost_model_regressions(
                cost_model,
                baseline_cost_model,
                args.regression_threshold,
            )
            comparable_regressions = [
                item["regressed"]
                for item in cost_model_comparison.values()
                if item["regressed"] is not None
            ]
            cost_model_regressed = any(comparable_regressions)
            regression["regressed"] = bool(regression["regressed"] or cost_model_regressed)
        total_time_sec = time.perf_counter() - case_start

        tinyfive_ok = (
            tinyfive_status == "PASS"
            and regression["regressed"] is not True
        )
        interpreter_ok = (
            result.returncode == 0
            and interpreter_result.get("status") == "PASS"
        )
        if args.verification_backend == "interpreter":
            ok = interpreter_ok
        elif args.verification_backend == "tinyfive":
            ok = tinyfive_ok
        else:
            interpreter_required = interpreter_result.get("status") != "UNSUPPORTED"
            ok = tinyfive_ok and (interpreter_ok or not interpreter_required)

        if ok:
            print("PASS")
            passed += 1
            status = "PASS"
        else:
            print("FAIL")
            failed += 1
            status = "FAIL"
            if tinyfive_failure_kind == "input_abi_unsupported":
                print(input_abi_reason)
            elif sim_result.get("error"):
                print(sim_result["error"])
            else:
                print(
                    "output mismatch: "
                    f"expected={expected_value}, "
                    f"tinyfive={actual_value}"
                )

        results.append({
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
            "verification_backend": args.verification_backend,
            "interpreter_status": interpreter_result.get("status"),
            "interpreter_actual": interpreter_actual,
            "interpreter_matched": interpreter_matched,
            "interpreter_error": interpreter_result.get("error"),
            "interpreter_failure_kind": interpreter_failure_kind,
            "interpreter_time_sec": interpreter_time_sec,
            "backend_outputs_match": backend_outputs_match,
            "tinyfive_status": tinyfive_status,
            "tinyfive_failure_kind": tinyfive_failure_kind,
            "tinyfive_input_abi_supported": input_abi_supported,
            "tinyfive_input_abi_reason": input_abi_reason,
            "simulation_error": sim_result.get("error"),
            "initial_registers": initial_registers,
            "register_map": _report_path(register_map_file),
            "backend": sim_result.get("backend", "none"),
            "instr_count": sim_result.get("instr_count", 0),
            "static_asm_instruction_count": static_asm_instruction_count,
            "machine_code_instruction_count": sim_result.get("machine_code_instructions"),
            "code_size_bytes": sim_result.get("code_size_bytes"),
            "perf_counters": sim_result.get("perf_counters") or {},
            "cost_model": cost_model,
            "cost_model_comparison": cost_model_comparison,
            "cost_model_regressed": cost_model_regressed,
            "compile_returncode": result.returncode,
            "compile_timed_out": result.returncode == 124,
            "compile_error": compile_error,
            "compile_command": (
                shlex.join(str(arg) for arg in result.args)
                if isinstance(result.args, (list, tuple)) and result.args
                else "(compiler was not started)"
            ),
            "compile_log": _report_path(compile_log) if compile_log else None,
            "compile_time_sec": compile_time_sec,
            "simulation_time_sec": simulation_time_sec,
            "total_time_sec": total_time_sec,
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

    if args.update_baseline:
        save_baseline(
            results,
            preserve_existing=bool(args.category or args.name_filter),
        )
        print(f"Baseline written to {BASELINE_FILE}")

    metadata_path = write_test_metadata(
        results,
        passed,
        failed,
        regression_threshold_pct=args.regression_threshold,
        selection_category=args.category,
        selection_filter=args.name_filter,
    )
    print(f"Metadata written to {metadata_path}")
    return 1 if args.fail_on_test_failure and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
