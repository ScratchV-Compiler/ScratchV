import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from scratchv.simulator.tinyfive import verify_assembly

TEST_DIR = Path("tests_main")
BUILD_DIR = Path("build")
REPORT_DIR = Path("reports")
REPORT_FILE = REPORT_DIR / "report.md"
HTML_REPORT_FILE = REPORT_DIR / "report.html"
CHART_FILE = REPORT_DIR / "course_report_instructions.png"
BASELINE_FILE = REPORT_DIR / "benchmark_baseline.json"
REGRESSION_THRESHOLD_PCT = 5.0


def run_compile(dsl_file: Path):
    output_file = BUILD_DIR / (dsl_file.stem + ".s")

    cmd = [
        sys.executable,
        "-m",
        "scratchv.main",
        str(dsl_file),
        "-o",
        str(output_file),
        "--optimize",
        "all",
        "--dump-ir",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    return result, output_file


def load_metadata(dsl_file: Path):
    meta_file = dsl_file.with_suffix(".meta.json")
    if not meta_file.exists():
        return {
            "description": "",
            "expected_output_type": "return_value",
            "expected_return": "",
        }
    return json.loads(meta_file.read_text(encoding="utf-8"))


def run_simulation(asm_file: Path):
    if not asm_file.exists():
        return {
            "success": False,
            "instr_count": 0,
            "return_value": None,
            "backend": "none",
            "error": "assembly file not found",
        }

    asm_code = asm_file.read_text(encoding="utf-8")
    return verify_assembly(asm_code)


def apply_add(lhs, rhs):
    if isinstance(lhs, list) and isinstance(rhs, list):
        return [apply_add(a, b) for a, b in zip(lhs, rhs)]
    if isinstance(lhs, list):
        return [apply_add(a, rhs) for a in lhs]
    if isinstance(rhs, list):
        return [apply_add(lhs, b) for b in rhs]
    return lhs + rhs


def apply_binary(lhs, rhs, op):
    if isinstance(lhs, list) and isinstance(rhs, list):
        return [apply_binary(a, b, op) for a, b in zip(lhs, rhs)]
    if isinstance(lhs, list):
        return [apply_binary(a, rhs, op) for a in lhs]
    if isinstance(rhs, list):
        return [apply_binary(lhs, b, op) for b in rhs]
    return op(lhs, rhs)


def apply_relu(value):
    if isinstance(value, list):
        return [apply_relu(v) for v in value]
    return value if value > 0 else 0


def apply_gelu(value):
    if isinstance(value, list):
        return [apply_gelu(v) for v in value]
    return 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))


def apply_softmax(value):
    if not isinstance(value, list):
        return 1.0
    max_value = max(value)
    exp_values = [math.exp(v - max_value) for v in value]
    total = sum(exp_values)
    return [v / total for v in exp_values]


def apply_maxpool(value, kernel, stride):
    if not isinstance(value, list):
        return value
    return [max(value[i:i + kernel]) for i in range(0, len(value) - kernel + 1, stride)]


def apply_dot(lhs, rhs, length):
    return sum(lhs[i] * rhs[i] for i in range(length))


def apply_matmul(lhs, rhs, m, n, k):
    result = []
    for i in range(m):
        row = []
        for j in range(n):
            cell = 0
            for kk in range(k):
                cell += lhs[i][kk] * rhs[kk][j]
            row.append(cell)
        result.append(row)
    return result


def resolve_value(token, env):
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return env[token]


def execute_block(lines, env, start_idx=0, end_idx=None):
    if end_idx is None:
        end_idx = len(lines)

    idx = start_idx
    while idx < end_idx:
        line = lines[idx]

        if line.startswith("for "):
            loop_var_text = line.replace("for ", "", 1)
            loop_var, bounds = [p.strip() for p in loop_var_text.split("=", 1)]
            loop_start_text, loop_end_text = [p.strip() for p in bounds.split(",", 1)]
            loop_start = int(loop_start_text)
            loop_end = int(loop_end_text)

            depth = 1
            body_start = idx + 1
            body_end = body_start
            while body_end < end_idx and depth > 0:
                current = lines[body_end]
                if current.startswith("for "):
                    depth += 1
                elif current == "endfor":
                    depth -= 1
                    if depth == 0:
                        break
                body_end += 1

            for i in range(loop_start, loop_end):
                env[loop_var] = i
                returned, value = execute_block(lines, env, body_start, body_end)
                if returned:
                    return True, value

            idx = body_end + 1
            continue

        if line.startswith("if "):
            cond_text = line.replace("if ", "", 1).strip()
            cond_value = resolve_value(cond_text, env)

            depth = 1
            body_start = idx + 1
            scan_idx = body_start
            else_idx = None
            endif_idx = None
            while scan_idx < end_idx:
                current = lines[scan_idx]
                if current.startswith("if "):
                    depth += 1
                elif current == "endif":
                    depth -= 1
                    if depth == 0:
                        endif_idx = scan_idx
                        break
                elif current == "else" and depth == 1:
                    else_idx = scan_idx
                scan_idx += 1

            if endif_idx is None:
                raise ValueError("if without matching endif")

            if cond_value:
                branch_start = body_start
                branch_end = else_idx if else_idx is not None else endif_idx
            else:
                branch_start = else_idx + 1 if else_idx is not None else endif_idx
                branch_end = endif_idx

            returned, value = execute_block(lines, env, branch_start, branch_end)
            if returned:
                return True, value

            idx = endif_idx + 1
            continue

        if line in {"else", "endif"}:
            return False, None

        if line == "endfor":
            return False, None

        if line.startswith("return "):
            return True, resolve_value(line.replace("return ", "", 1).strip(), env)

        dest_name, expr = [p.strip() for p in line.split("=", 1)]
        op_name = expr[:expr.index("(")]
        arg_text = expr[expr.index("(") + 1: expr.rindex(")")]
        args = [a.strip() for a in arg_text.split(",") if a.strip()]

        plain_args = []
        kwargs = {}
        for arg in args:
            if ":" in arg:
                key, value = arg.split(":", 1)
                kwargs[key.strip()] = int(value.strip())
            else:
                plain_args.append(resolve_value(arg, env))

        if op_name == "add":
            env[dest_name] = apply_add(plain_args[0], plain_args[1])
        elif op_name == "sub":
            env[dest_name] = apply_binary(plain_args[0], plain_args[1], lambda a, b: a - b)
        elif op_name == "mul":
            env[dest_name] = apply_binary(plain_args[0], plain_args[1], lambda a, b: a * b)
        elif op_name == "div":
            env[dest_name] = apply_binary(plain_args[0], plain_args[1], lambda a, b: a / b)
        elif op_name == "relu":
            env[dest_name] = apply_relu(plain_args[0])
        elif op_name == "gelu":
            env[dest_name] = apply_gelu(plain_args[0])
        elif op_name == "softmax":
            env[dest_name] = apply_softmax(plain_args[0])
        elif op_name == "maxpool":
            env[dest_name] = apply_maxpool(
                plain_args[0],
                kwargs.get("kernel", 2),
                kwargs.get("stride", 2),
            )
        elif op_name == "dot":
            env[dest_name] = apply_dot(plain_args[0], plain_args[1], kwargs["len"])
        elif op_name == "matmul":
            env[dest_name] = apply_matmul(
                plain_args[0],
                plain_args[1],
                kwargs["m"],
                kwargs["n"],
                kwargs["k"],
            )
        else:
            raise ValueError(f"Unsupported op in reference executor: {op_name}")

        idx += 1

    return False, None


def execute_dsl_reference(dsl_file: Path, inputs):
    raw_lines = dsl_file.read_text(encoding="utf-8").splitlines()
    lines = []
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)

    env = dict(inputs)
    returned, value = execute_block(lines, env)
    return value if returned else None


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
            "avg_instr_count": 0.0,
            "min_instr_count": 0,
            "max_instr_count": 0,
            "ci95_instr_count": 0.0,
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


def detect_regression(avg_instr_count, baseline_instr_count):
    delta = avg_instr_count - baseline_instr_count
    delta_pct = 0.0 if baseline_instr_count == 0 else (delta / baseline_instr_count) * 100.0
    return {
        "baseline_instr_count": baseline_instr_count,
        "delta": round(delta, 4),
        "delta_pct": round(delta_pct, 4),
        "threshold_pct": REGRESSION_THRESHOLD_PCT,
        "regressed": delta_pct > REGRESSION_THRESHOLD_PCT,
    }


def load_baseline():
    if not BASELINE_FILE.exists():
        return {}
    return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))


def save_baseline(results):
    REPORT_DIR.mkdir(exist_ok=True)
    payload = {}
    for r in results:
        payload[r["name"]] = {
            "category": r["category"],
            "avg_instr_count": r["avg_instr_count"],
            "runs": r["benchmark_runs"],
        }
    BASELINE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_report_text(results, passed, failed):
    lines = []
    lines.append("# ScratchV DSL 编译器性能测试报告\n\n")

    lines.append("## 测试概览\n\n")
    lines.append(f"- 用例总数: {len(results)}\n")
    lines.append(f"- 通过数量: {passed}\n")
    lines.append(f"- 失败数量: {failed}\n\n")

    benchmark_mode = any(r.get("benchmark_runs", 1) > 1 for r in results)
    if benchmark_mode:
        lines.append("## 性能基准概览\n\n")
        lines.append("- 运行模式: benchmark\n")
        lines.append(f"- 性能基线文件: `{BASELINE_FILE}`\n\n")

    lines.append("## 测试结果\n\n")
    if benchmark_mode:
        lines.append("| 测试用例 | 类别 | 状态 | 模拟后端 | 平均指令数 | 最小值 | 最大值 | 基线 | 变化率 | 是否退化 | 预期输出 | 实际输出 | 是否匹配 | 汇编文件 |\n")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|\n")
    else:
        lines.append("| 测试用例 | 类别 | 状态 | 模拟后端 | 指令数 | 预期输出 | 实际输出 | 是否匹配 | 汇编文件 |\n")
        lines.append("|---|---|---|---|---:|---|---|---|---|\n")

    for r in results:
        expected = str(r["expected"]).replace("\n", " ").replace("|", "\\|")
        actual = str(r["actual"]).replace("\n", " ").replace("|", "\\|")
        if benchmark_mode:
            lines.append(
                f"| {r['name']} | {r['category']} | {r['status']} | {r['backend']} | "
                f"{r['avg_instr_count']:.2f} | {r['min_instr_count']} | {r['max_instr_count']} | "
                f"{r['baseline_instr_count']:.2f} | {r['delta_pct']:.2f} | {r['regressed']} | "
                f"{expected} | {actual} | {r['matched']} | {r['asm']} |\n"
            )
        else:
            lines.append(
                f"| {r['name']} | {r['category']} | {r['status']} | {r['backend']} | "
                f"{r['instr_count']} | {expected} | {actual} | {r['matched']} | {r['asm']} |\n"
            )

    lines.append("\n## 性能图表\n\n")
    chart_cases = [r["name"] for r in results]
    chart_instr = [str(round(r.get("avg_instr_count", r["instr_count"]), 2)) for r in results]
    lines.append("### 各测试用例指令数\n\n")
    lines.append("```mermaid\n")
    lines.append("xychart-beta\n")
    lines.append('    title "各测试用例指令数"\n')
    lines.append("    x-axis [" + ", ".join(f'"{name}"' for name in chart_cases) + "]\n")
    max_instr = max((r.get("avg_instr_count", r["instr_count"]) for r in results), default=0)
    lines.append(f'    y-axis "指令数" 0 --> {max_instr + 2}\n')
    lines.append("    bar [" + ", ".join(chart_instr) + "]\n")
    lines.append("```\n\n")

    category_totals = {}
    for r in results:
        category_totals[r["category"]] = category_totals.get(r["category"], 0) + r.get("avg_instr_count", r["instr_count"])
    lines.append("### 各类别指令数占比\n\n")
    lines.append("```mermaid\n")
    lines.append("pie showData\n")
    lines.append('    title 各类别指令数占比\n')
    for category, total in sorted(category_totals.items()):
        lines.append(f'    "{category}" : {total}\n')
    lines.append("```\n")

    lines.append("\n## 用例详情\n\n")
    for r in results:
        lines.append(f"### {r['name']}\n\n")
        lines.append(f"- 类别: {r['category']}\n")
        lines.append(f"- 描述: {r['description']}\n")
        lines.append(f"- 预期输出 ({r['expected_type']}): {r['expected']}\n")
        lines.append(f"- 实际输出: {r['actual']}\n")
        lines.append(f"- 是否匹配: {r['matched']}\n")
        lines.append(f"- 模拟后端: {r['backend']}\n")
        if benchmark_mode:
            lines.append(f"- Benchmark 重复次数: {r['benchmark_runs']}\n")
            lines.append(f"- 平均指令数: {r['avg_instr_count']:.2f}\n")
            lines.append(f"- 最小指令数: {r['min_instr_count']}\n")
            lines.append(f"- 最大指令数: {r['max_instr_count']}\n")
            lines.append(f"- 基线指令数: {r['baseline_instr_count']:.2f}\n")
            lines.append(f"- 性能变化率 (%): {r['delta_pct']:.2f}\n")
            lines.append(f"- 是否性能退化: {r['regressed']}\n")
        else:
            lines.append(f"- 指令数: {r['instr_count']}\n")
        lines.append(f"- 汇编文件: {r['asm']}\n\n")

    return "".join(lines)


def write_report(results, passed, failed):
    REPORT_DIR.mkdir(exist_ok=True)

    REPORT_FILE.write_text(generate_report_text(results, passed, failed), encoding="utf-8")
    print(f"\nReport written to {REPORT_FILE}")


def _markdown_cell(value):
    return str(value).replace("\n", " ").replace("|", "\\|")


def generate_report_text(results, passed, failed):
    benchmark_mode = any(r.get("benchmark_runs", 1) > 1 for r in results)
    pass_rate = 0.0 if not results else passed / len(results) * 100.0
    lines = [
        "# ScratchV DSL 编译器性能测试报告\n\n",
        "## 测试概览\n\n",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"- 用例总数: {len(results)}\n",
        f"- 通过数量: {passed}\n",
        f"- 失败数量: {failed}\n",
        f"- 通过率: {pass_rate:.1f}%\n",
        f"- 测试目录: `{TEST_DIR}`\n",
        f"- 汇编输出目录: `{BUILD_DIR}`\n",
        f"- 性能基线文件: `{BASELINE_FILE}`\n",
        f"- 性能退化阈值: {REGRESSION_THRESHOLD_PCT:.1f}%\n\n",
        "## 测试结果\n\n",
    ]

    if benchmark_mode:
        lines.extend([
            "| 用例 | 类别 | 状态 | 模拟后端 | 平均指令数 | 95%置信区间 | 最小 | 最大 | 基线 | 变化率(%) | 是否退化 | 预期输出 | 实际输出 | 输出匹配 | 汇编文件 |\n",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|\n",
        ])
    else:
        lines.extend([
            "| 用例 | 类别 | 状态 | 模拟后端 | 指令数 | 预期输出 | 实际输出 | 输出匹配 | 汇编文件 |\n",
            "|---|---|---|---|---:|---|---|---|---|\n",
        ])

    for r in results:
        expected = _markdown_cell(r["expected"])
        actual = _markdown_cell(r["actual"])
        if benchmark_mode:
            lines.append(
                f"| {r['name']} | {r['category']} | {r['status']} | {r['backend']} | "
                f"{r['avg_instr_count']:.2f} | ±{r['ci95_instr_count']:.2f} | "
                f"{r['min_instr_count']} | {r['max_instr_count']} | "
                f"{r['baseline_instr_count']:.2f} | {r['delta_pct']:.2f} | "
                f"{r['regressed']} | {expected} | {actual} | {r['matched']} | {r['asm']} |\n"
            )
        else:
            lines.append(
                f"| {r['name']} | {r['category']} | {r['status']} | {r['backend']} | "
                f"{r['instr_count']} | {expected} | {actual} | {r['matched']} | {r['asm']} |\n"
            )

    lines.extend([
        "\n## 性能图表\n\n",
        f"![课程版指令数图表]({CHART_FILE.name})\n\n",
        "### Mermaid 图表\n\n",
        "```mermaid\n",
        "xychart-beta\n",
        '    title "各测试用例指令数"\n',
        "    x-axis [" + ", ".join(f'"{r["name"]}"' for r in results) + "]\n",
    ])
    max_instr = max((r.get("avg_instr_count", r["instr_count"]) for r in results), default=0)
    chart_values = [str(round(r.get("avg_instr_count", r["instr_count"]), 2)) for r in results]
    lines.extend([
        f'    y-axis "指令数" 0 --> {max_instr + 2}\n',
        "    bar [" + ", ".join(chart_values) + "]\n",
        "```\n\n",
        "## 用例详情\n\n",
    ])

    for r in results:
        lines.extend([
            f"### {r['name']}\n\n",
            f"- 类别: {r['category']}\n",
            f"- 描述: {r['description']}\n",
            f"- 预期输出 ({r['expected_type']}): {r['expected']}\n",
            f"- 实际输出: {r['actual']}\n",
            f"- 输出是否匹配: {r['matched']}\n",
            f"- 模拟后端: {r['backend']}\n",
            f"- 汇编文件: {r['asm']}\n",
        ])
        if benchmark_mode:
            lines.extend([
                f"- Benchmark 重复次数: {r['benchmark_runs']}\n",
                f"- 平均指令数: {r['avg_instr_count']:.2f}\n",
                f"- 95% 置信区间: ±{r['ci95_instr_count']:.2f}\n",
                f"- 最小指令数: {r['min_instr_count']}\n",
                f"- 最大指令数: {r['max_instr_count']}\n",
                f"- 基线指令数: {r['baseline_instr_count']:.2f}\n",
                f"- 性能变化率: {r['delta_pct']:.2f}%\n",
                f"- 性能退化阈值: {r['threshold_pct']:.2f}%\n",
                f"- 是否性能退化: {r['regressed']}\n",
            ])
        else:
            lines.append(f"- 指令数: {r['instr_count']}\n")
        lines.append("\n")

    return "".join(lines)


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

    names = [r["name"] for r in results]
    values = [r.get("avg_instr_count", r["instr_count"]) for r in results]
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


def write_html_report(results, passed, failed):
    try:
        from jinja2 import Template
    except ImportError:
        return None

    pass_rate = 0.0 if not results else passed / len(results) * 100.0
    template = Template("""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>ScratchV 课程版测试报告</title>
  <style>
    body { font-family: "Microsoft YaHei", sans-serif; margin: 32px; color: #1f2937; }
    .summary { background: #f8fafc; border: 1px solid #e5e7eb; padding: 14px 18px; }
    table { border-collapse: collapse; width: 100%; margin-top: 18px; }
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
    <p>用例总数：{{ total }}，通过：{{ passed }}，失败：{{ failed }}，通过率：{{ "%.1f"|format(pass_rate) }}%</p>
    <p>测试目录：{{ test_dir }}，性能退化阈值：{{ threshold }}%</p>
  </div>
  <img src="{{ chart_name }}" alt="课程版指令数图表">
  <table>
    <thead>
      <tr><th>用例</th><th>类别</th><th>状态</th><th>平均指令数</th><th>95%置信区间</th><th>变化率(%)</th><th>是否退化</th><th>描述</th></tr>
    </thead>
    <tbody>
      {% for r in results %}
      <tr>
        <td>{{ r.name }}</td>
        <td>{{ r.category }}</td>
        <td class="{{ 'pass' if r.status == 'PASS' else 'fail' }}">{{ r.status }}</td>
        <td>{{ "%.2f"|format(r.avg_instr_count) }}</td>
        <td>±{{ "%.2f"|format(r.ci95_instr_count) }}</td>
        <td>{{ "%.2f"|format(r.delta_pct) }}</td>
        <td>{{ r.regressed }}</td>
        <td>{{ r.description }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</body>
</html>
""")
    HTML_REPORT_FILE.write_text(
        template.render(
            total=len(results),
            passed=passed,
            failed=failed,
            pass_rate=pass_rate,
            test_dir=str(TEST_DIR),
            threshold=REGRESSION_THRESHOLD_PCT,
            chart_name=CHART_FILE.name,
            results=results,
        ),
        encoding="utf-8",
    )
    return HTML_REPORT_FILE


def write_report(results, passed, failed):
    REPORT_DIR.mkdir(exist_ok=True)
    write_chart(results)
    REPORT_FILE.write_text(generate_report_text(results, passed, failed), encoding="utf-8")
    html_path = write_html_report(results, passed, failed)
    print(f"\nMarkdown report written to {REPORT_FILE}")
    if html_path:
        print(f"HTML report written to {html_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run ScratchV DSL benchmark suite.")
    parser.add_argument("--benchmark", type=int, default=0, metavar="N",
                        help="Run each case N times and report average instruction count.")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Write current benchmark averages to the baseline file.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    BUILD_DIR.mkdir(exist_ok=True)
    baseline = load_baseline() if args.benchmark else {}

    dsl_files = list(TEST_DIR.rglob("*.dsl"))

    if not dsl_files:
        print("No DSL test files found.")
        return

    passed = 0
    failed = 0
    results = []

    print("Running DSL compiler tests...")
    print("=" * 50)

    for dsl_file in dsl_files:
        print(f"\n[TEST] {dsl_file}")

        meta = load_metadata(dsl_file)
        result, output_file = run_compile(dsl_file)
        sim_result = run_simulation(output_file) if result.returncode == 0 else {
            "success": False,
            "instr_count": 0,
            "return_value": None,
            "backend": "none",
            "error": "compile failed",
        }
        expected_value = meta.get("expected_return")
        actual_value = execute_dsl_reference(dsl_file, meta.get("inputs", {}))
        matched = values_equal(actual_value, expected_value)
        benchmark_counts = []
        benchmark_summary = {
            "runs": 1,
            "avg_instr_count": sim_result.get("instr_count", 0),
            "min_instr_count": sim_result.get("instr_count", 0),
            "max_instr_count": sim_result.get("instr_count", 0),
            "ci95_instr_count": 0.0,
        }
        regression = {
            "baseline_instr_count": 0.0,
            "delta": 0.0,
            "delta_pct": 0.0,
            "threshold_pct": REGRESSION_THRESHOLD_PCT,
            "regressed": False,
        }

        if args.benchmark > 0 and result.returncode == 0 and output_file.exists():
            for _ in range(args.benchmark):
                benchmark_counts.append(run_simulation(output_file).get("instr_count", 0))
            benchmark_summary = summarize_benchmark_runs(benchmark_counts)
            baseline_entry = baseline.get(dsl_file.stem)
            if baseline_entry:
                regression = detect_regression(
                    avg_instr_count=benchmark_summary["avg_instr_count"],
                    baseline_instr_count=baseline_entry.get("avg_instr_count", 0.0),
                )

        ok = (
            result.returncode == 0
            and output_file.exists()
            and sim_result["success"]
            and matched
            and not regression["regressed"]
        )

        if ok:
            print("PASS")
            passed += 1
            status = "PASS"
        else:
            print("FAIL")
            failed += 1
            status = "FAIL"
            print(sim_result.get("error") or result.stderr or result.stdout)

        results.append({
            "name": dsl_file.stem,
            "category": dsl_file.parent.name,
            "path": str(dsl_file),
            "status": status,
            "description": meta.get("description", ""),
            "expected_type": meta.get("expected_output_type", "return_value"),
            "expected": expected_value,
            "actual": actual_value,
            "matched": matched,
            "backend": sim_result.get("backend", "none"),
            "instr_count": sim_result.get("instr_count", 0),
            "benchmark_runs": benchmark_summary["runs"],
            "avg_instr_count": benchmark_summary["avg_instr_count"],
            "min_instr_count": benchmark_summary["min_instr_count"],
            "max_instr_count": benchmark_summary["max_instr_count"],
            "ci95_instr_count": benchmark_summary["ci95_instr_count"],
            "baseline_instr_count": regression["baseline_instr_count"],
            "delta": regression["delta"],
            "delta_pct": regression["delta_pct"],
            "threshold_pct": regression["threshold_pct"],
            "regressed": regression["regressed"],
            "asm": str(output_file),
        })

    print("\n" + "=" * 50)
    print(f"Total: {len(dsl_files)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if args.benchmark and args.update_baseline:
        save_baseline(results)
        print(f"Baseline written to {BASELINE_FILE}")

    write_report(results, passed, failed)


if __name__ == "__main__":
    main()
