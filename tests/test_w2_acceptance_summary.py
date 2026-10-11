"""The W2 acceptance summary must answer only operators, precision and timing.

It also has to stay honest when the upstream jobs produced no evidence: a
missing report is never a pass, and the page never claims more coverage than
the run recorded.
"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.summarize_w2_ci import (FAMILY_OPS, MAX_SUMMARY_BYTES, backend_opcodes,
                                     frontend_operators, main, opcode_name, render)

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("w2-backend-ops", "w2-runtime", "w2-runtime-model",
               "qwen3-small", "qwen3-riscv", "qwen3-parse")


def metric(value, name="logits.npy"):
    return {"name": name, "passed": True, "max_abs": value,
            "relative_l2": 2e-7, "cosine_similarity": 0.99999}


def backend_ops_report():
    return {"passed": True, "seconds": 100.0, "cases": [
        {"name": "matmul_matrix", "passed": True, "executions": [
            {"optimization": level, "passed": True, "compile_seconds": 1.0,
             "qemu_process_wall_seconds": 2.0,
             "ir_vs_ort": metric(1e-6), "ir_vs_numpy": metric(2e-6),
             "qemu_vs_ort": metric(3e-6), "qemu_vs_numpy": metric(4e-6)}
            for level in ("none", "all")]},
        {"name": "elementwise_broadcast", "passed": True, "executions": [
            {"optimization": "none", "passed": True, "compile_seconds": 1.0,
             "qemu_process_wall_seconds": 2.0,
             "ir_vs_ort": metric(5e-6), "ir_vs_numpy": metric(6e-6),
             "qemu_vs_ort": metric(7e-6), "qemu_vs_numpy": metric(8e-6)}]},
    ]}


def reports():
    return {
        "w2-backend-ops": backend_ops_report(),
        "w2-runtime": {"gate": "unit:runtime", "passed": True, "seconds": 5.0},
        "w2-runtime-model": {"gate": "integration:runtime-model", "passed": True, "seconds": 20.0,
                             "cases": [{"name": "c0", "executions": [
                                 {"engine": "ir", "passed": True, "vs_ort": metric(1e-6),
                                  "vs_torch": metric(2e-6)}]}]},
        "qwen3-small": {"passed": True, "seconds": 30.0,
                        "onnx": {"operators": {"MatMul": 2, "Add": 3}},
                        # The host IR comparison wraps one tensor_diff per checkpoint.
                        "cases": [{"name": "c0", "ir_vs_ort": {
                            "passed": True, "first_divergence": None,
                            "checkpoints": [dict(metric(1e-6), name="logits"),
                                            dict(metric(9e-6), name="layer_0.output")]}}]},
        "qwen3-riscv": {"passed": True,
                        "timing": {"pipeline_seconds": 60.0, "cross_compile_seconds_total": 10.0,
                                   "qemu_process_wall_seconds_total": 12.5},
                        "cases": [{"name": "c0", "qemu": [metric(1e-6)]}]},
        "qwen3-parse": {"gate": "frontend:parse-qwen3", "passed": True, "seconds": 40.0,
                        "ir": {"operators": {"matmul": 42}},
                        "graph": {"graph": {"operators": {"MatMul": 42, "Gather": 1}}}},
    }


def save(directory, payload):
    for name, report in payload.items():
        folder = directory / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")


def row_of(text, prefix):
    """The pipe-separated cells of the first table row starting with `prefix`."""
    for line in text.splitlines():
        if line.startswith(prefix):
            return [value.strip() for value in line.strip().strip("|").split("|")]
    raise AssertionError(f"no row starting with {prefix!r} in:\n{text}")


def test_three_sections_and_every_chain_are_present(tmp_path):
    save(tmp_path, reports())
    text = render(tmp_path)
    assert text.startswith("## W2 测试项\n")
    for heading in ("### 算子支持", "### 精度", "### 耗时"):
        assert heading in text
    for label in ("w2-backend-ops", "w2-runtime-model", "qwen3-small host IR",
                  "qwen3-small RV64 QEMU", "qwen3-parse"):
        assert any(line.startswith(f"| {label}") for line in text.splitlines()), label
    # Every measured chain contributes its seconds, and the total is their sum.
    assert "255.000 s" in row_of(text, "| **总计**")[2]


def test_exercised_operators_are_marked_and_unmeasured_ones_are_not(tmp_path):
    save(tmp_path, reports())
    text = render(tmp_path)
    frontend = row_of(text, "| IR 层")[2]
    assert "add" in frontend and "matmul" in frontend and "gather" in frontend
    # Supported by the parser, but no W2 report recorded it.
    assert "conv" not in frontend and "maxpool" not in frontend
    backend = row_of(text, "| 汇编层")[2]
    assert "add" in backend and "matmul" in backend
    assert "softmax" not in backend
    kernels = row_of(text, "| `scratchv/backend/kernels`")[2]
    assert kernels == "无（W2 不执行）"


def test_backend_layer_is_not_marked_without_a_passing_qemu_run(tmp_path):
    payload = reports()
    payload["qwen3-riscv"]["cases"] = [{"name": "c0", "qemu": [dict(metric(1e-6), passed=False)]}]
    for execution in payload["w2-backend-ops"]["cases"][0]["executions"]:
        execution["passed"] = False
    payload["w2-backend-ops"]["cases"][1]["executions"] = []
    save(tmp_path, payload)
    assert row_of(render(tmp_path), "| 汇编层")[2] == "无（本次未执行）"


def test_precision_legend_explains_every_name_before_the_table(tmp_path):
    save(tmp_path, reports())
    text = render(tmp_path)
    section = text.split("### 精度", 1)[1]
    for name in ("ORT", "IR", "QEMU", "NumPy", "最终 logits"):
        assert f"**{name}**" in section, name
    # A reader cannot use the table without the legend above it.
    assert section.index("**ORT**") < section.index("| 比较 |")


def test_structural_chains_stay_visible_without_numeric_metrics(tmp_path):
    save(tmp_path, reports())
    text = render(tmp_path)
    assert "w2-runtime（文本层，无数值比较）" in text
    assert "qwen3-parse（完整 28 层前端，无数值比较）" in text
    assert "未测（0 条比较）" in text


def test_precision_table_reports_the_worst_recorded_comparison(tmp_path):
    save(tmp_path, reports())
    text = render(tmp_path)
    qemu = row_of(text, "| w2-backend-ops / QEMU vs ORT")
    # Worst of 3e-6, 4e-6, 7e-6 leaves 8e-6 out: that is the NumPy row.
    assert qemu[1] == format(7e-6, ".16e")
    numpy_row = row_of(text, "| w2-backend-ops / QEMU vs NumPy")
    assert numpy_row[1] == format(8e-6, ".16e")
    # Only the final logits checkpoint counts, not the 9e-6 layer_0.output one.
    small = row_of(text, "| qwen3-small host IR / IR vs ORT")
    assert small[1] == format(1e-6, ".16e")


def test_missing_reports_are_reported_and_never_pass(tmp_path):
    text = render(tmp_path)
    assert text.count("报告缺失") == len(DIRECTORIES)
    assert "PASS" not in text
    assert "不完整" in text and "0.000 s" in text


@pytest.mark.parametrize("invalid", ["{", "[]", '"text"', '{"gate":"unit:runtime"}'])
def test_invalid_reports_do_not_become_a_pass(tmp_path, invalid):
    folder = tmp_path / "w2-runtime"
    folder.mkdir(parents=True)
    (folder / "report.json").write_text(invalid, encoding="utf-8")
    text = render(tmp_path)
    assert "报告无法解析" in text or "报告结构不符" in text
    assert "PASS" not in text


def test_a_failing_chain_is_never_hidden_by_its_report(tmp_path):
    payload = reports()
    payload["qwen3-small"]["passed"] = False
    save(tmp_path, payload)
    row = row_of(render(tmp_path), "| qwen3-small host IR |")
    assert row[1] == "FAIL"


def test_summary_budget_falls_back_without_losing_the_header(tmp_path, monkeypatch):
    import scripts.summarize_w2_ci as module
    save(tmp_path, reports())
    monkeypatch.setattr(module, "MAX_SUMMARY_BYTES", 200)
    text = render(tmp_path)
    assert text.startswith("## W2 测试项\n")
    assert "summary budget" in text
    assert len(text.encode("utf-8")) < 400
    assert MAX_SUMMARY_BYTES == 64 * 1024


def test_cli_appends_to_the_github_summary_and_always_succeeds(tmp_path, monkeypatch):
    destination = tmp_path / "summary.md"
    destination.write_text("Prior summary\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(destination))
    assert main(["--report-dir", str(tmp_path / "absent")]) == 0
    assert destination.read_text().startswith("Prior summary\n## W2 测试项\n")
    # -S excludes site-packages: the acceptance job installs no dependencies.
    environment = {key: value for key, value in os.environ.items() if key != "GITHUB_STEP_SUMMARY"}
    process = subprocess.run([sys.executable, "-S", "-X", "utf8", str(ROOT / "scripts/summarize_w2_ci.py"),
                              "--report-dir", str(tmp_path / "absent")],
                             cwd=tmp_path, env=environment, text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    assert process.stdout.startswith("## W2 测试项\n")


def test_backend_opcodes_are_pinned_to_the_codegen_set():
    """Changing the tensor C lowering set has to update this summary."""
    assert backend_opcodes() == sorted([
        "abs", "add", "cast", "concat", "cos", "div", "exp", "expand", "gather", "load_const",
        "matmul", "mul", "neg", "pow", "reciprocal", "reduce_mean", "relu", "reshape", "sigmoid",
        "sin", "slice", "softmax", "sqrt", "sub", "transpose", "unsqueeze"])


def test_frontend_operators_come_from_the_parser_handlers():
    operators = frontend_operators()
    assert len(operators) == len(set(operators))
    assert {"matmul", "softmax", "reducemean", "unsqueeze"} <= set(operators)
    assert operators == sorted(operators)


def test_family_ops_cover_every_operator_the_cases_build():
    """The probe records only families, so this map has to stay true."""
    tree = ast.parse((ROOT / "probes/w2_backend_ops/cases.py").read_text(encoding="utf-8"))
    built = {node.args[0].value for node in ast.walk(tree)
             if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "make_node"}
    assert built, "no ONNX nodes found in cases.py"
    assert {opcode_name(op) for op in built} == {op for ops in FAMILY_OPS.values() for op in ops}
