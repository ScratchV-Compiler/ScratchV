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
                                     frontend_operators, main, opcode_name, precision_comparisons, render)

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("w2-backend-ops", "w2-runtime", "w2-runtime-model",
               "qwen3-small", "qwen3-riscv", "qwen3-parse")


def metric(value, name="logits.npy"):
    return {"name": name, "passed": True, "max_abs": value,
            "relative_l2": 2e-7, "cosine_similarity": 0.99999}


def backend_ops_report():
    return {"passed": True, "seconds": 100.0, "levels": ["none", "all"],
            "planned_executions": 4, "cases": [
        {"name": "matmul_matrix", "family": "matmul", "passed": True, "stage": "complete",
         "frontend_parse": {"passed": True, "operators": {"MatMul": 1}},
         "ort_vs_numpy": metric(1e-6), "executions": [
            {"optimization": level, "passed": True, "stage": "complete", "compile_seconds": 1.0,
             "qemu_process_wall_seconds": 2.0,
             "ir_vs_ort": metric(1e-6), "ir_vs_numpy": metric(2e-6),
             "qemu_vs_ort": metric(3e-6), "qemu_vs_numpy": metric(4e-6)}
            for level in ("none", "all")]},
        {"name": "elementwise_broadcast", "family": "elementwise", "passed": True, "stage": "complete",
         "frontend_parse": {"passed": True, "operators": {"Add": 1, "Sub": 1, "Mul": 1, "Div": 1}},
         "ort_vs_numpy": metric(1e-6), "executions": [
            {"optimization": level, "passed": True, "stage": "complete", "compile_seconds": 1.0,
             "qemu_process_wall_seconds": 2.0,
             "ir_vs_ort": metric(5e-6), "ir_vs_numpy": metric(6e-6),
             "qemu_vs_ort": metric(7e-6), "qemu_vs_numpy": metric(8e-6)}
            for level in ("none", "all")]},
    ]}


def reports():
    return {
        "w2-backend-ops": backend_ops_report(),
        "w2-runtime": {"gate": "unit:runtime", "passed": True, "seconds": 5.0},
        "w2-runtime-model": {"gate": "integration:runtime-model", "passed": True, "seconds": 20.0,
                             "planned_cases": 2, "levels": ["none", "all"],
                             "cases": [{"name": name, "executions": [
                                 {"engine": "ort", "passed": True, "vs_torch": metric(1e-6)},
                                 {"engine": "ir-none", "passed": True, "vs_ort": metric(1e-6),
                                  "vs_torch": metric(2e-6)},
                                 {"engine": "ir-all", "passed": True, "vs_ort": metric(1e-6),
                                  "vs_torch": metric(2e-6)}]}
                                 for name in ("chinese", "special_pad_in_prompt")]},
        "qwen3-small": {"passed": True, "stage": "complete", "seconds": 30.0,
                        "planned_cases": 1,
                        "frontend_parse": {"passed": True, "operators": {"MatMul": 2, "Add": 3}},
                        "onnx": {"operators": {"MatMul": 2, "Add": 3}},
                        # Ordinary and diagnostic forwards are separate executions.
                        "cases": [{"name": "c0", "ordinary_logits": {"ir_vs_ort": metric(1e-6)},
                                   "ir_vs_ort": {"passed": True, "first_divergence": None,
                                                 "checkpoints": [dict(metric(2e-6), name="logits"),
                                                                 dict(metric(9e-6), name="layer_0.output")]}}]},
        "qwen3-riscv": {"passed": True,
                        "timing": {"pipeline_seconds": 60.0, "cross_compile_seconds_total": 10.0,
                                   "qemu_process_wall_seconds_total": 12.5},
                        "cases": [{"name": "c0", "qemu": [
                            dict(metric(1e-6 if graph == "normal" else 8e-6), graph=graph, optimization=level)
                            for graph in ("normal", "diagnostic") for level in ("none", "all")]}]},
        "qwen3-parse": {"gate": "frontend:parse-qwen3", "passed": True, "stage": "complete", "seconds": 40.0,
                        "frontend_parse": {"passed": True, "operators": {"MatMul": 42, "Gather": 1}},
                        "ir": {"functions": 1, "operators": {"matmul": 42, "gather": 1}},
                        "ir_verifier": {"passed": True},
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
    # W2 never executes the hand-written assembly kernels, so they get no row.
    assert "scratchv/backend/kernels" not in text


def test_backend_layer_is_not_marked_without_a_passing_qemu_run(tmp_path):
    payload = reports()
    payload["qwen3-riscv"]["cases"] = [{"name": "c0", "qemu": [dict(metric(1e-6), passed=False)]}]
    for execution in payload["w2-backend-ops"]["cases"][0]["executions"]:
        execution["passed"] = False
    payload["w2-backend-ops"]["cases"][1]["executions"] = []
    save(tmp_path, payload)
    assert row_of(render(tmp_path), "| 汇编层")[2] == "无（没有通过记录）"


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
    # Worst QEMU/ORT error is 7e-6; 8e-6 belongs only to the NumPy reference row.
    assert qemu[1] == format(7e-6, ".16e")
    numpy_row = row_of(text, "| w2-backend-ops / QEMU vs NumPy")
    assert numpy_row[1] == format(8e-6, ".16e")
    # Use ordinary logits (1e-6), not diagnostic logits (2e-6) or an intermediate (9e-6).
    small = row_of(text, "| 两层模型：IR 解释器与 ORT（最终输出）")
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
    destination.write_text("Prior summary\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(destination))
    assert main(["--report-dir", str(tmp_path / "absent")]) == 0
    assert destination.read_text(encoding="utf-8").startswith("Prior summary\n## W2 测试项\n")
    # -S excludes site-packages: the acceptance job installs no dependencies.
    environment = {key: value for key, value in os.environ.items() if key != "GITHUB_STEP_SUMMARY"}
    process = subprocess.run([sys.executable, "-S", "-X", "utf8", str(ROOT / "scripts/summarize_w2_ci.py"),
                              "--report-dir", str(tmp_path / "absent")],
                             cwd=tmp_path, env=environment, text=True, encoding="utf-8", capture_output=True)
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


@pytest.mark.parametrize("missing", [False, True])
def test_ordinary_logits_failure_or_absence_is_not_hidden_by_diagnostic_pass(tmp_path, missing):
    payload = reports()
    small = payload["qwen3-small"]
    small["passed"] = False
    case = small["cases"][0]
    case["ir_vs_ort"]["checkpoints"][0] = dict(metric(0), name="logits")
    if missing:
        case.pop("ordinary_logits")
    else:
        case["ordinary_logits"]["ir_vs_ort"] = dict(metric(1e-3), passed=False)
    save(tmp_path, payload)
    row = row_of(render(tmp_path), "| 两层模型：IR 解释器与 ORT（最终输出）")
    assert ("未完成" if missing else "未通过") in row[-1]
    if missing:
        assert "未记录" in row[1]
        assert row[1] != format(0, ".16e")
    else:
        assert row[1] == format(1e-3, ".16e")
        assert "失败 1" in row[-1]


def test_compile_failure_without_qemu_output_keeps_its_comparison_slot(tmp_path):
    payload = {"w2-backend-ops": {"passed": False, "levels": ["none", "all"],
        "planned_executions": 2, "cases": [{"name": "matmul_matrix", "executions": [
            {"optimization": "none", "passed": True, "qemu_vs_ort": metric(1e-6)},
            {"optimization": "all", "passed": False, "stage": "compile", "error": "compiler failed"}]}]}}
    selected = dict(precision_comparisons(payload))["w2-backend-ops / QEMU vs ORT"]
    assert len(selected) == 2
    assert selected[1].get("missing_reason")
    save(tmp_path, payload)
    row = row_of(render(tmp_path), "| w2-backend-ops / QEMU vs ORT")
    assert "1/2" in row[-1] and "未完成" in row[-1]
    assert "未记录 1/2" in row[1]


@pytest.mark.parametrize("recorded,planned", [(0, 2), (1, 2), (1, 4)])
def test_absent_executions_and_remaining_plan_are_not_counted_as_pass(tmp_path, recorded, planned):
    executions = ([{"optimization": "none", "passed": True, "qemu_vs_ort": metric(1e-6)}]
                  if recorded else [])
    report = {"passed": False, "levels": ["none", "all"], "planned_executions": planned,
              "cases": [{"name": "matmul_matrix", "executions": executions}]}
    payload = {"w2-backend-ops": report}
    selected = dict(precision_comparisons(payload))["w2-backend-ops / QEMU vs ORT"]
    assert len(selected) == planned
    assert sum(bool(row.get("missing_reason")) for row in selected) == planned - recorded
    save(tmp_path, payload)
    row = row_of(render(tmp_path), "| w2-backend-ops / QEMU vs ORT")
    assert "未完成" in row[-1] and f"{recorded}/{planned}" in row[-1]
    assert f"未记录 {planned - recorded}/{planned}" in row[1]


@pytest.mark.parametrize("chain,stage", [("qwen3-small", "parse"), ("qwen3-parse", "scratchv-parse")])
def test_failed_frontend_parse_does_not_claim_onnx_inventory_as_measured(tmp_path, chain, stage):
    report = reports()[chain]
    report.update(passed=False, stage=stage)
    report.pop("frontend_parse")
    report.pop("ir", None)
    report.pop("ir_verifier", None)
    report.pop("cases", None)
    save(tmp_path, {chain: report})
    assert row_of(render(tmp_path), "| IR 层")[2] == "无（没有通过记录）"


@pytest.mark.parametrize("explicit", [False, True])
def test_backend_only_successful_parse_counts_even_when_later_compile_fails(tmp_path, explicit):
    case = {"name": "rmsnorm_hidden_eps_1e-06", "family": "rmsnorm", "passed": False,
            "stage": "complete", "ort_vs_numpy": metric(0),
            "executions": [{"optimization": "none", "passed": False,
                            "stage": "compile", "error": "compiler failed"}]}
    if explicit:
        case["frontend_parse"] = {"passed": True, "operators": {
            "Pow": 1, "ReduceMean": 1, "Add": 1, "Sqrt": 1, "Div": 1, "Mul": 1}}
    save(tmp_path, {"w2-backend-ops": {"passed": False, "cases": [case]}})
    text = render(tmp_path)
    frontend = row_of(text, "| IR 层")[2]
    assert "reducemean" in frontend and "sqrt" in frontend and "pow" in frontend
    assert row_of(text, "| 汇编层")[2] == "无（没有通过记录）"


@pytest.mark.parametrize("explicit", [False, True])
def test_small_parse_success_is_preserved_after_numeric_runtime_failure(tmp_path, explicit):
    report = {"passed": False, "stage": "numeric", "onnx": {"operators": {"MatMul": 2}},
              "cases": [{"name": "first", "passed": False}], "error": "interpreter failed"}
    if explicit:
        report["frontend_parse"] = {"passed": True, "operators": {"MatMul": 2}}
    save(tmp_path, {"qwen3-small": report})
    frontend = row_of(render(tmp_path), "| IR 层")[2]
    assert "matmul" in frontend


def test_explicit_failed_parse_does_not_fall_back_to_a_later_stage(tmp_path):
    report = {"passed": False, "stage": "complete", "onnx": {"operators": {"MatMul": 2}},
              "frontend_parse": {"passed": False, "operators": {"MatMul": 2}}}
    save(tmp_path, {"qwen3-small": report})
    assert row_of(render(tmp_path), "| IR 层")[2] == "无（没有通过记录）"


def test_runtime_model_expected_engines_keep_missing_results_without_ort_self_comparison(tmp_path):
    report = reports()["w2-runtime-model"]
    baseline = dict(precision_comparisons({"w2-runtime-model": report}))
    assert len(baseline["w2-runtime-model / vs torch"]) == 6
    assert len(baseline["w2-runtime-model / vs ORT"]) == 4
    report["passed"] = False
    report["cases"][1]["executions"] = []
    selected = dict(precision_comparisons({"w2-runtime-model": report}))
    assert len(selected["w2-runtime-model / vs torch"]) == 6
    assert len(selected["w2-runtime-model / vs ORT"]) == 4
    assert sum(bool(row.get("missing_reason")) for row in selected["w2-runtime-model / vs ORT"]) == 2
    save(tmp_path, {"w2-runtime-model": report})
    row = row_of(render(tmp_path), "| w2-runtime-model / vs ORT")
    assert "未记录 2/4" in row[1] and "未完成" in row[-1]


def test_qemu_intentionally_retains_normal_and_diagnostic_statistics(tmp_path):
    # This PR explicitly keeps the combined scope; changing it is not part of these fixes.
    report = reports()["qwen3-riscv"]
    for row in report["cases"][0]["qemu"]:
        if row["graph"] == "diagnostic":
            row.update(relative_l2=8e-5, cosine_similarity=0.75)
    payload = {"qwen3-riscv": report}
    selected = dict(precision_comparisons(payload))["qwen3-small RV64 QEMU / QEMU vs ORT"]
    assert len(selected) == 4
    assert {row["graph"] for row in selected} == {"normal", "diagnostic"}
    save(tmp_path, payload)
    row = row_of(render(tmp_path), "| qwen3-small RV64 QEMU / QEMU vs ORT")
    assert row[1] == format(8e-6, ".16e")
    assert row[2] == format(8e-5, ".16e")
    assert row[3] == format(0.75, ".17g")

@pytest.mark.parametrize("legacy", [False, True])
def test_small_missing_cases_remain_in_expected_count(tmp_path, legacy):
    report = reports()["qwen3-small"]
    report["passed"] = False
    if legacy:
        report.pop("planned_cases")
        expected = 7
    else:
        report["planned_cases"] = 2
        expected = 2
    payload = {"qwen3-small": report}
    selected = dict(precision_comparisons(payload))["两层模型：IR 解释器与 ORT（最终输出）"]
    assert len(selected) == expected
    assert sum(bool(row.get("missing_reason")) for row in selected) == expected - 1
    save(tmp_path, payload)
    row = row_of(render(tmp_path), "| 两层模型：IR 解释器与 ORT（最终输出）")
    assert "未完成" in row[-1] and f"1/{expected}" in row[-1]


def test_runtime_model_missing_case_keeps_both_ir_engines(tmp_path):
    report = reports()["w2-runtime-model"]
    report["passed"] = False
    report["cases"].pop()
    payload = {"w2-runtime-model": report}
    selected = dict(precision_comparisons(payload))
    assert len(selected["w2-runtime-model / vs torch"]) == 6
    ort = selected["w2-runtime-model / vs ORT"]
    assert len(ort) == 4
    assert sum(bool(row.get("missing_reason")) for row in ort) == 2
    save(tmp_path, payload)
    row = row_of(render(tmp_path), "| w2-runtime-model / vs ORT")
    assert "未完成" in row[-1] and "2/4" in row[-1]


def test_qemu_plan_retains_both_graphs_and_unexecuted_comparisons(tmp_path):
    report = reports()["qwen3-riscv"]
    report["passed"] = False
    report["timing"]["planned_count"] = 6
    payload = {"qwen3-riscv": report}
    selected = dict(precision_comparisons(payload))["qwen3-small RV64 QEMU / QEMU vs ORT"]
    assert len(selected) == 6
    assert sum(bool(row.get("missing_reason")) for row in selected) == 2
    assert {row.get("graph") for row in selected if not row.get("missing_reason")} == {"normal", "diagnostic"}
    save(tmp_path, payload)
    row = row_of(render(tmp_path), "| qwen3-small RV64 QEMU / QEMU vs ORT")
    assert row[1].startswith(format(8e-6, ".16e"))
    assert "未完成" in row[-1] and "4/6" in row[-1]


@pytest.mark.parametrize("chain,key", [("w2-backend-ops", "planned_executions"),
                                       ("qwen3-small", "planned_cases"),
                                       ("w2-runtime-model", "planned_cases"),
                                       ("qwen3-riscv", "planned_count")])
@pytest.mark.parametrize("invalid", [True, -1, "2", 10**9])
def test_invalid_or_unbounded_planned_count_is_reported(tmp_path, chain, key, invalid):
    report = reports()[chain]
    target = report["timing"] if chain == "qwen3-riscv" else report
    target[key] = invalid
    save(tmp_path, {chain: report})
    text = render(tmp_path)
    assert "报告结构不符" in text
    assert "PASS" not in text


@pytest.mark.parametrize("explicit", [False, True])
def test_backend_onnx_inventory_without_successful_parse_is_not_measured(tmp_path, explicit):
    case = {"name": "rmsnorm_hidden_eps_1e-06", "family": "rmsnorm", "passed": False,
            "stage": "reference", "error": "parse failed", "executions": []}
    if explicit:
        # Explicit failed parsing cannot be overridden by stale legacy evidence.
        case["frontend_parse"] = {"passed": False, "operators": {"Pow": 1, "ReduceMean": 1}}
        case["ort_vs_numpy"] = metric(0)
    save(tmp_path, {"w2-backend-ops": {"passed": False, "cases": [case]}})
    assert row_of(render(tmp_path), "| IR 层")[2] == "无（没有通过记录）"
