"""Report summaries must preserve scope, missing evidence and visible failures."""
from copy import deepcopy

import pytest

from probes.w2_qwen3_small import run, riscv


def metric(error=1e-6, relative=2e-7, cosine=0.999999999):
    return {"passed": True, "max_abs": error, "relative_l2": relative,
            "cosine_similarity": cosine}


def host_report():
    return {"passed": True, "stage": "complete", "cases": [
        {"name": f"input_{index}", "valid_length": 256, "passed": True,
         "ordinary_logits": {"pytorch_vs_ort": metric(), "ir_vs_ort": metric(3e-6),
                             "ir_diagnostic_vs_ordinary": metric(0)},
         "ir_vs_ort": {"passed": True, "first_divergence": None,
                       "checkpoints": [{"name": "sentinel_checkpoint", **metric(123.456)}]}}
        for index in range(7)], "invariants": []}


def qemu_report():
    report = {"passed": True, "stage": "complete", "seconds": 45, "cases": [],
              "builds": [{"compile_seconds": 1}] * 4, "invariants": []}
    for index in range(7):
        case = {"name": f"input_{index}", "passed": True, "qemu": [], "optimized_ir": []}
        for graph in ("normal", "diagnostic"):
            for level in ("none", "basic", "all"):
                case["optimized_ir"].append({"graph": graph, "optimization": level,
                                              **metric(2e-6 if graph == "normal" else 123.456)})
            for level in ("none", "all"):
                case["qemu"].append({"graph": graph, "optimization": level, "status": "success",
                                     "qemu_process_wall_seconds": 1, "timeout_seconds": 180,
                                     **metric(3e-6 if graph == "normal" else 123.456)})
        report["cases"].append(case)
    report["timing"] = riscv.summarize_timing(report)
    return report


@pytest.mark.parametrize("view", [run._report_markdown, run._report_html])
def test_host_front_page_uses_ordinary_logits_not_diagnostic_maximum(view):
    rendered = view(host_report())
    front = rendered.split("<details", 1)[0]
    assert "3.0000000000000001e-06" in front
    assert "相对 L2" in front and "余弦相似度" in front
    assert "IR 原生 NumPy / ORT" in front
    assert "123.456" not in front and "sentinel_checkpoint" not in front
    assert "sentinel_checkpoint" in rendered
    assert "同一模型、同一权重和输入" in front
    assert "PPL、zero-shot 未启用" in front


@pytest.mark.parametrize("view", [run._report_markdown, run._report_html])
def test_old_host_metric_fields_are_explicitly_missing(view):
    report = host_report()
    for case in report["cases"]:
        for row in case["ordinary_logits"].values():
            row.pop("relative_l2")
            row.pop("cosine_similarity")
    front = view(report).split("<details", 1)[0]
    assert "未记录 7/7" in front and "指标不完整" in front
    assert "相对 L2" in front


@pytest.mark.parametrize("view", [run._report_markdown, run._report_html])
def test_host_diagnostic_failure_stays_visible_when_logits_pass(view):
    report = host_report()
    report["passed"] = report["cases"][0]["passed"] = False
    report["cases"][0]["ir_vs_ort"].update(passed=False, first_divergence="layer_1.rmsnorm")
    front = view(report).split("<details", 1)[0]
    assert "FAIL" in front and "layer_1.rmsnorm" in front
    assert "失败／未完成项" in front


@pytest.mark.parametrize("filename", ["report.md", "report.html"])
def test_qemu_front_page_uses_only_normal_logits_and_documents_execution(filename):
    rendered = riscv._report_views(qemu_report())[filename]
    front = rendered.split("<details", 1)[0]
    assert "生成 C → RV64/QEMU / ORT" in front
    assert "3.0000000000000001e-06" in front and "123.456" not in front
    assert "14/14" in front and "21/21" in front
    assert "相对 L2" in front and "余弦相似度" in front
    assert "不是单独的主机 C 测试" in front
    assert "不代表纯前向或目标硬件性能" in front
    assert "normal/none" not in front and "normal/none" in rendered
    assert "本探测的 C 编译固定使用 -O2" in rendered
    assert "diagnostic MaxAbs 包含全部打包检查点" in rendered


@pytest.mark.parametrize("filename", ["report.md", "report.html"])
def test_missing_normal_level_cannot_be_filled_by_diagnostic_result(filename):
    report = qemu_report()
    report["cases"][0]["qemu"] = [row for row in report["cases"][0]["qemu"]
                                  if (row["graph"], row["optimization"]) != ("normal", "none")]
    report["passed"] = report["cases"][0]["passed"] = False
    report["timing"] = riscv.summarize_timing(report)
    front = riscv._report_views(report)[filename].split("<details", 1)[0]
    assert "未记录 1/14" in front and "未确认 1" in front
    assert "部分结果" in front and "FAIL" in front
    assert "123.456" not in front


@pytest.mark.parametrize("filename", ["report.md", "report.html"])
def test_qemu_error_is_visible_and_escaped(filename):
    report = qemu_report()
    report["passed"] = False
    report["error"] = "execution failed <script>alert('x')</script>"
    front = riscv._report_views(report)[filename].split("<details", 1)[0]
    assert "execution failed" in front and "<script>" not in front
    assert "&lt;script&gt;" in front


def test_report_rendering_does_not_mutate_numeric_evidence():
    for report, render in ((host_report(), run._host_report_views), (qemu_report(), riscv._report_views)):
        before = deepcopy(report)
        render(report)
        assert report == before
