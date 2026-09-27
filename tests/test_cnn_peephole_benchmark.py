"""Contract tests for the real standalone CNN peephole benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.compare_peephole_cnn import compare_cnn_model, generate_cnn_html, main

MODEL = Path(__file__).resolve().parents[1] / "models" / "graph" / "cnn.onnx"


def test_cnn_ab_report_uses_same_input_and_compiles_real_machine_code():
    report = compare_cnn_model(MODEL)

    assert report["model"]["name"] == "cnn.onnx"
    assert report["compiler_parameters"]["baseline"] == {
        **report["compiler_parameters"]["optimized"],
        "peephole_enabled": False,
    }
    assert report["compiler_parameters"]["optimized"]["peephole_enabled"] is True
    assert (
        report["input_assembly_sha256"]["baseline"]
        == report["input_assembly_sha256"]["optimized"]
    )
    assert report["machine_code"]["baseline"]["success"] is True
    assert report["machine_code"]["optimized"]["success"] is True
    assert report["relocation_validation"]["baseline"] is True
    assert report["relocation_validation"]["optimized"] is True
    assert report["gp_patch_validation"]["baseline"] is True
    assert report["gp_patch_validation"]["optimized"] is True
    assert report["instructions"]["baseline"] > 0
    assert report["instructions"]["optimized"] > 0
    assert report["binary_sha256"]["baseline"] != report["binary_sha256"]["optimized"]


def test_cnn_html_reuses_report_contract_without_commit_or_rule_expectations():
    report = compare_cnn_model(MODEL)
    html = generate_cnn_html(report)

    assert "cnn.onnx" in html
    assert "Commit:" not in html
    assert "类别" not in html
    assert "输入摘要" not in html
    assert "预期规则" not in html
    assert "关闭" in html and "开启" in html and "节省" in html
    assert "代码大小" in html
    assert "A/B 编译参数" in html


def test_cnn_json_round_trip_keeps_machine_metrics():
    report = compare_cnn_model(MODEL)
    encoded = json.dumps(report, ensure_ascii=False)
    loaded = json.loads(encoded)

    assert loaded["instructions"]["baseline"] == report["instructions"]["baseline"]
    assert loaded["code_size"]["optimized"] == report["code_size"]["optimized"]
    assert "total_matches" in loaded["peephole"]
    assert "fixed_point_iterations" in loaded["peephole"]


def test_cnn_cli_writes_json_without_standalone_html(tmp_path):
    json_path = tmp_path / "cnn.json"

    assert main(["--model", str(MODEL), "--json", str(json_path)]) == 0

    assert json_path.is_file()
    assert (
        json.loads(json_path.read_text(encoding="utf-8"))["model"]["name"] == "cnn.onnx"
    )
