"""CNN benchmark evidence must detect side effects, including same-size changes."""

import json

import pytest

from benchmarks import bench_cnn_schedule as cnn
from scratchv.backend.inst_scheduler import schedule_assembly


def audit(source, candidate):
    return cnn.audit_side_effects(source, candidate, schedule_assembly(source).stats["regions"])


def test_resource_counts_include_opaque_ops_aliases_and_both_stack_syntaxes():
    source = "max t0, a0, zero\nsw sp(4), t0\nlw t1, 4(x2)\naddi sp, sp, -16\nret\n"
    stats = cnn.assembly_resources(source)
    assert stats["physical_registers"] == ["x1", "x10", "x2", "x5", "x6"]
    assert stats["stack_loads"] == stats["stack_stores"] == 1
    assert stats["stack_offsets"] == [4]
    assert stats["stack_pointer_operations"] == [("addi", ("sp", "sp", "-16"))]


def test_unknown_stack_address_is_an_error_not_zero_spills():
    with pytest.raises(ValueError, match="stack address"):
        cnn.assembly_resources("sw t0, symbol(sp)\n")


def test_same_instruction_and_register_counts_do_not_prove_safety():
    source = "mul t0, a0, a1\nadd t1, t0, a2\n"
    after = "add t1, t0, a2\nmul t0, a0, a1\n"
    report = audit(source, after)
    assert report["checks"]["instruction_multiset_equal"]
    assert report["checks"]["physical_registers_equal"]
    assert not report["checks"]["region_dataflow_equal"]
    assert report["status"] == "failed"


def test_memory_reordering_is_rejected_even_with_equal_stack_counts():
    source = "sw t0, 0(sp)\nsw t1, 4(sp)\n"
    after = "sw t1, 4(sp)\nsw t0, 0(sp)\n"
    report = audit(source, after)
    assert report["before"]["stack_stores"] == report["after"]["stack_stores"] == 2
    assert not report["checks"]["stack_accesses_equal"]
    assert not report["checks"]["region_dataflow_equal"]


def test_legal_reorder_can_increase_liveness_without_adding_spills():
    source = "add t0, a0, a1\nli t1, 1\n"
    after = "li t1, 1\nadd t0, a0, a1\n"
    report = audit(source, after)
    assert report["status"] == "passed"
    assert report["checks"]["physical_registers_equal"]
    assert report["before"]["stack_stores"] == report["after"]["stack_stores"] == 0
    live = report["local_liveness"]
    assert (live["peak_before"], live["peak_after"], live["increased_regions"]) == (2, 3, 1)


@pytest.mark.parametrize("after", [
    "max t1, a0, zero\n",  # Same number of physical registers, different set.
    "max t0, a0, zero\nsw t0, 0(sp)\n",  # A new spill must be detected.
])
def test_unmodeled_code_and_new_spills_cannot_escape_audit(after):
    report = audit("max t0, a0, zero\n", after)
    assert report["status"] == "failed"
    assert not report["checks"]["instruction_multiset_equal"]
    assert not report["checks"]["fixed_lines_equal"]


def test_real_schedule_preserves_layout_and_dataflow(tmp_path):
    source = ".text\nf:\nmul t0, a0, a1\nadd t1, t0, a2\nsw t1, 4(a0)\naddi t2, t2, 1\n"
    path = tmp_path / "input.s"
    path.write_text(source)
    report = cnn.analyze_assembly(path, "test")
    assert report["comparison_status"] == "changed"
    assert report["metric"]["saved"] > 0
    assert report["side_effects"]["status"] == "passed"
    assert report["side_effects"]["verified_regions"] > 0


def test_zero_coverage_uses_na_and_exact_identity(tmp_path):
    path = tmp_path / "numeric.s"
    path.write_text("addi t0, t0, 1\nbne t0, zero, -4\n")
    report = cnn.analyze_assembly(path, "numeric")
    assert report["status"] == "passed"
    assert report["comparison_status"] == "not_modeled"
    assert report["metric"]["before"] is None
    assert report["metric"]["reduction_percent"] is None
    assert report["side_effects"]["local_liveness"]["peak_before"] is None
    assert report["assembly"]["before"] == report["assembly"]["after"]


def test_cli_persists_failure_without_substituting_model(tmp_path):
    output = tmp_path / "report.json"
    rendered = tmp_path / "report.md"
    assert cnn.main(["--model", str(tmp_path / "missing.onnx"),
                     "--json", str(output), "--markdown", str(rendered)]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert "missing.onnx" in report["error"]


def test_tracked_cnn_all_paths_and_existing_standalone_identity(tmp_path):
    report = cnn.run_benchmark()
    assert report["status"] == "passed"
    assert report["input"]["sha256"] == cnn.sha256(cnn.DEFAULT_MODEL.read_bytes())
    standalone, greedy, linear = report["cases"]
    for case in (greedy, linear):
        saved = case["metric"]["saved"]
        assert saved >= 0
        assert (case["assembly"]["before"] == case["assembly"]["after"]) == (saved == 0)
        assert all(case["side_effects"]["checks"].values())
        assert case["simulation"]["output_equal"] is None
    assert standalone["comparison_status"] == "changed"
    assert report["schema_version"] == 4
    assert report["model"]["name"] == "llvm-mca/sifive-e76"
    from scratchv.backend.llvm_mca import model_metadata
    assert report["model"]["llvm_version"] == model_metadata()["llvm_version"]
    assert standalone["scheduling"]["metrics_after"]["peak_parallelism"] == 2
    assert standalone["scheduling"]["metrics_after"]["critical_path_max"] is None
    assert standalone["side_effects"]["checks"]["fixed_lines_equal"]
    assert "N/A" in cnn.markdown(report)
    # A supplied CI assembly must belong to this model, not another CNN.
    mismatch = tmp_path / "wrong.s"
    mismatch.write_text("addi t0, t0, 1\n")
    with pytest.raises(ValueError, match="does not match"):
        cnn.run_benchmark(standalone_asm=mismatch)


@pytest.mark.parametrize("execution_status", ["baseline_failed", "failed"])
def test_full_execution_failure_fails_the_benchmark(monkeypatch, execution_status):
    monkeypatch.setattr("benchmarks.cnn_schedule_execution.execute_pair",
                        lambda *args: {"status": execution_status})
    report = cnn.run_benchmark(execute=True)
    assert report["status"] == "failed"
    assert report["cases"][0]["acceptance"]["execution_equal"] is False
    assert report["whole_cnn_execution"] == execution_status
