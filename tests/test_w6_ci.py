"""W6 gate scope, thresholds and a small portable LM Head fixture."""
import json
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import helper, numpy_helper
import pytest
import yaml

from probes.w6_performance import prepare_lm_head as fixture
from probes.w6_performance import run as measure
from scripts import check_w6_ci as gate


def write(root, filename, value):
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def trials(case, variant, time, **values):
    return [dict(case=case, variant=variant, round=i, phase="measure" if i else "warmup", **values,
                 **{time[0]: time[1]}) for i in range(4)]


def matmul_report():
    rows = []
    for case, variant in gate.MATMUL.items():
        for v, t in (("1x4-original", 3), (variant, 2)):
            rows.extend(trials(case, v, ("forward_s", t), max_abs_vs_ort=0., bitwise_equal=True))
    return dict(commit="abc", cores=4, dtype="float32", workspace_layout="none",
                shapes={"dense": [[256, 1024], [1024, 1024]], "attention_scores": [[16, 256, 128], [16, 128, 256]]},
                runs=rows, required_variants=gate.MATMUL.copy())


def test_measurement_selects_the_same_fixed_scalar_candidate_as_acceptance():
    assert measure.REQUIRED_VARIANTS == gate.MATMUL
    assert gate.MATMUL["attention_scores"] == "4x4-checked-parallel-copy"
    variants = measure.configurations("attention_scores", compare_candidates=True)
    assert ("1x4-original", "1x4", "original") in variants
    assert (gate.MATMUL["attention_scores"], "4x4", "runtime-panel") in variants


@pytest.mark.parametrize("case, variant", [
    ("attention_scores", "4x4-runtime-panel"),
    ("attention_scores", "4x4-checked-publish64"),
    ("attention_chain", "fused"),
])
def test_faster_other_implementations_cannot_replace_selected_attention(tmp_path, case, variant):
    result = matmul_report()
    for row in result["runs"]:
        if row["variant"] == gate.MATMUL["attention_scores"]:
            row.update(case=case, variant=variant, forward_s=0.1)
    write(tmp_path, "matmul/result.json", result)
    report, ok = gate.summarize(tmp_path, "abc", "small")
    assert not ok and "attention_scores | FAIL" in report and "dense | PASS" in report


def test_small_reports_both_shapes_and_keeps_full_checks_unrun(tmp_path):
    result = matmul_report()
    write(tmp_path, "matmul/result.json", result)
    report, ok = gate.summarize(tmp_path, "abc", "small")
    assert ok and report.count("NOT RUN") == 3
    assert "dense | PASS" in report and "attention_scores | PASS" in report
    for r in result["runs"]:
        if r["variant"] == gate.MATMUL["attention_scores"]:
            r["forward_s"] = 2.1
    # A fast obsolete kernel or fused graph must not rescue a sub-1.5x MatMul.
    for case, variant in (("attention_scores", "4x4-runtime-panel"), ("attention_chain", "fused")):
        result["runs"].extend(trials(case, variant, ("forward_s", 0.1),
                                      max_abs_vs_ort=0., bitwise_equal=True))
    write(tmp_path, "matmul/result.json", result)
    report, ok = gate.summarize(tmp_path, "abc", "small")
    assert not ok and "attention_scores | FAIL" in report and "dense | PASS" in report


@pytest.mark.parametrize("bad", ["commit", "shape", "round", "nan", "threshold", "selection", "missing", "full"])
def test_gate_requires_scoped_complete_measurements(tmp_path, bad):
    result = matmul_report()
    if bad == "commit":
        result["commit"] = "older"
    elif bad == "shape":
        result["shapes"]["dense"][0][0] = 1
    elif bad == "round":
        result["runs"][3]["round"] = 2
    elif bad == "nan":
        result["runs"][3]["forward_s"] = float("nan")
    elif bad == "threshold":
        result["runs"][3]["max_abs_vs_ort"] = 1e-3
    elif bad == "selection":
        result["required_variants"]["attention_scores"] = "4x4-runtime-panel"
    if bad != "missing":
        write(tmp_path, "matmul/result.json", result)
    assert not gate.summarize(tmp_path, "abc", "full" if bad == "full" else "small")[1]


def test_real_vocab_and_measured_memory_threshold(tmp_path):
    lm = {}
    for k, vocab in (("small", 257), ("real", gate.VOCAB)):
        runs = []
        for v, elapsed in (("baseline", 4), ("vocab_blocked", 2)):
            runs.extend(trials(k, v, ("guest_s", elapsed), finite=True, greedy_matches=256,
                               greedy_mismatch_rows=[], elements=256*vocab, max_abs_vs_ort=0.0003))
        lm[k] = dict(setup=dict(vocab=vocab, cores=4, dtype="float32", padding="none"), runs=runs)
    write(tmp_path, "lm-head/result.json", dict(commit="abc", cases=lm))
    assert gate.lm_head(tmp_path, "abc")[0]
    lm["real"]["runs"][-1]["greedy_mismatch_rows"] = [3]
    write(tmp_path, "lm-head/result.json", dict(commit="abc", cases=lm))
    with pytest.raises(ValueError, match="numerical"):
        gate.lm_head(tmp_path, "abc")

    expected = [9]*256
    full = dict(git_head="abc", status="PASS", case="full_seed_0", layers=28,
                shape=[1, 256, gate.VOCAB], controls=["lm-head-only"],
                candidate_weight_transport="file-backed-ram", inputs_equal=True, ort_bitwise_equal=True,
                same_weight_manifest=True, same_other_rhs_panels=True, same_candidate_bundle=True,
                same_candidate_output=True, same_candidate_elf=True, ort_greedy_ids=expected, arms={}, numeric={},
                flags=["--case", "full_seed_0", "--cores", "4", "--matmul-policy", "blocked_fma",
                       "--matmul-tile", "2x4", "--matmul-rhs-layout", "column-panel", "--mean-policy",
                       "two-four-lane-last-axis", "--nonlinear-policy", "fp32-reference",
                       "--qemu-workspace-layout", "none"])
    for name in ("original", "lm-head-only", "vocab-blocked"):
        candidate = name == "vocab-blocked"
        full["arms"][name] = dict(combined_peak_bytes=80 if candidate else 100, samples=190, qemu_samples=48,
                                  qemu_s=10, read_errors=2, qemu_read_errors=1, controller_bytes_at_peak=20,
                                  qemu_bytes_at_peak=60 if candidate else 80, qemu_exit_peak_bytes=85,
                                  qemu_sampled_peak_bytes=75, lm_head_policy="original" if name == "original" else "vocab-blocked",
                                  qemu_weight_transport="file-backed-ram" if candidate else "loader", fma_passed=True)
        full["numeric"][name] = dict(finite=True, elements=256*gate.VOCAB, max_abs_vs_ort=0.0003,
                                     token_ids=expected.copy(), greedy_matches=256, greedy_mismatch_rows=[])
    write(tmp_path, "full/comparison.json", full)
    assert gate.memory(tmp_path, "abc")[0] and gate.regression(tmp_path, "abc")[0]
    candidate = full["arms"]["vocab-blocked"]
    for key, value in (("samples", 1), ("qemu_samples", 1), ("read_errors", 100), ("qemu_read_errors", 3),
                       ("controller_bytes_at_peak", 1), ("qemu_exit_peak_bytes", 10)):
        previous, candidate[key] = candidate[key], value
        write(tmp_path, "full/comparison.json", full)
        with pytest.raises(ValueError):
            gate.memory(tmp_path, "abc")
        candidate[key] = previous
    for key in ("same_candidate_output", "same_candidate_elf", "same_candidate_bundle"):
        full[key] = False
        write(tmp_path, "full/comparison.json", full)
        with pytest.raises(ValueError, match="differ"):
            gate.full_result(tmp_path, "abc")
        full[key] = True
    full["flags"][3] = "1"
    write(tmp_path, "full/comparison.json", full)
    with pytest.raises(ValueError, match="settings"):
        gate.full_result(tmp_path, "abc")
    full["flags"][3] = "4"
    candidate["lm_head_policy"] = "original"
    write(tmp_path, "full/comparison.json", full)
    with pytest.raises(ValueError, match="policy"):
        gate.full_result(tmp_path, "abc")
    candidate["lm_head_policy"] = "vocab-blocked"
    candidate["combined_peak_bytes"] = 81
    candidate["controller_bytes_at_peak"] = 21
    full["numeric"]["lm-head-only"]["token_ids"][-1] = 2
    write(tmp_path, "full/comparison.json", full)
    assert not gate.memory(tmp_path, "abc")[0]
    with pytest.raises(ValueError, match="token mismatch"):
        gate.regression(tmp_path, "abc")


def test_prepare_lm_head_uses_original_float32_weights_and_ort(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "acquire_model", lambda path: {"reused": True})
    for attr, size in (("VOCAB", 5), ("HIDDEN", 4), ("ROWS", 2)):
        monkeypatch.setattr(fixture, attr, size)
    weight = np.arange(20, dtype=np.float32).reshape(5, 4)
    stored = numpy_helper.from_array(weight, name="model.lm_head.weight")
    tensor = helper.make_tensor_value_info
    graph = helper.make_graph([helper.make_node("MatMul", ["x", stored.name], ["y"])],
                              "small", [tensor("x", 1, [1, 5])], [tensor("y", 1, [1, 4])], [stored])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8),
              tmp_path / "model.onnx")
    out = tmp_path / "head"
    fixture.prepare(tmp_path, out)
    np.testing.assert_array_equal(np.fromfile(out / "weight.bin", dtype="<f4"), weight.ravel())
    np.testing.assert_allclose(np.load(out / "ort.npy"), np.load(out / "input_x.npy") @ weight.T,
                               rtol=1e-6, atol=1e-6)
    with pytest.raises(FileExistsError):
        fixture.prepare(tmp_path, out)


def test_w6_workflow_only_runs_on_request_or_opted_in_schedule():
    workflow = Path(__file__).resolve().parents[1] / ".github/workflows/w6-performance.yml"
    body = yaml.load(workflow.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert set(body["on"]) == {"schedule", "workflow_dispatch"}
    assert body["on"]["workflow_dispatch"]["inputs"]["mode"]["default"] == "small"
    job = body["jobs"]["measure"]
    assert "vars.W6_NIGHTLY_ENABLED == 'true'" in job["if"]
    for step in job["steps"]:
        shell = step.get("run", "")
        if "prepare_lm_head" in shell or "full_forward" in shell or "w6_performance.lm_head " in shell:
            assert step["if"] == "env.W6_MODE == 'full'"
        if "check_w6_ci.py" in shell:
            assert step["if"] == "always()" and step["shell"] == "bash"
        if shell:
            subprocess.run(["bash", "-n"], input=shell, text=True, check=True)
