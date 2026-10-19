"""Two-layer official Qwen3: all inputs, checkpoints and execution paths."""
import pytest

from probes.w2_qwen3_small.diagnostics import compare_outputs, unpack_trace
from probes.w2_qwen3_small.run import attention_checks, input_cases, position_comparison

CASE_NAMES = [name for name, _, _ in input_cases()]
INVARIANTS = [("full_seed_0", "changed_future", 64), ("short_17", "changed_padding", 17)]


@pytest.mark.parametrize("name", CASE_NAMES)
def test_observation_preserves_torch_logits(name, qwen3_small_torch, assert_tensor):
    observed, plain = qwen3_small_torch(name)
    assert_tensor(observed["logits"], plain)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_independent_attention(name, qwen3_small_torch, qwen3_small_inputs, qwen3_small_model):
    reference, _ = qwen3_small_torch(name)
    checks = attention_checks(reference, qwen3_small_inputs[name][1], qwen3_small_model.config_metadata)
    assert len(checks) == 6
    for check in checks:
        assert check["passed"], check


@pytest.mark.parametrize("name", CASE_NAMES)
def test_ort_checkpoints(name, qwen3_small_torch, qwen3_small_ort, qwen3_small_artifacts,
                         qwen3_small_inputs, qwen3_small_model, assert_tensor):
    reference, plain = qwen3_small_torch(name)
    actual = qwen3_small_ort(name)
    packed = unpack_trace(actual["diagnostic"], qwen3_small_artifacts["schema"])
    result = compare_outputs(packed, actual["named"], qwen3_small_model.output_names)
    assert result["passed"], result
    result = position_comparison(actual["named"], reference, qwen3_small_model.output_names,
                                 qwen3_small_model.checkpoint_metadata, qwen3_small_inputs[name][0])
    assert result["passed"], result
    assert_tensor(actual["normal"], plain)
    assert_tensor(actual["normal"], actual["named"]["logits"])


@pytest.mark.parametrize("name", CASE_NAMES)
@pytest.mark.parametrize("kind", ["normal", "diagnostic"])
@pytest.mark.parametrize("level", ["none", "basic", "all"])
def test_small_ir(name, kind, level, qwen3_small_result, qwen3_small_ort, qwen3_small_torch,
                  qwen3_small_model, qwen3_small_inputs, assert_tensor):
    output, named, _ = qwen3_small_result(name, kind, "ir", level)
    reference = qwen3_small_ort(name)
    assert_tensor(output, reference[kind])
    assert_tensor(named["logits"], qwen3_small_torch(name)[0]["logits"])
    assert_tensor(named["logits"], reference["normal"])
    if kind == "diagnostic":
        result = position_comparison(named, reference["named"], qwen3_small_model.output_names,
                                     qwen3_small_model.checkpoint_metadata, qwen3_small_inputs[name][0])
        assert result["passed"], result
        assert_tensor(named["logits"], qwen3_small_result(name, "normal", "ir", level)[1]["logits"])


@pytest.mark.parametrize("name", CASE_NAMES)
@pytest.mark.parametrize("kind", ["normal", "diagnostic"])
@pytest.mark.parametrize("level", ["none", "all"])
def test_small_riscv(name, kind, level, qwen3_small_result, qwen3_small_ort, qwen3_small_torch,
                     qwen3_small_model, assert_tensor, record_property):
    output, named, seconds = qwen3_small_result(name, kind, "qemu", level)
    record_property("qemu_process_wall_seconds", seconds)
    reference = qwen3_small_ort(name)
    assert_tensor(output, reference[kind])
    assert_tensor(named["logits"], reference["normal"])
    torch_reference = qwen3_small_torch(name)[0]
    assert_tensor(named["logits"], torch_reference["logits"])
    if kind == "diagnostic":
        for expected in (reference["named"], torch_reference):
            comparison = compare_outputs(named, expected, qwen3_small_model.output_names)
            assert comparison["passed"], comparison
        assert_tensor(named["logits"], qwen3_small_result(name, "normal", "qemu", level)[1]["logits"])


@pytest.mark.parametrize("before,after,length", INVARIANTS, ids=["causal", "padding"])
@pytest.mark.parametrize("engine", ["torch", "ort", "ir", "qemu"])
def test_small_causality_and_padding(before, after, length, engine, qwen3_small_torch,
                                    qwen3_small_ort, qwen3_small_result, assert_tensor):
    if engine == "torch":
        assert_tensor(qwen3_small_torch(before)[0]["logits"][:, :length],
                      qwen3_small_torch(after)[0]["logits"][:, :length])
    elif engine == "ort":
        assert_tensor(qwen3_small_ort(before)["normal"][:, :length],
                      qwen3_small_ort(after)["normal"][:, :length])
    else:
        levels = ("none", "basic", "all") if engine == "ir" else ("none", "all")
        for kind in ("normal", "diagnostic"):
            for level in levels:
                left = qwen3_small_result(before, kind, engine, level)[1]["logits"]
                right = qwen3_small_result(after, kind, engine, level)[1]["logits"]
                assert_tensor(left[:, :length], right[:, :length])
