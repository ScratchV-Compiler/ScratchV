"""Probe-contract tests. Tiny explicit fixtures here are not pretrained evidence."""
from __future__ import annotations

import hashlib
import json

import numpy as np
from onnx import numpy_helper
import onnxruntime as ort
import pytest
import torch
from safetensors.torch import save_file

from probes.w3_qwen3_subgraphs import assets, cases, run


def test_strict_comparison_rejects_threshold_shape_dtype_and_nonfinite():
    zero = np.zeros((2,), np.float32)
    edge = np.nextafter(np.float32(run.ATOL), np.float32(np.inf))
    result = run.tensor_diff(np.array([0, edge], np.float32), zero)
    assert not result["passed"]
    assert result["firstdiff"]["index"] == [1]
    assert run.tensor_diff(zero, zero)["passed"]
    assert not run.tensor_diff(zero.reshape(1, 2), zero)["shape_matches"]
    assert not run.tensor_diff(zero.astype(np.float64), zero)["passed"]
    nonfinite = run.compare({"y": np.array([0, np.nan], np.float32)}, {"y": zero})
    assert not nonfinite["passed"] and nonfinite["max_abs_error"] is None
    assert nonfinite["first_divergence"]["firstdiff"]["actual"] == "nan"
    json.dumps(nonfinite, allow_nan=False)


def test_source_authentication_rejects_tampering_before_tensor_access(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"explicit authentication unit fixture, not a model")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(assets.DIMENSIONS), encoding="utf-8")
    manifest = {"source_checkpoint_sha256": assets.sha256(checkpoint),
                "source_files": [{"name": "config.json", "sha256": assets.sha256(config)}],
                "model_id": "unit-fixture", "revision": "unit-fixture"}
    evidence = assets.verify_source(tmp_path, manifest)
    assert evidence["revision"] == "unit-fixture"
    checkpoint.write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA256 mismatch: model.safetensors"):
        assets.verify_source(tmp_path, manifest)
    checkpoint.write_bytes(b"explicit authentication unit fixture, not a model")
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch: config.json"):
        assets.verify_source(tmp_path, manifest)


def test_source_authentication_rejects_path_escape(tmp_path):
    manifest = {"source_checkpoint_sha256": hashlib.sha256(b"fixture").hexdigest(),
                "source_files": [{"name": "../config.json", "sha256": "x"}]}
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="Non-flat"):
        assets.verify_source(tmp_path, manifest)


def test_selected_tensor_loading_records_source_dtype_and_fp32_hash(tmp_path):
    name = "model.layers.0.self_attn.q_norm.weight"
    data = torch.arange(128, dtype=torch.bfloat16)
    save_file({name: data, "model.layers.1.unused": torch.ones(3)}, str(tmp_path / "model.safetensors"))
    reader = assets.Weights(tmp_path)
    actual = reader.get("self_attn.q_norm.weight", (128,))
    np.testing.assert_array_equal(actual, np.arange(128, dtype=np.float32))
    assert list(reader.evidence) == [name]
    assert reader.evidence[name]["source_dtype"] == "torch.bfloat16"
    assert reader.evidence[name]["fp32_sha256"] == hashlib.sha256(actual.tobytes()).hexdigest()
    with pytest.raises(ValueError, match="Unexpected/nonfinite"):
        reader.get("self_attn.q_norm.weight", (64,))


def test_full_head_rope_rotates_coordinate_127_with_coordinate_63():
    x = np.zeros((1, 8, 2, 128), np.float32)
    x[..., 63] = 1
    x[..., 127] = 2
    positions = np.array([0, 255], np.float32)
    result = cases.rope_reference(x, positions)
    np.testing.assert_array_equal(result[:, :, 0], x[:, :, 0])
    angle = 255 / (1000000 ** (126 / 128))
    np.testing.assert_allclose(result[:, :, 1, 63], np.cos(angle) - 2 * np.sin(angle), atol=1e-7, rtol=0)
    np.testing.assert_allclose(result[:, :, 1, 127], 2 * np.cos(angle) + np.sin(angle), atol=3e-7, rtol=0)
    assert np.count_nonzero(result[..., :63]) == 0


@pytest.mark.parametrize("kind,heads", [("q", 16), ("k", 8)])
def test_rope_graph_and_independent_oracle_share_one_exact_parameter_buffer(monkeypatch, kind, heads):
    # An explicit fixture buffer distinguishes correct parameter reuse from
    # independently recomputing the nominal formula in either backend.
    inv = cases.rope_inverse_frequency()
    inv[3] = np.nextafter(inv[3], np.float32(np.inf))
    calls = []
    def supplied_buffer():
        calls.append(True)
        return inv.copy()
    monkeypatch.setattr(cases, "rope_inverse_frequency", supplied_buffer)
    monkeypatch.setattr(np, "power", lambda *a, **k: pytest.fail("RoPE parameters were recomputed in NumPy"))
    x = np.zeros((1, heads, 256, 128), np.float32)
    x[..., 3], x[..., 67] = 8, -8
    case = cases.rope_case(kind, x)
    assert calls == [True]
    stored = next(numpy_helper.to_array(value) for value in case.model.graph.initializer
                  if value.name == "inv_freq")
    np.testing.assert_array_equal(stored.reshape(-1), inv)
    assert stored.dtype == np.float32
    assert case.metadata["rotary_parameters"]["inv_freq_sha256"] == hashlib.sha256(inv.tobytes()).hexdigest()
    positions = torch.arange(256, dtype=torch.float32)
    angle = positions * torch.from_numpy(inv)[3]
    expected_first = 8 * angle.cos() + 8 * angle.sin()
    expected_second = -8 * angle.cos() + 8 * angle.sin()
    np.testing.assert_array_equal(case.expected["y"][0, 0, :, 3], expected_first.numpy())
    np.testing.assert_array_equal(case.expected["y"][0, 0, :, 67], expected_second.numpy())
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(case.model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    assert run.tensor_diff(session.run(None, case.feed)[0], case.expected["y"])["passed"]


def test_rope_parameter_buffer_matches_the_default_transformers_contract():
    # Qwen's buffer is constructed before forward/export. Sharing this buffer
    # does not share the rotation implementation or execute ORT as an oracle.
    from transformers import Qwen3Config
    from transformers.modeling_rope_utils import _compute_default_rope_parameters
    config = Qwen3Config(hidden_size=1024, num_attention_heads=16,
                         num_key_value_heads=8, head_dim=128, rope_theta=1000000)
    expected, scale = _compute_default_rope_parameters(config, torch.device("cpu"))
    assert scale == 1
    np.testing.assert_array_equal(cases.rope_inverse_frequency(), expected.numpy())


def test_real_gqa_head_mapping_causal_and_key_padding_have_manual_oracle():
    length, valid = 4, 2
    q = np.zeros((1, 16, length, 128), np.float32)
    k = np.zeros((1, 8, length, 128), np.float32)
    v = np.empty_like(k)
    for head in range(8):
        for position in range(length):
            v[:, head, position, :] = 10 * head + position
    feed = {"q": q, "k": k, "v": v, "mask": cases.causal_mask(length, valid)}
    case = cases.attention_case("unit_fixture", feed, valid)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(case.model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    probability, context = session.run(None, feed)
    for head in range(16):
        # With zero scores the only legal keys are uniform: key0 for query0,
        # then keys0/1 for every later query (padding queries are retained).
        expected = np.full((length, 128), 10 * (head // 2) + 0.5, np.float32)
        expected[0] -= 0.5
        np.testing.assert_array_equal(context[0, head], expected)
    np.testing.assert_array_equal(context, case.expected["y"])
    assert np.count_nonzero(probability[..., valid:]) == 0
    assert probability[0, 0, 0, 1] == 0
    v[:, :, valid:] = 1234
    np.testing.assert_array_equal(session.run(None, feed)[1], context)


def test_checkpoint_reporting_identifies_first_failure():
    correct = np.zeros((2,), np.float32)
    wrong = np.ones((2,), np.float32)
    diff = run.compare({"gate": correct, "hidden": wrong, "y": correct},
                       {"gate": correct, "hidden": correct, "y": correct})
    assert not diff["passed"]
    assert diff["first_divergence"]["checkpoint"] == "hidden"
    assert diff["first_divergence"]["firstdiff"]["index"] == [0]


def test_main_persists_failure_report_and_refuses_existing_output(tmp_path, monkeypatch):
    def fail(source, out, report, length):
        report["stage"] = "verify_source"
        raise ValueError("Source checkpoint SHA256 mismatch")
    monkeypatch.setattr(run, "run_probe", fail)
    output = tmp_path / "failure"
    argv = ["--source-dir", str(tmp_path), "--output-dir", str(output)]
    assert run.main(argv) == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL" and report["passed"] is False
    assert report["stage"] == "verify_source"
    assert "SHA256 mismatch" in report["error"]
    assert (output / "report.md").is_file() and (output / "report.html").is_file()
    with pytest.raises(FileExistsError):
        run.main(argv)


def test_formal_gate_contract_is_immutable():
    assert run.ATOL == 1e-4
    assert run.LEVELS == ("none", "basic", "all")
    assert len(run.CASE_NAMES) == 14 and len(set(run.CASE_NAMES)) == 14
    assert assets.DIMENSIONS["hidden_size"] == 1024
    assert assets.DIMENSIONS["head_dim"] == 128
    assert assets.DIMENSIONS["num_attention_heads"] == 16
    assert assets.DIMENSIONS["num_key_value_heads"] == 8


@pytest.mark.parametrize("length,status,passed,code", [(256, "PASS", True, 0), (4, "PARTIAL", False, 2)])
def test_debug_lengths_never_claim_formal_pass(tmp_path, monkeypatch, length, status, passed, code):
    def simulated_complete(source, out, report, seq_len):
        report["cases"] = [{"name": name, "passed": True} for name in run.CASE_NAMES]
        report["invariants"] = [{"passed": True} for _ in range(10)]
        run.finalize_report(report, seq_len)
    monkeypatch.setattr(run, "run_probe", simulated_complete)
    output = tmp_path / "debug_length"
    assert run.main(["--source-dir", str(tmp_path), "--output-dir", str(output),
                     "--seq-len", str(length)]) == code
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == status and report["passed"] is passed


@pytest.mark.parametrize("omit_case,omit_invariant", [(True, False), (False, True)])
def test_missing_coverage_never_passes(omit_case, omit_invariant):
    report = {"cases": [{"name": name, "passed": True} for name in run.CASE_NAMES],
              "invariants": [{"passed": True} for _ in range(10)]}
    if omit_case:
        report["cases"].pop()
    if omit_invariant:
        report["invariants"].pop()
    run.finalize_report(report, 256)
    assert report["status"] == "FAIL" and report["passed"] is False
