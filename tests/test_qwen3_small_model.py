"""Optional pinned-stack checks for the official Qwen3 diagnostic model.

Minimal compiler installations do not need torch/transformers. The full CI
and dedicated Qwen probe environments run these tests with transformers 4.51.3.
"""

import numpy as np
import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if transformers.__version__ != "4.51.3":
    pytest.skip("Qwen diagnostic tests require transformers==4.51.3", allow_module_level=True)

from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3 import modeling_qwen3

from probes.w2_qwen3_small.model import MODEL_CONFIG, build_model, export_onnx


@pytest.fixture(scope="module", autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def wrapper():
    return build_model(seed=193)


@pytest.fixture(scope="module")
def inputs():
    ids = (torch.arange(256).reshape(1, 256) % 127 + 1).long()
    ids[:, 37:] = 0
    mask = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
    future = torch.triu(torch.ones((256, 256), dtype=torch.bool), diagonal=1)
    mask.masked_fill_(future, torch.finfo(torch.float32).min)
    mask[:, :, :, 37:] = torch.finfo(torch.float32).min
    return ids, mask


@pytest.fixture(scope="module")
def checkpoints(wrapper, inputs):
    with torch.inference_mode(), wrapper.capture():
        result = wrapper(*inputs)
    return dict(zip(wrapper.output_names, result))


@pytest.fixture(scope="module")
def exported_model(wrapper, inputs, tmp_path_factory):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    path = tmp_path_factory.mktemp("qwen3-small-export") / "checkpoints.onnx"
    original = modeling_qwen3.eager_attention_forward
    metadata = export_onnx(wrapper, path, *inputs)
    assert_observers_removed(wrapper, original)
    return path, metadata


def raw_logits(model, inputs, positions):
    return model(
        input_ids=inputs[0], attention_mask=inputs[1], position_ids=positions,
        use_cache=False, output_attentions=False, output_hidden_states=False,
        return_dict=False, logits_to_keep=0,
    )[0]


def assert_observers_removed(wrapper, original):
    assert modeling_qwen3.eager_attention_forward is original
    assert not wrapper._capture_active
    assert not wrapper._captured
    assert all(not module._forward_hooks for module in wrapper.modules())
    assert all(not module._forward_pre_hooks for module in wrapper.modules())


def test_build_preserves_rng_and_reproduces_seed():
    state = torch.random.get_rng_state().clone()
    first = build_model(seed=31)
    assert torch.equal(torch.random.get_rng_state(), state)
    second = build_model(seed=31)
    different = build_model(seed=32)
    assert torch.equal(torch.random.get_rng_state(), state)
    first_weights = first.model.state_dict()
    second_weights = second.model.state_dict()
    assert first_weights.keys() == second_weights.keys()
    assert all(torch.equal(first_weights[name], second_weights[name]) for name in first_weights)
    assert not torch.equal(
        first.model.model.embed_tokens.weight, different.model.model.embed_tokens.weight
    )


def test_official_architecture_preserves_nonstandard_q_width_and_tying(wrapper):
    assert isinstance(wrapper.model, Qwen3ForCausalLM)
    config = wrapper.model.config
    assert config.num_hidden_layers == 2
    assert config.hidden_size == 32
    assert config.intermediate_size == 96
    assert config.num_attention_heads == 4
    assert config.num_key_value_heads == 2
    assert config.head_dim == 16
    assert config.num_attention_heads * config.head_dim != config.hidden_size
    assert config.rms_norm_eps == 1e-6
    assert config.rope_theta == 1_000_000.0
    assert config.rope_scaling is None
    assert config.hidden_act == "silu"
    assert config._attn_implementation == "eager"
    assert not config.use_cache
    assert not config.use_sliding_window
    assert config.attention_dropout == 0.0
    assert wrapper.model.lm_head.weight is wrapper.model.model.embed_tokens.weight
    assert tuple(wrapper.model.lm_head.weight.shape) == (128, 32)
    assert not wrapper.training and not wrapper.model.training
    assert all(parameter.dtype == torch.float32 for parameter in wrapper.parameters())
    assert torch.equal(wrapper.positions, torch.arange(256).reshape(1, 256))
    for layer in wrapper.model.model.layers:
        attention = layer.self_attn
        assert tuple(attention.q_proj.weight.shape) == (64, 32)
        assert tuple(attention.k_proj.weight.shape) == (32, 32)
        assert tuple(attention.v_proj.weight.shape) == (32, 32)
        assert tuple(attention.o_proj.weight.shape) == (32, 64)
        assert tuple(attention.q_norm.weight.shape) == (16,)
        assert tuple(attention.k_norm.weight.shape) == (16,)
        assert attention.num_key_value_groups == 2
        assert attention.scaling == 0.25


def test_checkpoint_schema_matches_observed_tensors(wrapper, checkpoints):
    assert wrapper.output_names[0] == "token_embedding"
    assert wrapper.output_names[-1] == "logits"
    assert len(wrapper.output_names) == len(set(wrapper.output_names)) == 29
    assert tuple(checkpoints) == wrapper.output_names
    for name, tensor in checkpoints.items():
        metadata = wrapper.checkpoint_metadata[name]
        assert list(tensor.shape) == metadata["shape"], name
        assert tensor.shape[metadata["sequence_axis"]] == 256, name
        assert tensor.dtype == torch.float32, name
        assert bool(torch.isfinite(tensor).all()), name
    assert tuple(checkpoints["layer_0.q_norm"].shape) == (1, 256, 4, 16)
    assert tuple(checkpoints["layer_0.k_norm"].shape) == (1, 256, 2, 16)
    assert tuple(checkpoints["layer_0.rope_q"].shape) == (1, 4, 256, 16)
    assert tuple(checkpoints["layer_0.rope_k"].shape) == (1, 2, 256, 16)
    assert wrapper.checkpoint_metadata["layer_0.attn_probs"]["key_sequence_axis"] == 3


@pytest.mark.parametrize("layer_index", [0, 1])
@pytest.mark.parametrize("kind,heads", [("q", 4), ("k", 2)])
def test_qk_norm_is_per_head_rmsnorm(wrapper, checkpoints, layer_index, kind, heads):
    attention = wrapper.model.model.layers[layer_index].self_attn
    normalized = checkpoints[f"layer_{layer_index}.input_norm"].numpy()
    projection = getattr(attention, f"{kind}_proj").weight.detach().numpy()
    weights = getattr(attention, f"{kind}_norm").weight.detach().numpy()
    projected = (normalized @ projection.T).reshape(1, 256, heads, 16)
    mean_square = np.mean(np.square(projected), axis=-1, keepdims=True)
    expected = projected / np.sqrt(mean_square + np.float32(1e-6)) * weights
    actual = checkpoints[f"layer_{layer_index}.{kind}_norm"].numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("layer_index", [0, 1])
@pytest.mark.parametrize("kind", ["q", "k"])
def test_rope_uses_all_head_dimensions_and_fixed_positions(checkpoints, layer_index, kind):
    before = checkpoints[f"layer_{layer_index}.{kind}_norm"].numpy().transpose(0, 2, 1, 3)
    frequencies = np.float32(1.0) / np.power(
        np.float32(1_000_000.0), np.arange(0, 16, 2, dtype=np.float32) / np.float32(16)
    )
    angles = np.arange(256, dtype=np.float32)[:, None] * frequencies[None, :]
    angles = np.concatenate((angles, angles), axis=-1)[None, None, :, :]
    rotated_half = np.concatenate((-before[..., 8:], before[..., :8]), axis=-1)
    expected = before * np.cos(angles) + rotated_half * np.sin(angles)
    actual = checkpoints[f"layer_{layer_index}.rope_{kind}"].numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    # Position zero must be unchanged, and every channel must rotate somewhere
    # in the fixed sequence (including the low-frequency second half).
    np.testing.assert_array_equal(actual[:, :, 0], before[:, :, 0])
    changes_per_channel = np.max(np.abs(actual[:, :, 1:] - before[:, :, 1:]), axis=(0, 1, 2))
    assert np.all(changes_per_channel > 1e-4)


def test_observers_leave_official_logits_bitwise_unchanged(wrapper, inputs, checkpoints):
    with torch.inference_mode():
        expected = raw_logits(wrapper.model, inputs, wrapper.positions)
    assert torch.equal(checkpoints["logits"], expected)


def test_diagnostic_forward_requires_capture(wrapper, inputs):
    with pytest.raises(RuntimeError, match="wrapper.capture"):
        wrapper(*inputs)


def test_capture_exception_restores_global_function_and_hooks(wrapper, inputs):
    original = modeling_qwen3.eager_attention_forward
    with pytest.raises(ValueError, match="deliberate failure"):
        with torch.inference_mode(), wrapper.capture():
            wrapper(*inputs)
            raise ValueError("deliberate failure")
    assert_observers_removed(wrapper, original)
    with torch.inference_mode(), wrapper.capture():
        assert wrapper(*inputs)[-1].shape == (1, 256, 128)
    assert_observers_removed(wrapper, original)


def test_capture_reentry_rejection_does_not_break_outer_capture(wrapper, inputs, checkpoints):
    other = build_model(seed=194)
    original = modeling_qwen3.eager_attention_forward
    with torch.inference_mode(), wrapper.capture():
        observer = modeling_qwen3.eager_attention_forward
        for target in (wrapper, other):
            with pytest.raises(RuntimeError, match="must not overlap"):
                with target.capture():
                    pytest.fail("Overlapping captures were accepted")
            assert modeling_qwen3.eager_attention_forward is observer
            assert wrapper._capture_active
        assert torch.equal(wrapper(*inputs)[-1], checkpoints["logits"])
    assert_observers_removed(wrapper, original)
    assert_observers_removed(other, original)


def test_capture_preserves_preexisting_user_hooks(wrapper, inputs):
    calls = []
    handle = wrapper.model.model.norm.register_forward_hook(
        lambda module, args, result: calls.append(result.shape)
    )
    original = modeling_qwen3.eager_attention_forward
    try:
        with torch.inference_mode(), wrapper.capture():
            wrapper(*inputs)
        assert handle.id in wrapper.model.model.norm._forward_hooks
        with torch.inference_mode():
            raw_logits(wrapper.model, inputs, wrapper.positions)
        assert calls == [torch.Size((1, 256, 32)), torch.Size((1, 256, 32))]
    finally:
        handle.remove()
    assert_observers_removed(wrapper, original)


def test_capture_does_not_intercept_other_official_model_with_more_layers(wrapper, inputs):
    config = Qwen3Config(**{**MODEL_CONFIG, "num_hidden_layers": 3})
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(295)
        other = Qwen3ForCausalLM(config).float().eval()
    original = modeling_qwen3.eager_attention_forward
    with torch.inference_mode():
        expected = raw_logits(other, inputs, wrapper.positions)
        with wrapper.capture():
            actual = raw_logits(other, inputs, wrapper.positions)
            # Layer indices 0/1 belong to different modules; index 2 exceeds
            # this wrapper's layer count. None may be observed or raise.
            assert wrapper._captured == {}
    assert torch.equal(actual, expected)
    assert_observers_removed(wrapper, original)


def test_dynamo_export_names_and_all_checkpoints_round_trip(wrapper, inputs, checkpoints, exported_model):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    original = modeling_qwen3.eager_attention_forward
    path, metadata = exported_model
    assert_observers_removed(wrapper, original)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    assert [(item.domain, item.version) for item in graph.opset_import] == [("", 18)]
    assert tuple(value.name for value in graph.graph.output) == wrapper.output_names
    assert metadata["external_data"] is False
    assert not any(value.external_data for value in graph.graph.initializer)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    feed = {"input_ids": inputs[0].numpy(), "attention_mask": inputs[1].numpy()}
    actual = session.run(list(wrapper.output_names), feed)
    for name, result in zip(wrapper.output_names, actual):
        np.testing.assert_allclose(result, checkpoints[name].numpy(), atol=1e-5, rtol=1e-5,
                                   err_msg=name)


@pytest.mark.parametrize("fault", ["normal_parse", "normal_verifier", "diagnostic_parse",
                                   "diagnostic_verifier", "numeric"])
def test_run_probe_records_only_verified_graphs(tmp_path, monkeypatch, wrapper, exported_model, fault):
    """Run the real small-model stages; reuse one export and stop at the fault."""
    from collections import Counter
    import shutil
    import onnx
    from probes.w2_qwen3_small import model, run
    from scratchv.analysis import ir_verifier
    from scratchv.frontend.onnx_parser import ONNXParser
    from scratchv.verification.ir_interpreter import IRInterpreter

    source, metadata = exported_model
    monkeypatch.setattr(run, "environment", lambda: {"fixture": "pinned small-model stack"})
    monkeypatch.setattr(model, "build_model", lambda **kwargs: wrapper)

    def reuse_export(wrapper, path, *inputs):
        shutil.copyfile(source, path)
        return metadata
    monkeypatch.setattr(model, "export_onnx", reuse_export)
    original_parse, original_verify = ONNXParser.parse, ir_verifier.verify_ir
    parsed = []

    def parse(self, path):
        kind = "normal" if str(path).endswith("model.onnx") else "diagnostic"
        parsed.append(kind)
        if fault == f"{kind}_parse":
            raise ValueError(f"injected {fault}")
        return original_parse(self, path)

    def verify(program):
        if fault == f"{parsed[-1]}_verifier":
            return False, [f"injected {fault}"]
        return original_verify(program)

    def numeric_failure(*args, **kwargs):
        raise ValueError("injected numeric")

    monkeypatch.setattr(ONNXParser, "parse", parse)
    monkeypatch.setattr(ir_verifier, "verify_ir", verify)
    if fault == "numeric":
        monkeypatch.setattr(IRInterpreter, "run", numeric_failure)
    report = {"passed": False}
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with pytest.raises(ValueError, match="injected"):
            run.run_probe(tmp_path, report, model_seed=193)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)
    assert not report["passed"]
    assert report["planned_cases"] == len(run.input_cases()) == 7
    assert report["stage"] == ("numeric" if fault == "numeric" else "parse")
    if fault.startswith("normal"):
        assert "frontend_parse" not in report
    else:
        graphs = ["normal", "diagnostic"] if fault == "numeric" else ["normal"]
        operators = Counter()
        for kind in graphs:
            path = tmp_path / ("model.onnx" if kind == "normal" else "diagnostics.onnx")
            operators.update(node.op_type for node in onnx.load(path).graph.node)
        assert report["frontend_parse"] == {
            "passed": True, "operators": dict(operators), "graphs": graphs}
