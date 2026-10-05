"""Pinned-stack regressions for config-driven observers and the medium preset."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if transformers.__version__ != "4.51.3":
    pytest.skip("Qwen observers require transformers==4.51.3", allow_module_level=True)

from probes.w2_qwen3_small.model import MODEL_CONFIG, build_model, export_onnx
from probes.w2_qwen3_small.diagnostics import tensor_diff
from probes.w3_qwen3_medium.preset import MEDIUM_CONFIG, build_medium_model
from probes.w3_qwen3_medium.run import trace_schema


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def execute(wrapper):
    length = wrapper.sequence_length
    ids = (torch.arange(length).reshape(1, length) % (wrapper.model.config.vocab_size - 1) + 1).long()
    mask = torch.zeros((1, 1, length, length), dtype=torch.float32)
    mask.masked_fill_(torch.triu(torch.ones((length, length), dtype=torch.bool), 1),
                      torch.finfo(torch.float32).min)
    with torch.inference_mode(), wrapper.capture():
        tensors = wrapper(ids, mask)
    return (ids, mask), dict(zip(wrapper.output_names, tensors))


def test_medium_preset_and_all_observed_shapes():
    wrapper = build_medium_model(seed=31)
    assert (wrapper.model.config.num_hidden_layers, wrapper.model.config.hidden_size,
            wrapper.model.config.intermediate_size) == (6, 64, 192)
    assert len(wrapper.output_names) == 3 + 13 * 6 == 81
    assert wrapper.config_metadata["weights"].startswith("fixed-seed random")
    assert wrapper.config_metadata["dtype"] == "float32"
    assert wrapper.config_metadata["use_cache"] is False
    assert wrapper.model.lm_head.weight is wrapper.model.model.embed_tokens.weight
    _, values = execute(wrapper)
    for name, value in values.items():
        assert list(value.shape) == wrapper.checkpoint_metadata[name]["shape"], name
        assert value.dtype == torch.float32 and bool(torch.isfinite(value).all())
    schema = trace_schema(wrapper.checkpoint_metadata)
    assert len(schema["checkpoints"]) == 81
    assert [row["layer"] for row in schema["checkpoints"] if row["checkpoint"] == "residual"] == list(range(6))
    assert wrapper.config_metadata["hidden_size"] == 64
    assert MODEL_CONFIG["hidden_size"] == 32


def test_config_changes_every_schema_dimension_and_export_contract(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    settings = {"hidden_size": 48, "intermediate_size": 80, "num_hidden_layers": 1,
                "num_attention_heads": 6, "num_key_value_heads": 3, "head_dim": 8,
                "vocab_size": 29, "max_position_embeddings": 11}
    original = dict(settings)
    wrapper = build_model(seed=41, model_config=settings, sequence_length=7)
    assert settings == original
    assert len(wrapper.output_names) == 16
    assert wrapper.config_metadata["position_ids"] == "0..6, including right-padding positions"
    inputs, values = execute(wrapper)
    assert list(values["layer_0.q_norm"].shape) == [1, 7, 6, 8]
    assert list(values["layer_0.k_norm"].shape) == [1, 7, 3, 8]
    assert list(values["layer_0.v_proj"].shape) == [1, 7, 24]
    assert list(values["layer_0.attn_context"].shape) == [1, 7, 48]
    assert list(values["logits"].shape) == [1, 7, 29]
    for name, value in values.items():
        assert list(value.shape) == wrapper.checkpoint_metadata[name]["shape"]
    path = tmp_path / "custom.onnx"
    export_onnx(wrapper, path, *inputs)
    model = onnx.load(path)
    assert [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim] == [1, 7]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input_ids": inputs[0].numpy(), "attention_mask": inputs[1].numpy()})
    assert all(tensor_diff(got, values[name].numpy(), atol=1e-5)["passed"]
               for name, got in zip(wrapper.output_names, actual))


@pytest.mark.parametrize("override,match", [
    ({"num_hidden_layers": 0}, "positive integer"),
    ({"num_attention_heads": 5}, "divisible"),
    ({"head_dim": 15}, "even"),
    ({"hidden_size": True}, "positive integer"),
    ({"use_cache": True}, "use_cache=False"),
    ({"mystery": 5}, "Unsupported"),
])
def test_invalid_probe_configs_fail_before_initialization(override, match):
    with pytest.raises(ValueError, match=match):
        build_model(model_config=override)


def test_sequence_length_cannot_exceed_model_positions():
    with pytest.raises(ValueError, match="exceeds"):
        build_model(sequence_length=257)
