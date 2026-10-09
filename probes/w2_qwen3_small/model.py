"""Official, randomly initialized Qwen3 with observational export checkpoints.

No pretrained checkpoint or tokenizer is downloaded. The forward computation is
the one shipped in transformers 4.51.3. Module hooks observe its intermediate
tensors; the scoped eager-attention adapter only observes its input Q/K and
delegates all arithmetic to the original implementation.
"""

from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3 import modeling_qwen3


TRANSFORMERS_VERSION = "4.51.3"
SEQUENCE_LENGTH = 256
MODEL_CONFIG = {
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 96,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "hidden_act": "silu",
    "max_position_embeddings": SEQUENCE_LENGTH,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "rope_scaling": None,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "use_sliding_window": False,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "use_cache": False,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
}


class DiagnosticQwen(nn.Module):
    """Wrap an official model without replacing any of its tensor operations.

    Use ``with wrapper.capture(): wrapper(input_ids, attention_mask)`` for
    reference execution. Hooks and the module-global eager adapter exist only
    inside that context and are removed even after an exception. The context is
    deliberately non-reentrant and must not overlap another capture/thread.
    ``wrapper.model`` can be run normally outside the context to verify that
    instrumentation preserves the logits.
    """

    def __init__(self, model: Qwen3ForCausalLM, seed: int, sequence_length=SEQUENCE_LENGTH):
        super().__init__()
        self.model = model
        self.sequence_length = sequence_length
        config = model.config
        length = sequence_length
        hidden = config.hidden_size
        heads = config.num_attention_heads
        kv_heads = config.num_key_value_heads
        width = config.head_dim
        self.register_buffer(
            "positions", torch.arange(length).reshape(1, length),
            persistent=False,
        )
        self.config_metadata = {
            **{name: getattr(config, name) for name in MODEL_CONFIG},
            "sequence_length": length,
            "batch_size": 1,
            "seed": seed,
            "dtype": "float32",
            "attention_implementation": "eager",
            "model_class": "transformers.Qwen3ForCausalLM",
            "transformers_version": TRANSFORMERS_VERSION,
            "weights": "fixed-seed random initialization; no pretrained weights",
            "position_ids": f"0..{length - 1}, including right-padding positions",
            "mask": "4D FP32 additive causal + key padding; 0 / finfo(float32).min",
        }
        self.checkpoint_metadata = {}
        # "embedding" is also an exporter-generated internal value name; a
        # distinct checkpoint name avoids a torch 2.7 output-alias SSA clash.
        self._describe("token_embedding", [1, length, hidden], 1, "Token embedding")
        for index in range(config.num_hidden_layers):
            prefix = f"layer_{index}."
            for name, shape, axis, description in (
                ("input_norm", [1, length, hidden], 1, "Input RMSNorm"),
                ("q_norm", [1, length, heads, width], 1, "Per-head Q RMSNorm before transpose"),
                ("k_norm", [1, length, kv_heads, width], 1, "Per-head K RMSNorm before transpose"),
                ("v_proj", [1, length, kv_heads * width], 1, "V projection before KV-head reshape"),
                ("rope_q", [1, heads, length, width], 2, "Q after official RoPE"),
                ("rope_k", [1, kv_heads, length, width], 2, "K after official RoPE, before GQA repetition"),
                ("attn_probs", [1, heads, length, length], 2, "Masked softmax probabilities"),
                ("attn_context", [1, length, heads * width], 1, "Attention context, input to output projection"),
                ("attn_output", [1, length, hidden], 1, "Attention output after output projection"),
                ("attn_residual", [1, length, hidden], 1, "Residual after attention"),
                ("post_attention_norm", [1, length, hidden], 1, "RMSNorm before MLP"),
                ("mlp", [1, length, hidden], 1, "Official SwiGLU MLP output"),
                ("residual", [1, length, hidden], 1, "Decoder output after MLP residual"),
            ):
                self._describe(prefix + name, shape, axis, description)
            self.checkpoint_metadata[prefix + "attn_probs"]["key_sequence_axis"] = 3
        self._describe("final_norm", [1, length, hidden], 1, "Final RMSNorm")
        self._describe("logits", [1, length, config.vocab_size], 1, "Tied language-model head logits")
        self.output_names = tuple(self.checkpoint_metadata)
        self._captured = {}
        self._capture_active = False

    def _describe(self, name, shape, sequence_axis, description):
        self.checkpoint_metadata[name] = {
            "shape": shape,
            "dtype": "float32",
            "sequence_axis": sequence_axis,
            "description": description,
        }

    @contextmanager
    def capture(self):
        """Install temporary observers around an official forward/export call."""
        original = modeling_qwen3.eager_attention_forward
        if self._capture_active or getattr(original, "_qwen_probe_capture", False):
            raise RuntimeError("Qwen diagnostic capture contexts must not overlap")
        self._capture_active = True
        handles = []

        def output_hook(name, item=None):
            def observe(module, inputs, output):
                self._captured[name] = output if item is None else output[item]
            return observe

        def input_hook(name):
            def observe(module, inputs):
                self._captured[name] = inputs[0]
            return observe

        def attention_observer(
            module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
        ):
            # This observer never computes attention or RoPE itself. Restrict
            # capture to this wrapper's modules while preserving other callers.
            if (
                0 <= module.layer_idx < len(self.model.model.layers)
                and module is self.model.model.layers[module.layer_idx].self_attn
            ):
                self._captured[f"layer_{module.layer_idx}.rope_q"] = query
                self._captured[f"layer_{module.layer_idx}.rope_k"] = key
            return original(
                module, query, key, value, attention_mask,
                scaling=scaling, dropout=dropout, **kwargs,
            )

        attention_observer._qwen_probe_capture = True
        try:
            base = self.model.model
            handles.append(base.embed_tokens.register_forward_hook(output_hook("token_embedding")))
            handles.append(base.norm.register_forward_hook(output_hook("final_norm")))
            for index, layer in enumerate(base.layers):
                prefix = f"layer_{index}."
                for module, name in (
                    (layer.input_layernorm, "input_norm"),
                    (layer.self_attn.q_norm, "q_norm"),
                    (layer.self_attn.k_norm, "k_norm"),
                    (layer.self_attn.v_proj, "v_proj"),
                    (layer.post_attention_layernorm, "post_attention_norm"),
                    (layer.mlp, "mlp"),
                ):
                    handles.append(module.register_forward_hook(output_hook(prefix + name)))
                handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                    input_hook(prefix + "attn_context")))
                handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
                    input_hook(prefix + "attn_residual")))
                handles.append(layer.self_attn.register_forward_hook(
                    output_hook(prefix + "attn_output", 0)))
                handles.append(layer.self_attn.register_forward_hook(
                    output_hook(prefix + "attn_probs", 1)))
                handles.append(layer.register_forward_hook(output_hook(prefix + "residual", 0)))
            modeling_qwen3.eager_attention_forward = attention_observer
            yield self
        finally:
            modeling_qwen3.eager_attention_forward = original
            for handle in handles:
                handle.remove()
            self._capture_active = False
            self._captured = {}

    def forward(self, input_ids, attention_mask):
        if not self._capture_active:
            raise RuntimeError("Use 'with wrapper.capture()' for diagnostic forward calls")
        self._captured = {}
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=self.positions,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            logits_to_keep=0,
        )[0]
        self._captured["logits"] = logits
        outputs = tuple(self._captured[name] for name in self.output_names)
        # Do not leave tensors or a changed pytree on the module: torch.export
        # checks module-attribute mutations after the forward has completed.
        self._captured = {}
        return outputs


def build_model(seed=0, model_config=None, sequence_length=SEQUENCE_LENGTH):
    """Construct an official FP32 probe; omitted options preserve the W2 preset.

    ``model_config`` overrides named ``MODEL_CONFIG`` fields. Checkpoint shapes
    derive from the constructed model, including independent Q/KV head widths.
    The sequence length is a static export contract, not a dynamic dimension.
    """
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"This probe requires transformers=={TRANSFORMERS_VERSION}; "
            f"found {transformers.__version__}"
        )
    settings = dict(MODEL_CONFIG)
    if model_config is not None:
        unknown = set(model_config) - set(settings)
        if unknown:
            raise ValueError(f"Unsupported probe configuration fields: {sorted(unknown)}")
        settings.update(model_config)
    for name in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                 "num_attention_heads", "num_key_value_heads", "head_dim",
                 "max_position_embeddings"):
        if isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")
    if sequence_length > settings["max_position_embeddings"]:
        raise ValueError("sequence_length exceeds max_position_embeddings")
    if settings["num_attention_heads"] % settings["num_key_value_heads"]:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if settings["head_dim"] % 2:
        raise ValueError("head_dim must be even for RoPE")
    if settings["use_cache"] or not settings["tie_word_embeddings"]:
        raise ValueError("The probe requires use_cache=False and tie_word_embeddings=True")
    config = Qwen3Config(**settings)
    config._attn_implementation = "eager"
    # Preserve the caller's random state; CPU initialization never downloads.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Qwen3ForCausalLM(config).float().eval()
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise RuntimeError("The official model did not tie its embedding and LM-head weights")
    return DiagnosticQwen(model, seed, sequence_length).eval()


def export_onnx(wrapper, path, input_ids, attention_mask):
    """Export all named outputs using the same Dynamo path as the full model.

    Inputs have fixed shapes [1,L] INT64 and [1,1,L,L] FP32. No dynamic
    shapes, fallback exporter, or ONNX optimizer is enabled. The small random
    weights are embedded in a single ONNX file rather than external data.
    """
    length = wrapper.sequence_length
    if input_ids.shape != (1, length) or input_ids.dtype != torch.int64:
        raise ValueError(f"input_ids must have fixed shape [1,{length}] and dtype INT64")
    if (
        attention_mask.shape != (1, 1, length, length)
        or attention_mask.dtype != torch.float32
    ):
        raise ValueError(f"attention_mask must have fixed shape [1,1,{length},{length}] and dtype FP32")
    if wrapper.training or wrapper.model.training:
        raise ValueError("The Qwen probe must be exported in eval mode")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), wrapper.capture():
        program = torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            dynamo=True,
            opset_version=18,
            input_names=["input_ids", "attention_mask"],
            output_names=list(wrapper.output_names),
            dynamic_shapes=None,
            optimize=False,
            fallback=False,
        )
        program.save(str(path), external_data=False)
    # Check the persisted artifact and output-name mapping, not only tracing.
    import onnx

    exported = onnx.load(str(path))
    onnx.checker.check_model(exported)
    actual = tuple(value.name for value in exported.graph.output)
    if actual != wrapper.output_names:
        raise RuntimeError(f"Exporter changed diagnostic output names: {actual!r}")
    return {
        "exporter": "torch.onnx.export(dynamo=True, optimize=False, fallback=False)",
        "opset": 18,
        "external_data": False,
        "output_names": list(actual),
        "instrumentation": "temporary module hooks plus delegating official eager-attention observer",
    }
