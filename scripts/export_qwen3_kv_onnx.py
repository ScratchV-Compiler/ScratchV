#!/usr/bin/env python3
"""Export functional, fixed-capacity Qwen3 prefill/decode (no hidden Cache state).

Run in the pinned requirements/qwen3-small-probe.txt environment. The small
model uses seeded random weights; --model-dir loads the pinned full checkpoint.
Artifacts, including external data, are written to a new output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_model(model_dir=None, seed=0):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from probes.w2_qwen3_small.run import PINNED
    mismatches = {name: importlib.metadata.version(name) for name, expected in PINNED.items()
                  if importlib.metadata.version(name).split('+')[0] != expected}
    if mismatches:
        raise RuntimeError(f"Use the pinned export environment; mismatches: {mismatches}")
    if model_dir is not None:
        from export_qwen3_onnx import CHECKPOINT_SHA256, sha256
        if sha256(Path(model_dir) / 'model.safetensors') != CHECKPOINT_SHA256:
            raise ValueError('Full checkpoint differs from the pinned Qwen3 revision')
        model = Qwen3ForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=torch.float32,
            attn_implementation='eager')
        expected = {'num_hidden_layers':28,'hidden_size':1024,'num_attention_heads':16,
                    'num_key_value_heads':8,'head_dim':128,'vocab_size':151936}
        for name,value in expected.items():
            if getattr(model.config,name)!=value:
                raise ValueError(f'Unexpected full-model config: {name}')
    else:
        torch.manual_seed(seed)
        from probes.w2_qwen3_small.model import MODEL_CONFIG
        config = Qwen3Config(**{**MODEL_CONFIG, 'use_cache': True, '_attn_implementation': 'eager'})
        model = Qwen3ForCausalLM(config)
    return model.eval().cpu()


def make_wrappers(model, capacity=256):
    import torch
    from transformers.models.qwen3.modeling_qwen3 import (
        apply_rotary_pos_emb,
    )
    if capacity != 256:
        raise ValueError('KV ABI v1 requires capacity=256')

    class Wrapper(torch.nn.Module):
        def __init__(self, decode):
            super().__init__()
            self.model = model
            self.decode = decode
            if model.config.rope_scaling is not None:
                raise ValueError('KV ABI v1 exports default fixed RoPE only')
            with torch.no_grad():
                cos, sin = model.model.rotary_emb(
                    torch.zeros(1, capacity, model.config.hidden_size),
                    torch.arange(capacity).reshape(1, capacity))
            self.register_buffer('cos_table', cos[0])
            self.register_buffer('sin_table', sin[0])
            if decode:
                # Gather one column of this identity at the runtime position.
                # This expresses a functional update with ordinary ONNX ops;
                # it needs neither Scatter nor a dynamic Slice.
                self.register_buffer('slot_table', torch.eye(capacity))

        def forward(self, input_ids, attention_mask, index, *past):
            hidden = self.model.model.embed_tokens(input_ids)
            cos_sin = ((torch.index_select(self.cos_table, 0, index.reshape(1)).unsqueeze(0),
                        torch.index_select(self.sin_table, 0, index.reshape(1)).unsqueeze(0)) if self.decode else
                       (self.cos_table.unsqueeze(0), self.sin_table.unsqueeze(0)))
            slot = torch.index_select(self.slot_table, 0, index.reshape(1)).reshape(1, 1, capacity, 1) if self.decode else None
            present = []
            for i, layer in enumerate(self.model.model.layers):
                residual = hidden
                x = layer.input_layernorm(hidden)
                attn = layer.self_attn
                shape = (*x.shape[:-1], -1, attn.head_dim)
                q = attn.q_norm(attn.q_proj(x).view(shape)).transpose(1, 2)
                k = attn.k_norm(attn.k_proj(x).view(shape)).transpose(1, 2)
                v = attn.v_proj(x).view(shape).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, *cos_sin)
                if self.decode:
                    k = past[2*i] * (1.0 - slot) + k * slot
                    v = past[2*i+1] * (1.0 - slot) + v * slot
                present.extend((k, v))
                groups = attn.num_key_value_groups
                def repeat(tensor):
                    return torch.cat([tensor.unsqueeze(2)] * groups, dim=2).reshape(
                        1, model.config.num_attention_heads, capacity, attn.head_dim)
                scores = torch.matmul(q, repeat(k).transpose(2, 3)) * attn.scaling
                probs = torch.softmax(scores + attention_mask, dim=-1, dtype=torch.float32)
                context = torch.matmul(probs, repeat(v)).transpose(1, 2).contiguous()
                hidden = residual + attn.o_proj(context.reshape(*x.shape[:-1], -1))
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            hidden = self.model.model.norm(hidden)
            if not self.decode:
                hidden = torch.index_select(hidden, 1, index)
            return (self.model.lm_head(hidden), *present)

    return Wrapper(False).eval(), Wrapper(True).eval()


def export_models(directory, *, model_dir=None, seed=0, mode=None, exporter=None):
    import numpy as np
    import onnx
    import torch
    from scratchv.runtime.kv_cache import KVCacheLayout, prepare_prefill, prepare_decode
    directory = Path(directory).resolve()
    if mode not in (None, 'prefill', 'decode'):
        raise ValueError('mode must be prefill or decode')
    directory.mkdir(parents=True, exist_ok=mode is not None)
    model = load_model(model_dir, seed)
    config = model.config
    layout = KVCacheLayout(config.num_hidden_layers, config.num_key_value_heads,
                           config.head_dim, config.vocab_size)
    wrappers = make_wrappers(model)
    feeds = [prepare_prefill([1, 2, 3], layout),
             prepare_decode(4, 3, layout, [np.zeros(layout.shape, np.float32)
                                         for _ in layout.cache_names])]
    report = {'schema': 'scratchv.kv-export.v1', 'layout': layout.to_dict(),
              'seed': seed, 'checkpoint': str(model_dir) if model_dir else 'seeded-small',
              'packages': {name: importlib.metadata.version(name) for name in
                           ('torch', 'transformers', 'onnx', 'onnxruntime')}, 'graphs': {}}
    exporter = exporter or ('dynamo' if model_dir is not None else 'legacy')
    if exporter not in ('legacy', 'dynamo'):
        raise ValueError('Unsupported exporter')
    report['exporter'] = exporter
    selected_mode = mode
    for mode, wrapper, feed in zip(('prefill', 'decode'), wrappers, feeds):
        if selected_mode is not None and selected_mode != mode:
            continue
        path = directory / f'{mode}.onnx'
        if path.exists():
            raise FileExistsError(path)
        print(f'Exporting {mode} with {exporter}', flush=True)
        with torch.no_grad():
            tensors = tuple(torch.from_numpy(v) for v in feed.values())
            if exporter == 'dynamo':
                program = torch.onnx.export(
                    wrapper, tensors, input_names=list(feed),
                    output_names=list(layout.output_names), opset_version=18,
                    dynamo=True, optimize=False, dynamic_shapes=None)
                program.save(str(path), external_data=True)
                del program
            else:
                torch.onnx.export(wrapper, tensors, path, input_names=list(feed),
                                  output_names=list(layout.output_names),
                                  opset_version=18, dynamo=False, do_constant_folding=True)
        graph = onnx.shape_inference.infer_shapes(onnx.load(path, load_external_data=False), strict_mode=True)
        onnx.checker.check_model(str(path))
        unknown = []
        for value in [*graph.graph.input, *graph.graph.output, *graph.graph.value_info]:
            if any(not d.HasField('dim_value') for d in value.type.tensor_type.shape.dim):
                unknown.append(value.name)
        if unknown:
            raise ValueError(f'{mode}: dynamic boundary/intermediate declarations: {unknown}')
        report['graphs'][mode] = {
            'path': str(path), 'operators': dict(sorted(Counter(
                f'{node.domain or "ai.onnx"}::{node.op_type}' for node in graph.graph.node).items())),
            'opsets': {op.domain or 'ai.onnx': op.version for op in graph.opset_import},
            'inputs': list(feed), 'outputs': list(layout.output_names), 'dynamic_shapes': unknown,
            'untyped_intermediates': sorted({name for node in graph.graph.node for name in node.output}
                - {v.name for v in [*graph.graph.input,*graph.graph.output,*graph.graph.value_info]})}
        if report['graphs'][mode]['untyped_intermediates']:
            raise ValueError(f'{mode}: shape inference missed intermediate tensors')
    report_path = directory / ('export.json' if selected_mode is None else f'export-{selected_mode}.json')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return model, wrappers, layout, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', choices=('prefill', 'decode'),
                        help='Export one graph per process, sharing a new output directory')
    parser.add_argument('--exporter', choices=('legacy', 'dynamo'))
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    import torch
    torch.set_num_threads(args.threads)
    _, _, _, report = export_models(args.output_dir, model_dir=args.model_dir, seed=args.seed,
                                   mode=args.mode, exporter=args.exporter)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
