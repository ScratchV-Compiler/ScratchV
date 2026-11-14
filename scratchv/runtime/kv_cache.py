"""Versioned fixed-capacity KV tensor contract and Host reference inputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np

from .llm_inputs import _integer, _token_sequence, _fp32_finite, prepare_inputs


@dataclass(frozen=True)
class KVCacheLayout:
    layers: int
    kv_heads: int
    head_dim: int
    vocab_size: int
    capacity: int = 256
    pad_token_id: int = 0

    def __post_init__(self):
        for name in ('layers', 'kv_heads', 'head_dim', 'vocab_size'):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=1, maximum=2**31-1))
        object.__setattr__(self, 'capacity', _integer(self.capacity, 'capacity'))
        if self.capacity != 256:
            raise ValueError('KV ABI v1 requires capacity=256')
        object.__setattr__(self, 'pad_token_id', _integer(
            self.pad_token_id, 'pad_token_id', maximum=self.vocab_size-1))
        if math.prod(self.shape) > 2**31-1:
            raise ValueError('Cache tensor exceeds static backend element limit')

    @property
    def shape(self):
        return (1, self.kv_heads, self.capacity, self.head_dim)

    @property
    def cache_names(self):
        return tuple(f'layer_{i}.{kind}' for i in range(self.layers) for kind in ('key', 'value'))

    @property
    def output_names(self):
        return ('logits', *self.cache_names)

    @property
    def bank_bytes(self):
        return len(self.cache_names) * math.prod(self.shape) * 4

    def to_dict(self):
        return asdict(self)

    def validate_graphs(self, prefill, decode):
        """Validate names/order/types/shapes of two TensorC artifacts."""
        from scratchv.ir.types import DataType
        def spec(name, dtype, shape):
            return (name, dtype, shape)
        expected_outputs = [spec('logits', DataType.FLOAT32, (1, 1, self.vocab_size))]
        expected_outputs += [spec(name, DataType.FLOAT32, self.shape) for name in self.cache_names]
        for artifact, is_decode in ((prefill, False), (decode, True)):
            inputs = [spec('input_ids', DataType.INT64, (1, 1 if is_decode else self.capacity)),
                      spec('attention_mask', DataType.FLOAT32,
                           (1, 1, 1 if is_decode else self.capacity, self.capacity)),
                      spec('position_ids' if is_decode else 'logit_index', DataType.INT64,
                           (1, 1) if is_decode else (1,))]
            if is_decode:
                inputs += [spec('past.'+name, DataType.FLOAT32, self.shape) for name in self.cache_names]
            actual = lambda values: [(v.name, v.dtype, tuple(v.shape)) for v in values]
            if actual(artifact.inputs) != inputs or actual(artifact.outputs) != expected_outputs:
                raise ValueError('Prefill/Decode tensors differ from KV ABI v1')
            if artifact.abi_version != 2 or artifact.constant_storage != 'external':
                raise ValueError('KV session requires external Tensor-C ABI v2')


def prepare_prefill(tokens, layout: KVCacheLayout):
    tokens = _token_sequence(tokens, vocab_size=layout.vocab_size, name='tokens')
    prepared = prepare_inputs(tokens, pad_token_id=layout.pad_token_id, vocab_size=layout.vocab_size)
    return {**prepared.as_feed(), 'logit_index': np.array([len(tokens)-1], np.int64)}


def prepare_decode(token, position, layout: KVCacheLayout, cache):
    token = _integer(token, 'token', maximum=layout.vocab_size-1)
    position = _integer(position, 'position', minimum=1, maximum=layout.capacity-1)
    if len(cache) != len(layout.cache_names):
        raise ValueError('Wrong number of cache tensors')
    mask = np.where(np.arange(layout.capacity) <= position, np.float32(0), np.finfo(np.float32).min)
    result = {'input_ids': np.array([[token]], np.int64),
              'attention_mask': mask.reshape(1, 1, 1, layout.capacity),
              'position_ids': np.array([[position]], np.int64)}
    for name, array in zip(layout.cache_names, cache):
        _fp32_finite(array, name)
        if array.shape != layout.shape:
            raise ValueError(f'Invalid cache shape for {name}')
        result['past.'+name] = array
    return result


def cached_logits(logits, vocab_size):
    _fp32_finite(logits, 'logits')
    if logits.shape != (1, 1, vocab_size):
        raise ValueError('Cached logits must be [1,1,vocab]')
    return logits[0, 0].copy()
