#!/usr/bin/env python3
"""Independent numpy implementation of the same 2-layer Transformer.

Written from the architecture description, not from the ONNX graph, so that
agreeing with ONNX Runtime is real evidence rather than a tautology. If both
implementations share a misunderstanding this will not catch it — but it does
catch transcription errors in the graph, which is where the bugs have been.
"""

from __future__ import annotations

import numpy as np

from tiny_transformer import Config, causal_mask, rope_tables, weights


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * w


def rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, rotary_dim: int) -> np.ndarray:
    """x: [1, heads, seq, head_dim]; cos/sin: [seq, rotary_dim]."""
    d = rotary_dim
    half = d // 2
    xr = x[..., :d]
    front, back = xr[..., :half], xr[..., half:d]
    rot = np.concatenate([-back, front], axis=-1)
    rotated = xr * cos + rot * sin
    if d == x.shape[-1]:
        return rotated
    return np.concatenate([rotated, x[..., d:]], axis=-1)


def softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def forward(cfg: Config, ids: np.ndarray, mask: np.ndarray,
            w: dict[str, np.ndarray] | None = None) -> np.ndarray:
    w = w if w is not None else weights(cfg)
    cos, sin = rope_tables(cfg)
    cos, sin = cos[None, None], sin[None, None]        # broadcast over batch/head

    h = w["embed"][ids]                                # [1, seq, hidden]
    scale = 1.0 / np.sqrt(cfg.head_dim)

    for i in range(cfg.layers):
        p = f"L{i}"
        n = rms_norm(h, w[f"{p}.ln1"], cfg.eps)

        def split(mat, bias, nheads):
            y = n @ w[mat] + w[bias]
            y = y.reshape(1, cfg.seq, nheads, cfg.head_dim)
            return y.transpose(0, 2, 1, 3)             # [1, H, seq, d]

        qh = split(f"{p}.Wq", f"{p}.bq", cfg.heads)
        kh = split(f"{p}.Wk", f"{p}.bk", cfg.kv_heads)
        vh = split(f"{p}.Wv", f"{p}.bv", cfg.kv_heads)

        qh = rope(qh, cos, sin, cfg.rotary_dim)
        kh = rope(kh, cos, sin, cfg.rotary_dim)

        # GQA: each KV head serves `group` consecutive query heads
        kh = np.repeat(kh, cfg.kv_group, axis=1)
        vh = np.repeat(vh, cfg.kv_group, axis=1)

        scores = (qh @ kh.transpose(0, 1, 3, 2)) * scale + mask
        ctx = softmax(scores) @ vh                     # [1, H, seq, d]

        o = ctx.transpose(0, 2, 1, 3).reshape(1, cfg.seq, cfg.q_dim)
        h = h + o @ w[f"{p}.Wo"] + w[f"{p}.bo"]

        n2 = rms_norm(h, w[f"{p}.ln2"], cfg.eps)
        gate = n2 @ w[f"{p}.Wgate"]
        up = n2 @ w[f"{p}.Wup"]
        silu = (1.0 / (1.0 + np.exp(-gate))) * gate
        h = h + (silu * up) @ w[f"{p}.Wdown"]

    h = rms_norm(h, w["ln_f"], cfg.eps)
    return h @ w["embed"].T                            # tied LM head
