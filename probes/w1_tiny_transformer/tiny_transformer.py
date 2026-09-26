#!/usr/bin/env python3
"""Build a 2-layer Qwen3-shaped Transformer directly as an ONNX graph.

No torch: the graph is assembled with onnx.helper and constant tensors are
baked in as initializers. That keeps the probe runnable anywhere `onnx` is
installed, and — more usefully for a probe — makes the op set explicit, which
is exactly what we need to compare against ScratchV's ONNX parser.

Architecture, mirroring Qwen3 (all sizes tiny so the probe iterates fast):

    pre-norm, RMSNorm (no mean subtraction)
    RoPE, partial — only the first `rotary_dim` of head_dim is rotated
    GQA — num_kv_heads < num_heads, KV repeated to match
    SwiGLU MLP — down(silu(gate(x)) * up(x))
    tied embeddings — the LM head is the transpose of the embedding table
    additive causal mask, supplied as an input

Fixed shapes throughout: [1, 256]. No dynamic axes, no KV cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17


@dataclass
class Config:
    vocab_size: int = 128
    hidden: int = 64
    layers: int = 2
    heads: int = 4
    kv_heads: int = 2
    head_dim: int = 16
    intermediate: int = 192
    seq: int = 256
    rope_theta: float = 10000.0
    # Fraction of head_dim that RoPE rotates. Qwen3 uses partial RoPE; 0.5 on a
    # 16-wide head rotates 8 dims, enough to exercise the slice/concat path.
    partial_rotary: float = 0.5
    eps: float = 1e-6
    seed: int = 0

    @property
    def q_dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def rotary_dim(self) -> int:
        d = int(self.head_dim * self.partial_rotary)
        return d - (d % 2)          # RoPE rotates pairs

    @property
    def kv_group(self) -> int:
        return self.heads // self.kv_heads


class _Builder:
    """Accumulates nodes and initializers with unique names."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.n = 0

    def name(self, hint: str) -> str:
        self.n += 1
        return f"{hint}_{self.n}"

    def init(self, hint: str, array: np.ndarray) -> str:
        nm = self.name(hint)
        self.inits.append(numpy_helper.from_array(array.astype(np.float32), nm))
        return nm

    def op(self, op_type: str, inputs: list[str], hint: str,
           out_name: str | None = None, **attrs) -> str:
        """out_name pins the tensor name — graph outputs must match exactly,
        otherwise the checker reports them as produced by no node."""
        out = out_name or self.name(hint)
        self.nodes.append(helper.make_node(op_type, inputs, [out], name=out, **attrs))
        return out

    # ── composites ─────────────────────────────────────────────────────────

    def ints(self, hint: str, values: list[int]) -> str:
        """An int64 initializer — what opset 13+ wants for Slice/Unsqueeze args."""
        nm = self.name(hint)
        self.inits.append(numpy_helper.from_array(np.array(values, dtype=np.int64), nm))
        return nm

    def slice_(self, x: str, hint: str, start: int, end: int, axis: int = -1) -> str:
        return self.op(
            "Slice",
            [x, self.ints(f"{hint}_s", [start]),
             self.ints(f"{hint}_e", [end]),
             self.ints(f"{hint}_a", [axis])],
            f"{hint}_out",
        )


    def rms_norm(self, x: str, weight: np.ndarray, hint: str) -> str:
        """x / sqrt(mean(x^2) + eps) * weight   — no mean subtraction."""
        w = self.init(f"{hint}_w", weight)
        sq = self.op("Mul", [x, x], f"{hint}_sq")
        mean = self.op("ReduceMean", [sq], f"{hint}_mean", axes=[-1], keepdims=1)
        eps = self.init(f"{hint}_eps", np.array(self.cfg.eps))
        add = self.op("Add", [mean, eps], f"{hint}_add")
        root = self.op("Sqrt", [add], f"{hint}_sqrt")
        div = self.op("Div", [x, root], f"{hint}_div")
        return self.op("Mul", [div, w], f"{hint}_out")

    def linear(self, x: str, w: np.ndarray, b: np.ndarray, hint: str) -> str:
        wi = self.init(f"{hint}_W", w)
        bi = self.init(f"{hint}_b", b)
        mm = self.op("MatMul", [x, wi], f"{hint}_mm")
        return self.op("Add", [mm, bi], f"{hint}_out")

    def rope(self, x: str, cos: np.ndarray, sin: np.ndarray, hint: str) -> str:
        """Partial rotary embedding over the last axis.

        Rotates the first `rotary_dim` dims (as pairs) and passes the rest
        through unchanged. cos/sin are [seq, rotary_dim] constants.
        """
        cfg = self.cfg
        d = cfg.rotary_dim
        if d == 0:
            return x

        cos_i = self.init(f"{hint}_cos", cos)
        sin_i = self.init(f"{hint}_sin", sin)

        half = d // 2
        # rotate_half: [-x[..., half:d], x[..., 0:half]] over the rotary part
        front = self.slice_(x, f"{hint}_front", 0, half)
        back = self.slice_(x, f"{hint}_back", half, d)
        neg = self.op("Neg", [back], f"{hint}_neg")
        rot = self.op("Concat", [neg, front], f"{hint}_rot", axis=-1)

        rot_part = self.slice_(x, f"{hint}_rp", 0, d)
        a = self.op("Mul", [rot_part, cos_i], f"{hint}_a")
        b = self.op("Mul", [rot, sin_i], f"{hint}_b")
        rotated = self.op("Add", [a, b], f"{hint}_rotated")

        if d == cfg.head_dim:
            return rotated
        # the unrotated tail passes through, concatenated back on
        rest = self.slice_(x, f"{hint}_rest", d, cfg.head_dim)
        return self.op("Concat", [rotated, rest], f"{hint}_out", axis=-1)

    def repeat_kv(self, x: str, hint: str) -> str:
        """[1, kv_heads, seq, head_dim] -> [1, heads, seq, head_dim]."""
        g = self.cfg.kv_group
        if g == 1:
            return x
        # Unsqueeze an axis then Expand along it, then reshape back.
        un = self.op("Unsqueeze", [x, self.ints(f"{hint}_ax", [2])], f"{hint}_un")
        shape = self.ints(f"{hint}_shape",
                          [1, self.cfg.kv_heads, g, self.cfg.seq, self.cfg.head_dim])
        exp = self.op("Expand", [un, shape], f"{hint}_exp")
        target = self.ints(f"{hint}_tgt",
                           [1, self.cfg.heads, self.cfg.seq, self.cfg.head_dim])
        return self.op("Reshape", [exp, target], f"{hint}_out")


def rope_tables(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin of shape [seq, rotary_dim] for absolute positions 0..seq-1."""
    d = cfg.rotary_dim
    inv = 1.0 / (cfg.rope_theta ** (np.arange(0, d, 2) / d))     # [d/2]
    pos = np.arange(cfg.seq, dtype=np.float64)                    # [seq]
    ang = np.outer(pos, inv)                                      # [seq, d/2]
    emb = np.concatenate([ang, ang], axis=-1)                     # [seq, d]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def weights(cfg: Config) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)

    def r(*shape, scale=0.02):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    w: dict[str, np.ndarray] = {"embed": r(cfg.vocab_size, cfg.hidden, scale=0.05)}
    for i in range(cfg.layers):
        p = f"L{i}"
        w[f"{p}.ln1"] = np.ones(cfg.hidden, dtype=np.float32)
        w[f"{p}.ln2"] = np.ones(cfg.hidden, dtype=np.float32)
        w[f"{p}.Wq"] = r(cfg.hidden, cfg.q_dim)
        w[f"{p}.bq"] = np.zeros(cfg.q_dim, dtype=np.float32)
        w[f"{p}.Wk"] = r(cfg.hidden, cfg.kv_dim)
        w[f"{p}.bk"] = np.zeros(cfg.kv_dim, dtype=np.float32)
        w[f"{p}.Wv"] = r(cfg.hidden, cfg.kv_dim)
        w[f"{p}.bv"] = np.zeros(cfg.kv_dim, dtype=np.float32)
        w[f"{p}.Wo"] = r(cfg.q_dim, cfg.hidden)
        w[f"{p}.bo"] = np.zeros(cfg.hidden, dtype=np.float32)
        w[f"{p}.Wgate"] = r(cfg.hidden, cfg.intermediate)
        w[f"{p}.Wup"] = r(cfg.hidden, cfg.intermediate)
        w[f"{p}.Wdown"] = r(cfg.intermediate, cfg.hidden)
    w["ln_f"] = np.ones(cfg.hidden, dtype=np.float32)
    return w


def build(cfg: Config | None = None) -> tuple[onnx.ModelProto, dict[str, np.ndarray]]:
    cfg = cfg or Config()
    w = weights(cfg)
    cos, sin = rope_tables(cfg)
    b = _Builder(cfg)

    embed = b.init("embed", w["embed"])

    h = b.op("Gather", [embed, "input_ids"], "embed_out", axis=0)   # [1,seq,hidden]

    for i in range(cfg.layers):
        p = f"L{i}"
        n = b.rms_norm(h, w[f"{p}.ln1"], f"{p}_ln1")

        q = b.linear(n, w[f"{p}.Wq"], w[f"{p}.bq"], f"{p}_q")       # [1,seq,q_dim]
        k = b.linear(n, w[f"{p}.Wk"], w[f"{p}.bk"], f"{p}_k")       # [1,seq,kv_dim]
        v = b.linear(n, w[f"{p}.Wv"], w[f"{p}.bv"], f"{p}_v")

        # Split into heads first: RoPE rotates within head_dim, so it has to run
        # on a tensor whose last axis *is* head_dim. Applying it to the packed
        # [1,seq,q_dim] would slice the wrong axis.
        shp_q = b.ints(f"{p}_qshape", [1, cfg.seq, cfg.heads, cfg.head_dim])
        qh = b.op("Reshape", [q, shp_q], f"{p}_qresh")
        shp_kv = b.ints(f"{p}_kvshape", [1, cfg.seq, cfg.kv_heads, cfg.head_dim])
        kh = b.op("Reshape", [k, shp_kv], f"{p}_kresh")
        vh = b.op("Reshape", [v, shp_kv], f"{p}_vresh")

        perm = [0, 2, 1, 3]                                          # -> [1,H,seq,d]
        qh = b.op("Transpose", [qh], f"{p}_qt", perm=perm)
        kh = b.op("Transpose", [kh], f"{p}_kt", perm=perm)
        vh = b.op("Transpose", [vh], f"{p}_vt", perm=perm)

        qh = b.rope(qh, cos, sin, f"{p}_qrope")
        kh = b.rope(kh, cos, sin, f"{p}_krope")

        kh = b.repeat_kv(kh, f"{p}_krepeat")
        vh = b.repeat_kv(vh, f"{p}_vrepeat")

        kt = b.op("Transpose", [kh], f"{p}_ktt", perm=[0, 1, 3, 2])  # [1,H,d,seq]
        scores = b.op("MatMul", [qh, kt], f"{p}_scores")             # [1,H,seq,seq]

        scale = b.init(f"{p}_scale", np.array(1.0 / np.sqrt(cfg.head_dim), dtype=np.float32))
        scores = b.op("Mul", [scores, scale], f"{p}_scaled")
        scores = b.op("Add", [scores, "attention_mask"], f"{p}_masked")

        probs = b.op("Softmax", [scores], f"{p}_softmax", axis=-1)
        ctx = b.op("MatMul", [probs, vh], f"{p}_ctx")                # [1,H,seq,d]

        ct = b.op("Transpose", [ctx], f"{p}_ctt", perm=[0, 2, 1, 3])
        shp_o = b.ints(f"{p}_oshape", [1, cfg.seq, cfg.q_dim])
        o = b.op("Reshape", [ct, shp_o], f"{p}_o")

        proj = b.linear(o, w[f"{p}.Wo"], w[f"{p}.bo"], f"{p}_wo")
        h = b.op("Add", [h, proj], f"{p}_resid1")

        n2 = b.rms_norm(h, w[f"{p}.ln2"], f"{p}_ln2")
        gate = b.linear(n2, w[f"{p}.Wgate"], np.zeros(cfg.intermediate, np.float32), f"{p}_gate")
        up = b.linear(n2, w[f"{p}.Wup"], np.zeros(cfg.intermediate, np.float32), f"{p}_up")
        sig = b.op("Sigmoid", [gate], f"{p}_sig")
        act = b.op("Mul", [sig, gate], f"{p}_silu")                  # silu = sigmoid(x)*x
        act = b.op("Mul", [act, up], f"{p}_swiglu")
        down = b.linear(act, w[f"{p}.Wdown"], np.zeros(cfg.hidden, np.float32), f"{p}_down")
        h = b.op("Add", [h, down], f"{p}_resid2")

    h = b.rms_norm(h, w["ln_f"], "ln_f")

    # Tied embeddings: the LM head is the transpose of the embedding table.
    embed_t = b.op("Transpose", [embed], "embed_T", perm=[1, 0])
    logits = b.op("MatMul", [h, embed_t], "logits", out_name="logits")

    graph = helper.make_graph(
        b.nodes,
        "tiny_transformer_2l",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64,
                                          [1, cfg.seq]),
            helper.make_tensor_value_info("attention_mask", TensorProto.FLOAT,
                                          [1, 1, cfg.seq, cfg.seq]),
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT,
                                       [1, cfg.seq, cfg.vocab_size])],
        initializer=b.inits,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="scratchv-w1-probe",
    )
    model.ir_version = 9          # keep it loadable by onnxruntime 1.x
    onnx.checker.check_model(model)
    return model, w


def causal_mask(cfg: Config) -> np.ndarray:
    """Additive mask, [1,1,seq,seq]: 0 where attended, -1e9 where masked."""
    m = np.triu(np.full((cfg.seq, cfg.seq), -1e9, dtype=np.float32), k=1)
    return m.reshape(1, 1, cfg.seq, cfg.seq)
