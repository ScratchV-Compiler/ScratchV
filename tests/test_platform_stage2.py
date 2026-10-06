"""Full Stage-2 (30 data point) verification of the size-independent
``--platform-asm`` kernels for the three contest problems.

Each problem's 10 data-point shapes (from ``stage2数据点.md`` v1.2) are
generated on the fly, the emitted ``.s`` is executed by the tiny RV32IM
interpreter from ``test_platform_kernels.py``, and the result is compared
against a NumPy reference. This is the end-to-end check that one
size-independent ``.s`` per problem covers every data point.

Skipped by default: the largest SpMM points take tens of seconds each under
the pure-Python interpreter (~3 min total). Run explicitly with::

    SCRATCHV_PLATFORM_STAGE2=1 PYTHONPATH=. python3.11 \
        -m pytest tests/test_platform_stage2.py -q
"""

import contextlib
import importlib.util
import io
import os
import pathlib

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SCRATCHV_PLATFORM_STAGE2") != "1",
    reason="set SCRATCHV_PLATFORM_STAGE2=1 to run the full 30-point "
           "platform-kernel check (~3 min)",
)

_here = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_tpk", _here / "test_platform_kernels.py")
_tpk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tpk)

_s32 = _tpk._s32
Q = 65536
_MASK = 0xFFFFFFFF

# ── Stage-2 data-point specs (stage2数据点.md v1.2) ─────────────────────────
FWHT = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
CONV = [(1, 3, 16, 16, 3, 8), (1, 8, 16, 16, 3, 16), (1, 8, 8, 8, 3, 16),
        (1, 8, 12, 12, 3, 16), (1, 6, 12, 12, 3, 12), (1, 3, 16, 16, 5, 8),
        (1, 4, 12, 12, 5, 8), (1, 8, 8, 8, 3, 8), (1, 8, 6, 6, 3, 16),
        (1, 4, 8, 8, 3, 8)]
SPMM = [(512, 512, 1, 0.90), (1024, 1024, 1, 0.95), (256, 512, 4, 0.85),
        (1024, 1024, 1, 0.98), (1280, 128, 16, 0.99), (512, 1024, 1, 0.90),
        (256, 256, 4, 0.80), (256, 2048, 1, 0.95), (512, 512, 2, 0.70),
        (256, 256, 8, 0.50)]


def _q(a):
    return np.trunc(np.asarray(a, np.float64) * Q).astype(np.int64)


def _emit(model, tmp_path):
    import onnx
    from scratchv.standalone.onnx_to_riscv_standalone import (
        ONNXModel, CNNRISCVGenerator, MemoryPlan, emit_platform_asm)
    path = tmp_path / "m.onnx"
    onnx.save(model, str(path))
    m = ONNXModel.from_file(str(path))
    mem = MemoryPlan()
    for vi in m.inputs:
        n = 1
        for d in m.get_shape(vi.name):
            n *= d
        mem.alloc_workspace(vi.name, n)
    wd = mem.layout_weights(m.initializers)
    gen = CNNRISCVGenerator(m, mem)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return emit_platform_asm(gen, wd, m)


def _run(asm, regs_init, mem, max_steps=400_000_000):
    prog, labels = _tpk._parse(asm)
    regs = [0] * 32
    for i, v in regs_init.items():
        regs[i] = v
    _tpk._run(prog, labels, regs, mem, max_steps=max_steps)


# ── builders ──────────────────────────────────────────────────────────────
def _fwht_model(n):
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, n])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, n])
    node = helper.make_node("Fwht", ["X"], ["Y"], domain="org.scratchv")
    return helper.make_model(
        helper.make_graph([node], "f", [x], [y]),
        opset_imports=[helper.make_opsetid("org.scratchv", 1),
                       helper.make_opsetid("", 13)])


def _conv_model(N, C, H, W, K, Cout):
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [N, C, H, W])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [N, Cout, H, W])
    rng = np.random.default_rng(abs(hash((N, C, H, W, K, Cout))) % 2**32)
    wt = helper.make_tensor(
        "w", TensorProto.FLOAT, [Cout, C, K, K],
        rng.normal(0, 0.3, size=Cout * C * K * K).astype(np.float32).ravel().tolist())
    bs = helper.make_tensor(
        "b", TensorProto.FLOAT, [Cout],
        rng.normal(0, 0.3, size=Cout).astype(np.float32).ravel().tolist())
    pad = K // 2
    node = helper.make_node("Conv", ["x", "w", "b"], ["y"],
                            pads=[pad, pad, pad, pad])
    return helper.make_model(
        helper.make_graph([node], "c", [x], [y], [wt, bs]),
        opset_imports=[helper.make_opsetid("", 13)])


def _spmm_model(M, K, N, dens, seed):
    from onnx import TensorProto, helper
    rng = np.random.default_rng(seed)
    mask = rng.random((M, K)) < dens
    A = (mask * rng.uniform(-0.4, 0.4, (M, K))).astype(np.float32)
    vals, col, row = [], [], [0]
    for i in range(M):
        for j in np.nonzero(A[i])[0]:
            col.append(int(j)); vals.append(float(A[i, j]))
        row.append(len(col))
    B = rng.uniform(-0.4, 0.4, (K, N)).astype(np.float32)
    x = helper.make_tensor_value_info("B", TensorProto.FLOAT, [K, N])
    y = helper.make_tensor_value_info("C", TensorProto.FLOAT, [M, N])
    v = helper.make_tensor("values", TensorProto.FLOAT, [len(vals)], vals)
    c = helper.make_tensor("col", TensorProto.INT32, [len(col)], col)
    r = helper.make_tensor("rowptr", TensorProto.INT32, [M + 1], row)
    node = helper.make_node("SpmmCsr", ["values", "col", "rowptr", "B"], ["C"])
    return helper.make_model(
        helper.make_graph([node], "s", [x], [y], [v, c, r]),
        opset_imports=[helper.make_opsetid("", 13)])


# ── reference helpers (per-product >>16 then accumulate, matching kernel) ──
def _fwht_ref(x):
    a = list(x)
    n = len(a)
    length = 1
    while length < n:
        for i in range(0, n, 2 * length):
            for j in range(length):
                u, v = a[i + j], a[i + j + length]
                a[i + j], a[i + j + length] = _s32(u + v), _s32(u - v)
        length <<= 1
    return a


def _conv_ref(xq, wq, bq, pad, K):
    C, H, W = xq.shape
    Cout, _, _, _ = wq.shape
    Hout, Wout = H, W
    xp = np.zeros((C, H + 2 * pad, W + 2 * pad), np.int64)
    xp[:, pad:pad + H, pad:pad + W] = xq
    acc = np.zeros((Cout, Hout, Wout), np.int64)
    for ic in range(C):
        for kh in range(K):
            for kw in range(K):
                patch = xp[ic, kh:kh + Hout, kw:kw + Wout]
                acc += (patch[None, :, :] * wq[:, ic, kh, kw][:, None, None]) >> 16
    acc += bq[:, None, None]
    return acc.reshape(-1)


def _spmm_ref(M, K, N, vq, col, row, Bq):
    rowid = np.repeat(np.arange(M), np.diff(row))
    contrib = (vq[:, None] * Bq[col]) >> 16
    C = np.zeros((M, N), np.int64)
    np.add.at(C, rowid, contrib)
    return C.reshape(-1)


# ── FWHT ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("n", FWHT, ids=[f"n{n}" for n in FWHT])
def test_stage2_fwht(tmp_path, n):
    asm = _emit(_fwht_model(n), tmp_path)
    rng = np.random.default_rng(0)
    x = [int(np.trunc(v * Q)) for v in rng.uniform(-0.4, 0.4, n)]
    mem = bytearray(0x200000)
    inp, out = 0x10000, 0x80000
    for i, v in enumerate(x):
        mem[inp + 4 * i:inp + 4 * i + 4] = int(v & _MASK).to_bytes(4, "little")
    _run(asm, {10: inp, 11: out, 12: n}, mem)
    got = [_s32(int.from_bytes(mem[out + 4 * i:out + 4 * i + 4], "little"))
           for i in range(n)]
    assert got == _fwht_ref(x)


# ── Conv ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cfg", CONV, ids=[f"c{i}" for i in range(1, 11)])
def test_stage2_conv(tmp_path, cfg):
    N, C, H, W, K, Cout = cfg
    model = _conv_model(*cfg)
    asm = _emit(model, tmp_path)
    from onnx import numpy_helper
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    wq = _q(inits["w"])
    bq = _q(inits["b"])
    pad = K // 2
    rng = np.random.default_rng(0)
    xq = _q(rng.uniform(-0.4, 0.4, (N, C, H, W)))
    mem = bytearray(0x200000)
    w = lambda a, v: mem.__setitem__(slice(a, a + 4), int(v & _MASK).to_bytes(4, "little"))
    r32 = lambda a: _s32(int.from_bytes(mem[a:a + 4], "little"))
    inp, out, wp, bp, par = 0x10000, 0x80000, 0x100000, 0x120000, 0x140000
    for i, v in enumerate(xq.reshape(-1)):
        w(inp + 4 * i, int(v))
    for i, v in enumerate(wq.reshape(-1)):
        w(wp + 4 * i, int(v))
    for i, v in enumerate(bq):
        w(bp + 4 * i, int(v))
    for off, val in enumerate([N, C, H, W, Cout, K, pad, 1, H, W, wp, bp]):
        w(par + 4 * off, int(val))
    _run(asm, {10: inp, 11: out, 12: par}, mem)
    got = [r32(out + 4 * i) for i in range(N * Cout * H * W)]
    ref = _conv_ref(xq[0], wq, bq, pad, K)
    assert got == list(ref)


# ── SpMM ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cfg", SPMM, ids=[f"s{i}" for i in range(1, 11)])
def test_stage2_spmm(tmp_path, cfg):
    M, K, N, dens = cfg
    model = _spmm_model(M, K, N, dens, SPMM.index(cfg) + 1)
    asm = _emit(model, tmp_path)
    from onnx import numpy_helper
    d = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    vq = _q(d["values"])
    col = d["col"]
    row = d["rowptr"]
    rng = np.random.default_rng(1)
    Bq = _q(rng.uniform(-0.4, 0.4, (K, N)))
    mem = bytearray(0x4000000)
    w = lambda a, v: mem.__setitem__(slice(a, a + 4), int(v & _MASK).to_bytes(4, "little"))
    r32 = lambda a: _s32(int.from_bytes(mem[a:a + 4], "little"))
    block, va, ca, ra, ba, out, par = (0x1000, 0x100000, 0x600000, 0xB00000,
                                       0x1000000, 0x1100000, 0x1200000)
    w(block, va); w(block + 4, ca); w(block + 8, ra)
    for i, v in enumerate(vq):
        w(va + 4 * i, int(v))
    for i, v in enumerate(col):
        w(ca + 4 * i, int(v))
    for i, v in enumerate(row):
        w(ra + 4 * i, int(v))
    for kk in range(K):
        for nn in range(N):
            w(ba + 4 * (kk * N + nn), int(Bq[kk][nn]))
    for off, val in enumerate([M, K, N, len(vq), ba]):
        w(par + 4 * off, int(val))
    _run(asm, {10: block, 11: out, 12: par}, mem)
    got = [r32(out + 4 * (i * N + nn)) for i in range(M) for nn in range(N)]
    ref = _spmm_ref(M, K, N, vq, col, row, Bq)
    assert got == list(ref)
