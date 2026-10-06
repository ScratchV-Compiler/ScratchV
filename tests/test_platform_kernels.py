"""Size-independent platform kernels: FWHT and CSR SpMM.

The generated ``.s`` is executed by a tiny in-test RV32IM interpreter (only the
instruction subset the kernels use), so the kernels' logic is verified without
a RISC-V toolchain.
"""

import io
import contextlib
import re

import numpy as np
import pytest


# ── Minimal RV32IM interpreter (subset used by the kernels) ────────────────
_RN = {"zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4, "t0": 5, "t1": 6,
       "t2": 7, "s0": 8, "s1": 9, "a0": 10, "a1": 11, "a2": 12, "a3": 13,
       "a4": 14, "a5": 15, "a6": 16, "a7": 17, "s2": 18, "s3": 19, "s4": 20,
       "s5": 21, "s6": 22, "s7": 23, "s8": 24, "s9": 25, "s10": 26, "s11": 27,
       "t3": 28, "t4": 29, "t5": 30, "t6": 31}


def _parse(asm):
    prog, labels = [], {}
    for raw in asm.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("."):
            continue
        if line.endswith(":"):
            labels[line[:-1]] = len(prog)
            continue
        prog.append(line)
    return prog, labels


def _s32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def _run(prog, labels, regs, mem, max_steps=10_000_000):
    pc = steps = 0
    while pc < len(prog) and steps < max_steps:
        steps += 1
        p = prog[pc].replace(",", " ").split()
        op, r = p[0], p[1:]
        nxt = pc + 1
        g = lambda i: regs[i]
        st = lambda i, v: regs.__setitem__(i, v & 0xFFFFFFFF)

        if op == "li":
            st(_RN[r[0]], int(r[1]))
        elif op == "mv":
            st(_RN[r[0]], g(_RN[r[1]]))
        elif op in ("lw", "sw"):
            off = int(r[1][:r[1].index("(")])
            bs = r[1][r[1].index("(") + 1:-1]
            addr = (g(_RN[bs]) + off) & 0xFFFFFFFF
            if op == "lw":
                st(_RN[r[0]], int.from_bytes(mem[addr:addr + 4], "little"))
            else:
                mem[addr:addr + 4] = int(g(_RN[r[0]])).to_bytes(4, "little")
        elif op == "add":
            st(_RN[r[0]], g(_RN[r[1]]) + g(_RN[r[2]]))
        elif op == "sub":
            st(_RN[r[0]], g(_RN[r[1]]) - g(_RN[r[2]]))
        elif op == "mul":
            st(_RN[r[0]], g(_RN[r[1]]) * g(_RN[r[2]]))
        elif op == "srai":
            st(_RN[r[0]], _s32(g(_RN[r[1]])) >> int(r[2]))
        elif op == "slli":
            st(_RN[r[0]], g(_RN[r[1]]) << int(r[2]))
        elif op == "addi":
            st(_RN[r[0]], g(_RN[r[1]]) + int(r[2]))
        elif op in ("blt", "bge", "beq", "bne"):
            a, b = _s32(g(_RN[r[0]])), _s32(g(_RN[r[1]]))
            cond = {"blt": a < b, "bge": a >= b,
                    "beq": a == b, "bne": a != b}[op]
            nxt = labels[r[2]] if cond else nxt
        elif op == "j":
            nxt = labels[r[0]]
        elif op == "ret":
            return steps
        else:
            raise AssertionError(f"unknown op {op}")
        pc = nxt
    return steps


def _asm_for(model_builder, tmp_path):
    import onnx
    from scratchv.standalone.onnx_to_riscv_standalone import (
        ONNXModel, CNNRISCVGenerator, MemoryPlan, emit_platform_asm,
    )
    path = tmp_path / "m.onnx"
    onnx.save(model_builder(), str(path))
    model = ONNXModel.from_file(str(path))
    mem = MemoryPlan()
    for vi in model.inputs:
        n = 1
        for d in model.get_shape(vi.name):
            n *= d
        mem.alloc_workspace(vi.name, n)
    wd = mem.layout_weights(model.initializers)
    gen = CNNRISCVGenerator(model, mem)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return emit_platform_asm(gen, wd, model)


def _fwht_model():
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])
    node = helper.make_node("Fwht", ["X"], ["Y"], domain="org.scratchv")
    return helper.make_model(
        helper.make_graph([node], "f", [x], [y]),
        opset_imports=[helper.make_opsetid("org.scratchv", 1),
                       helper.make_opsetid("", 13)])


def _spmm_model():
    from onnx import TensorProto, helper
    # empty weight arrays are fine; the platform kernel reads runtime memory
    x = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 2])
    y = helper.make_tensor_value_info("C", TensorProto.FLOAT, [4, 2])
    vals = helper.make_tensor("values", TensorProto.FLOAT, [4], [0.0] * 4)
    col = helper.make_tensor("col", TensorProto.INT32, [4], [0, 1, 2, 3])
    row = helper.make_tensor("rowptr", TensorProto.INT32, [5], [0, 1, 2, 3, 4])
    node = helper.make_node("SpmmCsr", ["values", "col", "rowptr", "B"], ["C"])
    return helper.make_model(
        helper.make_graph([node], "s", [x], [y], [vals, col, row]),
        opset_imports=[helper.make_opsetid("", 13)])


@pytest.mark.parametrize("n", [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
def test_platform_fwht_kernel_matches_reference(tmp_path, n):
    asm = _asm_for(_fwht_model, tmp_path)
    prog, labels = _parse(asm)

    Q = 65536
    rng = np.random.default_rng(0)
    x = [int(np.trunc(v * Q)) for v in rng.uniform(-0.4, 0.4, n)]
    mem = bytearray(0x200000)
    inp, out = 0x10000, 0x80000
    for i, v in enumerate(x):
        mem[inp + 4 * i:inp + 4 * i + 4] = int(v & 0xFFFFFFFF).to_bytes(4, "little")
    regs = [0] * 32
    regs[10], regs[11], regs[12] = inp, out, n
    _run(prog, labels, regs, mem)

    got = [_s32(int.from_bytes(mem[out + 4 * i:out + 4 * i + 4], "little"))
           for i in range(n)]
    a = [float(v) / Q for v in x]
    length = 1
    while length < n:
        for i in range(0, n, 2 * length):
            for j in range(length):
                u, v = a[i + j], a[i + j + length]
                a[i + j], a[i + j + length] = u + v, u - v
        length <<= 1
    ref = [int(np.trunc(v * Q)) for v in a]
    assert got == ref


def test_platform_spmm_kernel_matches_reference(tmp_path):
    asm = _asm_for(_spmm_model, tmp_path)
    prog, labels = _parse(asm)

    Q = 65536
    M, K, N = 4, 4, 2
    values = [0.25, -0.5, 0.75, 0.125]
    col = [0, 2, 1, 3]
    row = [0, 2, 2, 3, 4]
    rng = np.random.default_rng(1)
    B = rng.uniform(-0.4, 0.4, (K, N))
    vq = [int(np.trunc(v * Q)) for v in values]
    Bq = [[int(np.trunc(x * Q)) for x in r] for r in B]

    mem = bytearray(0x40000)
    w = lambda a, v: mem.__setitem__(slice(a, a + 4),
                                     int(v & 0xFFFFFFFF).to_bytes(4, "little"))
    r32 = lambda a: _s32(int.from_bytes(mem[a:a + 4], "little"))
    block, va, ca, ra, ba, out, par = (0x1000, 0x10000, 0x10100, 0x10200,
                                       0x10300, 0x10400, 0x11000)
    w(block, va); w(block + 4, ca); w(block + 8, ra)
    for i, v in enumerate(vq):
        w(va + 4 * i, v)
    for i, v in enumerate(col):
        w(ca + 4 * i, v)
    for i, v in enumerate(row):
        w(ra + 4 * i, v)
    for kk in range(K):
        for nn in range(N):
            w(ba + 4 * (kk * N + nn), Bq[kk][nn])
    w(par + 0, M); w(par + 4, K); w(par + 8, N); w(par + 12, len(vq)); w(par + 16, ba)

    regs = [0] * 32
    regs[10], regs[11], regs[12] = block, out, par
    _run(prog, labels, regs, mem)

    C = [[0] * N for _ in range(M)]
    for i in range(M):
        for j in range(row[i], row[i + 1]):
            a, kk = vq[j], col[j]
            for nn in range(N):
                prod = _s32((a * Bq[kk][nn]) & 0xFFFFFFFF) >> 16
                C[i][nn] = _s32(C[i][nn] + prod)
    got = [r32(out + 4 * (i * N + nn)) for i in range(M) for nn in range(N)]
    ref = [C[i][nn] for i in range(M) for nn in range(N)]
    assert got == ref


def _conv_model(cin=3, h=8, w=8, k=3, cout=8):
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, cin, h, w])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, cout, h, w])
    wt = helper.make_tensor("w", TensorProto.FLOAT, [cout, cin, k, k],
                            [0.0] * (cout * cin * k * k))
    b = helper.make_tensor("b", TensorProto.FLOAT, [cout], [0.0] * cout)
    pad = k // 2
    node = helper.make_node("Conv", ["x", "w", "b"], ["y"],
                            pads=[pad, pad, pad, pad])
    return helper.make_model(
        helper.make_graph([node], "c", [x], [y], [wt, b]),
        opset_imports=[helper.make_opsetid("", 13)])


@pytest.mark.parametrize("cin,h,w,k,cout", [(3, 8, 8, 3, 8), (3, 16, 16, 5, 8)])
def test_platform_conv_kernel_matches_reference(tmp_path, cin, h, w, k, cout):
    asm = _asm_for(lambda: _conv_model(cin, h, w, k, cout), tmp_path)
    prog, labels = _parse(asm)

    Q = 65536
    rng = np.random.default_rng(0)
    Nn, Cin, H, W = 1, cin, h, w
    Cout, K, pad = cout, k, k // 2
    Hout = Wout = (H + 2 * pad - K) + 1
    xin = rng.uniform(-0.4, 0.4, (Nn, Cin, H, W))
    wt = rng.uniform(-0.4, 0.4, (Cout, Cin, K, K))
    bias = rng.uniform(-0.4, 0.4, (Cout,))
    xq = np.trunc(xin * Q).astype(np.int64)
    wq = np.trunc(wt * Q).astype(np.int64)
    bq = np.trunc(bias * Q).astype(np.int64)

    mem = bytearray(0x800000)
    w = lambda a, v: mem.__setitem__(slice(a, a + 4),
                                     int(v & 0xFFFFFFFF).to_bytes(4, "little"))
    r32 = lambda a: _s32(int.from_bytes(mem[a:a + 4], "little"))
    inp, out, wp, bp, par = 0x10000, 0x80000, 0x100000, 0x120000, 0x140000

    def put(base, arr):
        for i, v in enumerate(np.asarray(arr).reshape(-1)):
            w(base + 4 * i, int(v))
    put(inp, xq)
    put(wp, wq)
    put(bp, bq)
    w(par + 0, Nn); w(par + 4, Cin); w(par + 8, H); w(par + 12, W)
    w(par + 16, Cout); w(par + 20, K); w(par + 24, pad); w(par + 28, 1)
    w(par + 32, Hout); w(par + 36, Wout); w(par + 40, wp); w(par + 44, bp)

    regs = [0] * 32
    regs[10], regs[11], regs[12] = inp, out, par
    _run(prog, labels, regs, mem)

    got = [r32(out + 4 * i) for i in range(Nn * Cout * Hout * Wout)]
    # reference: per-product arithmetic shift then 32-bit accumulate (matches kernel)
    xp = np.zeros((Cin, H + 2 * pad, W + 2 * pad), dtype=np.int64)
    xp[:, pad:pad + H, pad:pad + W] = xq[0]
    ref = []
    for oc in range(Cout):
        for oh in range(Hout):
            for ow in range(Wout):
                acc = int(bq[oc])
                for ic in range(Cin):
                    for kh in range(K):
                        for kw in range(K):
                            p = _s32((int(xp[ic, oh + kh, ow + kw])
                                      * int(wq[oc, ic, kh, kw])) & 0xFFFFFFFF) >> 16
                            acc = _s32(acc + p)
                ref.append(acc)
    assert got == ref
