"""Topic 16 LLVM codegen correctness tests.

Covers SSA uniqueness, canonical loop CFG, constant/type legality, real
lowering of the 8 NN ops, plus ``llvm-as``/``lli`` integration (skipped when
the LLVM tools are not installed).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from scratchv.backend.llvm_codegen import (
    LLVMCodegen, LLVMCodegenError, SSANamer, _float_literal,
)
from scratchv.frontend.dsl_extended import ExtendedDSLParser
from scratchv.frontend.dsl_parser import DSLParser
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, OpCode

LLVM_AS = shutil.which("llvm-as")
LLI = shutil.which("lli")
requires_asm = pytest.mark.skipif(
    LLVM_AS is None, reason="llvm-as not installed"
)
requires_lli = pytest.mark.skipif(
    LLI is None, reason="lli not installed"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CNN_MODEL = PROJECT_ROOT / "models" / "graph" / "cnn.onnx"

_OPS = (
    "dot", "matmul", "gemm", "maxpool", "conv", "softmax", "gelu", "sigmoid",
)

_LOOP_HEADERS = {
    "dot": 1,
    "matmul": 3,
    "gemm": 3,
    "maxpool": 5,
    "conv": 6,
    "softmax": 3,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _defs(ir: str) -> list[str]:
    return re.findall(r"^\s*(%[A-Za-z0-9_.]+)\s*=", ir, re.M)


def _labels(ir: str) -> list[str]:
    return re.findall(r"^([A-Za-z0-9_.]+):", ir, re.M)


def _assemble(ir: str, tmp_path) -> subprocess.CompletedProcess:
    ll = tmp_path / "m.ll"
    ll.write_text(ir)
    return subprocess.run(
        [LLVM_AS, str(ll), "-o", str(tmp_path / "m.bc")],
        capture_output=True, text=True,
    )


def _run(ir: str, tmp_path) -> subprocess.CompletedProcess:
    asm = _assemble(ir, tmp_path)
    assert asm.returncode == 0, asm.stderr
    return subprocess.run(
        [LLI, str(tmp_path / "m.bc")], capture_output=True, text=True,
    )


def _param(builder, func, name, shape=None):
    val = builder.make_value(name)
    if shape:
        val.shape = tuple(shape)
    func.params.append(val)
    return val


def _build_gelu_sigmoid():
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    x = _param(b, f, "x")
    b.ret(b.sigmoid(b.gelu(x)))
    return b.program


def _build_dot(length: int = 4, shape=(4,)):
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    a = _param(b, f, "a", shape)
    c = _param(b, f, "b", shape)
    b.ret(b.dot(a, c, length))
    return b.program


def _build_matmul(m: int = 2, n: int = 2, k: int = 2):
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    a = _param(b, f, "a", (m, k))
    c = _param(b, f, "b", (k, n))
    res = b.matmul(a, c, m, n, k)
    res.shape = (m, n)
    b.ret(res)
    return b.program


def _build_matmul_scalar():
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    a = _param(b, f, "a")
    c = _param(b, f, "b")
    b.ret(b.matmul(a, c, 1, 1, 1))
    return b.program


def _build_gemm(m: int = 2, n: int = 2, k: int = 2, trans_b: bool = False):
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    a = _param(b, f, "a", (m, k))
    w_shape = (n, k) if trans_b else (k, n)
    w = _param(b, f, "w", w_shape)
    bias = _param(b, f, "bias", (n,))
    res = b.gemm(a, w, bias, trans_b=trans_b)
    res.shape = (m, n)
    b.ret(res)
    return b.program


def _build_gemm_scalar():
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    a = _param(b, f, "a")
    w = _param(b, f, "w")
    bias = _param(b, f, "bias")
    b.ret(b.gemm(a, w, bias))
    return b.program


def _build_maxpool(kernel: int = 2, stride: int = 1, shape=(1, 2, 2)):
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    x = _param(b, f, "x", shape)
    b.ret(b.maxpool(x, kernel, stride))
    return b.program


def _build_conv(out_channels=2, kernel=2, stride=1, padding=0,
                shape=(1, 1, 4, 4)):
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    x = _param(b, f, "x", shape)
    cin = shape[-3]
    h = shape[-2]
    k = kernel
    out_h = (h + 2 * padding - k) // stride + 1
    w = _param(b, f, "w", (out_channels, cin, kernel, kernel))
    bias = _param(b, f, "bias", (out_channels,))
    res = b.conv(x, w, bias, out_channels, kernel, stride, padding)
    res.shape = (1, out_channels, out_h, out_h)
    b.ret(res)
    return b.program


def _build_conv_scalar():
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    x = _param(b, f, "x", (1, 1, 1))
    w = _param(b, f, "w", (1, 1, 1))
    bias = _param(b, f, "bias", (1,))
    b.ret(b.conv(x, w, bias, 1, 1, 1, 0))
    return b.program


def _build_softmax(n: int):
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    x = _param(b, f, "x", (n,))
    res = b.softmax(x)
    res.shape = (n,)
    b.ret(res)
    return b.program


def _build_nested_for_accumulator():
    b = IRBuilder()
    b.new_function("kernel")
    b.new_block("entry")
    acc = b.alloca(1)
    b.store(acc, b.load_const(0.0))
    outer = b.for_loop(0, 3)
    b.for_loop(0, 3)
    current = b.load(acc)
    b.store(acc, b.add(current, outer))
    b.endfor()
    b.endfor()
    b.ret(b.load(acc))
    return b.program


# ---------------------------------------------------------------------------
# Structure tests (no external tools)
# ---------------------------------------------------------------------------

def test_gelu_sigmoid_ssa_definitions_are_unique():
    ir = LLVMCodegen(_build_gelu_sigmoid()).emit()
    defs = _defs(ir)
    assert len(defs) == len(set(defs)), "duplicate SSA definitions"
    assert "call float @tanhf" in ir
    assert "call float @expf" in ir
    assert "fneg float" in ir
    assert "fdiv float 1.0" in ir


def test_float_constants_use_hex_encoding():
    ir = LLVMCodegen(_build_gelu_sigmoid()).emit()
    assert "0x" in ir
    assert not re.search(r"\d\.\d+[eE][+-]?\d+", ir), "exponent literal in IR"


def test_int_constants_do_not_use_float_ops():
    b = IRBuilder()
    b.new_function("kernel")
    b.new_block("entry")
    const = b.load_const(3, dtype=DataType.INT32)
    b.ret(const)
    ir = LLVMCodegen(b.program).emit()
    assert "add i32 0, 3" in ir
    assert "fadd i32" not in ir


def test_nested_for_labels_unique_and_canonical_cfg():
    program = DSLParser().parse(
        "for i = 0, 3\n"
        "for j = 0, 3\n"
        "s = add(s, i)\n"
        "endfor\n"
        "endfor\n"
        "return s"
    )
    ir = LLVMCodegen(program).emit()
    labels = _labels(ir)
    assert len(labels) == len(set(labels)), "duplicate block label"
    headers = [label for label in labels if "_hdr" in label]
    bodies = [label for label in labels if "_bdy" in label]
    exits = [label for label in labels if "_ext" in label]
    assert len(headers) == 2
    assert len(bodies) == 2
    assert len(exits) == 2
    # preheader must branch to the header (not the body)
    first_branch = ir.index("br label %")
    first_header = min(ir.index(f"{label}:") for label in headers)
    assert first_branch < first_header
    assert "icmp slt i32" in ir
    assert "br i1" in ir
    assert "sitofp i32" in ir  # mixed float/i32 arithmetic is coerced


def _op_program(name: str):
    if name == "dot":
        return _build_dot()
    if name == "matmul":
        return _build_matmul()
    if name == "gemm":
        return _build_gemm()
    if name == "maxpool":
        return _build_maxpool()
    if name == "conv":
        return _build_conv()
    if name == "softmax":
        return _build_softmax(2)
    if name == "gelu":
        return _build_gelu_sigmoid()
    if name == "sigmoid":
        b = IRBuilder()
        f = b.new_function("kernel")
        b.new_block("entry")
        x = _param(b, f, "x")
        b.ret(b.sigmoid(x))
        return b.program
    raise AssertionError(name)


@pytest.mark.parametrize("name", _OPS)
def test_no_placeholder_lowering_text(name):
    ir = LLVMCodegen(_op_program(name)).emit()
    lowered = ir.lower()
    assert "placeholder" not in lowered
    assert "passthrough" not in lowered
    assert "unsupported" not in lowered
    assert "requires" not in lowered


@pytest.mark.parametrize("name", sorted(_LOOP_HEADERS))
def test_tensor_ops_emit_geps_macs_and_loops(name):
    ir = LLVMCodegen(_op_program(name)).emit()
    assert "getelementptr" in ir
    if name in ("dot", "matmul", "gemm", "conv"):
        assert "fmul" in ir
        assert "fadd" in ir
    else:
        assert "fcmp" in ir
    assert "icmp slt i32" in ir
    assert "br i1" in ir
    header_count = len([x for x in _labels(ir) if "_hdr" in x])
    assert header_count == _LOOP_HEADERS[name]


def test_conv_and_maxpool_have_full_loop_nesting():
    conv_ir = LLVMCodegen(_build_conv()).emit()
    pool_ir = LLVMCodegen(_build_maxpool()).emit()
    assert len([x for x in _labels(conv_ir) if "_hdr" in x]) == 6
    assert len([x for x in _labels(pool_ir) if "_hdr" in x]) == 5
    assert "conv_mac" in conv_ir
    assert "conv_skip" in conv_ir


def test_softmax_has_three_passes():
    ir = LLVMCodegen(_build_softmax(2)).emit()
    headers = [x for x in _labels(ir) if "_hdr" in x]
    assert any("sm_max" in x for x in headers)
    assert any("sm_sum" in x for x in headers)
    assert any("sm_div" in x for x in headers)
    assert "fcmp ogt" in ir
    assert "call float @expf" in ir


def test_target_triple_is_configurable():
    program = _build_dot()
    default_ir = LLVMCodegen(program).emit()
    assert "target triple" not in default_ir
    riscv_ir = LLVMCodegen(program, "riscv64-unknown-elf").emit()
    assert 'target triple = "riscv64-unknown-elf"' in riscv_ir


def test_ssanamer_sanitizes_and_tracks_definitions():
    assert SSANamer.sanitize("foo.bar") == "foo.bar"
    assert SSANamer.sanitize("1bad") == "v_1bad"
    assert SSANamer.sanitize("a/b") == "a_b"
    assert SSANamer.sanitize("") == "v_"
    namer = SSANamer()
    first = namer.fresh("x")
    second = namer.fresh("x")
    assert first != second
    namer.register_definition(first)
    assert namer.registered(first)
    with pytest.raises(LLVMCodegenError):
        namer.register_definition(first)


def test_unmatched_endfor_raises():
    b = IRBuilder()
    b.new_function("kernel")
    b.new_block("entry")
    b.endfor()
    with pytest.raises(LLVMCodegenError):
        LLVMCodegen(b.program).emit()


# ---------------------------------------------------------------------------
# llvm-as integration
# ---------------------------------------------------------------------------

@requires_asm
@pytest.mark.parametrize("name", _OPS)
def test_asm_all_op_programs(name, tmp_path):
    result = _assemble(LLVMCodegen(_op_program(name)).emit(), tmp_path)
    assert result.returncode == 0, result.stderr


@requires_asm
def test_asm_dsl_nested_for(tmp_path):
    program = DSLParser().parse(
        "for i = 0, 3\n"
        "for j = 0, 3\n"
        "s = add(s, i)\n"
        "endfor\n"
        "endfor\n"
        "return s"
    )
    result = _assemble(LLVMCodegen(program).emit(), tmp_path)
    assert result.returncode == 0, result.stderr


@requires_asm
def test_asm_extended_if_only(tmp_path):
    source = (
        "i = add(i, 1.0)\n"
        "if (i == 10.0):\n"
        "i = add(i, 1.0)\n"
        "else:\n"
        "i = sub(i, 1.0)\n"
        "endif\n"
        "return i"
    )
    program = ExtendedDSLParser().parse(source)
    ir = LLVMCodegen(program).emit()
    result = _assemble(ir, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "fcmp oeq float" in ir


@requires_asm
def test_asm_extended_while_with_return(tmp_path):
    source = (
        "acc = add(acc, 1.0)\n"
        "while (i < 10.0):\n"
        "acc = add(acc, 1.0)\n"
        "return acc\n"
        "endwhile\n"
        "return acc"
    )
    program = ExtendedDSLParser().parse(source)
    ir = LLVMCodegen(program).emit()
    result = _assemble(ir, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "fcmp olt float" in ir


@requires_asm
def test_while_reassigned_condition_raises():
    """A while loop whose condition cannot change must fail loudly.

    DSL variables have static SSA semantics: ``i = add(i, 1.0)`` defines a
    new value and never updates the slot the header reads, so the loop would
    spin forever. Regression for review F1 (while deadlock).
    """
    source = (
        "i = add(0.0, 0.0)\n"
        "s = add(0.0, 0.0)\n"
        "while (i < 10.0):\n"
        "s = add(s, i)\n"
        "i = add(i, 1.0)\n"
        "endwhile\n"
        "return s"
    )
    program = ExtendedDSLParser().parse(source)
    with pytest.raises(LLVMCodegenError, match="can never exit"):
        LLVMCodegen(program).emit()


@requires_asm
def test_asm_onnx_cnn(tmp_path):
    if not CNN_MODEL.exists():
        pytest.skip("cnn.onnx not found")
    try:
        from scratchv.frontend.onnx_parser import ONNXParser
    except ImportError:  # pragma: no cover
        pytest.skip("onnx package not installed")
    program = ONNXParser().parse(str(CNN_MODEL))
    ir = LLVMCodegen(program).emit()
    result = _assemble(ir, tmp_path)
    assert result.returncode == 0, result.stderr


@requires_asm
def test_asm_int_arithmetic_uses_integer_ops(tmp_path):
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    x = _param(b, f, "x")
    y = _param(b, f, "y")
    x.dtype = DataType.INT32
    y.dtype = DataType.INT32
    product = b.mul(x, y)
    product.dtype = DataType.INT32
    total = b.add(product, x)
    total.dtype = DataType.INT32
    b.ret(total)
    ir = LLVMCodegen(b.program).emit()
    assert "mul i32" in ir
    assert "add i32" in ir
    assert "fmul i32" not in ir
    assert "fadd i32" not in ir
    result = _assemble(ir, tmp_path)
    assert result.returncode == 0, result.stderr


@requires_asm
def test_asm_double_dtype_softmax(tmp_path):
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    x = _param(b, f, "x", (3,))
    x.dtype = DataType.FLOAT64
    res = b.softmax(x)
    res.shape = (3,)
    b.ret(res)
    ir = LLVMCodegen(b.program).emit()
    assert "call double @exp(" in ir
    assert "define double* @kernel(double* %x)" in ir
    result = _assemble(ir, tmp_path)
    assert result.returncode == 0, result.stderr


@requires_asm
def test_asm_tensor_operand_degenerates_to_first_element(tmp_path):
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    a = _param(b, f, "a", (4,))
    scalar = _param(b, f, "s")
    b.ret(b.add(a, scalar))
    ir = LLVMCodegen(b.program).emit()
    assert "load float, float* %a" in ir
    result = _assemble(ir, tmp_path)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# lli numerical tests (harness: append @main calling @kernel)
# ---------------------------------------------------------------------------

def _scalar_main(call_expr: str, expected: str) -> str:
    return textwrap.dedent(f"""\
    define i32 @main() {{
    entry:
      %r = {call_expr}
      %ok = fcmp oeq float %r, {expected}
      %rc = select i1 %ok, i32 0, i32 1
      ret i32 %rc
    }}
    """)


def _store_array(reg: str, values) -> str:
    lines = [
        f"  %{reg} = alloca float, i32 {len(values)}",
        f"  store float {values[0]}, float* %{reg}",
    ]
    for i in range(1, len(values)):
        lines.append(
            f"  %{reg}{i} = getelementptr float, float* %{reg}, i32 {i}"
        )
        lines.append(f"  store float {values[i]}, float* %{reg}{i}")
    return "\n".join(lines)


def _check_numeric(program, main_ir: str, tmp_path):
    ir = LLVMCodegen(program).emit() + "\n" + main_ir
    result = _run(ir, tmp_path)
    assert result.returncode == 0, result.stderr


def _kernel_call(ir: str, name: str = "kernel") -> tuple[str, str]:
    """Build a call expression matching the emitted signature."""
    m = re.search(rf"define (\S+) @{name}\(([^)]*)\)", ir)
    assert m, f"definition of @{name} not found"
    ret_ty = m.group(1)
    params = [p.strip() for p in m.group(2).split(",") if p.strip()]
    args = ", ".join(f"{p.rsplit(' ', 1)[0]} 0.0" for p in params)
    return f"call {ret_ty} @{name}({args})", ret_ty


def _rename_kernel(program):
    program.functions[0].name = "kernel"
    return program


def _compare_array(reg: str, values, eps: str | None = None) -> str:
    """Emit per-element equality (or tolerance) checks; return IR lines."""
    lines: list[str] = []
    conds: list[str] = []
    for i, expected in enumerate(values):
        if i == 0:
            ptr = f"%{reg}"
        else:
            ptr = f"%{reg}{i}"
            lines.append(
                f"  {ptr} = getelementptr float, float* %{reg}, i32 {i}"
            )
        lines.append(f"  %v{i} = load float, float* {ptr}")
        if eps is None:
            lines.append(f"  %c{i} = fcmp oeq float %v{i}, {expected}")
        else:
            expected_lit = _float_literal(float(expected))
            lines.append(f"  %d{i} = fsub float %v{i}, {expected_lit}")
            lines.append(f"  %ad{i} = fcmp ogt float %d{i}, 0.0")
            lines.append(f"  %nd{i} = fneg float %d{i}")
            lines.append(
                f"  %aabs{i} = select i1 %ad{i}, float %d{i}, float %nd{i}"
            )
            lines.append(f"  %c{i} = fcmp olt float %aabs{i}, "
                         f"{_float_literal(float(eps))}")
        conds.append(f"%c{i}")
    acc = conds[0]
    for i, cond in enumerate(conds[1:], start=1):
        lines.append(f"  %t{i} = and i1 {acc}, {cond}")
        acc = f"%t{i}"
    lines.append(f"  %rc = select i1 {acc}, i32 0, i32 1")
    lines.append("  ret i32 %rc")
    return "\n".join(lines)


def _scalar_program(program, expected: str) -> str:
    ir = LLVMCodegen(program).emit()
    call_expr, _ = _kernel_call(ir)
    return _scalar_main(call_expr, expected)


@requires_lli
def test_lli_gelu_sigmoid_zero(tmp_path):
    _check_numeric(
        _build_gelu_sigmoid(),
        _scalar_main("call float @kernel(float 0.0)", "0.5"),
        tmp_path,
    )


@requires_lli
def test_lli_nested_for_accumulator(tmp_path):
    _check_numeric(
        _build_nested_for_accumulator(),
        _scalar_main("call float @kernel()", "9.0"),
        tmp_path,
    )


@requires_lli
def test_lli_dot(tmp_path):
    body = "\n".join([
        _store_array("a", ["1.0", "2.0", "3.0", "4.0"]),
        _store_array("b", ["1.0", "1.0", "1.0", "1.0"]),
    ])
    main = textwrap.dedent("""\
    define i32 @main() {
    entry:
    %s
      %%r = call float @kernel(float* %%a, float* %%b)
      %%ok = fcmp oeq float %%r, 10.0
      %%rc = select i1 %%ok, i32 0, i32 1
      ret i32 %%rc
    }
    """ % body)
    _check_numeric(_build_dot(), main, tmp_path)


@requires_lli
def test_lli_matmul_scalar(tmp_path):
    _check_numeric(
        _build_matmul_scalar(),
        _scalar_main("call float @kernel(float 2.0, float 3.0)", "6.0"),
        tmp_path,
    )


@requires_lli
def test_lli_matmul_2x2(tmp_path):
    body = "\n".join([
        _store_array("a", ["1.0", "2.0", "3.0", "4.0"]),
        _store_array("b", ["1.0", "0.0", "0.0", "1.0"]),
    ])
    main = textwrap.dedent("""\
    define i32 @main() {
    entry:
    %s
      %%out = call float* @kernel(float* %%a, float* %%b)
      %%o0 = load float, float* %%out
      %%op1 = getelementptr float, float* %%out, i32 1
      %%o1 = load float, float* %%op1
      %%op2 = getelementptr float, float* %%out, i32 2
      %%o2 = load float, float* %%op2
      %%op3 = getelementptr float, float* %%out, i32 3
      %%o3 = load float, float* %%op3
      %%c0 = fcmp oeq float %%o0, 1.0
      %%c1 = fcmp oeq float %%o1, 2.0
      %%c2 = fcmp oeq float %%o2, 3.0
      %%c3 = fcmp oeq float %%o3, 4.0
      %%t1 = and i1 %%c0, %%c1
      %%t2 = and i1 %%c2, %%c3
      %%all = and i1 %%t1, %%t2
      %%rc = select i1 %%all, i32 0, i32 1
      ret i32 %%rc
    }
    """ % body)
    _check_numeric(_build_matmul(), main, tmp_path)


@requires_lli
def test_lli_gemm_scalar_with_bias(tmp_path):
    _check_numeric(
        _build_gemm_scalar(),
        _scalar_main(
            "call float @kernel(float 2.0, float 3.0, float 0.5)", "6.5"
        ),
        tmp_path,
    )


@requires_lli
def test_lli_maxpool_2x2(tmp_path):
    body = _store_array("x", ["1.0", "2.0", "3.0", "4.0"])
    main = textwrap.dedent("""\
    define i32 @main() {
    entry:
    %s
      %%r = call float @kernel(float* %%x)
      %%ok = fcmp oeq float %%r, 4.0
      %%rc = select i1 %%ok, i32 0, i32 1
      ret i32 %%rc
    }
    """ % body)
    _check_numeric(_build_maxpool(), main, tmp_path)


@requires_lli
def test_lli_conv_1x1x1(tmp_path):
    body = "\n".join([
        _store_array("x", ["3.0"]),
        _store_array("w", ["2.0"]),
        _store_array("bias", ["1.0"]),
    ])
    main = textwrap.dedent("""\
    define i32 @main() {
    entry:
    %s
      %%r = call float @kernel(float* %%x, float* %%w, float* %%bias)
      %%ok = fcmp oeq float %%r, 7.0
      %%rc = select i1 %%ok, i32 0, i32 1
      ret i32 %%rc
    }
    """ % body)
    _check_numeric(_build_conv_scalar(), main, tmp_path)


@requires_lli
def test_lli_softmax_pair(tmp_path):
    body = _store_array("x", ["0.0", "0.0"])
    main = textwrap.dedent("""\
    define i32 @main() {
    entry:
    %s
      %%out = call float* @kernel(float* %%x)
      %%o0 = load float, float* %%out
      %%op1 = getelementptr float, float* %%out, i32 1
      %%o1 = load float, float* %%op1
      %%c0 = fcmp oeq float %%o0, 0.5
      %%c1 = fcmp oeq float %%o1, 0.5
      %%both = and i1 %%c0, %%c1
      %%rc = select i1 %%both, i32 0, i32 1
      ret i32 %%rc
    }
    """ % body)
    _check_numeric(_build_softmax(2), main, tmp_path)


@requires_lli
def test_lli_softmax_single(tmp_path):
    body = _store_array("x", ["7.0"])
    main = textwrap.dedent("""\
    define i32 @main() {
    entry:
    %s
      %%out = call float* @kernel(float* %%x)
      %%o0 = load float, float* %%out
      %%ok = fcmp oeq float %%o0, 1.0
      %%rc = select i1 %%ok, i32 0, i32 1
      ret i32 %%rc
    }
    """ % body)
    _check_numeric(_build_softmax(1), main, tmp_path)


# ---------------------------------------------------------------------------
# F1 regression: DSL variable semantics are static SSA
# (design doc "known limitations" — values below are the documented result)
# ---------------------------------------------------------------------------

@requires_lli
def test_lli_dsl_nested_for_static_ssa_value(tmp_path):
    """DSL accumulation is unsupported: each write is a fresh SSA value.

    ``s`` inside the loop always reads the pre-loop value, so the result is
    the last inner iteration's ``s + i`` = 2.0, not 9.0. The correct
    accumulator form is the IRBuilder alloca/load/store program covered by
    ``test_lli_nested_for_accumulator``.
    """
    program = _rename_kernel(DSLParser().parse(
        "s = add(s, 0.0)\n"
        "for i = 0, 3\n"
        "for j = 0, 3\n"
        "s = add(s, i)\n"
        "endfor\n"
        "endfor\n"
        "return s"
    ))
    _check_numeric(program, _scalar_program(program, "2.0"), tmp_path)


@requires_lli
def test_lli_dsl_loop_carried_static_ssa_value(tmp_path):
    """Loop-carried ``x`` never updates: 3.0 + 1.0 = 4.0, not 5.0."""
    program = _rename_kernel(DSLParser().parse(
        "x = add(3.0, 0.0)\n"
        "for i = 0, 2\n"
        "x = add(x, 1.0)\n"
        "endfor\n"
        "return x"
    ))
    _check_numeric(program, _scalar_program(program, "4.0"), tmp_path)


def test_dsl_if_else_merge_reads_last_parsed_branch():
    """If/else merge loads the slot of the last *parsed* assignment.

    The DSL has no phi/variable identity, so the read after ``endif`` always
    resolves to the ``else`` assignment even when the ``then`` branch ran
    (the else slot is then uninitialized → undefined value). Locking the
    structural defect (regression for review F1, if/else merge).
    """
    program = ExtendedDSLParser().parse(
        "y = add(0.0, 0.0)\n"
        "if (1.0 < 2.0):\n"
        "y = add(1.0, 0.0)\n"
        "else:\n"
        "y = add(2.0, 0.0)\n"
        "endif\n"
        "return y"
    )
    ir = LLVMCodegen(program).emit()
    blocks: dict[str, str] = {}
    current = "entry"
    for line in ir.splitlines():
        if re.match(r"^\w+:$", line.strip()):
            current = line.strip()[:-1]
            blocks[current] = ""
        blocks.setdefault(current, "")
        blocks[current] += line + "\n"
    else_block = next(name for name in blocks if name.startswith("if_else"))
    merge_block = next(
        name for name, body in blocks.items() if re.search(r"ret float", body)
    )
    merge_slot = re.search(
        r"load float, float\* (%\S+)\s*\n\s*ret float", blocks[merge_block]
    ).group(1)
    else_slot = re.findall(
        r"store float %\S+, float\* (%\S+)", blocks[else_block]
    )[-1]
    assert merge_slot == else_slot
    then_block = next(name for name in blocks if name.startswith("if_then"))
    then_slot = re.findall(
        r"store float %\S+, float\* (%\S+)", blocks[then_block]
    )[-1]
    assert then_slot != merge_slot


# ---------------------------------------------------------------------------
# F2 regression: shaped destinations must lower elementwise or fail loudly
# ---------------------------------------------------------------------------

def _unary_tensor_program(op: str, values=(4,)):
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    x = _param(b, f, "x", values)
    result = getattr(b, op)(x)
    result.shape = tuple(values)
    b.ret(result)
    return b.program


def _binary_tensor_program(op: str, values=(4,)):
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    x = _param(b, f, "x", values)
    y = _param(b, f, "y", values)
    result = getattr(b, op)(x, y)
    result.shape = tuple(values)
    b.ret(result)
    return b.program


@requires_asm
@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
def test_asm_binary_ops_with_tensor_dest(op, tmp_path):
    program = _binary_tensor_program(op)
    ir = LLVMCodegen(program).emit()
    assert "define float* @kernel(float* %x, float* %y)" in ir
    result = _assemble(ir, tmp_path)
    assert result.returncode == 0, result.stderr


@requires_asm
@pytest.mark.parametrize("op", ["relu", "exp", "neg", "gelu", "sigmoid"])
def test_asm_unary_ops_with_tensor_dest(op, tmp_path):
    program = _unary_tensor_program(op)
    ir = LLVMCodegen(program).emit()
    assert "define float* @kernel(float* %x)" in ir
    assert "ret float* %" in ir
    result = _assemble(ir, tmp_path)
    assert result.returncode == 0, result.stderr


@requires_lli
def test_lli_tensor_dest_relu(tmp_path):
    program = _unary_tensor_program("relu")
    main = textwrap.dedent("""\
    define i32 @main() {
    entry:
    %s
      %%out = call float* @kernel(float* %%arr)
    %s
    }
    """ % (_store_array("arr", ["1.0", "-2.0", "3.0", "-4.0"]),
           _compare_array("out", ["1.0", "0.0", "3.0", "0.0"])))
    _check_numeric(program, main, tmp_path)


@requires_lli
def test_lli_tensor_dest_add(tmp_path):
    program = _binary_tensor_program("add")
    main = textwrap.dedent("""\
    define i32 @main() {
    entry:
    %s
    %s
      %%out = call float* @kernel(float* %%x, float* %%y)
    %s
    }
    """ % (_store_array("x", ["1.0", "2.0", "3.0"]),
           _store_array("y", ["10.0", "20.0", "30.0"]),
           _compare_array("out", ["11.0", "22.0", "33.0"])))
    _check_numeric(program, main, tmp_path)


@requires_lli
def test_lli_tensor_dest_gelu_and_neg(tmp_path):
    program = _unary_tensor_program("neg")
    main = textwrap.dedent("""\
    define i32 @main() {
    entry:
    %s
      %%out = call float* @kernel(float* %%arr)
    %s
    }
    """ % (_store_array("arr", ["1.5", "-2.5"]),
           _compare_array("out", ["-1.5", "2.5"])))
    _check_numeric(program, main, tmp_path)

    program = _unary_tensor_program("gelu")
    main = textwrap.dedent("""\
    define i32 @main() {
    entry:
    %s
      %%out = call float* @kernel(float* %%arr)
    %s
    }
    """ % (_store_array("arr", ["0.0", "0.0", "0.0"]),
           _compare_array("out", ["0.0", "0.0", "0.0"])))
    _check_numeric(program, main, tmp_path)


@requires_asm
def test_tensor_dest_mismatched_operands_raise():
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    x = _param(b, f, "x", (4,))
    y = _param(b, f, "y", (3,))
    result = b.add(x, y)
    result.shape = (4,)
    b.ret(result)
    with pytest.raises(LLVMCodegenError, match="destination needs"):
        LLVMCodegen(b.program).emit()


# ---------------------------------------------------------------------------
# F3 regression: scalar operands cannot feed multi-element tensor loops
# ---------------------------------------------------------------------------

def test_dot_scalar_operands_with_length_raise():
    program = DSLParser().parse("y = dot(a, b, len:4)\nreturn y")
    with pytest.raises(LLVMCodegenError, match="scalar operand"):
        LLVMCodegen(program).emit()


def test_matmul_scalar_operands_with_dims_raise():
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    a = _param(b, f, "a")
    c = _param(b, f, "b")
    b.ret(b.matmul(a, c, 2, 2, 2))
    with pytest.raises(LLVMCodegenError, match="scalar operand"):
        LLVMCodegen(b.program).emit()


def test_gemm_scalar_operands_with_dims_raise():
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    a = _param(b, f, "a")
    w = _param(b, f, "w")
    bias = _param(b, f, "bias")
    dest = b.make_value("gemm_out")
    b._emit(OpCode.GEMM, dest, [a, w, bias], M=2, K=2, N=2)
    b.ret(dest)
    with pytest.raises(LLVMCodegenError, match="scalar operand"):
        LLVMCodegen(b.program).emit()


@requires_lli
def test_lli_dot_scalar_one_element_degenerate(tmp_path):
    """needed == 1 still allows the documented 1-element degeneration."""
    _check_numeric(
        _build_matmul_scalar(),
        _scalar_main("call float @kernel(float 2.0, float 3.0)", "6.0"),
        tmp_path,
    )


# ---------------------------------------------------------------------------
# F4 regression: softmax must cover every row of a 2D+ input
# ---------------------------------------------------------------------------

@requires_lli
def test_lli_softmax_2d_all_rows(tmp_path):
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    x = _param(b, f, "x", (2, 2))
    res = b.softmax(x)
    res.shape = (2, 2)
    b.ret(res)
    main = textwrap.dedent("""\
    define i32 @main() {
    entry:
    %s
      %%out = call float* @kernel(float* %%arr)
    %s
    }
    """ % (_store_array("arr", ["0.0"] * 4),
           _compare_array("out", ["0.5"] * 4)))
    _check_numeric(b.program, main, tmp_path)


@requires_asm
def test_softmax_2d_has_row_loop_and_full_buffer(tmp_path):
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    x = _param(b, f, "x", (2, 3))
    res = b.softmax(x)
    res.shape = (2, 3)
    b.ret(res)
    ir = LLVMCodegen(b.program).emit()
    assert "alloca float, i32 6" in ir
    assert "sm_row_hdr" in ir
    result = _assemble(ir, tmp_path)
    assert result.returncode == 0, result.stderr


def test_softmax_non_last_axis_raises():
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    x = _param(b, f, "x", (4, 4))
    b.ret(b.softmax(x, axis=0))
    with pytest.raises(LLVMCodegenError, match="only axis=-1"):
        LLVMCodegen(b.program).emit()


@requires_lli
def test_lli_softmax_three_values_tolerance(tmp_path):
    body = _store_array("x", ["1.0", "2.0", "3.0"])
    main = textwrap.dedent("""\
    define i32 @main() {
    entry:
    %s
      %%out = call float* @kernel(float* %%x)
    %s
    }
    """ % (body, _compare_array(
        "out",
        ["0.09003057", "0.24472847", "0.66524096"],
        eps="0.000001",
    )))
    _check_numeric(_build_softmax(3), main, tmp_path)


# ---------------------------------------------------------------------------
# F5 regression: mixed void/value returns must fail loudly
# ---------------------------------------------------------------------------

def test_mixed_return_types_raise_dsl():
    program = DSLParser().parse("return x\ni = add(i, 1.0)")
    with pytest.raises(LLVMCodegenError, match="mixed with value return"):
        LLVMCodegen(program).emit()


def test_mixed_return_types_raise_extended():
    program = ExtendedDSLParser().parse(
        "i = add(i, 1.0)\nif (i < 3.0):\nreturn i\nendif\n"
    )
    with pytest.raises(LLVMCodegenError, match="mixed with value return"):
        LLVMCodegen(program).emit()


@requires_asm
def test_uniform_early_return_assembles(tmp_path):
    program = ExtendedDSLParser().parse(
        "i = add(i, 1.0)\nif (i < 3.0):\nreturn i\nendif\nreturn 0.0"
    )
    ir = LLVMCodegen(program).emit()
    assert "ret float" in ir
    assert "ret void" not in ir
    result = _assemble(ir, tmp_path)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# F7 regression: unlowered opcodes fail loudly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("opcode", [OpCode.TRANSPOSE, OpCode.CONCAT])
def test_unlowered_opcode_raises(opcode):
    b = IRBuilder()
    b.new_function("kernel")
    b.new_block("entry")
    a = b.make_value("a")
    a.shape = (2, 2)
    dest = b.make_value("unlowered")
    b._emit(opcode, dest, [a])
    b.ret(dest)
    with pytest.raises(LLVMCodegenError, match="not lowered"):
        LLVMCodegen(b.program).emit()


# ---------------------------------------------------------------------------
# F9 regression: broader lli numeric matrix
# ---------------------------------------------------------------------------

@requires_lli
def test_lli_conv_padding_stride_nonuniform(tmp_path):
    shape = (1, 1, 6, 6)
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    x = _param(b, f, "x", shape)
    w = _param(b, f, "w", (1, 1, 2, 2))
    bias = _param(b, f, "bias", (1,))
    res = b.conv(x, w, bias, 1, 2, 2, 1)
    res.shape = (1, 1, 4, 4)
    b.ret(res)
    values = [float(i + 1) for i in range(36)]
    expected = []
    for oh in range(4):
        for ow in range(4):
            acc = 0.5
            for kh in range(2):
                for kw in range(2):
                    ih = oh * 2 + kh - 1
                    iw = ow * 2 + kw - 1
                    if 0 <= ih < 6 and 0 <= iw < 6 and kh == kw:
                        acc += values[ih * 6 + iw]
            expected.append(repr(acc))
    main = textwrap.dedent("""\
    define i32 @main() {
    entry:
    %s
    %s
    %s
      %%out = call float* @kernel(float* %%x, float* %%w, float* %%bias)
    %s
    }
    """ % (_store_array("x", [str(v) for v in values]),
           _store_array("w", ["1.0", "0.0", "0.0", "1.0"]),
           _store_array("bias", ["0.5"]),
           _compare_array("out", expected)))
    _check_numeric(b.program, main, tmp_path)


@requires_lli
def test_lli_gemm_trans_b_nonsquare(tmp_path):
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    a = _param(b, f, "a", (2, 2))
    w = _param(b, f, "w", (3, 2))
    bias = _param(b, f, "bias", (3,))
    res = b.gemm(a, w, bias, trans_b=True)
    res.shape = (2, 3)
    b.ret(res)
    main = textwrap.dedent("""\
    define i32 @main() {
    entry:
    %s
    %s
    %s
      %%out = call float* @kernel(float* %%a, float* %%w, float* %%bias)
    %s
    }
    """ % (_store_array("a", ["1.0", "2.0", "3.0", "4.0"]),
           _store_array("w", ["1.0", "0.0", "0.0", "1.0", "1.0", "1.0"]),
           _store_array("bias", ["0.0", "1.0", "2.0"]),
           _compare_array("out", ["1.0", "3.0", "5.0", "3.0", "5.0", "9.0"])))
    _check_numeric(b.program, main, tmp_path)


@requires_lli
def test_lli_matmul_nonsquare(tmp_path):
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    a = _param(b, f, "a", (2, 3))
    c = _param(b, f, "b", (3, 2))
    res = b.matmul(a, c, 2, 2, 3)
    res.shape = (2, 2)
    b.ret(res)
    main = textwrap.dedent("""\
    define i32 @main() {
    entry:
    %s
    %s
      %%out = call float* @kernel(float* %%a, float* %%b)
    %s
    }
    """ % (_store_array("a", ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0"]),
           _store_array("b", ["1.0", "0.0", "0.0", "1.0", "1.0", "1.0"]),
           _compare_array("out", ["4.0", "5.0", "10.0", "11.0"])))
    _check_numeric(b.program, main, tmp_path)


@requires_lli
def test_lli_maxpool_stride_two_all_elements(tmp_path):
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    x = _param(b, f, "x", (1, 4, 4))
    res = b.maxpool(x, 2, 2)
    res.shape = (1, 2, 2)
    b.ret(res)
    main = textwrap.dedent("""\
    define i32 @main() {
    entry:
    %s
      %%out = call float* @kernel(float* %%arr)
    %s
    }
    """ % (_store_array("arr", [str(float(i + 1)) for i in range(16)]),
           _compare_array("out", ["6.0", "8.0", "14.0", "16.0"])))
    _check_numeric(b.program, main, tmp_path)


@requires_lli
def test_lli_for_step_two(tmp_path):
    b = IRBuilder()
    b.new_function("kernel")
    b.new_block("entry")
    acc = b.alloca(1)
    b.store(acc, b.load_const(0.0))
    iv = b.for_loop(0, 6, 2)
    current = b.load(acc)
    b.store(acc, b.add(current, iv))
    b.endfor()
    b.ret(b.load(acc))
    main = textwrap.dedent("""\
    define i32 @main() {
    entry:
      %r = call float @kernel()
      %ok = fcmp oeq float %r, 6.0
      %rc = select i1 %ok, i32 0, i32 1
      ret i32 %rc
    }
    """)
    _check_numeric(b.program, main, tmp_path)


def test_gemm_trans_a_raises():
    b = IRBuilder()
    f = b.new_function("kernel")
    b.new_block("entry")
    a = _param(b, f, "a", (2, 2))
    w = _param(b, f, "w", (2, 2))
    bias = _param(b, f, "bias", (2,))
    res = b.gemm(a, w, bias, trans_a=True)
    res.shape = (2, 2)
    b.ret(res)
    with pytest.raises(LLVMCodegenError, match="trans_a"):
        LLVMCodegen(b.program).emit()
