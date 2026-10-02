"""Execute generated tensor C against ORT, plus static/numeric rejection cases.

Host compilation uses clang/gcc, or SCRATCHV_ZIG (a zig executable). Repository
local output/tools/zig-python/ziglang/zig.exe is also recognized on Windows.
Validation tests do not require a C toolchain; execution tests skip explicitly
when no compiler is available rather than reporting unexecuted cases as passes.
"""

from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from scratchv.backend.tensor_c_codegen import TensorCCodegen, TensorCCodegenError
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as D, OpCode, Value

ROOT = Path(__file__).resolve().parents[1]


def builder(*params):
    b = IRBuilder()
    b.new_function("main", list(params))
    b.new_block("entry")
    return b


@pytest.fixture(scope="module")
def compiler():
    zig = os.environ.get("SCRATCHV_ZIG")
    local = ROOT / "output/tools/zig-python/ziglang/zig.exe"
    if zig or local.is_file():
        return [zig or str(local), "cc"]
    for name in ("clang", "gcc", "cc"):
        found = shutil.which(name)
        if found:
            return [found]
    pytest.skip("No host C compiler: install clang/gcc or set SCRATCHV_ZIG")


@contextmanager
def compiled(tmp_path, artifact, compiler):
    source = tmp_path / "tensor.c"
    source.write_text(artifact.source, encoding="utf-8")
    library = tmp_path / ("tensor.dll" if sys.platform == "win32" else "tensor.so")
    env = dict(os.environ)
    env.setdefault("ZIG_GLOBAL_CACHE_DIR", str(ROOT / "output/zig-global-cache"))
    env.setdefault("ZIG_LOCAL_CACHE_DIR", str(tmp_path / "zig-cache"))
    command = [*compiler, "-shared", "-O2", "-std=c11", *artifact.compile_flags]
    if sys.platform != "win32":
        command.append("-fPIC")
    command += [str(source), "-o", str(library), "-lm"]
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    loaded = ctypes.CDLL(str(library))
    function = loaded.scratchv_run
    function.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    function.restype = ctypes.c_int

    def run(feed):
        # ascontiguousarray promotes a scalar to rank 1; preserve scalar inputs.
        arrays = [np.array(feed[spec.name], copy=True, order="C") for spec in artifact.inputs]
        for spec, array in zip(artifact.inputs, arrays):
            assert array.dtype == spec.numpy_dtype and array.shape == spec.shape
        pointers = (ctypes.c_void_p * len(arrays))(*(a.ctypes.data for a in arrays))
        output = np.empty(artifact.output.shape, dtype=artifact.output.numpy_dtype)
        status = function(pointers, output.ctypes.data)
        return status, output

    try:
        yield run
    finally:
        # Windows cannot delete a loaded DLL during pytest temporary cleanup.
        if sys.platform == "win32":
            import _ctypes
            _ctypes.FreeLibrary(loaded._handle)


@pytest.fixture(scope="module")
def operator_cases(tmp_path_factory):
    from benchmarks.onnx_operator_cases import build_cases
    return {case["name"]: case for case in build_cases(tmp_path_factory.mktemp("c-cases"))}


@pytest.mark.parametrize("name", [
    "add_float32", "matmul_32", "gather", "sqrt", "reduce_mean", "transpose",
    "concat", "slice", "unsqueeze", "expand", "abs", "cast", "constant", "cos",
    "identity", "pow", "reciprocal", "sin", "scalar_fp32_add_chain",
    "singleton_initializer", "constant_cast_abs_expand", "qwen_rmsnorm", "rope_numeric",
])
def test_generated_host_code_matches_onnx_runtime(tmp_path, compiler, operator_cases, name):
    ort = pytest.importorskip("onnxruntime")
    case = operator_cases[name]
    parser = ONNXParser()
    artifact = TensorCCodegen(parser.parse(case["model"]), parser.initializers).generate()
    with np.load(case["inputs"]) as archive:
        feed = dict(archive)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    expected = ort.InferenceSession(case["model"], options,
                                   providers=["CPUExecutionProvider"]).run(None, feed)[0]
    with compiled(tmp_path, artifact, compiler) as execute:
        status, actual = execute(feed)
    assert status == 0
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, atol=case["atol"], rtol=case["rtol"])


@pytest.mark.parametrize("left,right", [
    ((5,), (5,)), ((5,), (5, 3)), ((2, 5), (5,)),
    ((2, 1, 3, 5), (1, 4, 5, 2)), ((1, 3, 5), (5, 2)),
    ((2, 0), (0, 3)),
])
def test_batched_and_vector_matmul(tmp_path, compiler, left, right):
    a, w = Value("a", shape=left), Value("w", shape=right)
    b = builder(a, w)
    b.ret(b.matmul(a, w))
    rng = np.random.default_rng(42)
    arrays = {"a": rng.normal(size=left).astype("float32"),
              "w": rng.normal(size=right).astype("float32")}
    artifact = TensorCCodegen(b.program).generate()
    with compiled(tmp_path, artifact, compiler) as execute:
        status, result = execute(arrays)
    assert status == 0
    np.testing.assert_allclose(result, arrays["a"] @ arrays["w"], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_stable_softmax_non_last_axis(tmp_path, compiler, axis):
    x = Value("x", shape=(2, 3, 4))
    b = builder(x)
    b.ret(b.softmax(x, axis=axis))
    data = np.arange(24, dtype="float32").reshape(2, 3, 4) + 1000
    expected = np.exp(data - data.max(axis=axis, keepdims=True))
    expected /= expected.sum(axis=axis, keepdims=True)
    with compiled(tmp_path, TensorCCodegen(b.program).generate(), compiler) as execute:
        status, result = execute({"x": data})
    assert status == 0
    np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("axes,keepdims", [((0, 2), False), ((-1,), True), (None, False)])
def test_multi_axis_reduce_mean(tmp_path, compiler, axes, keepdims):
    x = Value("x", shape=(2, 3, 4))
    b = builder(x)
    b.ret(b.reduce_mean(x, axes, keepdims))
    data = np.arange(24, dtype="float32").reshape(2, 3, 4) / 7
    with compiled(tmp_path, TensorCCodegen(b.program).generate(), compiler) as execute:
        status, result = execute({"x": data})
    assert status == 0
    np.testing.assert_allclose(result, data.mean(axis=axes, keepdims=keepdims), atol=1e-6, rtol=1e-6)


def test_reverse_slice_clamps_too_negative_start(tmp_path, compiler):
    x = Value("x", shape=(5,))
    b = builder(x)
    b.ret(b.slice(x, (-20,), (np.iinfo(np.int64).min,), steps=(-1,)))
    with compiled(tmp_path, TensorCCodegen(b.program).generate(), compiler) as execute:
        status, result = execute({"x": np.arange(5, dtype="float32")})
    assert status == 0
    np.testing.assert_array_equal(result, np.array([0], dtype="float32"))


@pytest.mark.parametrize("shape,axis", [((0, 2), 1), ((2, 0), 0)])
def test_empty_gather_still_checks_indices_and_recovers(tmp_path, compiler, shape, axis):
    from scratchv.verification.ir_interpreter import IRExecutionError, IRInterpreter

    x = Value("x", shape=shape)
    index = Value("index", D.INT64, shape=(1,))
    b = builder(x, index)
    b.ret(b.gather(x, index, axis=axis))
    artifact = TensorCCodegen(b.program).generate()
    feed = {"x": np.empty(shape, np.float32), "index": np.array([2], np.int64)}
    with compiled(tmp_path, artifact, compiler) as execute:
        for bad_index in (2, -3):
            feed["index"][0] = bad_index
            with pytest.raises(IRExecutionError, match="GATHER index out of bounds"):
                IRInterpreter(b.program).run(feed)
            assert execute(feed)[0] == 3
        for valid_index in (1, -1):
            feed["index"][0] = valid_index
            expected = IRInterpreter(b.program).run(feed).return_value
            status, actual = execute(feed)
            assert status == 0
            np.testing.assert_array_equal(actual, expected)
            assert actual.shape == expected.shape and actual.size == 0


@pytest.mark.parametrize("dtype,np_type", [(D.INT32, "int32"), (D.INT64, "int64")])
def test_integer_wraparound_and_exact_abs(tmp_path, compiler, dtype, np_type):
    x = Value("x", dtype, shape=(3,))
    b = builder(x)
    b.ret(b.add(x, b.make_const(1, dtype)))
    data = np.array([np.iinfo(np_type).max, -3, 0], dtype=np_type)
    with compiled(tmp_path, TensorCCodegen(b.program).generate(), compiler) as execute:
        status, result = execute({"x": data})
    assert status == 0
    np.testing.assert_array_equal(result, data + np.array(1, dtype=np_type))


@pytest.mark.parametrize("kind", ["sqrt", "reciprocal", "gather", "exp", "input_nan", "cast", "abs", "pow",
                                 "softmax", "reduce_mean", "matmul", "mul"])
def test_runtime_rejects_invalid_numeric_data(tmp_path, compiler, kind):
    dtype = D.INT64 if kind == "abs" else D.FLOAT32
    x = Value("x", dtype, shape=(2,))
    b = builder(x)
    feed = {"x": np.array([1, 2], dtype="int64" if kind == "abs" else "float32")}
    if kind == "sqrt":
        feed["x"][0] = -1
        out = b.sqrt(x)
    elif kind == "reciprocal":
        feed["x"][0] = 0
        out = b.reciprocal(x)
    elif kind == "gather":
        index = Value("index", D.INT64, shape=(1,))
        b.current_func.params.append(index)
        feed["index"] = np.array([2], dtype="int64")
        out = b.gather(x, index)
    elif kind == "exp":
        feed["x"][0] = 1000
        out = b.exp(x)
    elif kind == "input_nan":
        feed["x"][0] = np.nan
        out = b.neg(x)
    elif kind == "cast":
        feed["x"][0] = 2**31
        out = b.cast(x, D.INT32)
    elif kind == "abs":
        feed["x"][0] = np.iinfo(np.int64).min
        out = b.abs(x)
    elif kind == "softmax":
        feed["x"][:] = [-np.finfo(np.float32).max, np.finfo(np.float32).max]
        out = b.softmax(x)
    elif kind == "reduce_mean":
        feed["x"][:] = np.finfo(np.float32).max
        out = b.reduce_mean(x)
    elif kind == "matmul":
        feed["x"][:] = np.finfo(np.float32).max
        rhs = Value("rhs", shape=(2,))
        b.current_func.params.append(rhs)
        feed["rhs"] = np.full(2, 2.0, dtype=np.float32)
        out = b.matmul(x, rhs)
    elif kind == "mul":
        feed["x"][0] = np.finfo(np.float32).max
        out = b.mul(x, b.make_const(2.0))
    else:
        feed["x"][0] = -1
        out = b.pow(x, b.make_const(0.5))
    b.ret(out)
    if kind in {"softmax", "reduce_mean", "matmul", "mul"}:
        from scratchv.verification.ir_interpreter import IRExecutionError, IRInterpreter

        with pytest.raises(IRExecutionError, match="NumericError"):
            IRInterpreter(b.program).run(feed)
    with compiled(tmp_path, TensorCCodegen(b.program).generate(), compiler) as execute:
        status, _ = execute(feed)
        assert status != 0
        # An earlier error must not poison the next call's reused arena.
        feed["x"][:] = 1
        if kind == "gather":
            feed["index"][:] = 0
        status, result = execute(feed)
        assert status == 0 and np.isfinite(result).all()


def test_static_arena_reuses_last_use_storage():
    x = Value("x", shape=(1024,))
    b = builder(x)
    value = x
    for _ in range(100):
        value = b.add(value, b.make_const(1))
    b.ret(value)
    artifact = TensorCCodegen(b.program).generate()
    assert artifact.workspace_bytes == 2 * 1024 * 4
    assert "static union" in artifact.source
    assert artifact.inputs[0].nbytes == 4096
    assert artifact.output.shape == (1024,)


@pytest.mark.parametrize("failure", ["dtype", "broadcast", "matmul", "attrs", "shape", "opcode",
                                      "missing_weight", "weight_shape", "weight_dtype",
                                      "workspace", "constant_limit", "control_flow"])
def test_invalid_static_graphs_fail_before_emission(failure):
    x = Value("x", D.FLOAT64 if failure == "dtype" else D.FLOAT32, shape=(2, 3))
    b = builder(x)
    bindings, kwargs = {}, {}
    if failure in ("broadcast", "matmul"):
        y = Value("y", shape=(4, 2))
        b.current_func.params.append(y)
        value = b.add(x, y) if failure == "broadcast" else b.matmul(x, y)
    elif failure in ("missing_weight", "weight_shape", "weight_dtype", "constant_limit"):
        y = Value("weight", shape=(2, 3))
        b.program.global_values.append(y)
        if failure != "missing_weight":
            bindings["weight"] = np.ones((1, 3) if failure == "weight_shape" else (2, 3),
                                          dtype="int64" if failure == "weight_dtype" else "float32")
        value = b.add(x, y)
        if failure == "constant_limit":
            kwargs["max_constant_bytes"] = 0
    elif failure == "opcode":
        value = b.gelu(x)
    else:
        value = b.neg(x)
    if failure == "attrs":
        b.current_block.instructions[-1].attrs["invented"] = 1
    if failure == "shape":
        value.shape = (100,)
    if failure == "workspace":
        kwargs["max_workspace_bytes"] = 8
    if failure == "control_flow":
        b.new_block("second")
    b.ret(value)
    with pytest.raises(TensorCCodegenError):
        TensorCCodegen(b.program, bindings, **kwargs).generate()


def test_same_generator_is_reusable_and_never_mutates_ir():
    x = Value("x", shape=(2, 3))
    b = builder(x)
    b.ret(b.neg(x))
    before = b.program.dump()
    generator = TensorCCodegen(b.program)
    first, second = generator.generate(), generator.generate()
    assert first == second
    assert b.program.dump() == before
