"""Shared tensor semantics required by W4, without an ONNX-origin dependency.

These tests exercise existing code generation and binary input contracts. They
are not evidence that the complete model executed in QEMU.
"""

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from scratchv.backend.tensor_c_codegen import TensorCCodegen, TensorCCodegenError
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Value
from scratchv.runtime.riscv_tensor import pack_inputs


def _builder(*parameters):
    builder = IRBuilder()
    builder.new_function("main", list(parameters))
    builder.new_block("entry")
    return builder


@pytest.mark.parametrize("shape", [(-1,), (True,), (1.5,), ("N",), (1,) * 17, (2**31,)])
def test_codegen_rejects_nonstatic_or_unrepresentable_parameter_shape(shape):
    value = Value("x", shape=shape)
    builder = _builder(value)
    builder.ret(value)
    with pytest.raises(TensorCCodegenError, match="static|rank/element"):
        TensorCCodegen(builder.program).generate()


@pytest.mark.parametrize("dtype", [DataType.FLOAT32, DataType.INT32, DataType.INT64])
def test_input_contract_rejects_equal_size_wrong_shape_and_dtype(dtype):
    value = Value("x", dtype=dtype, shape=(2, 3))
    builder = _builder(value)
    builder.ret(value)
    artifact = TensorCCodegen(builder.program).generate()
    spec = artifact.inputs[0]
    valid = np.arange(6, dtype=spec.numpy_dtype).reshape(2, 3)
    with pytest.raises(ValueError, match="must be"):
        pack_inputs(artifact.inputs, {"x": valid.reshape(3, 2)})
    incorrect_dtype = np.dtype("float64") if dtype == DataType.INT64 else np.dtype("uint32")
    wrong = valid.astype(incorrect_dtype)
    assert wrong.nbytes == valid.nbytes
    with pytest.raises(ValueError, match="must be"):
        pack_inputs(artifact.inputs, {"x": wrong})


def test_input_packing_preserves_readonly_strided_int64_ids_exactly():
    value = Value("input_ids", DataType.INT64, shape=(3,))
    builder = _builder(value)
    builder.ret(value)
    artifact = TensorCCodegen(builder.program).generate()
    storage = np.array([2**53 + 1, -9, 2**53 + 3, -8, 2**63 - 1, -7], np.int64)
    before = storage.copy()
    view = storage[::2]
    view.setflags(write=False)
    payload = pack_inputs(artifact.inputs, {"input_ids": view})
    np.testing.assert_array_equal(np.frombuffer(payload, dtype="<i8"), view)
    np.testing.assert_array_equal(storage, before)
    assert not view.flags.writeable


def test_codegen_requires_no_frontend_import_or_origin_metadata():
    # A fresh interpreter prevents prior ONNX imports from hiding a dependency.
    script = """
import importlib.abc
import sys
class RejectFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'onnx' or fullname.startswith(('onnx.', 'scratchv.frontend')):
            raise AssertionError('backend requested frontend: ' + fullname)
sys.meta_path.insert(0, RejectFrontend())
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value
from scratchv.backend.tensor_c_codegen import TensorCCodegen
x = Value('semantic_input', shape=(2, 3))
b = IRBuilder()
b.new_function('handbuilt', [x])
b.new_block('entry')
b.ret(b.neg(x))
a = TensorCCodegen(b.program).generate()
assert a.output.shape == (2, 3)
assert a.inputs[0].name == 'semantic_input'
assert a.workspace_bytes > 0
"""
    result = subprocess.run([sys.executable, "-B", "-c", script],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("offset", [0, 2])
def test_compiled_function_preserves_readonly_aliased_inputs(tmp_path, offset):
    zig = os.environ.get("SCRATCHV_ZIG")
    command = [zig, "cc"] if zig else None
    if command is None:
        found = next((shutil.which(name) for name in ("clang", "gcc", "cc")
                      if shutil.which(name)), None)
        if found is None:
            pytest.skip("No host C compiler; set SCRATCHV_ZIG or provide clang/gcc")
        command = [found]
    a, b = Value("a", shape=(4,)), Value("b", shape=(4,))
    builder = _builder(a, b)
    builder.ret(builder.add(a, b))
    artifact = TensorCCodegen(builder.program).generate()
    source = tmp_path / "readonly.c"
    source.write_text(artifact.source, encoding="utf-8")
    library = tmp_path / ("readonly.dll" if os.name == "nt" else "readonly.so")
    env = dict(os.environ)
    env.setdefault("ZIG_GLOBAL_CACHE_DIR", str(tmp_path / "zig-global"))
    env.setdefault("ZIG_LOCAL_CACHE_DIR", str(tmp_path / "zig-local"))
    flags = [] if os.name == "nt" else ["-fPIC"]
    result = subprocess.run([*command, "-shared", "-std=c11", "-O2", *flags,
                             *artifact.compile_flags, str(source), "-o", str(library), "-lm"],
                            env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    loaded = ctypes.CDLL(str(library))
    try:
        function = loaded.scratchv_run
        function.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        function.restype = ctypes.c_int
        storage = np.arange(6, dtype=np.float32) + 0.25
        before = storage.copy()
        left, right = storage[:4], storage[offset:offset + 4]
        left.setflags(write=False)
        right.setflags(write=False)
        pointers = (ctypes.c_void_p * 2)(left.ctypes.data, right.ctypes.data)
        output = np.empty((4,), dtype=np.float32)
        assert function(pointers, output.ctypes.data) == 0
        np.testing.assert_array_equal(output, before[:4] + before[offset:offset + 4])
        np.testing.assert_array_equal(storage, before)
        assert not left.flags.writeable and not right.flags.writeable
    finally:
        if os.name == "nt":
            import _ctypes
            _ctypes.FreeLibrary(loaded._handle)
