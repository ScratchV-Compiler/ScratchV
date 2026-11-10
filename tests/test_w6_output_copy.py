"""Exercise the W6-only copy at the normal checked external ABI."""
from dataclasses import replace

import numpy as np
import pytest

from probes.w6_performance.output_copy import with_block_output
from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Value
from tests.test_tensor_c_external import host_cc, library, pointers


def external_vector(dtype, square=False):
    value = Value("x", dtype=dtype, shape=(35,))
    builder = IRBuilder()
    builder.new_function("test", [value])
    builder.new_block("entry")
    builder.ret(builder.mul(value, value) if square else value)
    return TensorCCodegen(builder.program, constant_storage="external").generate()


@pytest.mark.parametrize("dtype", [DataType.FLOAT32, DataType.INT32, DataType.INT64])
def test_aligned_and_offset_copy_with_tail(tmp_path, host_cc, dtype):
    artifact = with_block_output(external_vector(dtype))
    kind = artifact.output.numpy_dtype
    src = np.arange(39, dtype=kind)
    dst = np.full(39, -37, dtype=kind)
    if dtype == DataType.FLOAT32:
        src[3] = -0.0  # Preserving the bytes includes the sign bit.
    with library(tmp_path, artifact, host_cc) as execute:
        for i in (1, 2):
            for j in (1, 2):
                dst.fill(-37)
                expected = dst.copy()
                expected[j:j + 35] = src[i:i + 35]
                assert execute(pointers([src[i:i + 35]]), None, None, 0, dst[j:].ctypes.data) == 0
                assert np.array_equal(dst.view(np.uint8), expected.view(np.uint8))


def test_output_only_written_after_success(tmp_path, host_cc):
    artifact = with_block_output(external_vector(DataType.FLOAT32, square=True))
    src = np.full(35, 3.0, dtype=np.float32)
    work = np.full(artifact.workspace_bytes, 193, dtype=np.uint8)
    dst = np.full(35, -1.0, dtype=np.float32)
    with library(tmp_path, artifact, host_cc) as execute:
        def call(destination, capacity=work.nbytes):
            return execute(pointers([src]), None, work.ctypes.data, capacity, destination.ctypes.data)

        assert call(dst, work.nbytes - 1) == 1
        assert call(work) == 1
        assert call(src) == 1
        assert np.all(src == 3.0)
        src[0] = np.nan
        assert call(dst) == 2
        src[0] = np.finfo(np.float32).max
        assert call(dst) == 2
        assert np.all(dst == -1.0)


def test_rejects_unrecognized_output_or_abi():
    source = external_vector(DataType.INT32)
    with pytest.raises(ValueError, match="external tensor ABI"):
        with_block_output(replace(source, constant_storage="inline"))
    with pytest.raises(ValueError, match="checked final output copy"):
        with_block_output(replace(source, output=replace(source.output, shape=(34,))))
