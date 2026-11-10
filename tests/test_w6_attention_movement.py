"""Check the Attention output checks and copy through the external C ABI."""
import ctypes
from dataclasses import replace
import os

import numpy as np
import pytest

from probes.w6_performance.attention_bounded_output import with_bounded_output
from probes.w6_performance.attention_input_checks import with_input_checks
from probes.w6_performance.attention_first_pack import with_first_pass_packing
from probes.w6_performance.attention_lattice_output import with_normal_output
from probes.w6_performance.attention_panel import RuntimeScorePanel
from probes.w6_performance.attention_paired_q import with_paired_q
from probes.w6_performance.attention_preboxed_k import with_paired_k, with_preboxed_k
from probes.w6_performance.attention_profile import decode_claims
from probes.w6_performance.attention_scalar_epilogue import with_bitwise_writeback
from probes.w6_performance.attention_static_schedule import with_static_schedule
from probes.w6_performance.attention_wide_scalar import with_wide_scalar
from probes.w6_performance.output_copy import with_block_output
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value
from tests.test_tensor_c_external import (
    _PARALLEL_HOST_RUNTIME, host_cc, library, pointers,
)


def attention_artifact(parallel_copy, *, claim_blocks=8, row_tile=4, cache_output=False,
                       loop_variant="shared-checked"):
    query = Value("x", shape=(16, 256, 128))
    key = Value("weight", shape=(16, 128, 256))
    builder = IRBuilder()
    builder.new_function("main", [query, key])
    builder.new_block("entry")
    builder.ret(builder.matmul(query, key))
    artifact = RuntimeScorePanel(
        builder.program, {}, constant_storage="external", kernel_calls=True,
        matmul_policy="blocked_fma", matmul_tile="2x4",
        matmul_rhs_layout="original", qemu_workspace_layout="none",
        loop_variant=loop_variant, claim_blocks=claim_blocks,
        row_tile=row_tile, cache_output=cache_output,
    ).generate()
    return with_block_output(artifact, parallel=parallel_copy)


@pytest.mark.skipif(os.name == "nt", reason="The parallel callback fixture requires pthreads")
@pytest.mark.parametrize("parallel", [False, True], ids=["serial-inputs", "parallel-inputs"])
def test_input_checks_reject_each_slice_before_matmul_and_allow_reuse(tmp_path, host_cc, parallel):
    artifact = with_input_checks(attention_artifact(True), parallel=parallel)
    artifact = replace(artifact, source=artifact.source + _PARALLEL_HOST_RUNTIME)
    query, key = (np.ones(spec.shape, dtype=np.float32) for spec in artifact.inputs)
    key.fill(0.5)
    inputs = pointers([query, key])
    workspace = np.full(artifact.workspace_bytes // 8, 193, dtype=np.uint64)
    output = np.full(artifact.output.shape, -999, dtype=np.float32)
    with library(tmp_path, artifact, [*host_cc, "-DSCRATCHV_CORES=4", "-pthread"]) as execute:
        dispatches = ctypes.c_size_t.in_dll(ctypes.CDLL(str(tmp_path / "model.so")), "sv_test_dispatches")

        def call(capacity=workspace.nbytes):
            return execute(inputs, None, workspace.ctypes.data, capacity, output.ctypes.data)

        # Invalid ABI arguments take precedence even when a valid input contains NaN.
        query.flat[0] = np.nan
        assert call(workspace.nbytes - 1) == 1
        inputs[1] = None
        assert call() == 1
        inputs[1] = key.ctypes.data
        query.flat[0] = 1

        # Poison one value at a time so an unchecked worker slice cannot be masked.
        slice_size = query.size // 4
        indices = (0, slice_size, 3 * slice_size - 1, query.size - 1)
        for index, infinity in zip(indices, (np.inf, -np.inf, np.inf, -np.inf)):
            for data, bad in ((query, np.nan), (key, infinity)):
                saved = data.flat[index]
                data.flat[index] = bad
                before = dispatches.value
                assert call() == 2
                assert (dispatches.value > before) == parallel
                assert np.all(workspace == 193)
                assert np.all(output == -999)
                data.flat[index] = saved
        assert call() == 0
        assert np.all(output == 64.0)  # 128 terms of 1 × 0.5, exactly representable in FP32.


@pytest.mark.skipif(os.name == "nt", reason="The parallel callback fixture requires pthreads")
@pytest.mark.parametrize("parallel_copy,claim_blocks,row_tile,cache_output,loop_variant,transform", [
    pytest.param(False, 8, 4, False, "shared-checked", None, id="serial-copy"),
    pytest.param(True, 4, 4, False, "shared-checked", None, id="claim4"),
    pytest.param(True, 8, 4, False, "shared-checked", None, id="claim8"),
    pytest.param(True, 16, 4, False, "shared-checked", None, id="claim16"),
    pytest.param(True, 8, 5, False, "shared-checked", None, id="5x4"),
    pytest.param(True, 8, 4, True, "shared-checked", None, id="4x4-cached-output"),
    pytest.param(True, 8, 5, True, "shared-checked", None, id="5x4-cached-output"),
    pytest.param(True, 8, 4, False, "shared-claim-panels", None, id="4x4-claim-panel"),
    pytest.param(True, 8, 5, True, "shared-claim-panels", None, id="5x4-cached-claim-panel"),
    pytest.param(True, 8, 6, True, "shared-checked", None, id="6x4-cached"),
    pytest.param(True, 8, 5, True, "shared-checked", with_bitwise_writeback, id="5x4-bitwise"),
    pytest.param(True, 8, 5, True, "shared-checked", with_preboxed_k, id="5x4-preboxed-k"),
    pytest.param(True, 8, 5, True, "shared-checked", with_paired_k, id="5x4-paired-k"),
    pytest.param(True, 8, 5, True, "shared-checked", lambda a: with_paired_k(with_bitwise_writeback(a)),
                 id="5x4-paired-bitwise"),
    pytest.param(True, 8, 5, True, "shared-checked", with_first_pass_packing, id="5x4-first-pack"),
    pytest.param(True, 8, 5, True, "shared-checked", lambda a: with_static_schedule(with_bitwise_writeback(a)),
                 id="5x4-static-bitwise"),
    pytest.param(True, 8, 5, True, "shared-checked", lambda a: with_bounded_output(with_bitwise_writeback(a)),
                 id="5x4-bounded-bitwise"),
    pytest.param(True, 8, 5, True, "shared-checked",
                 lambda a: with_paired_q(with_bounded_output(with_bitwise_writeback(a)), packed_load=False),
                 id="5x4-q-unroll2"),
    pytest.param(True, 8, 5, True, "shared-checked",
                 lambda a: with_paired_q(with_bounded_output(with_bitwise_writeback(a)), packed_load=True),
                 id="5x4-q-pair"),
    pytest.param(True, 8, 5, True, "shared-checked", lambda a: with_normal_output(with_bitwise_writeback(a)),
                 id="5x4-normal-output"),
    pytest.param(True, 8, 4, False, "shared-checked", lambda a: with_wide_scalar(a, rows=2), id="2x8"),
    pytest.param(True, 8, 4, False, "shared-checked", lambda a: with_wide_scalar(a, rows=3), id="3x8"),
])
def test_attention_overflow_preserves_output_and_workspace_can_be_reused(
        tmp_path, host_cc, parallel_copy, claim_blocks, row_tile, cache_output, loop_variant, transform):
    artifact = attention_artifact(parallel_copy, claim_blocks=claim_blocks,
                                  row_tile=row_tile, cache_output=cache_output, loop_variant=loop_variant)
    if transform is not None:
        artifact = transform(artifact)
    artifact = replace(artifact, source=artifact.source + _PARALLEL_HOST_RUNTIME)
    query_rows = (1 + np.arange(16, dtype=np.float32)[:, None]
                  + (np.arange(256, dtype=np.float32) % 16)[None, :] / 16)
    # Distinguish the final full five-row tile from its one-row tail on every head.
    query_rows[:, 254] = -32 - np.arange(16)
    query_rows[:, 255] = 32 + np.arange(16)
    query = np.broadcast_to(query_rows[:, :, None], (16, 256, 128)).copy()
    key_columns = (np.arange(256, dtype=np.float32) + 1) / 256
    key = np.broadcast_to(key_columns, (16, 128, 256)).copy()
    # Constant rows have exactly representable intermediate sums.
    expected = query_rows[:, :, None] * key_columns[None, None, :] * 128
    # Head 0 has one nonzero K term, so a single FP32 rounding is the exact
    # dot-product oracle. Exercise the main tile and the final row alike.
    tiny = np.array([1], dtype=np.uint32).view(np.float32)[0]
    largest = np.finfo(np.float32).max
    query[0] = 0
    query[0, :, -1] = 1
    query[0, 255, -1] = -tiny
    key[0] = 0
    key[0, -1, :7] = [0., -0., tiny, -tiny, largest, -largest, 0.25]
    products = query[0, :, -1, None].astype(np.float64) * key[0, -1, None, :].astype(np.float64)
    expected[0] = np.float32(0.0) + products.astype(np.float32)
    # The last FMA rounds -tiny * 0.25 to -0. The final add stores +0.
    assert expected[0, 255, 6].view(np.uint32) == 0
    # At column 255 the ordered dot product is 1; regrouping it can produce 2.
    query[-1, 255] = 0
    query[-1, 255, :4] = [2**24, 1, -(2**24), 1]
    tail = np.zeros_like(key_columns)
    for term in query[-1, 255, :4]:
        # These four exact binary products and sums fit in float64 before FP32 rounding.
        tail = (tail + float(term) * key_columns.astype(np.float64)).astype(np.float32)
    expected[-1, 255] = tail
    # Reserve the four rows a five-row tile would overwrite past its one-row tail.
    words = artifact.workspace_bytes // 8
    storage = np.full(words + 4 * 256 * 4 // 8, 193, dtype=np.uint64)
    workspace = storage[:words]
    output = np.full(artifact.output.shape, -999, dtype=np.float32)
    inputs = pointers([query, key])
    compiler = [*host_cc, "-DSCRATCHV_CORES=4", "-pthread"]
    with library(tmp_path, artifact, compiler) as execute:
        def call(destination=output, capacity=workspace.nbytes):
            return execute(inputs, None, workspace.ctypes.data, capacity, destination.ctypes.data)

        assert call(capacity=workspace.nbytes - 1) == 1
        assert call(destination=workspace) == 1
        assert np.all(workspace == 193)
        for invalid in (np.nan, np.inf):
            key[0, 0, 0] = invalid
            assert call() == 2
            assert np.all(output == -999) and np.all(workspace == 193)
        key[0, 0, 0] = 0

        saved_row = query[-1, -1].copy()
        query[-1, -1] = np.finfo(np.float32).max
        assert np.isfinite(query).all() and np.isfinite(key).all()
        assert call() == 2
        assert np.all(output == -999)
        query[-1, -1] = saved_row
        assert call() == 0
        np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))
        assert np.all(storage[words:] == 193)


@pytest.mark.skipif(os.name == "nt", reason="The parallel callback fixture requires pthreads")
def test_bounded_output_uses_workspace_only_above_the_inclusive_limit(tmp_path, host_cc):
    artifact = with_bounded_output(with_bitwise_writeback(
        attention_artifact(True, row_tile=5, cache_output=True)))
    artifact = replace(artifact, source=artifact.source + _PARALLEL_HOST_RUNTIME)
    limit = np.float32(2**32)
    outside = np.nextafter(limit, np.float32(np.inf))
    query = np.full(artifact.inputs[0].shape, -limit, dtype=np.float32)
    key = np.full(artifact.inputs[1].shape, limit, dtype=np.float32)
    workspace = np.full(artifact.workspace_bytes // 8, 193, dtype=np.uint64)
    output = np.full(artifact.output.shape, -999, dtype=np.float32)
    inputs = pointers([query, key])
    with library(tmp_path, artifact, [*host_cc, "-DSCRATCHV_CORES=4", "-pthread"]) as execute:
        def call():
            return execute(inputs, None, workspace.ctypes.data, workspace.nbytes, output.ctypes.data)

        # Every FMA lies on the exact worst-case bound and writes directly to the caller.
        assert call() == 0
        assert np.all(output == np.float32(-(2**71))) and np.all(workspace == 193)

        for q0, k0 in ((-outside, np.float32(1)), (np.float32(1), outside)):
            # One nonzero product per score lets us check every output bit at both boundaries.
            query.fill(0)
            key.fill(0)
            query[:, :, 0] = q0
            key[:, 0, :] = k0
            workspace.fill(193)
            assert call() == 0
            expected_bits = np.float32(q0 * k0).view(np.uint32)
            assert np.all(output.view(np.uint32) == expected_bits)
            assert np.all(workspace.view(np.uint32) == expected_bits)

        # An out-of-range Q must not bypass validation of the last K element.
        query[0, 0, 0] = outside
        key.flat[-1] = np.nan
        workspace.fill(193)
        output.fill(-999)
        assert call() == 2
        assert np.all(workspace == 193) and np.all(output == -999)
        query.fill(1)
        key.fill(1)
        assert call() == 0
        assert np.all(output == 128) and np.all(workspace == 193)


@pytest.mark.skipif(os.name == "nt", reason="The parallel callback fixture requires pthreads")
def test_normal_output_limits_cancellation_and_workspace_fallback(tmp_path, host_cc):
    artifact = with_normal_output(with_bitwise_writeback(
        attention_artifact(True, row_tile=5, cache_output=True)))
    downward = """
int w6_test_downward(const void *const *in, const void *const *weights, void *work, size_t bytes, void *out) {
    fenv_t saved;
    if (fegetenv(&saved)) return -1;
    int status = fesetround(FE_DOWNWARD) ? -1 : scratchv_run_external(in, weights, work, bytes, out);
    return fesetenv(&saved) ? -1 : status;
}
"""
    artifact = replace(artifact, source=artifact.source + _PARALLEL_HOST_RUNTIME + downward)
    low, high = np.float32(2**-30), np.float32(2**32)
    query = np.full(artifact.inputs[0].shape, -low, dtype=np.float32)
    key = np.full(artifact.inputs[1].shape, high, dtype=np.float32)
    workspace = np.full(artifact.workspace_bytes // 8, 193, dtype=np.uint64)
    output = np.full(artifact.output.shape, -999, dtype=np.float32)
    inputs = pointers([query, key])
    with library(tmp_path, artifact, [*host_cc, "-DSCRATCHV_CORES=4", "-pthread"]) as execute:
        def call():
            return execute(inputs, None, workspace.ctypes.data, workspace.nbytes, output.ctypes.data)

        # The inclusive endpoints give 128 exact products of -4.
        assert call() == 0
        assert np.all(output == -512) and np.all(workspace == 193)
        query.fill(-high)
        query[:, :, 1::2] = high
        key.fill(low)
        assert call() == 0
        assert np.all(output.view(np.uint32) == 0) and np.all(workspace == 193)

        # Keep every input in range, but cancel nearly equal products at FP32's lowest
        # attainable nonzero step for this input range (2**-106).
        u = np.nextafter(low, np.float32(np.inf))
        v = np.nextafter(u, np.float32(np.inf))
        query.fill(low)
        query[:, :, 1::2] = -low
        key.fill(low)
        key[:, -2, :] = v
        query[:, :, -1] = -u
        key[:, -1, :] = u
        assert call() == 0
        assert np.all(output.view(np.uint32) == 0x8A800000) and np.all(workspace == 193)
        query[:, :, -1] = -low
        key[:, -1, :] = v
        assert call() == 0
        assert np.all(output.view(np.uint32) == 0) and np.all(workspace == 193)

        # Keep the first 127 products nonzero and in range. They sum exactly to zero.
        query.fill(1)
        query[:, :, 0] = -2
        query[:, :, 3:127:2] = -1
        key.fill(1)
        below = np.nextafter(low, np.float32(0))
        above = np.nextafter(high, np.float32(np.inf))
        tiny = np.nextafter(np.float32(0), np.float32(1))
        for q_last, k_last in ((-below, high), (low, above), (np.float32(0), high),
                               (low, np.float32(-0.0)), (-tiny, np.float32(1))):
            query[:, :, -1] = q_last
            key[:, -1, :] = k_last
            workspace.fill(193)
            output.fill(-999)
            assert call() == 0
            # The original checked path adds +0 after the final FMA.
            value = np.float32(0) + np.float32(float(q_last) * float(k_last))
            assert np.all(output.view(np.uint32) == value.view(np.uint32))
            assert np.all(workspace.view(np.uint32) == value.view(np.uint32))

        query.fill(1)
        key.fill(1)
        workspace.fill(193)
        output.fill(-999)
        run_downward = ctypes.CDLL(str(tmp_path / "model.so")).w6_test_downward
        run_downward.argtypes = execute.argtypes
        assert run_downward(inputs, None, workspace.ctypes.data, workspace.nbytes, output.ctypes.data) == 0
        assert np.all(workspace.view(np.float32) == 128) and np.all(output == 128)


@pytest.mark.parametrize("parallel_copy", [False, True], ids=["serial-copy", "parallel-copy"])
def test_attention_candidate_compiles_for_single_core(tmp_path, host_cc, parallel_copy):
    artifact = attention_artifact(parallel_copy)
    with library(tmp_path, artifact, [*host_cc, "-DSCRATCHV_CORES=1"]):
        pass


@pytest.mark.parametrize("claim_blocks", [0, 3, 32, True, 8.0])
def test_claim_granularity_rejects_invalid_sizes(claim_blocks):
    with pytest.raises(ValueError, match="claims"):
        attention_artifact(True, claim_blocks=claim_blocks)


def claim_ticks(claim_blocks):
    ticks = [[0] * 8 for _ in range(10)]
    for core, finish in enumerate((185, 180, 195, 190)):
        ticks[core][:6] = [101+core, finish, 10, 60, 256, 256//claim_blocks]
        ticks[5+core][:3] = [211+core, 225+core, 16384]
    ticks[4][:4] = [100, 200, 0, 10]
    ticks[9][:4] = [210, 230, 0, 2]
    return ticks


@pytest.mark.parametrize("claim_blocks", [4, 8, 16])
def test_claim_timing_keeps_math_and_copy_waits_separate(claim_blocks):
    times = decode_claims(claim_ticks(claim_blocks), 1000, claim_blocks)
    assert times["dispatch_ms"] == 100
    assert times["copy_dispatch_ms"] == 20
    assert times["critical_core"] == 2
    assert times["main_wait_after_work_ms"] == 15
    assert times["main_join_wfi_ms"] == 10
    assert times["copy_main_join_wfi_ms"] == 2
    assert times["finish_spread_ms"] == 15
    assert sum(c["claimed_blocks"] for c in times["cores"]) == 1024
    assert sum(times[k] for k in ("critical_pack_ms", "critical_mul_store_ms",
                                 "critical_bookkeeping_ms", "critical_schedule_ms")) == 100


@pytest.mark.parametrize("row,slot,value", [(0, 4, 255), (4, 3, 16), (9, 0, 199), (5, 2, 16383)])
def test_claim_timing_rejects_missing_work_or_impossible_intervals(row, slot, value):
    ticks = claim_ticks(8)
    ticks[row][slot] = value
    with pytest.raises(ValueError):
        decode_claims(ticks, 1000, 8)


def test_claim_timing_allows_a_core_that_received_no_work():
    ticks = claim_ticks(8)
    ticks[0][4:6] = [512, 64]
    ticks[3][2:6] = [0, 0, 0, 0]
    times = decode_claims(ticks, 1000, 8)
    assert times["cores"][3]["claimed_blocks"] == 0
