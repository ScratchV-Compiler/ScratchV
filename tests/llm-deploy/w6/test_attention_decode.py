"""Check direct reads of shared K and publication through the external ABI."""
from dataclasses import replace
import os

import numpy as np
import pytest

from probes.w6_performance.attention_decode_native import native_artifact
from probes.w6_performance.attention_decode_sequence import sequence_artifact
from probes.w6_performance.attention_decode_sequence_full import full_sequence_artifact
from probes.w6_performance.attention_head_groups import head_groups_artifact
from tests.test_tensor_c_external import _PARALLEL_HOST_RUNTIME, host_cc, library, pointers


pytestmark = pytest.mark.skipif(os.name == "nt", reason="Four-core host fixture requires pthreads")


@pytest.mark.parametrize("length", [1, 255])
def test_shared_keys_tail_and_failed_call_do_not_corrupt_output(tmp_path, host_cc, length):
    artifact = native_artifact(length, "native-paired-dot")
    artifact = replace(artifact, source=artifact.source + _PARALLEL_HOST_RUNTIME)
    workspace = np.empty(artifact.workspace_bytes // 8, dtype=np.uint64)
    output = np.full((8, 2, length), -1, dtype=np.float32)
    q = np.zeros((8, 2, 128), dtype=np.float32)
    k = np.zeros((8, length, 128), dtype=np.float32)
    q[:, :, -1] = [1, 2]
    keys = np.arange(1, 8 * length + 1, dtype=np.float32).reshape(8, length)
    k[:, :, -1] = keys
    inputs = pointers([q, k])
    with library(tmp_path, artifact, [*host_cc, "-DSCRATCHV_CORES=4", "-pthread"]) as execute:
        def call(capacity):
            return execute(inputs, None, workspace.ctypes.data, capacity, output.ctypes.data)

        assert call(workspace.nbytes - 1) == 1
        k[-1, -1, -1] = np.finfo(np.float32).max
        assert call(workspace.nbytes) == 2
        assert np.all(output == -1)
        k[-1, -1, -1] = keys[-1, -1]
        assert call(workspace.nbytes) == 0
        np.testing.assert_array_equal(output[:, 0], keys)
        np.testing.assert_array_equal(output[:, 1], 2 * keys)


@pytest.mark.parametrize("variant", ["native-transpose-2x4", "native-paired-dot"])
def test_append_two_keys_preserves_prefix_and_output_boundaries(tmp_path, host_cc, variant):
    artifact = sequence_artifact(variant)
    artifact = replace(artifact, source=artifact.source + _PARALLEL_HOST_RUNTIME)
    # A unique integer at the last depth of each position identifies both its
    # KV head and its position. Each query selects it with a different scale.
    positions = np.arange(8 * 256, dtype=np.float32).reshape(8, 256) + 1
    history = np.zeros((8, 254, 128), dtype=np.float32)
    history[:, :, -1] = positions[:, :254]
    new_keys = np.zeros((2, 8, 128), dtype=np.float32)
    new_keys[:, :, -1] = positions[:, 254:].T
    scales = np.array([[1, 2], [-3, 4]], dtype=np.float32)
    queries = np.zeros((2, 8, 2, 128), dtype=np.float32)
    queries[:, :, :, -1] = scales[:, None, :]
    inputs = pointers([history, queries, new_keys])
    workspace = np.full(artifact.workspace_bytes // 4, np.nan, dtype=np.float32)
    lengths = (255, 256)
    expected = [positions[:, None, :n] * scale[None, :, None] for n, scale in zip(lengths, scales)]
    guarded = np.full(2 + sum(part.size for part in expected), -999, dtype=np.float32)
    output = guarded[1:-1]
    with library(tmp_path, artifact, [*host_cc, "-DSCRATCHV_CORES=4", "-pthread"]) as execute:
        def run():
            return execute(inputs, None, workspace.ctypes.data, workspace.nbytes, output.ctypes.data)

        # All inputs are finite, but step two overflows. Neither step should be
        # published until the complete call succeeds with the corrected key.
        new_keys[1, -1, -1] = np.finfo(np.float32).max
        assert run() == 2
        assert np.all(guarded == -999)
        new_keys[1, -1, -1] = positions[-1, -1]
        assert run() == 0
        offset = 0
        for scores in expected:
            np.testing.assert_array_equal(output[offset:offset + scores.size], scores.ravel())
            offset += scores.size
        np.testing.assert_array_equal(guarded[[0, -1]], [-999, -999])


@pytest.mark.parametrize("variant", ["native-transpose-2x4", "native-paired-dot", "head-groups"])
def test_two_step_attention_normalizes_the_valid_prefix_and_appends_values(tmp_path, host_cc, variant):
    artifact = head_groups_artifact() if variant == "head-groups" else full_sequence_artifact(variant)
    artifact = replace(artifact, source=artifact.source + _PARALLEL_HOST_RUNTIME)
    history = np.zeros((8, 254, 128), dtype=np.float32)
    keys = np.zeros((2, 8, 128), dtype=np.float32)
    q = np.zeros((2, 8, 2, 128), dtype=np.float32)
    # The common logit exceeds 180 after scaling. Taking exp without first
    # subtracting the row maximum would overflow float32.
    history[:, :, 0] = keys[:, :, 0] = 128
    q[:, :, :, 0] = 16
    query_scales = np.array([[8, -16], [-8, 16]], dtype=np.float32)
    q[:, :, :, -1] = query_scales[:, None, :]
    keys[:, :, -1] = [[1], [-2]]

    # All 254 history scores are equal; only the last history V is nonzero.
    # Use distinct values for every KV head, channel and append position.
    channels = np.arange(1, 1025, dtype=np.float32).reshape(8, 128)
    values = np.zeros_like(history)
    values[:, -1] = channels
    appended = np.stack([2 * channels + 1, -channels[:, ::-1]])
    expected = np.empty((2, 8, 2, 128), dtype=np.float64)
    for step in range(2):
        for head, scale in enumerate(query_scales[step]):
            first = np.exp(scale / np.sqrt(128.0))
            second = np.exp(-2 * scale / np.sqrt(128.0)) if step == 1 else 0.
            expected[step, :, head] = (
                channels + first * appended[0] + second * appended[1]
            ) / (254 + first + second)

    inputs = pointers([history, q, keys, values, appended])
    guarded_workspace = np.full(artifact.workspace_bytes // 8 + 2, 0x46564647, dtype=np.uint64)
    workspace = guarded_workspace[1:-1]
    workspace.view(np.float32).fill(np.nan)
    guarded = np.full(expected.size + 2, -999, dtype=np.float32)
    output = guarded[1:-1].reshape(expected.shape)
    with library(tmp_path, artifact, [*host_cc, "-DSCRATCHV_CORES=4", "-pthread"]) as run:
        assert run(inputs, None, workspace.ctypes.data, workspace.nbytes - 1, output.ctypes.data) == 1
        assert np.all(guarded == -999)
        saved = appended[1, -1, -1]
        appended[1, -1, -1] = np.nan
        assert run(inputs, None, workspace.ctypes.data, workspace.nbytes, output.ctypes.data) == 2
        assert np.all(guarded == -999)
        appended[1, -1, -1] = saved
        # Both inputs are finite. Only the last K/V group overflows on step two,
        # when its new K is multiplied by the queries.
        keys[1, -1, -1] = np.finfo(np.float32).max
        assert run(inputs, None, workspace.ctypes.data, workspace.nbytes, output.ctypes.data) == 2
        assert np.all(guarded == -999)
        keys[1, -1, -1] = -2
        assert run(inputs, None, workspace.ctypes.data, workspace.nbytes, output.ctypes.data) == 0
    np.testing.assert_allclose(output, expected, atol=2e-5, rtol=3e-5)
    np.testing.assert_array_equal(guarded[[0, -1]], [-999, -999])
    np.testing.assert_array_equal(guarded_workspace[[0, -1]], [0x46564647, 0x46564647])


@pytest.mark.parametrize("variant", ["native-transpose-2x4", "head-groups"])
def test_value_tile_preserves_128_position_accumulation_order(tmp_path, host_cc, variant):
    artifact = head_groups_artifact() if variant == "head-groups" else full_sequence_artifact(variant)
    artifact = replace(artifact, source=artifact.source + _PARALLEL_HOST_RUNTIME)
    keys = np.zeros((8, 254, 128), dtype=np.float32)
    queries = np.zeros((2, 8, 2, 128), dtype=np.float32)
    appended = np.zeros((2, 8, 128), dtype=np.float32)
    # Uniform probabilities. In even channels the +1 at position 129 rounds
    # away inside block two. A single accumulator would cancel positions 0
    # and 128 first and retain that +1. Odd channels are a nonzero control.
    values = np.zeros_like(keys)
    values[:, 0, ::2] = 2**30
    values[:, 128, ::2] = -(2**30)
    values[:, 129, ::2] = 1
    values[:, 0, 1::2] = 256
    inputs = pointers([keys, queries, appended, values, appended])
    workspace = np.empty(artifact.workspace_bytes // 8, dtype=np.uint64)
    output = np.full((2, 8, 2, 128), np.nan, dtype=np.float32)
    with library(tmp_path, artifact, [*host_cc, "-DSCRATCHV_CORES=4", "-pthread"]) as run:
        assert run(inputs, None, workspace.ctypes.data, workspace.nbytes, output.ctypes.data) == 0
    expected = np.zeros_like(output)
    for step, valid in enumerate((255, 256)):
        expected[step, :, :, 1::2] = np.float32(256) / np.float32(valid)
    np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))
