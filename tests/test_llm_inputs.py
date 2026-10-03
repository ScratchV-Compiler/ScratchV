"""Fixed Qwen3 host ABI, padding, finite greedy selection and stop boundaries."""

import numpy as np
import pytest

from scratchv.runtime.llm_inputs import (
    QWEN3_VOCAB_SIZE, SEQ_LENGTH, build_attention_mask, generation_stop_reason,
    greedy_next_token, greedy_token, last_valid_logits, prepare_inputs,
)


def test_prepare_inputs_preserves_prompt_and_does_not_infer_length_from_pad_id():
    original = np.array([7, 0, 9], dtype=np.int32)
    prepared = prepare_inputs(original, pad_token_id=0, vocab_size=10)
    original[:] = 1
    assert prepared.valid_length == 3
    assert prepared.input_ids.shape == (1, 256)
    assert prepared.input_ids.dtype == np.int64
    np.testing.assert_array_equal(prepared.input_ids[0, :3], [7, 0, 9])
    assert np.all(prepared.input_ids[0, 3:] == 0)
    assert prepared.input_ids.flags.c_contiguous
    assert prepared.attention_mask.flags.c_contiguous
    assert prepared.as_feed() == {
        "input_ids": prepared.input_ids, "attention_mask": prepared.attention_mask,
    }
    assert set(prepared.as_feed()) == {"input_ids", "attention_mask"}


def test_prepare_inputs_accepts_maximum_int64_without_rounding_through_float():
    maximum = int(np.iinfo(np.int64).max)
    prepared = prepare_inputs([maximum - 1, maximum], pad_token_id=maximum,
                              vocab_size=maximum + 1)
    assert int(prepared.input_ids[0, 0]) == maximum - 1
    assert int(prepared.input_ids[0, 1]) == maximum
    assert int(prepared.input_ids[0, -1]) == maximum


def test_prepare_inputs_defaults_to_full_qwen_vocabulary():
    prepared = prepare_inputs([QWEN3_VOCAB_SIZE - 1], pad_token_id=151643)
    assert prepared.input_ids[0, 0] == 151935
    assert prepared.input_ids[0, 1] == 151643


def test_prepare_inputs_accepts_full_capacity_without_truncation():
    prepared = prepare_inputs(tuple(range(256)), pad_token_id=0, vocab_size=256)
    assert prepared.valid_length == 256
    np.testing.assert_array_equal(prepared.input_ids[0], np.arange(256))


@pytest.mark.parametrize("tokens", [[], [1] * 257])
def test_prepare_inputs_rejects_empty_and_oversize_prompts(tokens):
    with pytest.raises(ValueError, match="valid_length"):
        prepare_inputs(tokens, pad_token_id=0, vocab_size=2)


@pytest.mark.parametrize("tokens,exception", [
    ([-1], ValueError), ([4], ValueError), ([2**64], ValueError),
    ([True], TypeError), ([np.bool_(False)], TypeError), ([1.0], TypeError),
    (["1"], TypeError), ([None], TypeError), ("123", TypeError),
    (b"123", TypeError), (iter([1]), TypeError), (1, TypeError),
    (np.array([[1]], dtype=np.int64), ValueError),
    (np.array([1.0], dtype=np.float32), TypeError),
    (np.array([1], dtype=object), TypeError),
    (np.array([2**64 - 1], dtype=np.uint64), ValueError),
])
def test_prepare_inputs_rejects_invalid_ids_without_casting_them(tokens, exception):
    with pytest.raises(exception):
        prepare_inputs(tokens, pad_token_id=0, vocab_size=4)


@pytest.mark.parametrize("kwargs,exception", [
    ({"pad_token_id": -1}, ValueError), ({"pad_token_id": 4}, ValueError),
    ({"pad_token_id": True}, TypeError), ({"pad_token_id": 0.0}, TypeError),
    ({"vocab_size": 0}, ValueError), ({"vocab_size": -1}, ValueError),
    ({"vocab_size": True}, TypeError), ({"vocab_size": 4.0}, TypeError),
    ({"vocab_size": 2**63 + 1}, ValueError),
])
def test_prepare_inputs_rejects_invalid_metadata(kwargs, exception):
    options = {"pad_token_id": 0, "vocab_size": 4}
    options.update(kwargs)
    with pytest.raises(exception):
        prepare_inputs([1], **options)


@pytest.mark.parametrize("length", [1, 17, 255, 256])
def test_attention_mask_blocks_future_and_padding_keys_but_not_all_padded_queries(length):
    mask = build_attention_mask(length)
    blocked = np.finfo(np.float32).min
    assert mask.shape == (1, 1, 256, 256)
    assert mask.dtype == np.float32
    assert np.isfinite(mask).all()
    assert set(np.unique(mask)) == {np.float32(0), blocked}
    # Counts independently characterize every row, including padded queries.
    counts = np.count_nonzero(mask[0, 0] == 0, axis=1)
    np.testing.assert_array_equal(counts, [min(q + 1, length) for q in range(256)])
    assert mask[0, 0, 0, 0] == 0
    assert np.all(mask[0, 0, 0, 1:] == blocked)
    assert np.all(mask[0, 0, 255, :length] == 0)
    if length < SEQ_LENGTH:
        assert np.all(mask[0, 0, :, length:] == blocked)
        assert mask[0, 0, length, length - 1] == 0


def test_mask_suppresses_large_padding_scores_and_never_hides_an_earlier_valid_key():
    mask = build_attention_mask(3)[0, 0]
    scores = np.full((256, 256), 1000, dtype=np.float32)
    scores[:, :3] = 0
    masked = scores + mask
    weights = np.exp(masked - masked.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    np.testing.assert_array_equal(weights[:, 3:], 0)
    np.testing.assert_array_equal(weights[0, :3], [1, 0, 0])
    np.testing.assert_array_equal(weights[1, :3], [0.5, 0.5, 0])
    np.testing.assert_allclose(weights[2:, :3], np.float32(1 / 3), rtol=0, atol=0)


@pytest.mark.parametrize("length,exception", [
    (0, ValueError), (-1, ValueError), (257, ValueError),
    (True, TypeError), (np.bool_(True), TypeError), (1.0, TypeError), (None, TypeError),
])
def test_mask_rejects_invalid_lengths(length, exception):
    with pytest.raises(exception, match="valid_length"):
        build_attention_mask(length)


@pytest.mark.parametrize("length", [1, 17, 255, 256])
def test_greedy_uses_last_valid_position_and_lowest_id_tie(length):
    logits = np.zeros((1, 256, 8), dtype=np.float32)
    logits[0, :, 7] = 100
    logits[0, length - 1, :] = [-9, 2, -1, 2, -5, 0, -3, 1]
    selected = last_valid_logits(logits, length, vocab_size=8)
    assert selected.shape == (8,)
    assert selected.dtype == np.float32
    assert not np.shares_memory(selected, logits)
    assert greedy_next_token(logits, length, vocab_size=8) == 1
    selected[:] = 0
    assert logits[0, length - 1, 1] == 2


def test_greedy_supports_noncontiguous_fp32_view_and_negative_logits():
    values = np.array([-9, 0, -1, 0, -3, 0, -1, 0], dtype=np.float32)[::2]
    assert not values.flags.c_contiguous
    assert greedy_token(values) == 1
    assert greedy_token(np.array([-0.0, 0.0], dtype=np.float32)) == 0


@pytest.mark.parametrize("logits,exception", [
    ([0, 1], TypeError), (np.zeros(3, dtype=np.float64), TypeError),
    (np.zeros(3, dtype=np.float16), TypeError), (np.zeros(3, dtype=np.int64), TypeError),
    (np.zeros(0, dtype=np.float32), ValueError), (np.zeros((), dtype=np.float32), ValueError),
    (np.zeros((1, 3), dtype=np.float32), ValueError),
    (np.array([np.nan, 0], dtype=np.float32), ValueError),
    (np.array([0, np.inf], dtype=np.float32), ValueError),
    (np.array([-np.inf, 0], dtype=np.float32), ValueError),
])
def test_greedy_rejects_malformed_or_nonfinite_vocabulary(logits, exception):
    with pytest.raises(exception):
        greedy_token(logits)


def test_masked_arrays_cannot_hide_invalid_ids_or_nan_logits():
    ids = np.ma.array([1, -1], mask=[False, True], dtype=np.int64)
    with pytest.raises(TypeError, match="masked array"):
        prepare_inputs(ids, pad_token_id=0, vocab_size=8)
    scores = np.ma.array([0, np.nan], mask=[False, True], dtype=np.float32)
    with pytest.raises(TypeError, match="masked array"):
        greedy_token(scores)


@pytest.mark.parametrize("shape", [(256, 8), (1, 255, 8), (2, 256, 8), (1, 256, 9)])
def test_last_valid_logits_rejects_invalid_model_output_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        last_valid_logits(np.zeros(shape, dtype=np.float32), 3, vocab_size=8)


def test_last_valid_logits_rejects_implicit_list_and_dtype_conversion():
    logits = np.zeros((1, 256, 8), dtype=np.float64)
    with pytest.raises(TypeError, match="float32"):
        last_valid_logits(logits, 3, vocab_size=8)
    with pytest.raises(TypeError, match="ndarray"):
        last_valid_logits(logits.tolist(), 3, vocab_size=8)


@pytest.mark.parametrize("position", [0, 2, 255])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_full_output_must_be_finite_including_unused_padding(position, invalid):
    logits = np.zeros((1, 256, 8), dtype=np.float32)
    logits[0, position, 7] = invalid
    with pytest.raises(ValueError, match="finite"):
        greedy_next_token(logits, 3, vocab_size=8)


def stop_reason(**kwargs):
    options = {"valid_length": 3, "generated_tokens": 0, "max_new_tokens": 10,
               "vocab_size": 8, "eos_token_ids": [6, 7]}
    options.update(kwargs)
    return generation_stop_reason(**options)


def test_stop_reason_keeps_prompt_eos_and_stops_on_either_generated_eos():
    assert stop_reason(last_token_id=7) is None
    assert stop_reason(generated_tokens=1, last_token_id=6) == "eos"
    assert stop_reason(generated_tokens=1, last_token_id=7) == "eos"
    assert stop_reason(generated_tokens=1, last_token_id=5) is None


def test_stop_reason_respects_zero_budget_exact_budget_and_capacity():
    assert stop_reason(max_new_tokens=0) == "max_new_tokens"
    assert stop_reason(generated_tokens=2, max_new_tokens=2, last_token_id=1) == "max_new_tokens"
    assert stop_reason(valid_length=255) is None
    assert stop_reason(valid_length=256) == "capacity"
    assert stop_reason(valid_length=256, generated_tokens=1, last_token_id=7,
                       max_new_tokens=1) == "eos"
    assert stop_reason(valid_length=256, generated_tokens=1, last_token_id=1,
                       max_new_tokens=1) == "capacity"


@pytest.mark.parametrize("kwargs,exception", [
    ({"valid_length": 257}, ValueError), ({"generated_tokens": -1}, ValueError),
    ({"generated_tokens": 3}, ValueError), ({"generated_tokens": True}, TypeError),
    ({"generated_tokens": 1}, ValueError), ({"max_new_tokens": -1}, ValueError),
    ({"max_new_tokens": 1.5}, TypeError), ({"last_token_id": 8}, ValueError),
    ({"eos_token_ids": [8]}, ValueError), ({"eos_token_ids": [False]}, TypeError),
    ({"eos_token_ids": 7}, TypeError),
])
def test_stop_reason_rejects_inconsistent_state_or_invalid_configuration(kwargs, exception):
    with pytest.raises(exception):
        stop_reason(**kwargs)
