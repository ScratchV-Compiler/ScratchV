"""Host generation contract boundaries without a model, compiler or QEMU."""

from dataclasses import FrozenInstanceError
from numbers import Integral

import numpy as np
import pytest

from scratchv.runtime import generation
from scratchv.runtime.generation import GenerationCancelledError, generate
from scratchv.runtime.llm_inputs import SEQ_LENGTH


class Tokenizer:
    """Small tokenizer with one undecodable model-vocabulary tail entry."""

    model_vocab_size = 8
    pad_token_id = 0
    generation_eos_token_ids = (5, 6)

    def __init__(self, tokens=(1, 2)):
        self.tokens = list(tokens)
        self.encode_calls = []
        self.decode_calls = []

    def encode(self, text, *, add_special_tokens):
        self.encode_calls.append((text, add_special_tokens))
        return self.tokens.copy()

    def validate_token_ids(self, tokens):
        checked = []
        for token in tokens:
            if isinstance(token, (bool, np.bool_)) or not isinstance(token, Integral):
                raise TypeError("token ID must be an integer")
            if not 0 <= token < 7:
                raise ValueError("token ID is absent from tokenizer vocabulary")
            checked.append(int(token))
        return checked

    def decode(self, tokens, *, skip_special_tokens, clean_up_tokenization_spaces):
        tokens = self.validate_token_ids(tokens)
        self.decode_calls.append((tuple(tokens), skip_special_tokens,
                                  clean_up_tokenization_spaces))
        return " ".join(str(token) for token in tokens if token not in (0, 5, 6))


def logits_for(prepared, token=3):
    logits = np.zeros((1, SEQ_LENGTH, 8), dtype=np.float32)
    # Padding rows deliberately prefer an undecodable model-vocabulary tail ID.
    logits[0, :, 7] = 10
    logits[0, prepared.valid_length - 1, :] = 0
    logits[0, prepared.valid_length - 1, token] = 20
    return logits


def no_forward(_prepared):
    raise AssertionError("forward must not be called")


def test_raw_prompt_last_valid_row_and_owned_result():
    tokenizer = Tokenizer()
    inputs = []

    def forward(prepared):
        inputs.append(prepared)
        return logits_for(prepared, 3 if len(inputs) == 1 else 4)

    result = generate(" raw\nCafe\u0301 ", tokenizer=tokenizer, forward=forward,
                      max_new_tokens=2)
    assert tokenizer.encode_calls == [(" raw\nCafe\u0301 ", False)]
    assert result.prompt_token_ids == (1, 2)
    assert result.generated_token_ids == (3, 4)
    assert result.generated_text == "3 4"
    assert result.stop_reason == "max_new_tokens"
    assert tokenizer.decode_calls == [((3, 4), True, False)]
    assert [item.valid_length for item in inputs] == [2, 3]
    np.testing.assert_array_equal(inputs[1].input_ids[0, :3], [1, 2, 3])
    assert inputs[1].attention_mask[0, 0, 2, 2] == 0
    assert inputs[1].attention_mask[0, 0, 2, 3] < 0
    tokenizer.tokens[:] = [4]
    assert result.prompt_token_ids == (1, 2)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.stop_reason = "eos"


def test_zero_budget_has_no_forward_or_generated_tokens():
    result = generate("prompt", tokenizer=Tokenizer(), forward=no_forward,
                      max_new_tokens=0)
    assert result.generated_token_ids == ()
    assert result.generated_text == ""
    assert result.stop_reason == "max_new_tokens"


@pytest.mark.parametrize("budget", [0, 1, 1000])
def test_full_prompt_stops_at_capacity_without_forward(budget):
    result = generate("prompt", tokenizer=Tokenizer([1] * SEQ_LENGTH),
                      forward=no_forward, max_new_tokens=budget)
    assert result.prompt_token_ids == (1,) * SEQ_LENGTH
    assert result.generated_token_ids == ()
    assert result.stop_reason == "capacity"


@pytest.mark.parametrize("tokens,exception", [
    ([], ValueError), ([1] * 257, ValueError), ([-1], ValueError),
    ([7], ValueError), ([True], TypeError), ([1.5], TypeError),
])
def test_invalid_prompt_rejected_even_with_zero_budget(tokens, exception):
    with pytest.raises(exception):
        generate("prompt", tokenizer=Tokenizer(tokens), forward=no_forward,
                 max_new_tokens=0)


@pytest.mark.parametrize("prompt", [None, b"prompt", 123, ["prompt"]])
def test_prompt_must_be_text(prompt):
    with pytest.raises(TypeError):
        generate(prompt, tokenizer=Tokenizer(), forward=no_forward,
                 max_new_tokens=0)


@pytest.mark.parametrize("budget,exception", [
    (-1, ValueError), (True, TypeError), (np.bool_(False), TypeError),
    (1.5, TypeError), (None, TypeError),
])
def test_invalid_budget_rejected_before_forward(budget, exception):
    with pytest.raises(exception):
        generate("prompt", tokenizer=Tokenizer(), forward=no_forward,
                 max_new_tokens=budget)


@pytest.mark.parametrize("eos", [5, 6])
def test_each_generation_eos_stops_and_is_retained_in_ids(eos):
    calls = []

    def forward(prepared):
        calls.append(prepared.valid_length)
        return logits_for(prepared, eos)

    result = generate("prompt", tokenizer=Tokenizer(), forward=forward,
                      max_new_tokens=5)
    assert calls == [2]
    assert result.generated_token_ids == (eos,)
    assert result.generated_text == ""
    assert result.stop_reason == "eos"


def test_eos_in_prompt_does_not_stop_generation():
    result = generate("prompt", tokenizer=Tokenizer([1, 5]), forward=logits_for,
                      max_new_tokens=1)
    assert result.prompt_token_ids == (1, 5)
    assert result.generated_token_ids == (3,)
    assert result.stop_reason == "max_new_tokens"


@pytest.mark.parametrize("token,budget,reason", [
    (5, 1, "eos"), (5, 3, "eos"), (3, 1, "capacity"), (3, 3, "capacity"),
])
def test_last_slot_stop_priority_and_no_extra_forward(token, budget, reason):
    calls = []

    def forward(prepared):
        calls.append(prepared.valid_length)
        return logits_for(prepared, token)

    result = generate("prompt", tokenizer=Tokenizer([1] * 255), forward=forward,
                      max_new_tokens=budget)
    assert calls == [255]
    assert result.generated_token_ids == (token,)
    assert result.stop_reason == reason


def test_greedy_tie_uses_lowest_id_without_sampling():
    def forward(prepared):
        logits = logits_for(prepared, 4)
        logits[0, prepared.valid_length - 1, 3] = 20
        return logits

    result = generate("prompt", tokenizer=Tokenizer(), forward=forward,
                      max_new_tokens=1)
    assert result.generated_token_ids == (3,)


@pytest.mark.parametrize("kind,exception", [
    ("shape", ValueError), ("dtype", TypeError),
    ("valid_nan", ValueError), ("padding_inf", ValueError),
])
def test_malformed_or_nonfinite_output_is_rejected(kind, exception):
    tokenizer = Tokenizer()

    def forward(prepared):
        logits = logits_for(prepared)
        if kind == "shape":
            return logits[:, :-1, :]
        if kind == "dtype":
            return logits.astype(np.float64)
        if kind == "valid_nan":
            logits[0, prepared.valid_length - 1, 3] = np.nan
        else:
            logits[0, -1, 0] = np.inf
        return logits

    with pytest.raises(exception):
        generate("prompt", tokenizer=tokenizer, forward=forward, max_new_tokens=1)
    assert tokenizer.decode_calls == []


def test_undecodable_model_tail_argmax_is_error_not_filtered_or_resampled():
    tokenizer = Tokenizer()
    with pytest.raises(ValueError, match="absent"):
        generate("prompt", tokenizer=tokenizer,
                 forward=lambda prepared: logits_for(prepared, 7), max_new_tokens=1)
    assert tokenizer.decode_calls == []


@pytest.mark.parametrize("error", [RuntimeError("backend failed"), TimeoutError("backend timeout")])
def test_backend_exception_identity_is_preserved(error):
    tokenizer = Tokenizer()

    def forward(_prepared):
        raise error

    with pytest.raises(type(error)) as caught:
        generate("prompt", tokenizer=tokenizer, forward=forward, max_new_tokens=2)
    assert caught.value is error
    assert tokenizer.decode_calls == []


def test_cancellation_before_first_forward_and_after_forward():
    with pytest.raises(GenerationCancelledError):
        generate("prompt", tokenizer=Tokenizer(), forward=no_forward,
                 max_new_tokens=1, cancelled=lambda: True)

    cancelled = False
    tokenizer = Tokenizer()
    calls = []

    def forward(prepared):
        nonlocal cancelled
        calls.append(prepared.valid_length)
        cancelled = True
        return logits_for(prepared)

    with pytest.raises(GenerationCancelledError):
        generate("prompt", tokenizer=tokenizer, forward=forward,
                 max_new_tokens=2, cancelled=lambda: cancelled)
    assert calls == [2]
    assert tokenizer.decode_calls == []


@pytest.mark.parametrize("tokens,budget", [((1, 2), 0), ((1,) * 256, 2)])
def test_cancellation_applies_to_no_forward_completion(tokens, budget):
    with pytest.raises(GenerationCancelledError):
        generate("prompt", tokenizer=Tokenizer(tokens), forward=no_forward,
                 max_new_tokens=budget, cancelled=lambda: True)


def test_cancellation_after_decode_does_not_return_success():
    tokenizer = Tokenizer()
    cancelled = False
    original_decode = tokenizer.decode

    def decode(*args, **kwargs):
        nonlocal cancelled
        text = original_decode(*args, **kwargs)
        cancelled = True
        return text

    tokenizer.decode = decode
    with pytest.raises(GenerationCancelledError):
        generate("prompt", tokenizer=tokenizer, forward=logits_for,
                 max_new_tokens=1, cancelled=lambda: cancelled)


@pytest.mark.parametrize("timeout,exception", [
    (-1, ValueError), (float("nan"), ValueError), (float("inf"), ValueError),
    (True, TypeError), ("1", TypeError),
])
def test_invalid_timeout_rejected_before_forward(timeout, exception):
    with pytest.raises(exception):
        generate("prompt", tokenizer=Tokenizer(), forward=no_forward,
                 max_new_tokens=0, timeout_seconds=timeout)


@pytest.mark.parametrize("budget", [0, 1])
def test_zero_timeout_fails_before_forward_even_at_zero_budget(monkeypatch, budget):
    monkeypatch.setattr(generation.time, "monotonic", lambda: 10.0)
    with pytest.raises(TimeoutError):
        generate("prompt", tokenizer=Tokenizer(), forward=no_forward,
                 max_new_tokens=budget, timeout_seconds=0)


def test_elapsed_forward_is_rejected_without_exposing_partial_success(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(generation.time, "monotonic", lambda: now[0])
    tokenizer = Tokenizer()

    def forward(prepared):
        now[0] = 12.0
        return logits_for(prepared, 5)

    with pytest.raises(TimeoutError):
        generate("prompt", tokenizer=tokenizer, forward=forward,
                 max_new_tokens=2, timeout_seconds=1)
    assert tokenizer.decode_calls == []


def test_timeout_after_decode_does_not_return_success(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(generation.time, "monotonic", lambda: now[0])
    tokenizer = Tokenizer()
    original_decode = tokenizer.decode

    def decode(*args, **kwargs):
        text = original_decode(*args, **kwargs)
        now[0] = 11.0
        return text

    tokenizer.decode = decode
    with pytest.raises(TimeoutError):
        generate("prompt", tokenizer=tokenizer, forward=logits_for,
                 max_new_tokens=1, timeout_seconds=1)


def test_workspace_reuse_and_backend_input_mutation_do_not_change_history():
    tokenizer = Tokenizer()
    buffer = np.empty((1, SEQ_LENGTH, 8), dtype=np.float32)
    seen = []
    inputs = []

    def forward(prepared):
        seen.append(tuple(prepared.input_ids[0, :prepared.valid_length]))
        inputs.append(prepared)
        buffer[:] = logits_for(prepared, 3 if len(seen) == 1 else 4)
        prepared.input_ids[:] = 0
        prepared.attention_mask[:] = 0
        return buffer

    result = generate("prompt", tokenizer=tokenizer, forward=forward,
                      max_new_tokens=2)
    assert seen == [(1, 2), (1, 2, 3)]
    assert inputs[0] is not inputs[1]
    assert not np.shares_memory(inputs[0].input_ids, inputs[1].input_ids)
    assert not np.shares_memory(inputs[0].attention_mask, inputs[1].attention_mask)
    buffer[:] = np.nan
    assert result.generated_token_ids == (3, 4)
    assert result.prompt_token_ids == (1, 2)


def test_independent_calls_do_not_share_context_after_success_or_error():
    tokenizer = Tokenizer([1])
    seen = []

    def forward(prepared):
        seen.append(tuple(prepared.input_ids[0, :prepared.valid_length]))
        return logits_for(prepared)

    first = generate("first", tokenizer=tokenizer, forward=forward, max_new_tokens=1)
    tokenizer.tokens[:] = [2, 4]
    with pytest.raises(RuntimeError, match="failed"):
        generate("failed", tokenizer=tokenizer,
                 forward=lambda _prepared: (_ for _ in ()).throw(RuntimeError("failed")),
                 max_new_tokens=1)
    second = generate("second", tokenizer=tokenizer, forward=forward, max_new_tokens=1)
    assert seen == [(1,), (2, 4)]
    assert first.prompt_token_ids == (1,)
    assert second.prompt_token_ids == (2, 4)
    assert first.generated_token_ids == second.generated_token_ids == (3,)


def test_decode_error_propagates_without_conversion_to_stop_reason():
    tokenizer = Tokenizer()
    error = ValueError("decode failed")

    def decode(*_args, **_kwargs):
        raise error

    tokenizer.decode = decode
    with pytest.raises(ValueError) as caught:
        generate("prompt", tokenizer=tokenizer, forward=logits_for, max_new_tokens=1)
    assert caught.value is error


@pytest.mark.parametrize("error", [
    RuntimeError("second step failed"), TimeoutError("second step timeout"),
    KeyboardInterrupt("interrupted second step"),
])
def test_second_step_error_keeps_identity_and_reusable_backend_has_no_history_leak(error):
    tokenizer = Tokenizer()
    seen = []

    def forward(prepared):
        seen.append(tuple(prepared.input_ids[0, :prepared.valid_length]))
        if len(seen) == 2:
            raise error
        return logits_for(prepared, 3 if len(seen) == 1 else 4)

    with pytest.raises(type(error)) as caught:
        generate("first", tokenizer=tokenizer, forward=forward, max_new_tokens=3)
    assert caught.value is error
    assert seen == [(1, 2), (1, 2, 3)]
    assert tokenizer.decode_calls == []

    retry = generate("retry", tokenizer=tokenizer, forward=forward, max_new_tokens=1)
    assert seen[-1] == (1, 2)
    assert retry.generated_token_ids == (4,)
    assert tokenizer.decode_calls == [((4,), True, False)]


def test_second_step_cancellation_discards_partial_result_and_allows_backend_reuse():
    tokenizer = Tokenizer()
    cancelled = False
    seen = []

    def forward(prepared):
        nonlocal cancelled
        seen.append(tuple(prepared.input_ids[0, :prepared.valid_length]))
        if len(seen) == 2:
            cancelled = True
        return logits_for(prepared, 3 if len(seen) == 1 else 4)

    with pytest.raises(GenerationCancelledError):
        generate("first", tokenizer=tokenizer, forward=forward,
                 max_new_tokens=3, cancelled=lambda: cancelled)
    assert seen == [(1, 2), (1, 2, 3)]
    assert tokenizer.decode_calls == []

    cancelled = False
    retry = generate("retry", tokenizer=tokenizer, forward=forward,
                     max_new_tokens=1, cancelled=lambda: cancelled)
    assert seen[-1] == (1, 2)
    assert retry.generated_token_ids == (4,)
    assert tokenizer.decode_calls == [((4,), True, False)]
