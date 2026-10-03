"""Host input preparation and greedy sampling for fixed-length Qwen3 graphs.

These helpers do not tokenize text, invoke a model, or run a generation loop.
The model ABI remains batch 1, sequence length 256, INT64 IDs and FP32 tensors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np


SEQ_LENGTH = 256
QWEN3_VOCAB_SIZE = 151936
StopReason = Literal["eos", "capacity", "max_new_tokens"]


def _integer(value: object, name: str, *, minimum: int = 0,
             maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be in {bound}; got {result}")
    return result


def _vocabulary_size(vocab_size: object) -> int:
    # A vocabulary size is a count; its largest valid ID must fit INT64.
    return _integer(vocab_size, "vocab_size", minimum=1,
                    maximum=int(np.iinfo(np.int64).max) + 1)


def _valid_length(valid_length: object) -> int:
    return _integer(valid_length, "valid_length", minimum=1, maximum=SEQ_LENGTH)


def _token_sequence(token_ids: Sequence[int] | np.ndarray, *,
                    vocab_size: int, name: str) -> list[int]:
    if isinstance(token_ids, np.ndarray):
        if np.ma.isMaskedArray(token_ids):
            raise TypeError(f"{name} must not be a masked array")
        if token_ids.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional sequence")
        if token_ids.dtype.kind not in "iu":
            raise TypeError(f"{name} array must have an integer dtype")
    elif isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
        raise TypeError(f"{name} must be a sequence of integer token IDs")
    return [_integer(value, f"{name}[{index}]", maximum=vocab_size - 1)
            for index, value in enumerate(token_ids)]


@dataclass(frozen=True)
class PreparedInputs:
    """Owned model input arrays plus the unpadded token count.

    Arrays are mutable for the model caller. ``as_feed`` returns references to
    these arrays, so changing the feed also changes this instance's tensors.
    Construct instances through :func:`prepare_inputs` for validated inputs.
    """

    input_ids: np.ndarray
    attention_mask: np.ndarray
    valid_length: int

    def as_feed(self) -> dict[str, np.ndarray]:
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}


def build_attention_mask(valid_length: int) -> np.ndarray:
    """Return causal + key-padding additive mask ``[1, 1, 256, 256]``.

    Query row ``q`` may attend to key ``k`` exactly when ``k <= q`` and
    ``k < valid_length``. Padded query rows still see earlier valid keys; fully
    masking those rows is not the fixed Qwen3 export's mask contract.
    """
    length = _valid_length(valid_length)
    positions = np.arange(SEQ_LENGTH)
    allowed = (positions[None, :] <= positions[:, None]) & (positions[None, :] < length)
    mask = np.where(allowed, np.float32(0), np.finfo(np.float32).min)
    return mask.reshape(1, 1, SEQ_LENGTH, SEQ_LENGTH)


def prepare_inputs(token_ids: Sequence[int] | np.ndarray, *, pad_token_id: int,
                   vocab_size: int = QWEN3_VOCAB_SIZE) -> PreparedInputs:
    """Validate unpadded IDs and return right-padded, contiguous ABI inputs.

    The length comes from the supplied sequence, never from occurrences of the
    pad ID: a valid prompt may itself contain that ID. Empty or over-capacity
    prompts fail instead of being silently truncated. Integer arrays of other
    widths are accepted only after every ID passes the INT64 range checks.
    """
    vocab = _vocabulary_size(vocab_size)
    pad = _integer(pad_token_id, "pad_token_id", maximum=vocab - 1)
    tokens = _token_sequence(token_ids, vocab_size=vocab, name="token_ids")
    length = _valid_length(len(tokens))
    ids = np.full((1, SEQ_LENGTH), pad, dtype=np.int64)
    ids[0, :length] = tokens
    return PreparedInputs(ids, build_attention_mask(length), length)


def _fp32_finite(array: object, name: str) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy ndarray")
    if np.ma.isMaskedArray(array):
        raise TypeError(f"{name} must not be a masked array")
    if array.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} must have dtype float32; got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def last_valid_logits(logits: np.ndarray, valid_length: int, *,
                      vocab_size: int = QWEN3_VOCAB_SIZE) -> np.ndarray:
    """Validate the full model output and copy the last valid token's logits.

    All positions, including padding, must be finite. Returning an owned vector
    lets a runner release its full output buffer after selecting this row.
    """
    length = _valid_length(valid_length)
    vocab = _vocabulary_size(vocab_size)
    if not isinstance(logits, np.ndarray):
        raise TypeError("logits must be a NumPy ndarray")
    expected = (1, SEQ_LENGTH, vocab)
    if logits.shape != expected:
        raise ValueError(f"logits must have shape {expected}; got {logits.shape}")
    checked = _fp32_finite(logits, "logits")
    return checked[0, length - 1, :].copy()


def greedy_token(logits: np.ndarray) -> int:
    """Argmax of one finite FP32 vocabulary vector; ties choose the lowest ID."""
    if not isinstance(logits, np.ndarray):
        raise TypeError("logits must be a NumPy ndarray")
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("logits must be a nonempty one-dimensional vocabulary vector")
    checked = _fp32_finite(logits, "logits")
    return int(np.argmax(checked))


def greedy_next_token(logits: np.ndarray, valid_length: int, *,
                      vocab_size: int = QWEN3_VOCAB_SIZE) -> int:
    """Select the next ID from ``logits[0, valid_length - 1]``, without softmax."""
    return greedy_token(last_valid_logits(logits, valid_length, vocab_size=vocab_size))


def generation_stop_reason(*, valid_length: int, generated_tokens: int,
                           max_new_tokens: int, last_token_id: int | None = None,
                           eos_token_ids: Sequence[int] | np.ndarray = (),
                           vocab_size: int = QWEN3_VOCAB_SIZE) -> StopReason | None:
    """Report whether a host loop must stop before its next forward call.

    Call initially with ``generated_tokens=0`` and after appending each newly
    sampled ID. ``valid_length`` includes the nonempty prompt and generated IDs.
    EOS applies only to a generated token, not an EOS-like token in the prompt.
    Simultaneous stop conditions resolve as EOS, capacity, then token budget.
    No truncation, model execution or token append is performed here.
    """
    length = _valid_length(valid_length)
    generated = _integer(generated_tokens, "generated_tokens", maximum=length - 1)
    budget = _integer(max_new_tokens, "max_new_tokens")
    vocab = _vocabulary_size(vocab_size)
    eos = _token_sequence(eos_token_ids, vocab_size=vocab, name="eos_token_ids")
    if generated and last_token_id is None:
        raise ValueError("last_token_id is required after generating a token")
    last = None if last_token_id is None else _integer(last_token_id, "last_token_id",
                                                      maximum=vocab - 1)
    if generated and last in eos:
        return "eos"
    if length == SEQ_LENGTH:
        return "capacity"
    if generated >= budget:
        return "max_new_tokens"
    return None
