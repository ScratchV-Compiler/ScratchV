"""Synchronous Host greedy generation for the fixed-length Qwen3 ABI.

The caller owns the forward backend and its resources. This module neither
builds a model nor starts a process, and never changes hart or workspace policy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from numbers import Real
import time
from typing import TYPE_CHECKING

import numpy as np

from .llm_inputs import (
    PreparedInputs,
    StopReason,
    generation_stop_reason,
    greedy_token,
    last_valid_logits,
    prepare_inputs,
)

if TYPE_CHECKING:
    from .qwen3_tokenizer import Qwen3Tokenizer


class GenerationCancelledError(RuntimeError):
    """The caller cancelled generation; no successful result is returned."""


@dataclass(frozen=True)
class GenerationResult:
    """Owned token IDs and the generated continuation, excluding the prompt.

    IDs retain generated EOS/special tokens. Display text skips special tokens
    and disables whitespace cleanup; IDs are the authoritative comparison.
    """

    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    generated_text: str
    stop_reason: StopReason


def generate(
    prompt: str,
    *,
    tokenizer: Qwen3Tokenizer,
    forward: Callable[[PreparedInputs], np.ndarray],
    max_new_tokens: int,
    cancelled: Callable[[], bool] | None = None,
    timeout_seconds: float | None = None,
) -> GenerationResult:
    """Generate a raw completion without KV cache, templates or truncation.

    ``forward`` receives fresh right-padded inputs at each step and returns the
    full finite FP32 logits. It may reuse its output buffer between calls; this
    loop retains no output arrays. Each call to ``generate`` has isolated state.

    Cancellation and timeout are cooperative checks before/after forward and
    before returning success. They cannot interrupt a blocking backend. The
    backend must enforce its own execution timeout and clean up its processes;
    backend exceptions (including KeyboardInterrupt) propagate unchanged.
    A finite nonnegative timeout starts after input/configuration validation.
    Zero expires immediately, even when the token budget is zero.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be str")
    if not callable(forward):
        raise TypeError("forward must be callable")
    if cancelled is not None and not callable(cancelled):
        raise TypeError("cancelled must be callable or None")
    if timeout_seconds is not None:
        if isinstance(timeout_seconds, (bool, np.bool_)) or not isinstance(timeout_seconds, Real):
            raise TypeError("timeout_seconds must be a real number or None")
        timeout_seconds = float(timeout_seconds)
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("timeout_seconds must be finite and nonnegative")

    prompt_ids = tuple(tokenizer.validate_token_ids(
        tokenizer.encode(prompt, add_special_tokens=False)
    ))
    vocab_size = tokenizer.model_vocab_size
    pad_token_id = tokenizer.pad_token_id
    eos_ids = tuple(tokenizer.validate_token_ids(tokenizer.generation_eos_token_ids))
    history = list(prompt_ids)
    prepared = prepare_inputs(history, pad_token_id=pad_token_id, vocab_size=vocab_size)
    # Validate the budget even if capacity or EOS would otherwise end the loop.
    stop = generation_stop_reason(
        valid_length=len(history), generated_tokens=0,
        max_new_tokens=max_new_tokens, eos_token_ids=eos_ids, vocab_size=vocab_size,
    )
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds

    def check_abort() -> None:
        if cancelled is not None and cancelled():
            raise GenerationCancelledError("generation cancelled")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("generation timeout expired")

    generated: list[int] = []
    while True:
        check_abort()
        if stop is not None:
            text = tokenizer.decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            check_abort()
            return GenerationResult(prompt_ids, tuple(generated), text, stop)

        logits = forward(prepared)
        check_abort()
        row = last_valid_logits(logits, len(history), vocab_size=vocab_size)
        # Release the full output before the next call allocates/reuses a buffer.
        del logits
        next_id = greedy_token(row)
        del row
        # The model has unused vocabulary rows: never mask or resample them.
        tokenizer.validate_token_ids([next_id])
        check_abort()
        generated.append(next_id)
        history.append(next_id)
        stop = generation_stop_reason(
            valid_length=len(history), generated_tokens=len(generated),
            max_new_tokens=max_new_tokens, last_token_id=next_id,
            eos_token_ids=eos_ids, vocab_size=vocab_size,
        )
        if stop is None:
            prepared = prepare_inputs(history, pad_token_id=pad_token_id, vocab_size=vocab_size)
