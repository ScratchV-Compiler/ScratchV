"""Greedy generation using one prefill followed by single-token decode calls."""
from __future__ import annotations
import math
from numbers import Real
import time
import numpy as np
from .generation import GenerationCancelledError, GenerationResult
from .kv_cache import cached_logits
from .llm_inputs import generation_stop_reason, greedy_token, prepare_inputs


def generate_cached(prompt, *, tokenizer, session, max_new_tokens,
                    cancelled=None, timeout_seconds=None, timings=None):
    """One isolated request. The caller owns and closes the resident session.

    The last selected token remains pending: generating N>0 tokens performs
    exactly one prefill and N-1 decode calls. Cache length advances only in the
    backend, after successful validation and commit.
    """
    if timings is not None and not isinstance(timings, dict):
        raise TypeError('timings must be a dict or None')
    started = time.perf_counter()
    selected_seconds = 0.0
    if not isinstance(prompt, str):
        raise TypeError('prompt must be str')
    if cancelled is not None and not callable(cancelled):
        raise TypeError('cancelled must be callable or None')
    if timeout_seconds is not None:
        if isinstance(timeout_seconds, (bool, np.bool_)) or not isinstance(timeout_seconds, Real):
            raise TypeError('timeout_seconds must be a real number or None')
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError('timeout_seconds must be finite and nonnegative')
    prompt_ids = tuple(tokenizer.validate_token_ids(tokenizer.encode(prompt, add_special_tokens=False)))
    vocab = tokenizer.model_vocab_size
    prepare_inputs(prompt_ids, pad_token_id=tokenizer.pad_token_id, vocab_size=vocab)
    eos = tuple(tokenizer.validate_token_ids(tokenizer.generation_eos_token_ids))
    stop = generation_stop_reason(valid_length=len(prompt_ids), generated_tokens=0,
                                  max_new_tokens=max_new_tokens, eos_token_ids=eos, vocab_size=vocab)
    deadline = None if timeout_seconds is None else time.monotonic()+timeout_seconds
    def check():
        if cancelled is not None and cancelled():
            raise GenerationCancelledError('generation cancelled')
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('generation timeout expired')
    generated = []
    check()
    if stop is None:
        session.reset()
        check()
        logits = session.prefill(prompt_ids)
        while True:
            check()
            selected_start = time.perf_counter()
            token = greedy_token(cached_logits(logits, vocab))
            tokenizer.validate_token_ids([token])
            selected_seconds += time.perf_counter()-selected_start
            generated.append(token)
            stop = generation_stop_reason(
                valid_length=len(prompt_ids)+len(generated), generated_tokens=len(generated),
                max_new_tokens=max_new_tokens, last_token_id=token, eos_token_ids=eos, vocab_size=vocab)
            check()
            if stop is not None:
                break
            logits = session.decode(token)
    text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    check()
    if timings is not None:
        timings.update(token_selection_s=selected_seconds,total_s=time.perf_counter()-started,
                       generated_tokens=len(generated),pending_tokens=1 if generated else 0)
    return GenerationResult(prompt_ids, tuple(generated), text, stop)
