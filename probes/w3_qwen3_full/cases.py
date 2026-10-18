"""Fixed numerical input suite for the released FP32 L256 model."""
from __future__ import annotations

import numpy as np

from scratchv.runtime.llm_inputs import prepare_inputs, QWEN3_VOCAB_SIZE

CASE_NAMES = ("full_seed_0", "full_seed_42", "one_token", "short_17", "short_255",
              "changed_future", "changed_padding")
SEQ = 256


def input_cases():
    """Raw embedding IDs; this suite makes no tokenizer or language-quality claim."""
    result = []
    for name, seed, length in ((CASE_NAMES[0], 0, 256), (CASE_NAMES[1], 42, 256),
                               (CASE_NAMES[2], 7, 1), (CASE_NAMES[3], 7, 17),
                               (CASE_NAMES[4], 7, 255)):
        tokens = np.random.default_rng(seed).integers(1, QWEN3_VOCAB_SIZE, length, dtype=np.int64)
        feed = prepare_inputs(tokens, pad_token_id=0).as_feed()
        result.append((name, length, feed))
    future = {key: value.copy() for key, value in result[0][2].items()}
    future["input_ids"][:, 64:] = future["input_ids"][:, 64:] % (QWEN3_VOCAB_SIZE - 1) + 1
    result.append((CASE_NAMES[5], 256, future))
    padding = {key: value.copy() for key, value in result[3][2].items()}
    padding["input_ids"][:, 17:] = np.random.default_rng(99).integers(
        1, QWEN3_VOCAB_SIZE, (1, 239), dtype=np.int64)
    result.append((CASE_NAMES[6], 17, padding))
    return result
