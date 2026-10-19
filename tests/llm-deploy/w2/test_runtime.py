"""Official offline tokenizer comparisons and host input/sampling behavior."""
import json
from pathlib import Path

import numpy as np
import pytest

from probes.w2_runtime.run import MANIFEST, tokenizer_metadata
from scratchv.runtime.llm_inputs import prepare_inputs, greedy_next_token, generation_stop_reason

CORPUS = json.loads(
    (Path(__file__).resolve().parents[3] / "probes/w2_runtime/corpus.json").read_text(encoding="utf-8")
)


def test_tokenizer_assets_and_metadata(tokenizers, tokenizer_dir):
    assert CORPUS["revision"] == MANIFEST["revision"]
    assert CORPUS["model_id"] == MANIFEST["model_id"]
    assert len(CORPUS["cases"]) == len({case["name"] for case in CORPUS["cases"]}) == 20
    metadata = tokenizer_metadata(*tokenizers, tokenizer_dir)
    assert metadata["model_vocab_size"] == 151936


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("reference_index", [1, 2], ids=["fast", "slow"])
def test_tokenizer_defaults(case, reference_index, tokenizers):
    adapter, reference = tokenizers[0], tokenizers[reference_index]
    ids = adapter.encode(case["text"])
    assert ids == reference.encode(case["text"]) == case["ids"]
    assert adapter.decode(ids) == reference.decode(ids) == case["decoded"]


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("special", [False, True], ids=["no-special", "special"])
@pytest.mark.parametrize("reference_index", [1, 2], ids=["fast", "slow"])
def test_tokenizer_explicit_flags(case, special, reference_index, tokenizers):
    adapter, reference = tokenizers[0], tokenizers[reference_index]
    ids = adapter.encode(case["text"], add_special_tokens=special)
    assert ids == reference.encode(case["text"], add_special_tokens=special) == case["ids"]
    for skip in (False, True):
        for cleanup in (False, True):
            options = {"skip_special_tokens": skip, "clean_up_tokenization_spaces": cleanup}
            actual = adapter.decode(ids, **options)
            assert actual == reference.decode(ids, **options), options
            if not skip and not cleanup:
                assert actual == case["decoded"]


@pytest.mark.parametrize("length", [1, 17, 255, 256])
def test_input_mask_and_greedy(length, tokenizers):
    adapter = tokenizers[0]
    ids = [adapter.pad_token_id] + [42] * (length - 1)
    prepared = prepare_inputs(ids, pad_token_id=adapter.pad_token_id, vocab_size=adapter.model_vocab_size)
    expected = np.full((1, 1, 256, 256), np.finfo(np.float32).min, dtype=np.float32)
    for query in range(256):
        for key in range(min(query + 1, length)):
            expected[0, 0, query, key] = 0
    assert prepared.valid_length == length
    assert prepared.input_ids.shape == (1, 256)
    assert prepared.input_ids.dtype == np.int64
    np.testing.assert_array_equal(prepared.input_ids[0, :length], ids)
    assert np.all(prepared.input_ids[0, length:] == adapter.pad_token_id)
    assert prepared.attention_mask.dtype == np.float32
    np.testing.assert_array_equal(prepared.attention_mask, expected)
    logits = np.zeros((1, 256, 8), dtype=np.float32)
    logits[0, -1, 7] = 100
    logits[0, length - 1, 7] = 0
    logits[0, length - 1, [2, 5]] = 1
    assert greedy_next_token(logits, length, vocab_size=8) == 2


def test_generation_stops(tokenizers):
    for eos in tokenizers[0].generation_eos_token_ids:
        assert generation_stop_reason(
            valid_length=2, generated_tokens=1, max_new_tokens=3,
            last_token_id=eos, eos_token_ids=tokenizers[0].generation_eos_token_ids,
        ) == "eos"
    assert generation_stop_reason(valid_length=256, generated_tokens=0, max_new_tokens=1) == "capacity"
    assert generation_stop_reason(valid_length=1, generated_tokens=0, max_new_tokens=0) == "max_new_tokens"
