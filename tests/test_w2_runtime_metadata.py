"""Negative controls for runtime metadata and public-default gate coverage."""

import json
from types import SimpleNamespace

import pytest

from probes.w2_runtime.run import tokenizer_cases, tokenizer_metadata


class Reference:
    vocab_size = 3
    pad_token_id = 3
    eos_token_id = 4
    bos_token_id = None
    unk_token_id = None

    def __len__(self):
        return 5

    def get_vocab(self):
        return {str(i): i for i in range(5)}

    def encode(self, text, *, add_special_tokens=True):
        return [1]

    def decode(self, ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=None):
        return "text"


@pytest.fixture
def metadata(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 8}), encoding="utf-8")
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [4, 3], "pad_token_id": 3}), encoding="utf-8"
    )
    adapter = SimpleNamespace(
        model_vocab_size=8, base_vocab_size=3, tokenizer_vocab_size=5,
        valid_token_ids=frozenset(range(5)), generation_eos_token_ids=(4, 3),
        pad_token_id=3, eos_token_id=4, bos_token_id=None, unk_token_id=None,
    )
    return adapter, tmp_path


def test_model_rows_and_two_reference_vocabularies_are_independently_checked(metadata):
    adapter, directory = metadata
    assert tokenizer_metadata(adapter, Reference(), Reference(), directory)["model_vocab_size"] == 8
    adapter.model_vocab_size = adapter.tokenizer_vocab_size
    with pytest.raises(ValueError, match="model_vocab_size"):
        tokenizer_metadata(adapter, Reference(), Reference(), directory)


@pytest.mark.parametrize(("name", "value"), [
    ("base_vocab_size", 5), ("tokenizer_vocab_size", 8),
    ("valid_token_ids", frozenset(range(8))), ("generation_eos_token_ids", (4,)),
    ("pad_token_id", 4), ("eos_token_id", 3), ("bos_token_id", 3), ("unk_token_id", 3),
])
def test_bad_metadata_cannot_be_reported_as_a_pass(metadata, name, value):
    adapter, directory = metadata
    setattr(adapter, name, value)
    with pytest.raises(ValueError):
        tokenizer_metadata(adapter, Reference(), Reference(), directory)


def test_second_reference_is_not_skipped(metadata):
    adapter, directory = metadata
    slow = Reference()
    slow.vocab_size = 4
    with pytest.raises(ValueError, match="vocabulary metadata"):
        tokenizer_metadata(adapter, Reference(), slow, directory)


def test_single_generation_eos_is_supported(metadata):
    adapter, directory = metadata
    adapter.generation_eos_token_ids = (4,)
    (directory / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 4, "pad_token_id": 3}), encoding="utf-8"
    )
    assert tokenizer_metadata(adapter, Reference(), Reference(), directory)["generation_eos_token_ids"] == [4]


@pytest.mark.parametrize("method", ["encode", "decode"])
def test_default_only_regression_fails_even_when_explicit_options_are_correct(method):
    class BrokenDefault(Reference):
        def encode(self, text, **kwargs):
            return [2] if method == "encode" and not kwargs else super().encode(text, **kwargs)

        def decode(self, ids, **kwargs):
            return "wrong" if method == "decode" and not kwargs else super().decode(ids, **kwargs)

    rows = []
    corpus = {"cases": [{"name": "example", "text": "text", "ids": [1], "decoded": "text"}]}
    with pytest.raises(ValueError, match="default"):
        tokenizer_cases(BrokenDefault(), Reference(), Reference(), corpus, rows)
    assert rows[0]["passed"] is False


def test_gate_covers_default_and_explicit_calls():
    rows = []
    corpus = {"cases": [{"name": "example", "text": "text", "ids": [1], "decoded": "text"}]}
    tokenizer_cases(Reference(), Reference(), Reference(), corpus, rows)
    assert rows[0]["passed"] is True
    assert rows[0]["checks"] == 24
