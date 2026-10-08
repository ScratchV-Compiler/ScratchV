"""Offline contract tests; official assets are checked by the runtime probe."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tokenizers")
from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

from scratchv.runtime.qwen3_tokenizer import Qwen3Tokenizer


@pytest.fixture
def source(tmp_path):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    engine = Tokenizer(models.BPE(vocab={s: i for i, s in enumerate(alphabet)}, merges=[]))
    engine.normalizer = normalizers.NFC()
    engine.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    engine.decoder = decoders.ByteLevel()
    engine.post_processor = processors.ByteLevel(trim_offsets=False)
    engine.add_special_tokens([
        AddedToken("<|endoftext|>", special=True, normalized=False),
        AddedToken("<|im_start|>", special=True, normalized=False),
        AddedToken("<|im_end|>", special=True, normalized=False),
    ])
    engine.add_tokens([AddedToken("<think>", special=False, normalized=False)])
    engine.save(str(tmp_path / "tokenizer.json"))
    added = {
        str(token_id): {name: getattr(token, name) for name in (
            "content", "lstrip", "rstrip", "single_word", "normalized", "special"
        )}
        for token_id, token in engine.get_added_tokens_decoder().items()
    }
    _write(tmp_path, "tokenizer_config.json", {
        "tokenizer_class": "Qwen2Tokenizer", "add_bos_token": False,
        "bos_token": None, "unk_token": None, "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>", "clean_up_tokenization_spaces": False,
        "added_tokens_decoder": added, "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
    })
    _write(tmp_path, "config.json", {"model_type": "qwen3", "vocab_size": 288, "bos_token_id": 256})
    _write(tmp_path, "generation_config.json", {"eos_token_id": [258, 256], "pad_token_id": 256})
    return tmp_path


def _write(directory: Path, filename: str, value: dict):
    (directory / filename).write_text(json.dumps(value), encoding="utf-8")


def _change(directory, filename, **changes):
    data = json.loads((directory / filename).read_text(encoding="utf-8"))
    data.update(changes)
    _write(directory, filename, data)


def test_metadata_distinguishes_vocab_and_generation_eos(source):
    tokenizer = Qwen3Tokenizer.from_directory(source)
    assert tokenizer.base_vocab_size == 256
    assert tokenizer.tokenizer_vocab_size == 260
    assert tokenizer.model_vocab_size == 288
    assert tokenizer.pad_token_id == 256
    assert tokenizer.eos_token_id == 258
    assert tokenizer.generation_eos_token_ids == (258, 256)
    assert tokenizer.bos_token_id is None
    assert tokenizer.unk_token_id is None
    assert tokenizer.valid_token_ids == frozenset(range(260))
    assert tokenizer.is_valid_token_id(np.int64(259))
    assert not tokenizer.is_valid_token_id(260)
    assert not tokenizer.is_valid_token_id(True)


def test_missing_unk_uses_official_default_but_explicit_null_disables_it(source):
    config = json.loads((source / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert Qwen3Tokenizer.from_directory(source).unk_token_id is None
    del config["unk_token"]
    _write(source, "tokenizer_config.json", config)
    assert Qwen3Tokenizer.from_directory(source).unk_token_id == 256


@pytest.mark.parametrize("text", ["", "Hello, world!", "  a\t\r\nb  ", "中文日本語", "😀👩‍💻", "a\x00b"])
def test_text_round_trip_and_no_implicit_special_tokens(source, text):
    tokenizer = Qwen3Tokenizer.from_directory(source)
    ids = tokenizer.encode(text)
    assert ids == tokenizer.encode(text, add_special_tokens=False)
    assert not set(ids) & {256, 257, 258}
    assert tokenizer.decode(ids) == text


def test_normalization_is_official_json_behavior(source):
    tokenizer = Qwen3Tokenizer.from_directory(source)
    assert tokenizer.encode("e\u0301") == tokenizer.encode("é")
    assert tokenizer.decode(tokenizer.encode("e\u0301")) == "é"


def test_special_tokens_and_ordinary_added_tokens(source):
    tokenizer = Qwen3Tokenizer.from_directory(source)
    text = "a<|im_start|><think><|im_end|><|endoftext|>b"
    ids = tokenizer.encode(text)
    assert all(token_id in ids for token_id in (256, 257, 258, 259))
    assert tokenizer.decode(ids) == text
    assert tokenizer.decode(ids, skip_special_tokens=True) == "a<think>b"
    assert tokenizer.decode(np.int64(258)) == "<|im_end|>"
    assert tokenizer.decode(np.array([257, 258], dtype=np.int64)) == "<|im_start|><|im_end|>"


def test_cleanup_is_opt_in_and_matches_transformers_rules(source):
    tokenizer = Qwen3Tokenizer.from_directory(source)
    text = "I 'm sure , you 're here . We 've seen it ! He 's fine ? I do n't know ."
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert tokenizer.decode(ids, clean_up_tokenization_spaces=True) == "I'm sure, you're here. We've seen it! He's fine? I don't know."
    _change(source, "tokenizer_config.json", clean_up_tokenization_spaces=True)
    configured = Qwen3Tokenizer.from_directory(source)
    assert configured.decode(ids) == tokenizer.decode(ids, clean_up_tokenization_spaces=True)
    assert configured.decode(ids, clean_up_tokenization_spaces=False) == text


@pytest.mark.parametrize("text", [None, 12, b"text", ["text"]])
def test_encode_rejects_non_text(source, text):
    with pytest.raises(TypeError, match="text must be str"):
        Qwen3Tokenizer.from_directory(source).encode(text)


def test_encode_rejects_invalid_unicode(source):
    with pytest.raises(UnicodeEncodeError):
        Qwen3Tokenizer.from_directory(source).encode("\ud800")


@pytest.mark.parametrize("ids", [
    [True], [1.0], ["1"], [[1]], [None], "123", b"123", bytearray(b"123"),
    None, True, {1, 2}, {1: "one"},
])
def test_decode_rejects_noninteger_ids(source, ids):
    with pytest.raises(TypeError):
        Qwen3Tokenizer.from_directory(source).decode(ids)


@pytest.mark.parametrize("ids", [[-1], [260], [287], [288], [2**80]])
def test_decode_refuses_unknown_or_unused_model_rows(source, ids):
    with pytest.raises(ValueError, match="not in the tokenizer vocabulary"):
        Qwen3Tokenizer.from_directory(source).decode(ids)


@pytest.mark.parametrize("option", ["add_special_tokens", "skip_special_tokens", "clean_up_tokenization_spaces"])
def test_flags_are_boolean(source, option):
    tokenizer = Qwen3Tokenizer.from_directory(source)
    with pytest.raises(TypeError, match="must be bool"):
        if option == "add_special_tokens":
            tokenizer.encode("text", **{option: 1})
        else:
            tokenizer.decode([], **{option: 1})


@pytest.mark.parametrize(("filename", "changes", "message"), [
    ("config.json", {"model_type": "other"}, "model_type"),
    ("config.json", {"vocab_size": True}, "vocab_size"),
    ("config.json", {"vocab_size": 259}, "vocab_size"),
    ("tokenizer_config.json", {"tokenizer_class": "OtherTokenizer"}, "tokenizer_class"),
    ("tokenizer_config.json", {"add_bos_token": True}, "add_bos_token"),
    ("tokenizer_config.json", {"add_prefix_space": True}, "add_prefix_space"),
    ("tokenizer_config.json", {"split_special_tokens": True}, "split_special_tokens"),
    ("tokenizer_config.json", {"mask_token": "<think>"}, "mask_token"),
    ("tokenizer_config.json", {"sep_token": "<think>"}, "sep_token"),
    ("tokenizer_config.json", {"cls_token": "<think>"}, "cls_token"),
    ("tokenizer_config.json", {"extra_special_tokens": {"image_token": "<think>"}}, "extra_special_tokens"),
    ("tokenizer_config.json", {"pad_token": "absent"}, "pad_token"),
    ("tokenizer_config.json", {"additional_special_tokens": ["<think>"]}, "special token"),
    ("tokenizer_config.json", {"added_tokens_decoder": {}}, "added token IDs differ"),
    ("generation_config.json", {"eos_token_id": []}, "eos_token_id"),
    ("generation_config.json", {"eos_token_id": [256]}, "include the tokenizer EOS"),
    ("generation_config.json", {"eos_token_id": [258, 280]}, "not in the tokenizer vocabulary"),
    ("generation_config.json", {"pad_token_id": 257}, "pad_token_id"),
])
def test_configuration_mismatch_is_not_silently_ignored(source, filename, changes, message):
    _change(source, filename, **changes)
    with pytest.raises(ValueError, match=message):
        Qwen3Tokenizer.from_directory(source)


def test_inconsistent_added_token_flags_rejected(source):
    data = json.loads((source / "tokenizer_config.json").read_text(encoding="utf-8"))
    data["added_tokens_decoder"]["259"]["special"] = True
    _write(source, "tokenizer_config.json", data)
    with pytest.raises(ValueError, match="inconsistent special"):
        Qwen3Tokenizer.from_directory(source)


@pytest.mark.parametrize("setting", ["padding", "truncation", "post_processor"])
def test_no_implicit_padding_truncation_or_bos(source, setting):
    engine = Tokenizer.from_file(str(source / "tokenizer.json"))
    if setting == "padding":
        engine.enable_padding(length=10, pad_id=256, pad_token="<|endoftext|>")
    elif setting == "truncation":
        engine.enable_truncation(max_length=2)
    else:
        engine.post_processor = processors.TemplateProcessing(single="<|im_start|> $A", special_tokens=[("<|im_start|>", 257)])
    engine.save(str(source / "tokenizer.json"))
    with pytest.raises(ValueError, match="automatic"):
        Qwen3Tokenizer.from_directory(source)


@pytest.mark.parametrize("nested", [False, True])
def test_json_cannot_insert_prefix_space_even_when_config_disables_it(source, nested):
    engine = Tokenizer.from_file(str(source / "tokenizer.json"))
    prefix = pre_tokenizers.ByteLevel(add_prefix_space=True)
    engine.pre_tokenizer = pre_tokenizers.Sequence([prefix]) if nested else prefix
    engine.save(str(source / "tokenizer.json"))
    with pytest.raises(ValueError, match="add_prefix_space"):
        Qwen3Tokenizer.from_directory(source)


def test_generation_single_eos_supported(source):
    _change(source, "generation_config.json", eos_token_id=258)
    assert Qwen3Tokenizer.from_directory(source).generation_eos_token_ids == (258,)


def test_files_are_local_and_required(source):
    (source / "generation_config.json").unlink()
    with pytest.raises(FileNotFoundError):
        Qwen3Tokenizer.from_directory(source)
