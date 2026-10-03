"""Local Qwen3 text tokenization, without a Torch or Transformers dependency.

The caller supplies a verified local snapshot containing tokenizer.json,
tokenizer_config.json, config.json and generation_config.json. This adapter never
downloads files or renders chat templates. Encoding follows tokenizer.json,
including its NFC normalization; a text round trip need not preserve decomposed
Unicode byte-for-byte. Model output width and the set of decodable token IDs are
different contracts (151936 versus 151669 entries in Qwen3-0.6B).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Set
from numbers import Integral
from pathlib import Path

from tokenizers import Tokenizer


def _read_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _cleanup(text: str) -> str:
    # Match PreTrainedTokenizerBase.clean_up_tokenization in Transformers 4.51.3.
    for before, after in (
        (" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"),
        (" n't", "n't"), (" 'm", "'m"), (" 's", "'s"), (" 've", "'ve"),
        (" 're", "'re"),
    ):
        text = text.replace(before, after)
    return text


class Qwen3Tokenizer:
    """Tokenizer for a local Qwen3 snapshot; use :meth:`from_directory`.

    ``eos_token_id`` is the tokenizer's EOS. ``generation_eos_token_ids`` comes
    from generation_config.json and includes *all* generation stop IDs. The
    model's BOS metadata does not cause encoding to insert a BOS token.
    """

    @classmethod
    def from_directory(cls, directory: str | Path) -> Qwen3Tokenizer:
        directory = Path(directory)
        config = _read_config(directory / "tokenizer_config.json")
        model = _read_config(directory / "config.json")
        generation = _read_config(directory / "generation_config.json")
        if config.get("tokenizer_class") not in ("Qwen2Tokenizer", "Qwen2TokenizerFast"):
            raise ValueError("tokenizer_class must identify the Qwen2 tokenizer used by Qwen3")
        if model.get("model_type") != "qwen3":
            raise ValueError("config.json model_type must be qwen3")
        model_vocab_size = model.get("vocab_size")
        if type(model_vocab_size) is not int or model_vocab_size <= 0:
            raise ValueError("model vocab_size must be a positive integer")
        for name in ("add_bos_token", "add_eos_token", "split_special_tokens", "add_prefix_space"):
            if _boolean(config.get(name, False), name):
                raise ValueError(f"Qwen3 text adapter does not support {name}=True")
        # Transformers can add/mark these tokens during initialization. Loading
        # tokenizer.json alone would silently ignore those extra semantics.
        for name in ("mask_token", "sep_token", "cls_token"):
            if config.get(name) is not None:
                raise ValueError(f"Qwen3 text adapter does not support {name}")
        if config.get("extra_special_tokens") not in (None, {}):
            raise ValueError("Qwen3 text adapter does not support extra_special_tokens")

        self = cls()
        self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        if self._tokenizer.padding is not None or self._tokenizer.truncation is not None:
            raise ValueError("tokenizer.json must not enable automatic padding or truncation")
        self._validate_prefix_space()
        self.model_vocab_size = model_vocab_size
        self.base_vocab_size = self._tokenizer.get_vocab_size(with_added_tokens=False)
        self._valid_token_ids = frozenset(self._tokenizer.get_vocab(with_added_tokens=True).values())
        self.tokenizer_vocab_size = len(self._valid_token_ids)
        if not self._valid_token_ids or any(
            token_id < 0 or token_id >= model_vocab_size for token_id in self._valid_token_ids
        ):
            raise ValueError("tokenizer vocabulary does not fit the model vocab_size")
        self.clean_up_tokenization_spaces = _boolean(
            config.get("clean_up_tokenization_spaces", False), "clean_up_tokenization_spaces"
        )
        self._validate_added_tokens(config)
        self.pad_token_id = self._special_id(config, "pad_token", required=True)
        self.eos_token_id = self._special_id(config, "eos_token", required=True)
        self.bos_token_id = self._special_id(config, "bos_token", required=False)
        # The official Qwen2 fast class defaults a missing unk_token to
        # endoftext; an explicit JSON null intentionally disables it instead.
        self.unk_token_id = self._special_id(
            {"unk_token": config.get("unk_token", "<|endoftext|>")}, "unk_token", required=False
        )
        eos_ids = generation.get("eos_token_id")
        if type(eos_ids) is int:
            eos_ids = [eos_ids]
        if not isinstance(eos_ids, list) or not eos_ids:
            raise ValueError("generation eos_token_id must be an integer or nonempty list")
        self.generation_eos_token_ids = tuple(self.validate_token_ids(eos_ids))
        if self.eos_token_id not in self.generation_eos_token_ids:
            raise ValueError("generation EOS IDs must include the tokenizer EOS")
        if (
            generation.get("pad_token_id") != self.pad_token_id
            or type(generation.get("pad_token_id")) is not int
        ):
            raise ValueError("generation pad_token_id must match the tokenizer pad token")
        # Qwen3's JSON post-processor is ByteLevel and inserts no token IDs.
        if self._tokenizer.num_special_tokens_to_add(is_pair=False) != 0:
            raise ValueError("Qwen3 text encoding must not automatically insert BOS/EOS tokens")
        return self

    def _validate_prefix_space(self) -> None:
        # The fixed snapshot uses a Sequence containing ByteLevel(False).
        # Transformers may override a top-level ByteLevel setting from config;
        # refusing prefix insertion keeps this narrow adapter unambiguous for
        # both that layout and the official nested layout.
        pre_tokenizer = self._tokenizer.pre_tokenizer
        pending = [] if pre_tokenizer is None else [json.loads(pre_tokenizer.__getstate__())]
        while pending:
            state = pending.pop()
            if state.get("type") == "ByteLevel" and state.get("add_prefix_space"):
                raise ValueError("tokenizer.json must not enable add_prefix_space")
            if state.get("type") == "Sequence":
                pending.extend(state["pretokenizers"])

    @property
    def valid_token_ids(self) -> frozenset[int]:
        """Actual vocabulary members, including added tokens; not model rows."""
        return self._valid_token_ids

    def is_valid_token_id(self, token_id: object) -> bool:
        return (
            isinstance(token_id, Integral)
            and not isinstance(token_id, bool)
            and int(token_id) in self._valid_token_ids
        )

    def validate_token_ids(self, token_ids: Iterable[int]) -> list[int]:
        """Copy an ordered iterable to ints, refusing unknown IDs or bools."""
        if (
            isinstance(token_ids, (str, bytes, bytearray, Mapping, Set))
            or not isinstance(token_ids, Iterable)
        ):
            raise TypeError("token_ids must be an ordered iterable of integer IDs")
        result = []
        for index, token_id in enumerate(token_ids):
            if not isinstance(token_id, Integral) or isinstance(token_id, bool):
                raise TypeError(f"token_ids[{index}] must be an integer (not bool)")
            if not self.is_valid_token_id(token_id):
                raise ValueError(f"token_ids[{index}]={token_id} is not in the tokenizer vocabulary")
            result.append(int(token_id))
        return result

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode one string, with no padding, truncation, BOS/EOS, or chat template."""
        if not isinstance(text, str):
            raise TypeError("text must be str")
        text.encode("utf-8", errors="strict")  # Reject lone surrogates explicitly.
        _boolean(add_special_tokens, "add_special_tokens")
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(
        self,
        token_ids: Iterable[int] | int,
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
    ) -> str:
        """Decode valid IDs; defaults match the official fast tokenizer API.

        Unlike raw tokenizers.decode, unknown IDs are errors, including unused
        rows below model_vocab_size. A sampled unused row must not disappear.
        """
        _boolean(skip_special_tokens, "skip_special_tokens")
        cleanup = self.clean_up_tokenization_spaces
        if clean_up_tokenization_spaces is not None:
            cleanup = _boolean(clean_up_tokenization_spaces, "clean_up_tokenization_spaces")
        if isinstance(token_ids, Integral):
            token_ids = [token_ids]
        ids = self.validate_token_ids(token_ids)
        text = self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        return _cleanup(text) if cleanup else text

    def _validate_added_tokens(self, config: dict) -> None:
        decoder = config.get("added_tokens_decoder")
        if not isinstance(decoder, dict):
            raise ValueError("added_tokens_decoder must be an object")
        actual = self._tokenizer.get_added_tokens_decoder()
        if set(decoder) != {str(token_id) for token_id in actual}:
            raise ValueError("tokenizer.json and tokenizer_config.json added token IDs differ")
        for token_id, expected in decoder.items():
            if not isinstance(expected, dict):
                raise ValueError(f"added token {token_id} must be an object")
            token = actual[int(token_id)]
            for name in ("content", "lstrip", "rstrip", "single_word", "normalized", "special"):
                value = expected.get(name)
                if type(value) is not type(getattr(token, name)) or value != getattr(token, name):
                    raise ValueError(f"added token {token_id} has inconsistent {name}")
        additional = config.get("additional_special_tokens", [])
        if not isinstance(additional, list):
            raise ValueError("additional_special_tokens must be a list")
        for content in additional:
            self._special_id({"token": content}, "token", required=True)

    def _special_id(self, config: dict, name: str, *, required: bool) -> int | None:
        content = config.get(name)
        if content is None and not required:
            return None
        if not isinstance(content, str) or not content:
            raise ValueError(f"{name} must be a nonempty token string")
        token_id = self._tokenizer.token_to_id(content)
        token = self._tokenizer.get_added_tokens_decoder().get(token_id)
        if token_id is None or token is None or not token.special:
            raise ValueError(f"{name}={content!r} is not a configured special token")
        return token_id
