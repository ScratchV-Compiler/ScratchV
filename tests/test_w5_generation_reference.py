"""Small, model-free checks of the independent W5 reference oracle."""

from dataclasses import replace

import numpy as np
import pytest

from probes.w5_generation import run
from scratchv.runtime.llm_inputs import prepare_inputs


class Tokenizer:
    model_vocab_size = 8
    pad_token_id = 0
    generation_eos_token_ids = (5, 6)

    def encode(self, text, *, add_special_tokens):
        assert text == "raw prompt" and add_special_tokens is False
        return [1, 2]

    def validate_token_ids(self, tokens):
        if any(not 0 <= token < 7 for token in tokens):
            raise ValueError("undecodable token ID")
        return list(tokens)

    def decode(self, tokens, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert clean_up_tokenization_spaces is False
        pieces = {0: "<pad>", 1: "raw", 2: "prompt", 3: "Paris", 4: "London",
                  5: "<eos>", 6: "<end>"}
        return "".join(pieces[token] for token in tokens
                       if not skip_special_tokens or token not in (0, 5, 6))


class Session:
    def __init__(self):
        self.calls = []

    def run(self, names, feed):
        assert names == ["logits"]
        # The final query exposes all valid keys, including any valid pad ID.
        length = int(np.count_nonzero(feed["attention_mask"][0, 0, -1] == 0))
        self.calls.append(tuple(feed["input_ids"][0, :length]))
        logits = np.zeros((1, 256, 8), dtype=np.float32)
        logits[0, :, 7] = 100  # An incorrect padding-row selection cannot pass.
        row = logits[0, length - 1]
        row[:] = 0
        if length == 2:
            row[3] = row[4] = 10  # The independent argmax must prefer lower ID.
        else:
            row[6] = 10
        return [logits]


def case():
    return {"name": "small", "prompt": "raw prompt", "max_new_tokens": 3,
            "required_substring": "Paris"}


def frozen():
    return {**case(), "prompt_utf8_hex": "raw prompt".encode().hex(),
            "prompt_token_ids": [1, 2], "generated_token_ids": [3, 6],
            "generated_text": "Paris", "stop_reason": "eos"}


@pytest.mark.parametrize("key,replacement", [
    ("prompt", "different"), ("prompt_utf8_hex", "00"),
    ("required_substring", "London"), ("max_new_tokens", 4),
    ("prompt_token_ids", [2, 1]), ("generated_token_ids", [4, 6]),
    ("generated_text", "London"), ("stop_reason", "max_new_tokens"),
])
def test_frozen_comparison_rejects_each_contract_field_change(key, replacement):
    expected = frozen()
    actual = {**expected, key: replacement}
    assert run.compare_frozen(actual, expected) == {
        "passed": False, "different_fields": [key],
    }
    assert expected[key] != replacement


def test_frozen_comparison_ignores_measurement_fields():
    expected = frozen()
    actual = {**expected, "wall_seconds": 2.0, "cpu_seconds": 3.0,
              "process_peak_rss_kib": 100, "steps": [{"timing": 99}]}
    assert run.compare_frozen(actual, expected) == {
        "passed": True, "different_fields": [],
    }


def test_run_case_checks_actual_greedy_rows_eos_and_host_append():
    session = Session()
    result = run.run_case(case(), Tokenizer(), session)
    assert session.calls == [(1, 2), (1, 2, 3)]
    assert result["generated_token_ids"] == [3, 6]
    assert result["generated_text"] == "Paris"
    assert result["stop_reason"] == "eos"
    assert result["semantic_passed"] and result["host_oracle_passed"]
    assert [step["valid_length"] for step in result["steps"]] == [2, 3]
    assert [step["token_id"] for step in result["steps"]] == [3, 6]
    assert result["steps"][0]["top1_top2_margin"] == 0
    assert run.compare_frozen(result, frozen())["passed"]


def test_independent_input_oracle_rejects_wrong_host_append(monkeypatch):
    def broken_generate(_prompt, *, tokenizer, forward, max_new_tokens):
        forward(prepare_inputs([1, 2], pad_token_id=0, vocab_size=8))
        forward(prepare_inputs([1, 2, 4], pad_token_id=0, vocab_size=8))

    monkeypatch.setattr(run, "generate", broken_generate)
    session = Session()
    with pytest.raises(ValueError, match="input_ids"):
        run.run_case(case(), Tokenizer(), session)
    # checked_inputs must fail before a second session call.
    assert session.calls == [(1, 2)]


def test_independent_stop_oracle_rejects_extra_host_forward(monkeypatch):
    original_generate = run.generate

    def broken_generate(prompt, *, tokenizer, forward, max_new_tokens):
        result = original_generate(prompt, tokenizer=tokenizer, forward=forward,
                                   max_new_tokens=max_new_tokens)
        forward(prepare_inputs([1, 2, 3, 6], pad_token_id=0, vocab_size=8))
        return result

    monkeypatch.setattr(run, "generate", broken_generate)
    session = Session()
    with pytest.raises(AssertionError, match="after the independent stop"):
        run.run_case(case(), Tokenizer(), session)
    assert session.calls == [(1, 2), (1, 2, 3)]


@pytest.mark.parametrize("field,value", [
    ("prompt_token_ids", (2, 1)), ("generated_token_ids", (4, 6)),
    ("generated_text", "London"), ("stop_reason", "max_new_tokens"),
])
def test_independent_final_oracle_rejects_corrupted_host_result(monkeypatch, field, value):
    original_generate = run.generate

    def broken_generate(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        return replace(result, **{field: value})

    monkeypatch.setattr(run, "generate", broken_generate)
    with pytest.raises(AssertionError, match="Host"):
        run.run_case(case(), Tokenizer(), Session())


def test_reference_diagnostic_margin_handles_all_finite_fp32_logits():
    import json

    class ExtremeSession:
        def run(self, names, feed):
            logits = np.full((1, 256, 8), -np.finfo(np.float32).max, dtype=np.float32)
            logits[0, 1, 3] = np.finfo(np.float32).max
            return [logits]

    with np.errstate(over="raise"):
        result = run.run_case({**case(), "max_new_tokens": 1}, Tokenizer(), ExtremeSession())
    assert result["generated_token_ids"] == [3]
    assert result["generated_text"] == "Paris"
    assert result["steps"][0]["top1_top2_margin"] == 2.0 * float(np.finfo(np.float32).max)
    json.dumps(result, allow_nan=False)
