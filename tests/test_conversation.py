"""Host text-history tests; optional official parity loads tokenizers only."""

from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jinja2")
from scratchv.runtime import conversation
from scratchv.runtime.conversation import ChatMessage, Conversation
from scratchv.runtime.generation import GenerationCancelledError, GenerationResult


TEMPLATE = "{%- for m in messages %}{{ m.role }}:{{ m.content }}|{% endfor -%}{{ 'assistant:' if add_generation_prompt else '' }}{{ 'nonthinking:' if enable_thinking is false else 'thinking:' }}"


class Tokenizer:
    model_vocab_size = 258
    pad_token_id = 256
    generation_eos_token_ids = (257,)

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))

    def validate_token_ids(self, tokens):
        if any(type(x) is not int or not 0 <= x < 258 for x in tokens):
            raise ValueError("invalid token")
        return list(tokens)

    def decode(self, tokens, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return bytes(x for x in tokens if x < 256).decode("utf-8")


def logits(prepared, token=97):
    value = np.zeros((1, 256, 258), dtype=np.float32)
    value[0, prepared.valid_length - 1, token] = 1
    return value


def no_forward(_prepared):
    raise AssertionError("backend must not run")


def test_three_rounds_preserve_exact_order_and_nonthinking_prompt():
    chat = Conversation(Tokenizer(), TEMPLATE, system_prompt="brief")
    inputs = []
    for index, question in enumerate(("first", "second", "third")):
        expected = "system:brief|" + "".join(
            "user:" + q + "|assistant:a|" for q in ("first", "second", "third")[:index]
        ) + "user:" + question + "|assistant:nonthinking:"
        assert chat.render_prompt(question) == expected

        def forward(prepared):
            inputs.append(bytes(prepared.input_ids[0, :prepared.valid_length].tolist()).decode("utf-8"))
            return logits(prepared)

        result = chat.turn(question, forward=forward, max_new_tokens=1)
        assert result.generated_text == "a"
        assert result.stop_reason == "max_new_tokens"
        assert inputs[-1] == expected
    assert chat.history == (
        ChatMessage("system", "brief"), ChatMessage("user", "first"), ChatMessage("assistant", "a"),
        ChatMessage("user", "second"), ChatMessage("assistant", "a"),
        ChatMessage("user", "third"), ChatMessage("assistant", "a"),
    )


def test_history_snapshot_is_immutable_and_sessions_reset_independently():
    first = Conversation(Tokenizer(), TEMPLATE, system_prompt="one")
    second = Conversation(Tokenizer(), TEMPLATE)
    before = first.history
    first.turn("first", forward=logits, max_new_tokens=1)
    assert before == (ChatMessage("system", "one"),)
    assert second.history == ()
    with pytest.raises(FrozenInstanceError):
        first.history[0].content = "changed"
    first.reset()
    assert first.history == before
    second.turn("second", forward=logits, max_new_tokens=1)
    assert first.history == before
    assert len(second.history) == 2


@pytest.mark.parametrize("role", ["tool", "developer", "function", "", None, 1])
def test_rejects_unsupported_roles(role):
    with pytest.raises(ValueError, match="role"):
        ChatMessage(role, "text")


@pytest.mark.parametrize("content", [None, b"text", [], {"type": "image"}, 1, True])
def test_rejects_nontext_content_before_backend(content):
    chat = Conversation(Tokenizer(), TEMPLATE)
    with pytest.raises(TypeError, match="content"):
        chat.turn(content, forward=no_forward, max_new_tokens=1)
    assert chat.history == ()


def test_rejects_invalid_unicode():
    with pytest.raises(UnicodeEncodeError):
        ChatMessage("user", "\ud800")


@pytest.mark.parametrize("template", [None, [], "", "   "])
def test_requires_explicit_nonempty_template(template):
    with pytest.raises((TypeError, ValueError), match="chat_template"):
        Conversation(Tokenizer(), template)


def test_invalid_system_prompt():
    with pytest.raises(TypeError, match="content"):
        Conversation(Tokenizer(), TEMPLATE, system_prompt=[])


@pytest.mark.parametrize("failure", [RuntimeError("backend failed"), KeyboardInterrupt(), SystemExit(2)])
def test_error_after_partial_generation_rolls_back_and_preserves_original_exception(failure):
    chat = Conversation(Tokenizer(), TEMPLATE)
    chat.turn("committed", forward=logits, max_new_tokens=1)
    before = chat.history
    calls = 0

    def forward(prepared):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        return logits(prepared)

    with pytest.raises(type(failure)) as caught:
        chat.turn("failed", forward=forward, max_new_tokens=3)
    assert caught.value is failure
    assert chat.history == before
    chat.turn("retry", forward=logits, max_new_tokens=1)
    assert len(chat.history) == len(before) + 2


def test_cancel_during_forward_rolls_back():
    chat = Conversation(Tokenizer(), TEMPLATE)
    aborted = False

    def forward(prepared):
        nonlocal aborted
        aborted = True
        return logits(prepared)

    with pytest.raises(GenerationCancelledError):
        chat.turn("cancel", forward=forward, max_new_tokens=1, cancelled=lambda: aborted)
    assert chat.history == ()


def test_timeout_after_forward_rolls_back(monkeypatch):
    from scratchv.runtime import generation
    moments = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(generation.time, "monotonic", lambda: next(moments))
    chat = Conversation(Tokenizer(), TEMPLATE)
    with pytest.raises(TimeoutError):
        chat.turn("timeout", forward=logits, max_new_tokens=1, timeout_seconds=1)
    assert chat.history == ()


def test_nonfinite_output_rolls_back():
    chat = Conversation(Tokenizer(), TEMPLATE)

    def forward(prepared):
        value = logits(prepared)
        value[0, -1, 0] = np.nan
        return value

    with pytest.raises(ValueError):
        chat.turn("bad", forward=forward, max_new_tokens=1)
    assert chat.history == ()


@pytest.mark.parametrize("budget", [0, False, -1])
def test_zero_budget_no_fake_turn_and_bad_budget_no_mutation(budget):
    chat = Conversation(Tokenizer(), TEMPLATE)
    if type(budget) is int and budget == 0:
        result = chat.turn("valid", forward=no_forward, max_new_tokens=budget)
        assert result.generated_token_ids == ()
    else:
        with pytest.raises((TypeError, ValueError)):
            chat.turn("valid", forward=no_forward, max_new_tokens=budget)
    assert chat.history == ()


def test_eos_only_reply_commits_legitimate_empty_assistant():
    chat = Conversation(Tokenizer(), TEMPLATE)
    result = chat.turn("hello", forward=lambda p: logits(p, 257), max_new_tokens=2)
    assert result.stop_reason == "eos"
    assert chat.history == (ChatMessage("user", "hello"), ChatMessage("assistant", ""))


def test_capacity_checks_full_rendered_history_without_truncation():
    chat = Conversation(Tokenizer(), TEMPLATE)
    chat.turn("prior", forward=logits, max_new_tokens=1)
    before = chat.history
    with pytest.raises(ValueError):
        chat.turn("x" * 256, forward=no_forward, max_new_tokens=1)
    assert chat.history == before
    assert "prior" in chat.render_prompt("next")


def test_exact_capacity_returns_no_fake_turn_and_last_slot_commits():
    chat = Conversation(Tokenizer(), "{{ messages[-1].content }}")
    result = chat.turn("x" * 256, forward=no_forward, max_new_tokens=1)
    assert result.stop_reason == "capacity"
    assert chat.history == ()
    result = chat.turn("x" * 255, forward=logits, max_new_tokens=2)
    assert result.stop_reason == "capacity"
    assert chat.history[-1] == ChatMessage("assistant", "a")


@pytest.mark.parametrize("action", ["turn", "reset"])
def test_reentrant_state_mutation_rejected(action):
    chat = Conversation(Tokenizer(), TEMPLATE)

    def forward(prepared):
        if action == "turn":
            chat.turn("nested", forward=no_forward, max_new_tokens=1)
        else:
            chat.reset()
        return logits(prepared)

    with pytest.raises(RuntimeError, match="active turn"):
        chat.turn("outer", forward=forward, max_new_tokens=1)
    assert chat.history == ()
    chat.reset()


@pytest.mark.parametrize("config", [{}, {"chat_template": []}, [], {"chat_template": ""}])
def test_from_directory_rejects_missing_or_ambiguous_template(tmp_path, config):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        Conversation.from_directory(tmp_path)


@pytest.mark.parametrize("system_prompt", [None, "Give brief answers."])
def test_official_three_round_rendering_matches_transformers(monkeypatch, system_prompt):
    """Runs when local assets exist; does not load weights or execute a model."""
    pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    root = Path(os.environ.get("SCRATCHV_QWEN3_TOKENIZER_DIR", Path(__file__).resolve().parents[1] / "assets/qwen3-0.6b/tokenizer"))
    if not (root / "tokenizer.json").is_file():
        pytest.skip("official local Qwen3 tokenizer assets are not installed")
    official = transformers.AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    chat = Conversation.from_directory(root, system_prompt=system_prompt)
    replies = iter(("Paris", "Berlin", "Tokyo"))

    def fake_generate(prompt, *, tokenizer, **kwargs):
        reply = next(replies)
        return GenerationResult(tuple(tokenizer.encode(prompt, add_special_tokens=False)),
                                tuple(tokenizer.encode(reply, add_special_tokens=False)), reply, "max_new_tokens")

    monkeypatch.setattr(conversation, "generate", fake_generate)
    for text in ("What is France's capital?", "And Germany's?", "What about Japan?"):
        messages = [dict(role=m.role, content=m.content) for m in chat.history]
        messages.append(dict(role="user", content=text))
        expected = official.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        rendered = chat.render_prompt(text)
        assert rendered == expected
        assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert chat.tokenizer.encode(rendered, add_special_tokens=False) == official.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        assert len(chat.tokenizer.encode(rendered, add_special_tokens=False)) < 256
        chat.turn(text, forward=no_forward, max_new_tokens=1)
    assert len(chat.history) == (7 if system_prompt is not None else 6)
