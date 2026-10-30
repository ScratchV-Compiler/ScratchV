"""Host text conversations using the supplied Qwen3 chat template.

The template is loaded locally and rendered without Transformers or Torch. This
layer owns text history only: the existing generation loop and forward backend
continue to own token generation and execution resources.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .generation import GenerationResult, generate
from .llm_inputs import PreparedInputs

if TYPE_CHECKING:
    from .qwen3_tokenizer import Qwen3Tokenizer


@dataclass(frozen=True)
class ChatMessage:
    """One text-only message; tools and multimodal content are unsupported."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError("role must be system, user or assistant")
        if not isinstance(self.content, str):
            raise TypeError("message content must be str")
        self.content.encode("utf-8", errors="strict")


class Conversation:
    """Transactional text history with explicit non-thinking generation.

    A successful turn with at least one generated token commits its user and
    assistant messages together. Errors, cancellation, timeout, zero budget and
    a full prompt with no room for output leave history unchanged. A response
    stopped by token budget or capacity is retained as the returned partial
    text; its ``GenerationResult.stop_reason`` remains explicit.

    History is re-rendered and encoded for every turn, without truncation or
    implicit removal of old messages. There is no guest session or KV cache.
    The same instance must not be used concurrently or reentrantly.
    """

    def __init__(
        self,
        tokenizer: Qwen3Tokenizer,
        chat_template: str,
        *,
        system_prompt: str | None = None,
    ) -> None:
        if not isinstance(chat_template, str):
            raise TypeError("chat_template must be str")
        if not chat_template.strip():
            raise ValueError("chat_template must not be empty")
        # Match the official template renderer's whitespace and sandbox rules;
        # undefined optional message fields intentionally remain false/empty.
        from jinja2 import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(message: str) -> None:
            raise TemplateError(message)

        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True,
            extensions=["jinja2.ext.loopcontrols"],
        )
        environment.globals["raise_exception"] = raise_exception
        self._template = environment.from_string(chat_template)
        self.tokenizer = tokenizer
        self._initial = (() if system_prompt is None else
                         (ChatMessage("system", system_prompt),))
        self._history = self._initial
        self._busy = False

    @classmethod
    def from_directory(
        cls, directory: str | Path, *, system_prompt: str | None = None,
    ) -> Conversation:
        """Load the official template and tokenizer from the same snapshot."""
        from .qwen3_tokenizer import Qwen3Tokenizer

        directory = Path(directory)
        config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("tokenizer_config.json must contain a JSON object")
        template = config.get("chat_template")
        if not isinstance(template, str) or not template.strip():
            raise ValueError("tokenizer_config.json must contain one nonempty chat_template")
        return cls(Qwen3Tokenizer.from_directory(directory), template,
                   system_prompt=system_prompt)

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        """An immutable snapshot of committed text messages."""
        return self._history

    def reset(self) -> None:
        """Discard completed turns while retaining the initial system prompt."""
        if self._busy:
            raise RuntimeError("conversation has an active turn")
        self._history = self._initial

    def render_prompt(self, user_text: str) -> str:
        """Render committed history plus a pending user turn without mutation."""
        user = ChatMessage("user", user_text)
        messages = [dict(role=item.role, content=item.content)
                    for item in (*self._history, user)]
        return self._template.render(
            messages=messages, tools=None,
            add_generation_prompt=True, enable_thinking=False,
        )

    def turn(
        self,
        user_text: str,
        *,
        forward: Callable[[PreparedInputs], np.ndarray],
        max_new_tokens: int,
        cancelled: Callable[[], bool] | None = None,
        timeout_seconds: float | None = None,
    ) -> GenerationResult:
        """Generate one reply and atomically commit it on success.

        Timeout/cancellation semantics are those of :func:`generate`; a backend
        remains responsible for interrupting and cleaning up blocking work.
        The full rendered prompt must fit the fixed 256-token ABI.
        """
        if self._busy:
            raise RuntimeError("conversation has an active turn")
        self._busy = True
        try:
            prompt = self.render_prompt(user_text)
            result = generate(
                prompt, tokenizer=self.tokenizer, forward=forward,
                max_new_tokens=max_new_tokens, cancelled=cancelled,
                timeout_seconds=timeout_seconds,
            )
            if result.generated_token_ids:
                self._history += (ChatMessage("user", user_text),
                                  ChatMessage("assistant", result.generated_text))
            return result
        finally:
            self._busy = False
