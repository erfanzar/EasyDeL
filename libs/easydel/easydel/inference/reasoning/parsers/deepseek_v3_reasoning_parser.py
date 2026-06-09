# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reasoning parser for DeepSeek V3 (and look-alike) models.

DeepSeek V3 only emits ``<think>``/``</think>`` reasoning when the chat
template enables it. This module provides a thin wrapper parser that
inspects the tokenizer's chat template at construction time and chooses
the appropriate delegate:

* When the chat template references ``thinking`` / ``enable_thinking`` or
  literally contains ``<think>``, :class:`DeepSeekR1ReasoningParser` is
  used for full chain-of-thought extraction.
* Otherwise :class:`IdentityReasoningParser` is selected as a no-op
  passthrough so the rest of the pipeline behaves as if reasoning was
  disabled.

The wrapper is also registered for GLM-4.5, Holo2 and Kimi-K2 since those
families ship with the same conditional grammar.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...openai_api_modules import DeltaMessage
from ..abstract_reasoning import ReasoningParser, ReasoningParserManager
from .deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from .identity_reasoning_parser import IdentityReasoningParser


@ReasoningParserManager.register_module(["deepseek_v3", "glm45", "holo2", "kimi_k2"])  # pyright: ignore[reportUntypedClassDecorator]
class DeepSeekV3ReasoningParser(ReasoningParser):
    """Conditional reasoning parser that delegates based on tokenizer config.

    Selects :class:`DeepSeekR1ReasoningParser` when the tokenizer's chat
    template indicates that ``<think>`` reasoning is enabled (by referring
    to ``thinking`` / ``enable_thinking`` or literally containing the
    start tag), otherwise falls through to :class:`IdentityReasoningParser`
    so output is passed through unchanged. The delegate can also be
    promoted to R1 at runtime by :meth:`configure_prompt_context` if the
    prompt ends with ``<think>``.

    All :class:`ReasoningParser` methods forward to the selected delegate
    after first mirroring compatibility flags (currently
    ``assume_reasoning``) so manual overrides on the wrapper are honoured.

    Attributes:
        _delegate (ReasoningParser): The currently active delegate parser.
            Starts as either :class:`DeepSeekR1ReasoningParser` or
            :class:`IdentityReasoningParser`; may be promoted to R1 inside
            :meth:`configure_prompt_context`.
    """

    def __init__(self, tokenizer):
        """Initialize and select delegate parser based on tokenizer chat template.

        Inspects the tokenizer's chat template for thinking/enable_thinking
        support. If found, uses DeepSeekR1ReasoningParser; otherwise falls
        back to IdentityReasoningParser.

        Args:
            tokenizer: Tokenizer whose chat template determines the delegate.
        """
        super().__init__(tokenizer)
        # Check if tokenizer's chat template supports thinking
        chat_template = getattr(tokenizer, "chat_template", "") or ""
        has_thinking = (
            "thinking" in chat_template
            or "enable_thinking" in chat_template
            or DeepSeekR1ReasoningParser.start_token in chat_template
        )
        if has_thinking:
            self._delegate = DeepSeekR1ReasoningParser(tokenizer)
        else:
            self._delegate = IdentityReasoningParser(tokenizer)

    def _sync_delegate_state(self) -> None:
        """Mirror compatibility flags from this wrapper onto the delegate.

        ``assume_reasoning`` may be toggled on the wrapper by callers that
        want to force prompt-gated reasoning; this helper keeps the
        delegate observable to such overrides.
        """
        # Keep compatibility with manual overrides on wrapper instances.
        self._delegate.assume_reasoning = self.assume_reasoning

    def configure_prompt_context(self, prompt_text: str, prompt_token_ids: Sequence[int]) -> None:
        """Configure prompt context and possibly promote the delegate to R1.

        When the active delegate is :class:`IdentityReasoningParser` but the
        prompt itself ends with the ``<think>`` start tag (either as text
        or as the trailing token ID), the delegate is replaced with a
        fresh :class:`DeepSeekR1ReasoningParser` so the subsequent
        generation is parsed as chain-of-thought.

        Args:
            prompt_text: Raw prompt text rendered by the chat template.
            prompt_token_ids: Tokenised prompt; the last token ID is
                inspected for an explicit ``<think>`` open tag.
        """
        super().configure_prompt_context(prompt_text, prompt_token_ids)
        if isinstance(self._delegate, IdentityReasoningParser):
            start_token = DeepSeekR1ReasoningParser.start_token
            start_id = self.vocab.get(start_token)
            prompt_has_reasoning_start = bool(prompt_text) and prompt_text.rstrip().endswith(start_token)
            prompt_has_reasoning_start_by_id = bool(prompt_token_ids) and (
                start_id is not None and prompt_token_ids[-1] == start_id
            )
            if prompt_has_reasoning_start or prompt_has_reasoning_start_by_id:
                self._delegate = DeepSeekR1ReasoningParser(self.model_tokenizer)
        self._sync_delegate_state()
        self._delegate.configure_prompt_context(prompt_text, prompt_token_ids)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Forward the end-of-reasoning check to the selected delegate.

        Args:
            input_ids: Generated token-ID sequence inspected by the
                delegate.

        Returns:
            The delegate's verdict on whether reasoning has finished.
        """
        self._sync_delegate_state()
        return self._delegate.is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Forward content-ID extraction to the selected delegate.

        Args:
            input_ids: Full generated token-ID sequence.

        Returns:
            The visible-content token IDs as returned by the delegate.
        """
        self._sync_delegate_state()
        return self._delegate.extract_content_ids(input_ids)

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        """Forward batch reasoning extraction to the selected delegate.

        Args:
            model_output: Full decoded text produced by the model.
            request: Optional inference request forwarded verbatim.

        Returns:
            Tuple ``(reasoning, content)`` as produced by the delegate.
        """
        self._sync_delegate_state()
        return self._delegate.extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request=None,
    ) -> DeltaMessage | None:
        """Forward streaming extraction to the selected delegate.

        Args:
            previous_text: Cumulative text before this chunk.
            current_text: Cumulative text including this chunk.
            delta_text: Newly produced text in this chunk.
            previous_token_ids: Token IDs before this chunk.
            current_token_ids: Token IDs including this chunk.
            delta_token_ids: Token IDs for ``delta_text``.
            request: Optional inference request forwarded verbatim.

        Returns:
            The :class:`DeltaMessage` produced by the delegate (or
            ``None`` when the delegate has nothing to emit).
        """
        self._sync_delegate_state()
        return self._delegate.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )
