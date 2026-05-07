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

"""Reasoning parser for Qwen3 models.

Format: <think>reasoning content</think>response

Qwen3 is strict about requiring both tags unless prompt context indicates
that the start tag was already injected by the chat template.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...openai_api_modules import DeltaMessage
from ..abstract_reasoning import ReasoningParserManager
from ..basic_parsers import BaseThinkingReasoningParser


@ReasoningParserManager.register_module(["qwen3", "qwen3_reasoning"])  # pyright: ignore[reportUntypedClassDecorator]
class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Qwen3 chain-of-thought outputs.

    Qwen3 wraps reasoning in literal ``<think>`` / ``</think>`` markers,
    matching the DeepSeek-R1 grammar. The parser differs from R1 only in
    its strictness:

    - **Missing start tag** -> the output is treated as visible content,
      unless the prompt context indicates that the chat template injected
      ``<think>`` (asymmetric prompt-gated mode).
    - **Missing end tag** -> the unfinished ``<think>...`` segment is kept
      hidden as reasoning when the start tag is present, otherwise the
      text is surfaced as content.

    Expected model output format::

        <think>chain-of-thought reasoning</think>visible response

    Parsed output schema (returned by :meth:`extract_reasoning`):

    - ``reasoning_content``: the text between ``<think>`` and ``</think>``.
    - ``visible_content``: everything after ``</think>`` (and any text
      preceding the start tag prepended).

    Edge cases handled:

    - Both tags absent -> all text becomes visible content.
    - Only start tag -> entire post-start segment hidden as reasoning.
    - Only end tag -> visible content unless prompt-gated reasoning is on.
    - Tags split across streaming chunks -> base-class state machine
      reassembles them.

    Attributes:
        start_token: The opening reasoning marker ``"<think>"``.
        end_token: The closing reasoning marker ``"</think>"``.
    """

    start_token = "<think>"
    end_token = "</think>"

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        """Split a complete Qwen3 generation into reasoning and content.

        Applies Qwen3's stricter tag rules before delegating to the base
        ``<think>``/``</think>`` parser.

        Args:
            model_output: Full decoded text produced by the model.
            request: Optional request context (unused here, accepted for
                interface compatibility).

        Returns:
            Tuple ``(reasoning_content, visible_content)``. Either element
            may be ``None`` when that portion is absent. When neither tag
            is present and prompt-gated mode is off, the whole output is
            returned as ``visible_content``.
        """
        if self.end_token not in model_output:
            # If an explicit reasoning start tag is present, keep the whole
            # unfinished segment hidden as reasoning rather than surfacing it
            # as visible content.
            if self.start_token in model_output or self._is_prompt_reasoning_active():
                return super().extract_reasoning(model_output, request)
            return None, model_output

        # Missing start tag is only allowed when prompt context indicates
        # reasoning already started in the prompt.
        if self.start_token not in model_output:
            if self._is_prompt_reasoning_active():
                return super().extract_reasoning(model_output, request)
            return None, model_output

        return super().extract_reasoning(model_output, request)

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
        """Stream-aware reasoning extraction with Qwen3 strict-tag rules.

        If no ``<think>`` start tag has been observed and prompt-gated
        mode is inactive once the cumulative text exceeds the start tag's
        length, the output is conclusively classified as content and
        forwarded as such. Otherwise the base ``<think>``/``</think>``
        streaming state machine handles the chunk.

        Args:
            previous_text: Cumulative text seen before this chunk.
            current_text: Cumulative text including the new chunk.
            delta_text: Newly produced text in the current chunk.
            previous_token_ids: Token IDs prior to the chunk.
            current_token_ids: Token IDs after the chunk has been added.
            delta_token_ids: Token IDs corresponding to ``delta_text``.
            request: Optional request context.

        Returns:
            A :class:`DeltaMessage` carrying ``reasoning_content`` and/or
            ``content``, or ``None`` when only a boundary token arrived
            and there is nothing meaningful to emit yet.
        """
        # Strict behavior: if no start tag is observed and we are not in prompt-aware
        # asymmetric mode, treat streaming output as content.
        if not self._is_prompt_reasoning_active():
            has_start_in_current = self.start_token in current_text or (
                self._start_token_id is not None and self._start_token_id in current_token_ids
            )
            if current_text and not has_start_in_current and len(current_text) > len(self.start_token):
                return DeltaMessage(content=delta_text) if delta_text else None

        return super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )
