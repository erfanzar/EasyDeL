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

"""Reasoning parser for GptOss (OpenAI open-weights) chain-of-thought models.

GptOss models emit reasoning inside a ``<|channel|>``-tagged analysis section
followed by a ``<|message|>`` delimiter that switches to the visible response.
Cumulative format::

    <|channel|>analysis text<|message|>visible response

This parser splits the cumulative output at those two markers and emits the
two halves as ``reasoning_content`` and ``content`` respectively, both in
batch mode (:meth:`extract_reasoning`) and during streaming
(:meth:`extract_reasoning_streaming`).
"""

from __future__ import annotations

from collections.abc import Sequence

from transformers import AutoTokenizer as AnyTokenizer

from ...openai_api_modules import DeltaMessage
from ..abstract_reasoning import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module(["openai_gptoss", "gptoss"])  # pyright: ignore[reportUntypedClassDecorator]
class GptOssReasoningParser(ReasoningParser):
    """Reasoning parser for GptOss models using channel/message marker tags.

    Splits cumulative model output into a hidden reasoning section and a
    visible content section using the canonical GptOss grammar:

        ``<|channel|>analysis_text<|message|>response_text``

    The parser holds a small two-flag state machine (``_in_reasoning`` /
    ``_reasoning_done``) that survives across streaming deltas so it can
    correctly attribute partial chunks to either the reasoning or content
    side even when the ``<|channel|>`` or ``<|message|>`` boundary tokens
    straddle a chunk.

    Attributes:
        CHANNEL_TAG (str): Marker that opens the analysis (reasoning)
            channel.
        MESSAGE_TAG (str): Marker that closes reasoning and switches the
            output stream to visible content.
    """

    CHANNEL_TAG = "<|channel|>"
    MESSAGE_TAG = "<|message|>"

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the parser and resolve marker token IDs.

        Args:
            tokenizer: HuggingFace tokenizer whose vocabulary is consulted
                to resolve the integer token IDs of ``CHANNEL_TAG`` and
                ``MESSAGE_TAG`` for id-level boundary scans.
        """
        super().__init__(tokenizer)
        self._channel_token_id = self.vocab.get(self.CHANNEL_TAG)
        self._message_token_id = self.vocab.get(self.MESSAGE_TAG)
        self._in_reasoning = False
        self._reasoning_done = False

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Check whether the reasoning section has ended.

        Args:
            input_ids: Sequence of decoded token IDs scanned for the
                ``<|message|>`` marker that closes reasoning.

        Returns:
            ``True`` when ``<|message|>`` has been emitted at least once,
            ``False`` while the model is still inside the analysis channel
            (or when the marker is not in the tokenizer vocabulary).
        """
        if self._message_token_id is not None:
            return self._message_token_id in input_ids
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Return token IDs emitted after the ``<|message|>`` delimiter.

        Args:
            input_ids: Full generated token-ID sequence.

        Returns:
            The slice of ``input_ids`` that follows the first
            ``<|message|>`` token, or a copy of the whole list when the
            delimiter has not appeared yet.
        """
        if self._message_token_id is None or self._message_token_id not in input_ids:
            return list(input_ids)
        idx = input_ids.index(self._message_token_id)
        return input_ids[idx + 1 :]

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        """Split a complete generation into reasoning and visible content.

        Args:
            model_output: Full decoded text produced by the model.
            request: Optional inference request; unused, kept for interface
                parity with sibling parsers.

        Returns:
            Tuple ``(reasoning, content)``. ``reasoning`` is ``None`` when
            the ``<|channel|>`` marker is absent; ``content`` is ``None``
            when the ``<|message|>`` marker has not yet been emitted.
        """
        if self.CHANNEL_TAG not in model_output:
            return None, model_output

        after_channel = model_output.split(self.CHANNEL_TAG, 1)[1]

        if self.MESSAGE_TAG not in after_channel:
            return after_channel.strip() or None, None

        reasoning, content = after_channel.split(self.MESSAGE_TAG, 1)
        return reasoning.strip() or None, content.strip() or None

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
        """Route a streaming delta into ``reasoning_content`` or ``content``.

        Maintains an internal phase flag that flips when ``<|channel|>`` or
        ``<|message|>`` appears in the delta. Once ``<|message|>`` has been
        seen, all subsequent deltas are surfaced as visible ``content``.

        Args:
            previous_text: Cumulative decoded text before this chunk
                (unused, kept for interface parity).
            current_text: Cumulative decoded text including this chunk
                (unused, kept for interface parity).
            delta_text: Newly produced text in this chunk.
            previous_token_ids: Token IDs before this chunk (unused).
            current_token_ids: Token IDs including this chunk (unused).
            delta_token_ids: Token IDs corresponding to ``delta_text``
                (unused).
            request: Optional inference request (unused).

        Returns:
            A :class:`DeltaMessage` carrying ``reasoning_content`` and/or
            ``content`` for this step, or ``None`` when there is nothing
            meaningful to emit (e.g. an empty delta or a delta that only
            contained a partial marker).
        """
        if not delta_text:
            return None

        if self._reasoning_done:
            return DeltaMessage(content=delta_text)

        if self.MESSAGE_TAG in delta_text:
            self._reasoning_done = True
            parts = delta_text.split(self.MESSAGE_TAG, 1)
            reasoning_part = parts[0] if self._in_reasoning else None
            content_part = parts[1] if len(parts) > 1 else None
            return DeltaMessage(
                reasoning_content=reasoning_part if reasoning_part else None,
                content=content_part if content_part else None,
            )

        if self.CHANNEL_TAG in delta_text:
            self._in_reasoning = True
            after = delta_text.split(self.CHANNEL_TAG, 1)[1]
            return DeltaMessage(reasoning_content=after) if after else None

        if self._in_reasoning:
            return DeltaMessage(reasoning_content=delta_text)

        return DeltaMessage(content=delta_text)
