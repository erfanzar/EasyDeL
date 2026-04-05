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

"""Reasoning parser for Gemma4 chat outputs.

Gemma4 uses channel markers in its chat template:

    <|channel>thought
    <channel|>...reasoning...
    <|channel>final
    <channel|>...content...

The prompt can already open the ``thought`` channel, so the generated text may
start with raw reasoning text before the first new channel marker appears.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from transformers import AutoTokenizer as AnyTokenizer

from ...openai_api_modules import DeltaMessage
from ..abstract_reasoning import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module("gemma4")  # pyright: ignore[reportUntypedClassDecorator]
class Gemma4ReasoningParser(ReasoningParser):
    """Extract thought/final channel content from Gemma4 model outputs.

    Gemma4 uses explicit channel markers (``<|channel>thought<channel|>``,
    ``<|channel>analysis<channel|>``, ``<|channel>final<channel|>``) to
    delineate reasoning vs. user-facing content. The prompt's chat template
    may already open a ``thought`` channel before generation begins, so the
    parser inspects the rendered prompt to detect the initial active channel.

    Both batch (``extract_reasoning``) and streaming
    (``extract_reasoning_streaming``) extraction are supported. Reasoning
    text from ``thought`` and ``analysis`` channels is collected separately
    from ``final``-channel content.

    Attributes:
        CHANNEL_START: Opening tag prefix for channel markers.
        CHANNEL_END: Closing tag suffix for channel markers.
        REASONING_CHANNELS: Set of channel names treated as reasoning.
        THOUGHT_START_TOKEN: Full opening marker for the ``thought`` channel.
        ANALYSIS_START_TOKEN: Full opening marker for the ``analysis`` channel.
        reasoning_start_tokens: Tuple of all reasoning channel open markers.
        start_token: Canonical reasoning start marker (``thought`` channel).
        end_token: Marker that terminates reasoning (``final`` channel open).
    """

    CHANNEL_START = "<|channel>"
    CHANNEL_END = "<channel|>"
    REASONING_CHANNELS = frozenset({"thought", "analysis"})
    THOUGHT_START_TOKEN = f"{CHANNEL_START}thought{CHANNEL_END}"
    ANALYSIS_START_TOKEN = f"{CHANNEL_START}analysis{CHANNEL_END}"
    reasoning_start_tokens = (THOUGHT_START_TOKEN, ANALYSIS_START_TOKEN)
    # Generic reasoning helpers default to a single start/end span, so keep the
    # canonical thought -> final boundary pair while also exposing the full
    # reasoning start-token set via ``reasoning_start_tokens``.
    start_token = THOUGHT_START_TOKEN
    end_token = f"{CHANNEL_START}final{CHANNEL_END}"

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the Gemma4 reasoning parser.

        Args:
            tokenizer: Tokenizer instance used for token-level operations
                inherited from ``ReasoningParser``.
        """
        super().__init__(tokenizer)
        self._initial_channel: str | None = None
        self._channel_pattern = re.compile(
            re.escape(self.CHANNEL_START) + r"(.*?)" + re.escape(self.CHANNEL_END),
            re.DOTALL,
        )

    def configure_prompt_context(self, prompt_text: str, prompt_token_ids: Sequence[int]) -> None:
        """Infer the active starting channel from the rendered prompt.

        Scans *prompt_text* for the last channel marker. If it appears at the
        very end of the prompt (no trailing non-whitespace), the parser
        assumes generation will continue inside that channel.

        Args:
            prompt_text: Full rendered prompt sent to the model.
            prompt_token_ids: Token IDs for the prompt (unused).
        """
        del prompt_token_ids
        matches = list(self._channel_pattern.finditer(prompt_text))
        self._initial_channel = None
        if not matches:
            return

        last = matches[-1]
        trailing = prompt_text[last.end() :].strip()
        if trailing:
            return
        self._initial_channel = last.group(1).strip() or None

    def _parse_channels(self, text: str) -> tuple[str | None, str | None]:
        """Split generated text into reasoning and content by channel markers.

        Walks through *text* tracking channel switches. Text under
        ``thought``/``analysis`` channels accumulates as reasoning; everything
        else (including ``final``) accumulates as content.

        Args:
            text: Raw model output text to parse.

        Returns:
            A ``(reasoning, content)`` tuple. Each element is a stripped
            string or ``None`` if no text fell into that category.
        """
        cursor = 0
        current_channel = self._initial_channel
        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        while cursor < len(text):
            match = self._channel_pattern.search(text, cursor)
            next_tag = len(text) if match is None else match.start()
            if next_tag > cursor:
                chunk = text[cursor:next_tag]
                if current_channel in self.REASONING_CHANNELS:
                    reasoning_parts.append(chunk)
                else:
                    content_parts.append(chunk)
            if match is None:
                break
            current_channel = match.group(1).strip() or None
            cursor = match.end()

        reasoning = "".join(reasoning_parts).strip() or None
        content = "".join(content_parts).strip() or None
        return reasoning, content

    @staticmethod
    def _suffix(previous: str | None, current: str | None) -> str | None:
        """Compute the incremental text delta between two cumulative strings.

        Args:
            previous: Cumulative text from the prior streaming step (may be
                ``None``).
            current: Cumulative text from the current streaming step (may be
                ``None``).

        Returns:
            The new suffix added in *current* beyond *previous*, or ``None``
            if there is no new text.
        """
        previous = previous or ""
        current = current or ""
        if not current:
            return None
        if current.startswith(previous):
            delta = current[len(previous) :]
            return delta or None
        return current

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Check whether the reasoning section has ended.

        Gemma4 does not rely on a single end-of-reasoning token; channel
        switches are detected by marker parsing instead. This always returns
        ``True`` so the generic pipeline does not block on a sentinel.

        Args:
            input_ids: Generated token IDs so far (unused).

        Returns:
            Always ``True``.
        """
        del input_ids
        return True

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Return content token IDs from the generated sequence.

        Gemma4 channel separation is text-based, so no token-level filtering
        is performed — the full sequence is returned as-is.

        Args:
            input_ids: Complete list of generated token IDs.

        Returns:
            A copy of *input_ids* unchanged.
        """
        return list(input_ids)

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        """Extract reasoning and content from a complete model output.

        Args:
            model_output: Full decoded model output text.
            request: Optional request context (unused).

        Returns:
            A ``(reasoning, content)`` tuple where each element is a string
            or ``None`` if absent.
        """
        del request
        return self._parse_channels(model_output)

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
        """Compute the streaming delta for reasoning and content channels.

        Compares the cumulative channel parse of *previous_text* and
        *current_text* to produce incremental reasoning and content deltas.

        Args:
            previous_text: Cumulative decoded text from the prior step.
            current_text: Cumulative decoded text including the latest tokens.
            delta_text: Newly generated text segment (unused).
            previous_token_ids: Token IDs from the prior step (unused).
            current_token_ids: Token IDs including the latest tokens (unused).
            delta_token_ids: Newly generated token IDs (unused).
            request: Optional request context (unused).

        Returns:
            A ``DeltaMessage`` with ``reasoning_content`` and/or ``content``
            set to incremental text, or ``None`` if nothing changed.
        """
        del delta_text, previous_token_ids, current_token_ids, delta_token_ids, request

        previous_reasoning, previous_content = self._parse_channels(previous_text)
        current_reasoning, current_content = self._parse_channels(current_text)

        reasoning_delta = self._suffix(previous_reasoning, current_reasoning)
        content_delta = self._suffix(previous_content, current_content)
        if reasoning_delta is None and content_delta is None:
            return None
        return DeltaMessage(reasoning_content=reasoning_delta, content=content_delta)
