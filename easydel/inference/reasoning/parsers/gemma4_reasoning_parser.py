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

"""Reasoning parser for Gemma4 thinking models.

Gemma4 uses ``<|channel>`` / ``<channel|>`` tokens to delimit reasoning
content. When thinking is enabled, the model generates::

    <|channel>thought
    ...chain of thought reasoning...<channel|>
    Final answer text here.

The ``thought\\n`` role label inside the channel delimiters is a structural
artefact that must be stripped so downstream consumers see only actual
reasoning text.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import cached_property

from transformers import AutoTokenizer as AnyTokenizer

from ...openai_api_modules import DeltaMessage
from ..abstract_reasoning import ReasoningParser, ReasoningParserManager

# Role label that Gemma4 emits at the start of the thinking channel.
_THOUGHT_PREFIX = "thought\n"

CHANNEL_START = "<|channel>"
CHANNEL_END = "<channel|>"


@ReasoningParserManager.register_module("gemma4")  # pyright: ignore[reportUntypedClassDecorator]
class Gemma4ReasoningParser(ReasoningParser):
    """Extract reasoning content from Gemma4 model outputs.

    Gemma4 uses ``<|channel>thought\\n...reasoning...<channel|>`` for
    chain-of-thought content. The parser handles:

    - Stripping the ``thought\\n`` role label prefix from reasoning
    - Proper ``is_reasoning_end`` based on token ID scanning
    - ``skip_special_tokens=False`` to preserve channel markers
    - Both batch and streaming extraction
    """

    start_token = CHANNEL_START
    end_token = CHANNEL_END

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self._reasoning_text: str = ""
        self._prefix_stripped: bool = False
        self._channel_pattern = re.compile(
            re.escape(CHANNEL_START) + r"(.*?)" + re.escape(CHANNEL_END),
            re.DOTALL,
        )

    @cached_property
    def start_token_id(self) -> int:
        return self.vocab[CHANNEL_START]

    @cached_property
    def end_token_id(self) -> int:
        return self.vocab[CHANNEL_END]

    @cached_property
    def _tool_call_token_id(self) -> int | None:
        return self.vocab.get("<|tool_call>")

    @cached_property
    def _new_turn_token_id(self) -> int | None:
        return self.vocab.get("<|turn>")

    @cached_property
    def _tool_response_token_id(self) -> int | None:
        return self.vocab.get("<|tool_response>")

    def adjust_request(self, request) -> object:
        """Disable special-token stripping to preserve channel markers."""
        request.skip_special_tokens = False
        return request

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Check whether reasoning has ended by scanning token IDs backwards.

        Returns True when the last relevant special token is ``<channel|>``
        (reasoning ended) or ``<|tool_call>`` (tool call started, implying
        reasoning is over). Returns False when still inside a
        ``<|channel>`` block or after ``<|turn>``/``<|tool_response>``
        (new reasoning may begin).
        """
        start_id = self.start_token_id
        end_id = self.end_token_id
        tool_call_id = self._tool_call_token_id
        new_turn_id = self._new_turn_token_id
        tool_response_id = self._tool_response_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            tid = input_ids[i]
            if tid == start_id:
                return False
            if tid == end_id:
                return True
            if tool_call_id is not None and tid == tool_call_id:
                return True
            if tid == new_turn_id or tid == tool_response_id:
                return False
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return list(input_ids)

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        """Extract reasoning and content from complete model output.

        Parses channel markers, collects text from ``thought``/``analysis``
        channels as reasoning, everything else as content. Strips the
        ``thought\\n`` prefix from reasoning.
        """
        del request
        if CHANNEL_START not in model_output and CHANNEL_END not in model_output:
            return None, model_output

        reasoning, content = self._parse_channels(model_output)
        if reasoning is not None:
            reasoning = _strip_thought_label(reasoning)
        return reasoning, content

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
        """Extract streaming reasoning, stripping ``thought\\n`` prefix.

        The prefix may arrive split across multiple deltas. We buffer
        early reasoning tokens until the prefix is fully consumed, then
        emit the remainder.
        """
        del previous_token_ids, delta_token_ids, request

        previous_reasoning, previous_content = self._parse_channels(previous_text)
        current_reasoning, current_content = self._parse_channels(current_text)

        reasoning_delta = _suffix(previous_reasoning, current_reasoning)
        content_delta = _suffix(previous_content, current_content)

        # Strip the thought\n prefix from the reasoning stream
        if reasoning_delta is not None:
            self._reasoning_text += reasoning_delta

            if not self._prefix_stripped:
                if self._reasoning_text.startswith(_THOUGHT_PREFIX):
                    prefix_len = len(_THOUGHT_PREFIX)
                    prev_reasoning_len = len(self._reasoning_text) - len(reasoning_delta)
                    if prev_reasoning_len >= prefix_len:
                        self._prefix_stripped = True
                    else:
                        chars_of_prefix_in_delta = prefix_len - prev_reasoning_len
                        stripped = reasoning_delta[chars_of_prefix_in_delta:]
                        self._prefix_stripped = len(self._reasoning_text) >= prefix_len
                        reasoning_delta = stripped or None
                elif _THOUGHT_PREFIX.startswith(self._reasoning_text):
                    # Still buffering — might be partial prefix
                    reasoning_delta = None
                else:
                    # Doesn't match prefix at all — emit everything buffered
                    self._prefix_stripped = True
                    reasoning_delta = self._reasoning_text

        if reasoning_delta is None and content_delta is None:
            return None
        return DeltaMessage(reasoning_content=reasoning_delta, content=content_delta)

    _REASONING_CHANNELS = frozenset({"thought", "analysis"})

    def _parse_channels(self, text: str) -> tuple[str | None, str | None]:
        """Split text into reasoning and content by channel markers."""
        cursor = 0
        current_channel: str | None = None
        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        while cursor < len(text):
            match = self._channel_pattern.search(text, cursor)
            next_tag = len(text) if match is None else match.start()
            if next_tag > cursor:
                chunk = text[cursor:next_tag]
                if current_channel in self._REASONING_CHANNELS:
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


def _strip_thought_label(text: str) -> str:
    """Remove the ``thought\\n`` role label from the beginning of text."""
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX) :]
    return text


def _suffix(previous: str | None, current: str | None) -> str | None:
    """Compute the incremental text delta between two cumulative strings."""
    previous = previous or ""
    current = current or ""
    if not current:
        return None
    if current.startswith(previous):
        delta = current[len(previous) :]
        return delta or None
    return current
