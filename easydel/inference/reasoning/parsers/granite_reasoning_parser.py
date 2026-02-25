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

"""Reasoning parser for IBM Granite models.

Format uses text delimiters:
    Here's my thought process:
    <reasoning>
    Here's my response:
    <content>
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from transformers import AutoTokenizer as AnyTokenizer

from ...openai_api_modules import DeltaMessage
from ..abstract_reasoning import ReasoningParser, ReasoningParserManager

_THOUGHT_STARTERS = [
    "Here's my thought process:",
    "Here is my thought process:",
]
_RESPONSE_STARTERS = [
    "Here's my response:",
    "Here is my response:",
]


@ReasoningParserManager.register_module(["granite"])
class GraniteReasoningParser(ReasoningParser):
    """Reasoning parser for Granite models using text delimiters."""

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        thought_pattern = "|".join(re.escape(s) for s in _THOUGHT_STARTERS)
        response_pattern = "|".join(re.escape(s) for s in _RESPONSE_STARTERS)
        self._regex = re.compile(
            rf"(?:{thought_pattern})\s*(.*?)\s*(?:{response_pattern})\s*(.*)",
            re.DOTALL,
        )
        self._thought_starters = _THOUGHT_STARTERS
        self._response_starters = _RESPONSE_STARTERS
        self._in_reasoning = False
        self._reasoning_done = False

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        text = self.model_tokenizer.decode(list(input_ids), skip_special_tokens=False)
        return any(s in text for s in self._response_starters)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return list(input_ids)

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        match = self._regex.search(model_output)
        if not match:
            return None, model_output
        reasoning = match.group(1).strip()
        content = match.group(2).strip()
        return reasoning or None, content or None

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
        if not delta_text:
            return None

        # Check if we've already found the response delimiter
        if self._reasoning_done:
            return DeltaMessage(content=delta_text)

        # Check if response delimiter appears in current text
        for starter in self._response_starters:
            if starter in current_text:
                self._reasoning_done = True
                # Check if it's in the delta
                if starter in delta_text:
                    parts = delta_text.split(starter, 1)
                    reasoning_part = parts[0] if self._in_reasoning else None
                    content_part = parts[1] if len(parts) > 1 else None
                    return DeltaMessage(
                        reasoning_content=reasoning_part if reasoning_part else None,
                        content=content_part if content_part else None,
                    )
                return DeltaMessage(content=delta_text)

        # Check if thought delimiter appears
        for starter in self._thought_starters:
            if starter in current_text:
                self._in_reasoning = True
                if starter in delta_text:
                    after = delta_text.split(starter, 1)[1]
                    return DeltaMessage(reasoning_content=after) if after else None
                if self._in_reasoning:
                    return DeltaMessage(reasoning_content=delta_text)

        if self._in_reasoning:
            return DeltaMessage(reasoning_content=delta_text)

        return DeltaMessage(content=delta_text)
