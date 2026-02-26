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

"""Reasoning parser for GptOss models.

Format uses channel tags: <|channel|>analysis<reasoning><|message|><content>
"""

from __future__ import annotations

from collections.abc import Sequence

from transformers import AutoTokenizer as AnyTokenizer

from ...openai_api_modules import DeltaMessage
from ..abstract_reasoning import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module(["openai_gptoss", "gptoss"])  # pyright: ignore[reportUntypedClassDecorator]
class GptOssReasoningParser(ReasoningParser):
    """Reasoning parser for GptOss models using channel tags.

    Format: <|channel|>analysis_text<|message|>response_text
    """

    CHANNEL_TAG = "<|channel|>"
    MESSAGE_TAG = "<|message|>"

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self._channel_token_id = self.vocab.get(self.CHANNEL_TAG)
        self._message_token_id = self.vocab.get(self.MESSAGE_TAG)
        self._in_reasoning = False
        self._reasoning_done = False

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self._message_token_id is not None:
            return self._message_token_id in input_ids
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self._message_token_id is None or self._message_token_id not in input_ids:
            return list(input_ids)
        idx = input_ids.index(self._message_token_id)
        return input_ids[idx + 1 :]

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
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
