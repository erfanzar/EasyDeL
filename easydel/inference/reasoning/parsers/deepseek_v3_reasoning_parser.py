# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reasoning parser for DeepSeek V3 models with conditional thinking support.

Delegates to DeepSeekR1ReasoningParser when thinking is enabled,
or IdentityReasoningParser when disabled. Also used as the parser
for GLM-4.5, Holo2, and Kimi-K2 (they share the same format).
"""

from __future__ import annotations

from collections.abc import Sequence

from ...openai_api_modules import DeltaMessage
from ..abstract_reasoning import ReasoningParser, ReasoningParserManager
from .deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from .identity_reasoning_parser import IdentityReasoningParser


@ReasoningParserManager.register_module(["deepseek_v3", "glm45", "holo2", "kimi_k2"])
class DeepSeekV3ReasoningParser(ReasoningParser):
    """Conditional reasoning parser: delegates to R1 or Identity based on tokenizer config.

    If the tokenizer has a chat template with thinking/enable_thinking support,
    this parser uses DeepSeekR1ReasoningParser. Otherwise, it falls through
    to IdentityReasoningParser (no reasoning extraction).
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        # Check if tokenizer's chat template supports thinking
        chat_template = getattr(tokenizer, "chat_template", "") or ""
        has_thinking = "thinking" in chat_template or "enable_thinking" in chat_template
        if has_thinking:
            self._delegate = DeepSeekR1ReasoningParser(tokenizer)
        else:
            self._delegate = IdentityReasoningParser(tokenizer)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return self._delegate.is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return self._delegate.extract_content_ids(input_ids)

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
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
        return self._delegate.extract_reasoning_streaming(
            previous_text, current_text, delta_text,
            previous_token_ids, current_token_ids, delta_token_ids, request,
        )
