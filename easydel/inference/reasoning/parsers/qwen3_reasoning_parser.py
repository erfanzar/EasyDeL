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

"""Reasoning parser for Qwen3 models.

Format: <think>reasoning content</think>response

Qwen3 is strict about requiring both tags. If the start tag is missing,
the entire output is treated as content.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...openai_api_modules import DeltaMessage
from ..abstract_reasoning import ReasoningParserManager
from ..basic_parsers import BaseThinkingReasoningParser


@ReasoningParserManager.register_module(["qwen3", "qwen3_reasoning"])
class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Qwen3 models using <think>...</think>.

    Strict mode: if <think> tag is missing, all output is content.
    """

    start_token = "<think>"
    end_token = "</think>"

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        # Qwen3 is strict: must have both start and end tags
        # But when assume_reasoning is set (prompt already has <think>), relax to end-only
        if self.start_token not in model_output:
            if self.assume_reasoning and self.end_token in model_output:
                # Prompt had <think>, model output has </think> — asymmetric OK
                parts = model_output.split(self.end_token, 1)
                reasoning = parts[0].strip()
                content = parts[1].strip() if len(parts) > 1 else None
                return reasoning or None, content
            return None, model_output
        # Start tag present — still require end tag in strict mode
        if self.end_token not in model_output:
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
        # If we've accumulated text and no start token found, treat as content
        # Unless assume_reasoning is set (prompt already has <think>)
        if (
            not self.assume_reasoning
            and current_text
            and self.start_token not in current_text
            and len(current_text) > len(self.start_token)
        ):
            return DeltaMessage(content=delta_text) if delta_text else None
        return super().extract_reasoning_streaming(
            previous_text, current_text, delta_text,
            previous_token_ids, current_token_ids, delta_token_ids, request,
        )
