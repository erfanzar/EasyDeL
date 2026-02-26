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
    """Reasoning parser for Qwen3 models using <think>...</think>.

    Strict mode:
    - Missing start tag -> content, unless prompt context already started reasoning.
    - Missing end tag -> content.
    """

    start_token = "<think>"
    end_token = "</think>"

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        # Qwen3 strictness: missing end tag is always content.
        if self.end_token not in model_output:
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
        has_start_in_current = self.start_token in current_text or (
            self._start_token_id is not None and self._start_token_id in current_token_ids
        )

        # Strict behavior: if no start tag is observed and we are not in prompt-aware
        # asymmetric mode, treat streaming output as content.
        if (
            not self._is_prompt_reasoning_active()
            and current_text
            and not has_start_in_current
            and len(current_text) > len(self.start_token)
        ):
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
