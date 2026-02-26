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

"""Reasoning parser for Step3 models.

Step3 uses only the </think> end token (no explicit start token).
All content before </think> is treated as reasoning.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...openai_api_modules import DeltaMessage
from ..abstract_reasoning import ReasoningParserManager
from ..basic_parsers import BaseThinkingReasoningParser


@ReasoningParserManager.register_module(["step3"])  # pyright: ignore[reportUntypedClassDecorator]
class Step3ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Step3 models. Uses only </think> end token."""

    start_token = "<think>"
    end_token = "</think>"

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        if self.end_token not in model_output:
            # No end token: entire output could be reasoning-in-progress or just content
            return None, model_output
        # Split at end token; everything before is reasoning
        parts = model_output.split(self.end_token, 1)
        reasoning = parts[0].replace(self.start_token, "").strip()
        content = parts[1].strip() if len(parts) > 1 else None
        return reasoning or None, content

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

        if self.end_token in previous_text:
            return DeltaMessage(content=delta_text)

        if self.end_token in delta_text:
            parts = delta_text.split(self.end_token, 1)
            reasoning_part = parts[0].replace(self.start_token, "")
            content_part = parts[1] if len(parts) > 1 else None
            return DeltaMessage(
                reasoning_content=reasoning_part if reasoning_part else None,
                content=content_part if content_part else None,
            )

        # Still accumulating reasoning
        cleaned = delta_text.replace(self.start_token, "")
        return DeltaMessage(reasoning_content=cleaned) if cleaned else None
