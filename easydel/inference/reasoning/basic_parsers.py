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

"""Base reasoning parser for token-delimited formats.

Provides BaseThinkingReasoningParser, a reusable helper for the most common
reasoning format: start_token + reasoning + end_token + content.

Example formats:
    <think>reasoning here</think>actual response
    [THINK]reasoning[/THINK]response
"""

from __future__ import annotations

from collections.abc import Sequence

from transformers import AutoTokenizer as AnyTokenizer

from ..openai_api_modules import DeltaMessage
from .abstract_reasoning import ReasoningParser


class BaseThinkingReasoningParser(ReasoningParser):
    """Base class for reasoning parsers using token delimiters.

    Subclasses only need to set ``start_token`` and ``end_token``.

    Handles:
    - Batch extraction via simple string splitting
    - Streaming with state tracking for multi-token delimiter boundaries
    - Asymmetric cases (e.g., only end_token present in model output
      when the chat template injects start_token into the prompt via
      ``add_generation_prompt``). Set ``assume_reasoning = True`` to
      enable this mode.
    """

    start_token: str = "<think>"
    end_token: str = "</think>"

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self._start_token_id: int | None = self.vocab.get(self.start_token)
        self._end_token_id: int | None = self.vocab.get(self.end_token)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self._end_token_id is not None:
            return self._end_token_id in input_ids
        return False

    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids: Sequence[int]) -> bool:
        """Check if end token appears in delta (more efficient for streaming)."""
        if self._end_token_id is not None:
            return self._end_token_id in delta_ids
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self._end_token_id is None or self._end_token_id not in input_ids:
            return input_ids
        end_idx = input_ids.index(self._end_token_id)
        return input_ids[end_idx + 1 :]

    def extract_reasoning(
        self,
        model_output: str,
        request=None,
    ) -> tuple[str | None, str | None]:
        if self.start_token not in model_output:
            if self.end_token in model_output:
                # Asymmetric: no start token, treat everything before end as reasoning
                parts = model_output.split(self.end_token, 1)
                reasoning = parts[0].strip()
                content = parts[1].strip() if len(parts) > 1 else None
                return reasoning or None, content
            return None, model_output

        # Split at start token
        before_start, after_start = model_output.split(self.start_token, 1)

        if self.end_token not in after_start:
            # Incomplete reasoning (no end token) — treat all after start as reasoning
            return after_start.strip() or None, before_start.strip() or None

        reasoning_part, content_part = after_start.split(self.end_token, 1)
        reasoning = reasoning_part.strip()
        content = content_part.strip()

        # Prepend any text before the start token to content
        if before_start.strip():
            content = before_start.strip() + ("\n" + content if content else "")

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

        has_start_in_prev = self.start_token in previous_text
        has_end_in_prev = self.end_token in previous_text
        has_start_in_current = self.start_token in current_text
        has_end_in_delta = self.end_token in delta_text
        has_end_in_current = self.end_token in current_text

        # Case 1: Both start and end appear in the delta itself
        if self.start_token in delta_text and self.end_token in delta_text:
            after_start = delta_text.split(self.start_token, 1)[1]
            reasoning_part, content_part = after_start.split(self.end_token, 1)
            return DeltaMessage(
                reasoning_content=reasoning_part if reasoning_part else None,
                content=content_part if content_part else None,
            )

        # Case 2: We're past the end token already — everything is content
        if has_end_in_prev:
            return DeltaMessage(content=delta_text)

        # Case 3: Start seen previously (or assumed from prompt), end token in this delta
        if (has_start_in_prev or self.assume_reasoning) and has_end_in_delta:
            parts = delta_text.split(self.end_token, 1)
            reasoning_part = parts[0]
            content_part = parts[1] if len(parts) > 1 else None
            return DeltaMessage(
                reasoning_content=reasoning_part if reasoning_part else None,
                content=content_part if content_part else None,
            )

        # Case 4: Start seen previously (or assumed from prompt), still accumulating reasoning
        if (has_start_in_prev or self.assume_reasoning) and not has_end_in_current:
            return DeltaMessage(reasoning_content=delta_text)

        # Case 5: Start token appears in this delta (reasoning begins)
        if self.start_token in delta_text:
            after_start = delta_text.split(self.start_token, 1)[1]
            if after_start:
                return DeltaMessage(reasoning_content=after_start)
            return None  # Just the start token itself, wait for more

        # Case 6: No start token seen yet — could be content before reasoning,
        # or the model might not use reasoning tags at all.
        # Check if it could be a partial start token at the boundary
        if not has_start_in_current:
            # No reasoning tags seen at all — return as content
            return DeltaMessage(content=delta_text)

        return None
