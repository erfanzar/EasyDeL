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
    - Batch extraction via simple string splitting.
    - Streaming with state tracking for delimiter boundaries.
    - Prompt-gated asymmetric parsing when chat templates inject ``start_token``
      into the prompt via ``add_generation_prompt`` (model output may then contain
      only ``end_token``).

    Backward compatibility:
    - ``assume_reasoning`` is still supported as a manual override.
    """

    start_token: str = "<think>"
    end_token: str = "</think>"

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize with tokenizer and resolve start/end token IDs from vocabulary.

        Args:
            tokenizer: Tokenizer instance used to resolve delimiter token IDs
                and decode token sequences.
        """
        super().__init__(tokenizer)
        self._start_token_id: int | None = self.vocab.get(self.start_token)
        self._end_token_id: int | None = self.vocab.get(self.end_token)
        self._prompt_started_reasoning: bool = False
        self._stream_seen_start: bool = False
        self._stream_seen_end: bool = False
        self._stream_state_initialized: bool = False

    def configure_prompt_context(self, prompt_text: str, prompt_token_ids: Sequence[int]) -> None:
        """Infer whether the prompt already entered reasoning mode.

        The primary signal is token-ID suffix (if available). Text suffix is used
        as a fallback and to support tokenizers without direct token-ID mapping.
        """
        super().configure_prompt_context(prompt_text, prompt_token_ids)
        has_start_suffix_by_token = False
        if self._start_token_id is not None and prompt_token_ids:
            has_start_suffix_by_token = prompt_token_ids[-1] == self._start_token_id

        has_start_suffix_by_text = bool(prompt_text) and prompt_text.rstrip().endswith(self.start_token)
        self._prompt_started_reasoning = has_start_suffix_by_token or has_start_suffix_by_text
        self._stream_seen_start = self._prompt_started_reasoning
        self._stream_seen_end = False
        self._stream_state_initialized = True

    def _is_prompt_reasoning_active(self) -> bool:
        """Return ``True`` when the prompt already opened the reasoning block.

        ``assume_reasoning`` remains a manual override that callers can flip
        on to force prompt-gated parsing without an explicit detection.
        """
        # `assume_reasoning` remains as a compatibility override.
        return self._prompt_started_reasoning or self.assume_reasoning

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Check if the end-of-reasoning token is present in the token IDs."""
        if self._end_token_id is not None:
            return self._end_token_id in input_ids
        return False

    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids: Sequence[int]) -> bool:
        """Check if end token appears in delta (more efficient for streaming)."""
        if self._end_token_id is not None:
            return self._end_token_id in delta_ids
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Extract content token IDs by removing everything up to and including the end token."""
        if self._end_token_id is None or self._end_token_id not in input_ids:
            return input_ids
        end_idx = input_ids.index(self._end_token_id)
        return input_ids[end_idx + 1 :]

    def extract_reasoning(
        self,
        model_output: str,
        request=None,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning and content by splitting on start/end token delimiters.

        Handles both explicit delimiter mode (start_token in output) and
        prompt-gated asymmetric mode (start_token injected by chat template).

        Returns:
            Tuple of (reasoning_content, visible_content). Either may be None.
        """
        if self.start_token not in model_output:
            if self._is_prompt_reasoning_active():
                # Prompt-aware asymmetric mode: start token came from prompt.
                # If end token is present, split into reasoning/content.
                if self.end_token in model_output:
                    parts = model_output.split(self.end_token, 1)
                    reasoning = parts[0].strip()
                    content = parts[1] if len(parts) > 1 else None
                    return reasoning or None, content if content else None
                # If end token is missing, we are still inside reasoning.
                reasoning = model_output.strip()
                return reasoning or None, None
            return None, model_output

        # Split at start token
        before_start, after_start = model_output.split(self.start_token, 1)

        if self.end_token not in after_start:
            # Incomplete reasoning (no end token) — treat all after start as reasoning
            return after_start.strip() or None, before_start.strip() or None

        reasoning_part, content = after_start.split(self.end_token, 1)
        reasoning = reasoning_part.strip()

        # Prepend any text before the start token to content
        if before_start.strip():
            content = before_start.strip() + ("\n" + content if content else "")

        return reasoning or None, content if content else None

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
        """Extract reasoning content from a streaming text delta.

        Uses state tracking to determine whether the current delta belongs
        to the reasoning section, content section, or straddles a boundary.
        Supports prompt-gated mode where the start token was in the prompt.

        Returns:
            DeltaMessage with reasoning_content and/or content set,
            or None if the delta is a boundary token with no text to emit.
        """
        if not delta_text:
            return None

        if not self._stream_state_initialized:
            self._stream_seen_start = self._is_prompt_reasoning_active() or self.start_token in previous_text or (
                self._start_token_id is not None and self._start_token_id in previous_token_ids
            )
            self._stream_seen_end = self.end_token in previous_text or (
                self._end_token_id is not None and self._end_token_id in previous_token_ids
            )
            self._stream_state_initialized = True

        def _marker_in_delta(marker: str) -> tuple[bool, bool]:
            in_delta = marker in delta_text
            if in_delta:
                return True, True
            if previous_text and len(marker) > 1:
                boundary = previous_text[-(len(marker) - 1) :] + delta_text
                if marker in boundary:
                    return True, False
            return False, False

        has_start_marker, has_start_text_in_delta = _marker_in_delta(self.start_token)
        has_end_marker, has_end_text_in_delta = _marker_in_delta(self.end_token)
        has_start_token_in_delta = self._start_token_id is not None and self._start_token_id in delta_token_ids
        has_end_token_in_delta = self._end_token_id is not None and self._end_token_id in delta_token_ids
        has_start_in_delta = has_start_marker or has_start_token_in_delta
        has_end_in_delta = has_end_marker or has_end_token_in_delta
        has_start_in_prev = self._stream_seen_start
        has_end_in_prev = self._stream_seen_end
        has_start_in_current = has_start_in_prev or has_start_in_delta
        has_end_in_current = has_end_in_prev or has_end_in_delta
        reasoning_started = has_start_in_prev or self._is_prompt_reasoning_active()

        if has_start_in_delta:
            self._stream_seen_start = True
        if has_end_in_delta:
            self._stream_seen_end = True

        # Case 1: Both start and end appear in the delta itself
        if has_start_text_in_delta and has_end_text_in_delta:
            after_start = delta_text.split(self.start_token, 1)[1]
            reasoning_part, content_part = after_start.split(self.end_token, 1)
            return DeltaMessage(
                reasoning_content=reasoning_part if reasoning_part else None,
                content=content_part if content_part else None,
            )

        # Case 2: We're past the end token already — everything is content
        if has_end_in_prev:
            return DeltaMessage(content=delta_text)

        # Case 3: Start token appears in this delta (reasoning begins). Keep this
        # before inferred-reasoning checks so literal start tokens are stripped.
        if has_start_text_in_delta:
            after_start = delta_text.split(self.start_token, 1)[1]
            if after_start:
                return DeltaMessage(reasoning_content=after_start)
            return None  # Just the start token itself, wait for more

        # Case 4: Reasoning already active (explicitly started or prompt-gated),
        # and end token arrives in this chunk.
        if reasoning_started and has_end_in_delta:
            if not has_end_text_in_delta:
                return None
            parts = delta_text.split(self.end_token, 1)
            reasoning_part = parts[0]
            content_part = parts[1] if len(parts) > 1 else None
            return DeltaMessage(
                reasoning_content=reasoning_part if reasoning_part else None,
                content=content_part if content_part else None,
            )

        # Case 5: Reasoning already active and still accumulating.
        if reasoning_started and not has_end_in_current:
            return DeltaMessage(reasoning_content=delta_text)

        # Case 6: No reasoning start seen — treat as content.
        if not has_start_in_current:
            return DeltaMessage(content=delta_text)

        return None
