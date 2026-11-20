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

"""Token-delimited reasoning with boundary-safe cumulative streaming."""

from __future__ import annotations

from collections.abc import Sequence

from transformers import AutoTokenizer as AnyTokenizer

from ..openai_api_modules import DeltaMessage
from .abstract_reasoning import ReasoningParser


class BaseThinkingReasoningParser(ReasoningParser):
    """Split symmetric or prompt-started reasoning without emitting partial markers."""

    start_token: str = "<think>"
    end_token: str = "</think>"
    asymmetric_reasoning: bool = False
    streaming_all_phases = True

    def __init__(self, tokenizer: AnyTokenizer):
        """Resolve marker IDs and initialize emitted-channel snapshots."""
        super().__init__(tokenizer)
        self._start_token_id = self.vocab.get(self.start_token)
        self._end_token_id = self.vocab.get(self.end_token)
        self._prompt_started_reasoning = False
        self._stream_seen_start = False
        self._stream_seen_end = False
        self._stream_state_initialized = False
        self._streaming_entered = False
        self._emitted_reasoning = ""
        self._emitted_content = ""

    def configure_prompt_context(self, prompt_text: str, prompt_token_ids: Sequence[int]) -> None:
        """Detect a reasoning opener at the end of the prompt and reset streaming state."""
        super().configure_prompt_context(prompt_text, prompt_token_ids)
        self._prompt_started_reasoning = bool(
            (self._start_token_id is not None and prompt_token_ids and prompt_token_ids[-1] == self._start_token_id)
            or (prompt_text and prompt_text.rstrip().endswith(self.start_token))
        )
        self._stream_seen_start = self._prompt_started_reasoning
        self._stream_seen_end = False
        self._stream_state_initialized = True
        self._streaming_entered = False
        self._emitted_reasoning = self._emitted_content = ""

    def _is_prompt_reasoning_active(self) -> bool:
        """Return whether reasoning starts before the first generated character."""
        return self.asymmetric_reasoning or self._prompt_started_reasoning or self.assume_reasoning

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Check for the closing marker ID."""
        return self._end_token_id is not None and self._end_token_id in input_ids

    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids: Sequence[int]) -> bool:
        """Check the newly generated IDs for the closing marker."""
        return self.is_reasoning_end(delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Return IDs following the closing marker, or the original IDs if absent."""
        if self._end_token_id is None or self._end_token_id not in input_ids:
            return input_ids
        return input_ids[input_ids.index(self._end_token_id) + 1 :]

    @staticmethod
    def _without_marker_prefix(text: str, marker: str) -> str:
        """Withhold a proper marker prefix until it is disambiguated or finalized."""
        for length in range(min(len(text), len(marker) - 1), 0, -1):
            if text.endswith(marker[:length]):
                return text[:-length]
        return text

    def _channel_snapshot(self, text: str, *, final: bool = False) -> tuple[str, str]:
        """Return stable channel text, retaining whitespace and withholding incomplete delimiters.

        Reasoning opens only at the start of the generation (after leading whitespace), or in the prompt; it closes at
        the first closing marker. Markers anywhere else are ordinary text: an answer that writes ``<think>...</think>``
        keeps them verbatim in content, and reasoning that mentions the opener keeps it.
        """
        stripped = text.lstrip()
        leading = text[: len(text) - len(stripped)]
        if self._is_prompt_reasoning_active():
            content = ""
            if stripped.startswith(self.start_token):
                # The model may repeat the opener the prompt already emitted.
                reasoning_text = stripped[len(self.start_token) :]
            elif not final and stripped and self.start_token.startswith(stripped):
                return "", ""
            else:
                reasoning_text = text
        elif stripped.startswith(self.start_token):
            content = leading
            reasoning_text = stripped[len(self.start_token) :]
        elif not final and stripped and self.start_token.startswith(stripped):
            return "", leading  # may still become the opener
        else:
            return "", text
        reasoning, ended, after_end = reasoning_text.partition(self.end_token)
        if ended:
            return reasoning, content + after_end
        if not final:
            reasoning = self._without_marker_prefix(reasoning, self.end_token)
        return reasoning, content

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        """Extract complete channels, including unfinished reasoning on token-limit termination."""
        reasoning, content = self._channel_snapshot(model_output, final=True)
        return reasoning.strip() or None, content or None

    def extract_reasoning_final(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        """Return lossless terminal channel snapshots for streaming catch-up."""
        reasoning, content = self._channel_snapshot(model_output, final=True)
        openers = getattr(self, "reasoning_start_tokens", (self.start_token,))
        recognized = self._is_prompt_reasoning_active() or any(
            model_output.lstrip().startswith(marker) for marker in openers
        )
        return reasoning if recognized or reasoning else None, content or None

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
        """Emit only newly disambiguated text, even when delimiters span arbitrary chunks."""
        # A stream may have entered reasoning while every byte is still buffered.
        self._streaming_entered = True
        return self._emit_snapshot(*self._channel_snapshot(current_text))

    def _emit_snapshot(self, reasoning: str, content: str) -> DeltaMessage | None:
        """Track actual emitted text independently of batch-derived snapshots."""
        reasoning_delta = (
            reasoning[len(self._emitted_reasoning) :] if reasoning.startswith(self._emitted_reasoning) else ""
        )
        content_delta = content[len(self._emitted_content) :] if content.startswith(self._emitted_content) else ""
        self._emitted_reasoning += reasoning_delta
        self._emitted_content += content_delta
        if not reasoning_delta and not content_delta:
            return None
        return DeltaMessage(reasoning_content=reasoning_delta or None, content=content_delta or None)

    def finalize_streaming(
        self, current_text: str, current_token_ids: Sequence[int] = (), request=None
    ) -> DeltaMessage | None:
        """Release an unresolved ordinary marker prefix once, in its original channel."""
        return self._emit_snapshot(*self._channel_snapshot(current_text, final=True))
