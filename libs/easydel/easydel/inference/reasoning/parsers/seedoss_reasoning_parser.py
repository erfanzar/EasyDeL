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

"""Reasoning parser for Seed OSS models.

Supports both Seed-OSS native ``<seed:think>...</seed:think>`` tags and
standard ``<think>...</think>`` tags for backwards compatibility.
"""

from __future__ import annotations

from collections.abc import Sequence

from transformers import AutoTokenizer as AnyTokenizer

from ..abstract_reasoning import ReasoningParserManager
from ..basic_parsers import BaseThinkingReasoningParser


@ReasoningParserManager.register_module(["seed_oss"])  # pyright: ignore[reportUntypedClassDecorator]
class SeedOSSReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Seed-OSS chain-of-thought outputs.

    Recognizes both ``<seed:think>...</seed:think>`` (Seed-OSS native format)
    and ``<think>...</think>`` across batch extraction, streaming extraction,
    prompt-context detection, token-ID boundary checks, and ``adjust_request``.
    """

    start_token = "<seed:think>"
    end_token = "</seed:think>"
    reasoning_start_tokens = ("<seed:think>", "<think>")
    reasoning_end_tokens = ("</seed:think>", "</think>")
    _TAG_PAIRS: tuple[tuple[str, str], ...] = (
        ("<seed:think>", "</seed:think>"),
        ("<think>", "</think>"),
    )

    def __init__(self, tokenizer: AnyTokenizer):
        """Resolve token IDs for both ``<seed:think>`` and ``<think>`` tag pairs."""
        super().__init__(tokenizer)
        self._pair_ids: list[tuple[str, str, int | None, int | None]] = [
            (start, end, self.vocab.get(start), self.vocab.get(end)) for start, end in self._TAG_PAIRS
        ]
        self._active_pair: tuple[str, str] | None = None

    def configure_prompt_context(self, prompt_text: str, prompt_token_ids: Sequence[int]) -> None:
        """Detect either ``<seed:think>`` or ``<think>`` at the end of the prompt."""
        super().configure_prompt_context(prompt_text, prompt_token_ids)
        self._active_pair = None
        self._prompt_started_reasoning = False
        for start, end, start_id, _end_id in self._pair_ids:
            if (start_id is not None and prompt_token_ids and prompt_token_ids[-1] == start_id) or (
                prompt_text and prompt_text.rstrip().endswith(start)
            ):
                self._prompt_started_reasoning = True
                self._active_pair = (start, end)
                break
        self._stream_seen_start = self._prompt_started_reasoning
        self._stream_seen_end = False
        self._stream_state_initialized = True
        self._streaming_entered = False
        self._emitted_reasoning = self._emitted_content = ""

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Return whether either ``</seed:think>`` or ``</think>`` appears in ``input_ids``."""
        return any(end_id is not None and end_id in input_ids for _s, _e, _sid, end_id in self._pair_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Return token IDs following the earliest recognized reasoning end token."""
        positions = [
            input_ids.index(end_id)
            for _s, _e, _sid, end_id in self._pair_ids
            if end_id is not None and end_id in input_ids
        ]
        if not positions:
            return input_ids
        return input_ids[min(positions) + 1 :]

    def _channel_snapshot(self, text: str, *, final: bool = False) -> tuple[str, str]:
        """Split reasoning/content for either ``<seed:think>`` or ``<think>`` delimiters."""
        stripped = text.lstrip()
        leading = text[: len(text) - len(stripped)]
        active_pair: tuple[str, str] | None = None

        if self._is_prompt_reasoning_active():
            content = ""
            if not final and not stripped:
                return "", ""
            for start, end in self._TAG_PAIRS:
                if stripped.startswith(start):
                    active_pair = (start, end)
                    reasoning_text = stripped[len(start) :]
                    break
            else:
                if not final and stripped and any(start.startswith(stripped) for start, _ in self._TAG_PAIRS):
                    return "", ""
                reasoning_text = text
                if self._active_pair is not None:
                    active_pair = self._active_pair
                else:
                    earliest: tuple[int, tuple[str, str]] | None = None
                    for start, end in self._TAG_PAIRS:
                        idx = reasoning_text.find(end)
                        if idx != -1 and (earliest is None or idx < earliest[0]):
                            earliest = (idx, (start, end))
                    if earliest is not None:
                        active_pair = earliest[1]
        else:
            for start, end in self._TAG_PAIRS:
                if stripped.startswith(start):
                    active_pair = (start, end)
                    content = leading
                    reasoning_text = stripped[len(start) :]
                    break
            else:
                if not final and stripped and any(start.startswith(stripped) for start, _ in self._TAG_PAIRS):
                    return "", leading
                return "", text

        if active_pair is not None:
            _start_tok, end_tok = active_pair
            reasoning, ended, after_end = reasoning_text.partition(end_tok)
            if ended:
                return reasoning, content + after_end
            if not final:
                reasoning = self._without_marker_prefix(reasoning, end_tok)
            return reasoning, content

        # Prompt-active reasoning where neither end tag has arrived yet: withhold
        # the longest proper prefix of any supported end tag.
        reasoning = reasoning_text
        if not final:
            shortest = reasoning
            for _start_tok, end_tok in self._TAG_PAIRS:
                candidate = self._without_marker_prefix(reasoning, end_tok)
                if len(candidate) < len(shortest):
                    shortest = candidate
            reasoning = shortest
        return reasoning, content
