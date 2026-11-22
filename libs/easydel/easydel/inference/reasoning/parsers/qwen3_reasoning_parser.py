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
    """Reasoning parser for Qwen3 chain-of-thought outputs.

    Qwen3 wraps reasoning in literal ``<think>`` / ``</think>`` markers,
    matching the DeepSeek-R1 grammar. The parser differs from R1 only in
    its strictness:

    - **Missing start tag** -> the output is treated as visible content,
      unless the prompt context indicates that the chat template injected
      ``<think>`` (asymmetric prompt-gated mode).
    - **Missing end tag** -> the unfinished ``<think>...`` segment is kept
      hidden as reasoning when the start tag is present, otherwise the
      text is surfaced as content.

    Expected model output format::

        <think>chain-of-thought reasoning</think>visible response

    Parsed output schema (returned by :meth:`extract_reasoning`):

    - ``reasoning_content``: the text between ``<think>`` and ``</think>``.
    - ``visible_content``: everything after ``</think>`` (and any text
      preceding the start tag prepended).

    Edge cases handled:

    - Both tags absent -> all text becomes visible content.
    - Only start tag -> entire post-start segment hidden as reasoning.
    - Only end tag -> visible content unless prompt-gated reasoning is on.
    - Tags split across streaming chunks -> base-class state machine
      reassembles them.

    Attributes:
        start_token: The opening reasoning marker ``"<think>"``.
        end_token: The closing reasoning marker ``"</think>"``.
    """

    start_token = "<think>"
    end_token = "</think>"

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        """Split a complete Qwen3 generation into reasoning and content.

        Delegates to :class:`BaseThinkingReasoningParser`, which enforces
        leading-only ``<think>`` detection (unless prompt context or
        ``assume_reasoning`` indicates thinking already opened) and keeps
        unfinished ``<think>...`` segments hidden as reasoning.
        """
        if not model_output:
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
        """Stream-aware reasoning extraction with Qwen3 strict-tag rules.

        Delegates directly to :class:`BaseThinkingReasoningParser`, which
        handles leading-only ``<think>`` detection, split-prefix withholding,
        and monotonic ``_emitted_content`` tracking without dropping withheld
        prefixes or replaying content when literal ``<think>`` appears mid-text.
        """
        return super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )
