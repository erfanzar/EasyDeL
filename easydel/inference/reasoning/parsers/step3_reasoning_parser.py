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

"""Reasoning parser for StepFun Step3 thinking models.

Step3 uses an asymmetric chain-of-thought grammar: the opening ``<think>``
marker is *never* emitted by the model, only the closing ``</think>`` marker
appears. The parser therefore treats every text chunk before the first
``</think>`` as reasoning and everything after as visible content. If a
stray ``<think>`` literal does appear in the output it is stripped before
the reasoning is surfaced.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...openai_api_modules import DeltaMessage
from ..abstract_reasoning import ReasoningParserManager
from ..basic_parsers import BaseThinkingReasoningParser


@ReasoningParserManager.register_module(["step3"])  # pyright: ignore[reportUntypedClassDecorator]
class Step3ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Step3 chain-of-thought outputs (asymmetric grammar).

    Subclasses :class:`BaseThinkingReasoningParser` but overrides
    :meth:`extract_reasoning` and :meth:`extract_reasoning_streaming` so
    that *only* the ``</think>`` closing marker is required. Any
    occurrences of ``<think>`` in the output (which Step3 normally does
    not emit but might appear if a chat template injects one) are stripped
    from the reasoning before being surfaced.

    Attributes:
        start_token: Reasoning open tag ``"<think>"``. Kept for compatibility
            with the base class but not required to be present in the
            model output.
        end_token: Reasoning close tag ``"</think>"``. This is the only
            marker the parser actually relies on.
    """

    start_token = "<think>"
    end_token = "</think>"

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        """Split the output at ``</think>``.

        Args:
            model_output: Full decoded text produced by the model.
            request: Optional inference request; unused, kept for interface
                parity with sibling parsers.

        Returns:
            Tuple ``(reasoning, content)`` where ``reasoning`` is the text
            before the first ``</think>`` (with any stray ``<think>``
            literals removed) and ``content`` is the text after it. When
            ``</think>`` is absent: if ``<think>`` is present in the output
            it is stripped and the rest is treated as a still-unfinished
            reasoning block (``(reasoning, None)``); otherwise the entire
            text is treated as visible content (``(None, model_output)``).
        """
        if self.end_token not in model_output:
            if self.start_token in model_output:
                cleaned = model_output.replace(self.start_token, "").strip()
                return cleaned or None, None
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
        """Stream reasoning until ``</think>`` is observed, then switch to content.

        Args:
            previous_text: Cumulative text before this chunk; used only to
                detect whether ``</think>`` has already been seen in a
                prior delta.
            current_text: Cumulative text including this chunk (unused).
            delta_text: Newly produced text in this chunk.
            previous_token_ids: Token IDs before this chunk (unused).
            current_token_ids: Token IDs including this chunk (unused).
            delta_token_ids: Token IDs for ``delta_text`` (unused).
            request: Optional inference request (unused).

        Returns:
            A :class:`DeltaMessage` carrying ``reasoning_content`` while
            still inside the reasoning section, both fields when the
            ``</think>`` boundary is straddled by this chunk, then plain
            ``content`` on subsequent chunks; ``None`` when ``delta_text``
            is empty or only carried a stripped ``<think>`` literal.
        """
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
