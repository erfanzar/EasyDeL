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

"""Identity reasoning parser — pass-through that treats all output as content."""

from __future__ import annotations

from collections.abc import Sequence

from ...openai_api_modules import DeltaMessage
from ..abstract_reasoning import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module(["identity", "none", "passthrough"])  # pyright: ignore[reportUntypedClassDecorator]
class IdentityReasoningParser(ReasoningParser):
    """No-op reasoning parser that surfaces every token as visible content.

    Used by the orchestrator (and by :class:`DeepSeekV3ReasoningParser` as
    a delegate) when reasoning extraction must be disabled — for example
    when the requested model does not have a chain-of-thought channel,
    when ``enable_reasoning=False``, or when an unrecognised parser name
    falls back to a safe pass-through. Registering under three aliases
    (``identity`` / ``none`` / ``passthrough``) lets configuration files
    spell the same intent in different conventional ways.

    The parser holds no state and emits exactly one streaming event per
    delta: a :class:`DeltaMessage` whose ``content`` field is the
    incoming ``delta_text``. ``is_reasoning_end`` always returns ``True``
    so the rest of the pipeline correctly believes there is nothing to
    wait for.
    """

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Report that the (non-existent) reasoning section is already over.

        Args:
            input_ids: Decoded token IDs (ignored).

        Returns:
            Always ``True``; the pass-through parser has no boundary to
            wait for.
        """
        return True

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Return ``input_ids`` verbatim — no token is treated as reasoning.

        Args:
            input_ids: Full decoded token sequence.

        Returns:
            A list copy of ``input_ids`` so callers can mutate it without
            disturbing the caller's view.
        """
        return list(input_ids)

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        """Treat the entire model output as visible content.

        Args:
            model_output: Complete decoded text from the model.
            request: Unused; kept for interface parity.

        Returns:
            Tuple ``(None, model_output)`` — no reasoning portion, the
            full text is visible content.
        """
        return None, model_output

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
        """Emit each streaming delta verbatim as visible content.

        Args:
            previous_text: Cumulative text prior to this delta (unused).
            current_text: Cumulative text including this delta (unused).
            delta_text: New text produced in this chunk; surfaced as
                ``content`` on the returned :class:`DeltaMessage`.
            previous_token_ids: Token IDs prior to this delta (unused).
            current_token_ids: Token IDs including this delta (unused).
            delta_token_ids: Token IDs corresponding to ``delta_text``
                (unused).
            request: Unused; kept for interface parity.

        Returns:
            A :class:`DeltaMessage` carrying ``delta_text`` as
            ``content`` when non-empty, or ``None`` when nothing new
            arrived this step (so the streamer can suppress an empty
            event).
        """
        return DeltaMessage(content=delta_text) if delta_text else None
