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

"""Reasoning parser for IBM Granite chain-of-thought models.

Granite does not use bracket/special tokens to delimit reasoning. Instead, it
emits literal English phrases as delimiters::

    Here's my thought process:
    ...chain-of-thought reasoning...
    Here's my response:
    ...visible response...

Both delimiters tolerate the "Here's"/"Here is" variants. The parser performs
text-level matching (via a single compiled regex for batch mode and substring
scans for streaming mode) since the delimiter strings are not single tokens
in the vocabulary.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from transformers import AutoTokenizer as AnyTokenizer

from ...openai_api_modules import DeltaMessage
from ..abstract_reasoning import ReasoningParser, ReasoningParserManager

_THOUGHT_STARTERS = [
    "Here's my thought process:",
    "Here is my thought process:",
]
_RESPONSE_STARTERS = [
    "Here's my response:",
    "Here is my response:",
]


@ReasoningParserManager.register_module(["granite"])  # pyright: ignore[reportUntypedClassDecorator]
class GraniteReasoningParser(ReasoningParser):
    """Reasoning parser for IBM Granite outputs using English-phrase delimiters.

    Granite chain-of-thought outputs are split with literal English phrases
    rather than special tokens. The parser compiles a regex that matches the
    ``Here's my thought process: ... Here's my response: ...`` shape, then
    keeps a tiny two-flag state machine (``_in_reasoning`` /
    ``_reasoning_done``) so streaming deltas survive partial delimiter
    arrival across chunk boundaries.

    Attributes:
        _regex (re.Pattern[str]): Compiled regex matching the full
            thought/response sandwich across newlines (``re.DOTALL``).
        _thought_starters (list[str]): Accepted opening phrases.
        _response_starters (list[str]): Accepted closing phrases.
        _in_reasoning (bool): Streaming flag — ``True`` once the thought
            delimiter has been observed and reasoning content is being
            collected.
        _reasoning_done (bool): Streaming flag — ``True`` once the response
            delimiter has been observed and subsequent deltas are visible
            content.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize parser state and compile the thought/response regex.

        Args:
            tokenizer: HuggingFace tokenizer used by
                :meth:`is_reasoning_end` to decode token IDs back into text
                for delimiter matching.
        """
        super().__init__(tokenizer)
        thought_pattern = "|".join(re.escape(s) for s in _THOUGHT_STARTERS)
        response_pattern = "|".join(re.escape(s) for s in _RESPONSE_STARTERS)
        self._regex = re.compile(
            rf"(?:{thought_pattern})\s*(.*?)\s*(?:{response_pattern})\s*(.*)",
            re.DOTALL,
        )
        self._thought_starters = _THOUGHT_STARTERS
        self._response_starters = _RESPONSE_STARTERS
        self._in_reasoning = False
        self._reasoning_done = False

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Report whether a ``Here's my response:`` phrase has been emitted.

        Args:
            input_ids: Sequence of generated token IDs which are decoded
                back to text for substring matching against the response
                starters.

        Returns:
            ``True`` when any of the response-starter phrases appears in
            the decoded text, ``False`` otherwise.
        """
        text = self.model_tokenizer.decode(list(input_ids), skip_special_tokens=False)
        return any(s in text for s in self._response_starters)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Return ``input_ids`` verbatim.

        Granite splits at the text level rather than via dedicated content
        tokens, so all token IDs are forwarded; downstream consumers should
        operate on the decoded text instead.

        Args:
            input_ids: Full generated token-ID sequence.

        Returns:
            A list copy of ``input_ids``.
        """
        return list(input_ids)

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        """Split the model output into reasoning and content via the compiled regex.

        Args:
            model_output: Full decoded text produced by the model.
            request: Optional inference request; unused, kept for interface
                parity with sibling parsers.

        Returns:
            Tuple ``(reasoning, content)`` extracted from the regex
            capture groups. When the regex does not match (no
            thought/response sandwich found), returns
            ``(None, model_output)`` so the whole text is treated as
            visible content. Either element may be ``None`` when the
            captured group was empty after stripping.
        """
        match = self._regex.search(model_output)
        if not match:
            return None, model_output
        reasoning = match.group(1).strip()
        content = match.group(2).strip()
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
        """Route streaming deltas using thought/response delimiter scanning.

        Maintains the parser's two-flag state machine so the correct half
        of the output is emitted for each delta even when the delimiter
        straddles a chunk.

        Args:
            previous_text: Cumulative text before this chunk (unused).
            current_text: Cumulative text including this chunk; scanned
                for the response-starter phrases.
            delta_text: Newly produced text in this chunk.
            previous_token_ids: Token IDs before this chunk (unused).
            current_token_ids: Token IDs including this chunk (unused).
            delta_token_ids: Token IDs for ``delta_text`` (unused).
            request: Optional inference request (unused).

        Returns:
            A :class:`DeltaMessage` carrying ``reasoning_content`` and/or
            ``content`` for this step, or ``None`` when the delta is empty.
        """
        if not delta_text:
            return None

        # Check if we've already found the response delimiter
        if self._reasoning_done:
            return DeltaMessage(content=delta_text)

        # Check if response delimiter appears in current text
        for starter in self._response_starters:
            if starter in current_text:
                self._reasoning_done = True
                # Check if it's in the delta
                if starter in delta_text:
                    parts = delta_text.split(starter, 1)
                    reasoning_part = parts[0] if self._in_reasoning else None
                    content_part = parts[1] if len(parts) > 1 else None
                    return DeltaMessage(
                        reasoning_content=reasoning_part if reasoning_part else None,
                        content=content_part if content_part else None,
                    )
                return DeltaMessage(content=delta_text)

        # Check if thought delimiter appears
        for starter in self._thought_starters:
            if starter in current_text:
                self._in_reasoning = True
                if starter in delta_text:
                    after = delta_text.split(starter, 1)[1]
                    return DeltaMessage(reasoning_content=after) if after else None
                if self._in_reasoning:
                    return DeltaMessage(reasoning_content=delta_text)

        if self._in_reasoning:
            return DeltaMessage(reasoning_content=delta_text)

        return DeltaMessage(content=delta_text)
