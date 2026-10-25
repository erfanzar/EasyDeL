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

    streaming_all_phases = True

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
        self._finalized_text: str | None = None

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
            if self._find_earliest(model_output, self._thought_starters) is not None:
                # An unfinished thought section still belongs to reasoning, including a
                # terminal prefix that can no longer become a response delimiter.
                reasoning, content = self._split_stream(model_output, final=True)
                return reasoning or None, content or None
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
        straddles a chunk. The reasoning/content split is recomputed from the
        cumulative text on every call, so a delimiter spread over several
        deltas never leaks into either channel; a tail that may still grow
        into a delimiter is held back until it resolves.

        Args:
            previous_text: Cumulative text before this chunk.
            current_text: Cumulative text including this chunk; scanned
                for the thought/response starter phrases.
            delta_text: Newly produced text in this chunk.
            previous_token_ids: Token IDs before this chunk (unused).
            current_token_ids: Token IDs including this chunk (unused).
            delta_token_ids: Token IDs for ``delta_text`` (unused).
            request: Optional inference request (unused).

        Returns:
            A :class:`DeltaMessage` carrying ``reasoning_content`` and/or
            ``content`` for this step, or ``None`` when there is no new text.
            Empty decoded updates retain buffered prefixes; consumers explicitly
            call :meth:`finalize_streaming` when generation has finished.
        """
        if not delta_text:
            return None
        self._finalized_text = None

        if self._reasoning_done:
            # The response delimiter is already fully inside ``previous_text``: everything that follows is content.
            return DeltaMessage(content=delta_text)

        # Both views are derived from the cumulative text, so a delimiter that is split across any number of deltas
        # is located at its real offset and never leaks into either channel. Text that could still turn out to be
        # the start of a delimiter is held back until it resolves.
        previous_reasoning, previous_content = self._split_stream(previous_text)
        current_reasoning, current_content = self._split_stream(current_text)

        if self._find_earliest(current_text, self._thought_starters) is not None:
            self._in_reasoning = True
        if self._find_earliest(current_text, self._response_starters) is not None:
            self._reasoning_done = True

        reasoning_delta = self._monotone_suffix(previous_reasoning, current_reasoning)
        content_delta = self._monotone_suffix(previous_content, current_content)
        if not reasoning_delta and not content_delta:
            return None
        return DeltaMessage(reasoning_content=reasoning_delta or None, content=content_delta or None)

    def extract_reasoning_final(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        """Return complete channel snapshots, preserving already-streamed whitespace.

        Args:
            model_output: Complete decoded output.
            request: Optional request context (unused).

        Returns:
            Full reasoning/content text with terminal ordinary-text prefixes released.
        """
        reasoning, content = self._split_stream(model_output, final=True)
        # An empty content string is intentional (for example a delimiter-only
        # output), not a request for the consumer to fall back to raw model text.
        return reasoning or None, content

    def finalize_streaming(
        self, current_text: str, current_token_ids: Sequence[int] = (), request=None
    ) -> DeltaMessage | None:
        """Release an unresolved delimiter prefix in its original text channel.

        Args:
            current_text: Complete decoded output.
            current_token_ids: Generated tokens (unused for text delimiters).
            request: Optional request context (unused).

        Returns:
            Withheld reasoning/content suffixes, or ``None`` if already flushed.
        """
        if self._finalized_text == current_text:
            return None
        self._finalized_text = current_text
        safe_reasoning, safe_content = self._split_stream(current_text)
        reasoning, content = self._split_stream(current_text, final=True)
        reasoning_delta = self._monotone_suffix(safe_reasoning, reasoning)
        content_delta = self._monotone_suffix(safe_content, content)
        if not reasoning_delta and not content_delta:
            return None
        return DeltaMessage(reasoning_content=reasoning_delta or None, content=content_delta or None)

    @staticmethod
    def _find_earliest(text: str, starters: Sequence[str]) -> tuple[int, int] | None:
        """Locate the earliest occurrence of any starter phrase.

        Args:
            text: Cumulative text to search.
            starters: Candidate delimiter phrases.

        Returns:
            ``(start, end)`` offsets of the earliest match (the longest phrase wins a tie), or ``None``.
        """
        best: tuple[int, int] | None = None
        for starter in starters:
            idx = text.find(starter)
            if idx < 0:
                continue
            candidate = (idx, idx + len(starter))
            if best is None or candidate[0] < best[0] or (candidate[0] == best[0] and candidate[1] > best[1]):
                best = candidate
        return best

    @staticmethod
    def _partial_tail(text: str, starters: Sequence[str]) -> int:
        """Length of the longest suffix of ``text`` that is a proper prefix of some starter phrase.

        Args:
            text: Cumulative text whose tail is inspected.
            starters: Candidate delimiter phrases.

        Returns:
            Number of trailing characters that might still grow into a delimiter (``0`` when none).
        """
        longest = 0
        for starter in starters:
            for k in range(min(len(starter) - 1, len(text)), longest, -1):
                if text.endswith(starter[:k]):
                    longest = k
                    break
        return longest

    def _split_stream(self, text: str, *, final: bool = False) -> tuple[str, str]:
        """Split cumulative text into the reasoning and content that are safe to emit.

        Args:
            text: Cumulative decoded text (may end inside a delimiter phrase).
            final: Release proper delimiter prefixes as ordinary text at end of stream.

        Returns:
            ``(reasoning, content)``. Both grow monotonically as ``text`` grows.
            Complete delimiters are stripped; unresolved prefixes are withheld until
            they resolve or ``final`` releases them in their original channel.
        """
        thought = self._find_earliest(text, self._thought_starters)
        response = self._find_earliest(text, self._response_starters)

        if response is not None and (thought is None or response[0] < thought[0]):
            # Response delimiter without a preceding thought: everything around it is visible content.
            return "", text[: response[0]] + text[response[1] :]

        if thought is not None:
            content = text[: thought[0]]
            if response is None:
                hold = 0 if final else self._partial_tail(text, self._response_starters)
                return text[thought[1] : len(text) - hold], content
            return text[thought[1] : response[0]], content + text[response[1] :]

        hold = 0 if final else self._partial_tail(text, (*self._thought_starters, *self._response_starters))
        return "", text[: len(text) - hold]

    @staticmethod
    def _monotone_suffix(previous: str, current: str) -> str:
        """Return the part of ``current`` that extends ``previous``.

        Args:
            previous: Channel text derived from the previous cumulative text.
            current: Channel text derived from the current cumulative text.

        Returns:
            The newly appended text; the whole ``current`` if it does not extend ``previous``.
        """
        if current.startswith(previous):
            return current[len(previous) :]
        return current
