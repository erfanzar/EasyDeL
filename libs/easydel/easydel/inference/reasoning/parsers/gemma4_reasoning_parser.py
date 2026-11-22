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

"""Reasoning parser for Gemma4 thinking models.

Gemma4 uses ``<|channel>`` / ``<channel|>`` tokens to delimit reasoning
content. When thinking is enabled, the model generates::

    <|channel>thought
    ...chain of thought reasoning...<channel|>
    Final answer text here.

The ``thought\\n`` role label inside the channel delimiters is a structural
artefact that must be stripped so downstream consumers see only actual
reasoning text.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import cached_property

from transformers import AutoTokenizer as AnyTokenizer

from ...openai_api_modules import DeltaMessage
from ..abstract_reasoning import ReasoningParserManager
from ..basic_parsers import BaseThinkingReasoningParser

# Role label that Gemma4 emits at the start of the thinking channel.
_THOUGHT_PREFIX = "thought\n"
_NATIVE_REASONING_PREFIXES = ("thought\r\n", "thought\n", "analysis\r\n", "analysis\n")
_KNOWN_HEADER_CHANNELS = frozenset({"thought", "analysis", "final", "commentary"})

CHANNEL_START = "<|channel>"
CHANNEL_END = "<channel|>"


def _extract_native_reasoning_body(text: str) -> str | None:
    """Return the body after a native inline ``thought\\n``/``analysis\\n`` opener, else ``None``."""
    for prefix in _NATIVE_REASONING_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :]
    return None


def _could_be_header_prefix(tail: str) -> bool:
    """Return whether an unclosed ``<|channel>`` tail might still become a channel header or native opener."""
    if not tail:
        return True
    if any(prefix.startswith(tail) for prefix in _NATIVE_REASONING_PREFIXES):
        return True
    header_candidates = tuple(f"{name}{CHANNEL_END}" for name in _KNOWN_HEADER_CHANNELS)
    if any(candidate.startswith(tail) for candidate in header_candidates):
        return True
    return False


@ReasoningParserManager.register_module("gemma4")  # pyright: ignore[reportUntypedClassDecorator]
class Gemma4ReasoningParser(BaseThinkingReasoningParser):
    """Extract reasoning content from Gemma4 model outputs.

    Supports both Gemma4 channel formats coherently across batch, streaming,
    prompt-context, and token-ID checks:

    - Native inline format: ``<|channel>thought\\n...reasoning...<channel|>answer``
    - Header-tag format: ``<|channel>thought<channel|>...reasoning...<|channel>final<channel|>answer``
    """

    start_token = f"{CHANNEL_START}thought{CHANNEL_END}"
    end_token = f"{CHANNEL_START}final{CHANNEL_END}"
    reasoning_start_tokens = (
        f"{CHANNEL_START}thought\n",
        f"{CHANNEL_START}analysis\n",
        f"{CHANNEL_START}thought{CHANNEL_END}",
        f"{CHANNEL_START}analysis{CHANNEL_END}",
    )

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize parser state and compile the channel-detection regex.

        Args:
            tokenizer: HuggingFace tokenizer whose vocabulary supplies the
                channel start/end token IDs.
        """
        super().__init__(tokenizer)
        self._reasoning_text: str = ""
        self._prefix_stripped: bool = False
        self._channel_pattern = re.compile(
            re.escape(CHANNEL_START) + r"(.*?)" + re.escape(CHANNEL_END),
            re.DOTALL,
        )
        self._prompt_channel: str | None = None
        self._prompt_inline_reasoning: bool = False

    @cached_property
    def start_token_id(self) -> int | None:
        """Token ID corresponding to the channel-open marker (``<|channel>``), or ``None``."""
        return self.vocab.get(CHANNEL_START)

    @cached_property
    def end_token_id(self) -> int | None:
        """Token ID corresponding to the channel-close marker (``<channel|>``), or ``None``."""
        return self.vocab.get(CHANNEL_END)

    @cached_property
    def _composite_reasoning_start_ids(self) -> frozenset[int]:
        """Composite header token IDs that open reasoning channels, if present in vocab."""
        ids = {
            self.vocab.get(f"{CHANNEL_START}{name}{CHANNEL_END}")
            for name in self._REASONING_CHANNELS
        }
        return frozenset(tid for tid in ids if tid is not None)

    @cached_property
    def _composite_final_id(self) -> int | None:
        """Composite ``<|channel>final<channel|>`` token ID if present in vocab."""
        return self.vocab.get(f"{CHANNEL_START}final{CHANNEL_END}")

    @cached_property
    def _tool_call_token_id(self) -> int | None:
        """Token ID for ``<|tool_call>`` if present in the vocabulary, else ``None``."""
        return self.vocab.get("<|tool_call>")

    @cached_property
    def _new_turn_token_id(self) -> int | None:
        """Token ID for ``<|turn>`` if present in the vocabulary, else ``None``."""
        return self.vocab.get("<|turn>")

    @cached_property
    def _tool_response_token_id(self) -> int | None:
        """Token ID for ``<|tool_response>`` if present in the vocabulary, else ``None``."""
        return self.vocab.get("<|tool_response>")

    def adjust_request(self, request) -> object:
        """Disable special-token stripping to preserve channel markers.

        Args:
            request: The mutable inference request whose
                ``skip_special_tokens`` flag is forced to ``False``.

        Returns:
            The same ``request`` after mutation, for fluent chaining.
        """
        request.skip_special_tokens = False
        return request

    def configure_prompt_context(self, prompt_text: str, prompt_token_ids: Sequence[int]) -> None:
        """Track the channel opened by the prompt, supporting both native and header-tag forms."""
        del prompt_token_ids
        self._emitted_reasoning = self._emitted_content = ""
        self._prompt_channel = None
        self._prompt_inline_reasoning = False
        text = prompt_text or ""
        cursor = 0
        while cursor < len(text):
            start_idx = text.find(CHANNEL_START, cursor)
            if start_idx < 0:
                break
            end_idx = text.find(CHANNEL_END, start_idx + len(CHANNEL_START))
            if end_idx < 0:
                tail = text[start_idx + len(CHANNEL_START) :]
                if _extract_native_reasoning_body(tail) is not None:
                    self._prompt_channel = "thought"
                    self._prompt_inline_reasoning = True
                break
            inner = text[start_idx + len(CHANNEL_START) : end_idx]
            if _extract_native_reasoning_body(inner) is not None:
                self._prompt_channel = None
                self._prompt_inline_reasoning = False
            else:
                self._prompt_channel = inner.strip() or None
                self._prompt_inline_reasoning = False
            cursor = end_idx + len(CHANNEL_END)

    def _decode_tokens_best_effort(self, token_ids: Sequence[int]) -> str | None:
        """Decode a small token slice when the tokenizer supports ``decode``."""
        decode = getattr(self.model_tokenizer, "decode", None)
        if not callable(decode):
            return None
        try:
            try:
                return decode(list(token_ids), skip_special_tokens=False)
            except TypeError:
                return decode(list(token_ids))
        except Exception:
            return None

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Check whether reasoning has ended across both native and header-tag Gemma4 formats.

        Handles:
        - Native inline ``<|channel>thought\\n...<channel|>`` (ended on ``<channel|>``)
        - Header-tag ``<|channel>thought<channel|>...<|channel>final<channel|>`` (open after
          ``<|channel>thought<channel|>``, ended once a non-reasoning channel or ``<|tool_call>`` arrives)
        - Missing vocabulary tokens (never raises ``KeyError``)
        """
        start_id = self.start_token_id
        end_id = self.end_token_id
        tool_call_id = self._tool_call_token_id
        new_turn_id = self._new_turn_token_id
        tool_response_id = self._tool_response_token_id
        comp_reasoning = self._composite_reasoning_start_ids
        comp_final = self._composite_final_id

        # Scan from the last turn/tool-response reset point forward so header-tag
        # pairs (<|channel>thought<channel|>) are distinguished from native inline
        # blocks (<|channel>thought\n...<channel|>).
        scan_start = 0
        for i in range(len(input_ids) - 1, -1, -1):
            tid = input_ids[i]
            if (new_turn_id is not None and tid == new_turn_id) or (
                tool_response_id is not None and tid == tool_response_id
            ):
                scan_start = i + 1
                break

        in_reasoning = self._prompt_channel in self._REASONING_CHANNELS
        ended = False
        open_channel_idx: int | None = None

        for i in range(scan_start, len(input_ids)):
            tid = input_ids[i]
            if comp_reasoning and tid in comp_reasoning:
                in_reasoning = True
                ended = False
                open_channel_idx = None
                continue
            if comp_final is not None and tid == comp_final:
                in_reasoning = False
                ended = True
                open_channel_idx = None
                continue
            if tool_call_id is not None and tid == tool_call_id:
                in_reasoning = False
                ended = True
                open_channel_idx = None
                continue
            if start_id is not None and tid == start_id:
                open_channel_idx = i
                in_reasoning = True
                ended = False
                continue
            if end_id is not None and tid == end_id:
                if open_channel_idx is None:
                    # Closing a prompt-opened inline channel.
                    in_reasoning = False
                    ended = True
                else:
                    between = input_ids[open_channel_idx + 1 : i]
                    inner = self._decode_tokens_best_effort(between)
                    if inner is not None:
                        if _extract_native_reasoning_body(inner) is not None:
                            in_reasoning = False
                            ended = True
                        elif inner.strip() in self._REASONING_CHANNELS:
                            # Header-tag opener (<|channel>thought<channel|>): reasoning JUST started!
                            in_reasoning = True
                            ended = False
                        else:
                            # Header-tag switch to non-reasoning channel (e.g. final).
                            in_reasoning = False
                            ended = True
                    else:
                        in_reasoning = False
                        ended = True
                    open_channel_idx = None

        return ended and not in_reasoning

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Return the input IDs unchanged."""
        return list(input_ids)

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        """Extract reasoning and content from complete model output."""
        del request
        if CHANNEL_START not in model_output and CHANNEL_END not in model_output and self._prompt_channel is None:
            return None, model_output

        reasoning, content = self._channel_snapshot(model_output, final=True)
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
        """Extract streaming reasoning for both native and header-tag Gemma4 formats."""
        return self._emit_snapshot(*self._channel_snapshot(current_text, final=False))

    def _channel_snapshot(self, text: str, *, final: bool = False) -> tuple[str, str]:
        """Compute stable ``(reasoning, content)`` snapshots without dropping unfinished reasoning."""
        reasoning, content, last_channel = self._parse_channels_with_state(
            text,
            initial_channel=self._prompt_channel,
            initial_inline_reasoning=self._prompt_inline_reasoning,
            final=final,
        )
        reasoning = reasoning or ""
        if not final and last_channel in self._REASONING_CHANNELS and reasoning and _THOUGHT_PREFIX.startswith(reasoning):
            reasoning = ""
        else:
            reasoning = _strip_thought_label(reasoning)
        return reasoning, content or ""

    def extract_reasoning_final(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        """Return terminal channel snapshots, preserving unfinished reasoning and literal tags."""
        reasoning, content = self._channel_snapshot(model_output, final=True)
        recognized = (content != model_output) or self._prompt_channel is not None
        return reasoning if recognized or reasoning else None, content or None

    _REASONING_CHANNELS = frozenset({"thought", "analysis"})

    def _parse_channels(
        self,
        text: str,
        *,
        initial_channel: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Split text into reasoning and content by channel markers."""
        reasoning, content, _last_channel = self._parse_channels_with_state(
            text,
            initial_channel=initial_channel,
            initial_inline_reasoning=self._prompt_inline_reasoning,
            final=True,
        )
        return reasoning, content

    def _parse_channels_with_state(
        self,
        text: str,
        *,
        initial_channel: str | None = None,
        initial_inline_reasoning: bool = False,
        final: bool = True,
    ) -> tuple[str | None, str | None, str | None]:
        """Parse both native inline and header-tag Gemma4 channel markers."""
        cursor = 0
        current_channel: str | None = initial_channel
        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        if initial_inline_reasoning:
            end_idx = text.find(CHANNEL_END)
            if end_idx < 0:
                body = text if final else self._without_marker_prefix(text, CHANNEL_END)
                if body:
                    reasoning_parts.append(body)
                return "".join(reasoning_parts) or None, None, current_channel
            body = text[:end_idx]
            if body:
                reasoning_parts.append(body)
            cursor = end_idx + len(CHANNEL_END)
            current_channel = None

        while cursor < len(text):
            start_idx = text.find(CHANNEL_START, cursor)
            if start_idx < 0:
                chunk = text[cursor:]
                if not final:
                    chunk = self._without_marker_prefix(chunk, CHANNEL_START)
                if chunk:
                    if current_channel in self._REASONING_CHANNELS:
                        reasoning_parts.append(chunk)
                    else:
                        content_parts.append(chunk)
                break

            if start_idx > cursor:
                chunk = text[cursor:start_idx]
                if current_channel in self._REASONING_CHANNELS:
                    reasoning_parts.append(chunk)
                else:
                    content_parts.append(chunk)

            after_start = start_idx + len(CHANNEL_START)
            end_idx = text.find(CHANNEL_END, after_start)
            if end_idx < 0:
                tail = text[after_start:]
                native_body = _extract_native_reasoning_body(tail)
                if native_body is not None:
                    # Confirmed native inline reasoning block still streaming or truncated at EOS.
                    body = native_body if final else self._without_marker_prefix(native_body, CHANNEL_END)
                    if body:
                        reasoning_parts.append(body)
                    current_channel = "thought"
                elif not final:
                    if not _could_be_header_prefix(tail):
                        literal = self._without_marker_prefix(text[start_idx:], CHANNEL_START)
                        if current_channel in self._REASONING_CHANNELS:
                            reasoning_parts.append(literal)
                        else:
                            content_parts.append(literal)
                else:
                    # At finalization, suppress a truncated incomplete header (e.g. "<|channel>fin")
                    # while preserving ordinary text following a literal "<|channel>".
                    stripped_tail = tail.strip()
                    is_truncated_header = not stripped_tail or any(
                        name.startswith(stripped_tail) for name in _KNOWN_HEADER_CHANNELS
                    )
                    if not is_truncated_header:
                        if current_channel in self._REASONING_CHANNELS:
                            reasoning_parts.append(text[start_idx:])
                        else:
                            content_parts.append(text[start_idx:])
                break

            inner = text[after_start:end_idx]
            native_body = _extract_native_reasoning_body(inner)
            if native_body is not None:
                if native_body:
                    reasoning_parts.append(native_body)
                current_channel = None
            else:
                current_channel = inner.strip() or None
            cursor = end_idx + len(CHANNEL_END)

        reasoning = "".join(reasoning_parts) or None
        content = "".join(content_parts) or None
        return reasoning, content, current_channel


def _strip_thought_label(text: str) -> str:
    """Remove the ``thought\\n`` role label from the beginning of text.

    Args:
        text: Reasoning text that may begin with the structural role label.

    Returns:
        ``text`` with the leading prefix stripped if present, otherwise
        unchanged.
    """
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX) :]
    return text


def _suffix(previous: str | None, current: str | None) -> str | None:
    """Compute the incremental text delta between two cumulative strings.

    Args:
        previous: Cumulative text from the previous snapshot.
        current: Cumulative text from the current snapshot.

    Returns:
        The newly appended portion when ``current`` extends ``previous``,
        ``current`` itself when prefix matching fails, or ``None`` when
        nothing was produced.
    """
    previous = previous or ""
    current = current or ""
    if not current:
        return None
    if current.startswith(previous):
        delta = current[len(previous) :]
        return delta or None
    return current
