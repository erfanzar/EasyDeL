# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hermes model tool call parser.

This module provides a tool call parser for NousResearch Hermes models and
similar architectures that use XML-style delimiters (`<tool_call>` and
`</tool_call>`) for function calling. The parser extracts JSON content from
within these XML-style tags.

The parser includes sophisticated handling for multi-token delimiters,
ensuring accurate boundary detection when the start/end tags are split
across multiple streaming tokens. Reasoning scratch pads are not executable,
and markup inside JSON strings is argument data, not a protocol boundary.

Example:
    >>> from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser
    >>> parser = HermesToolParser(tokenizer)
    >>> result = parser.extract_tool_calls(
    ...     '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>',
    ...     request
    ... )
    >>> result.tools_called
    True

Supported Format:
    <tool_call>{"name": "function_name", "arguments": {...}}</tool_call>

See Also:
    - NousResearch Hermes models documentation
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from uuid import uuid4

import partial_json_parser  # pyright: ignore[reportMissingTypeStubs]
from eformer.loggings import get_logger
from partial_json_parser.core.options import Allow  # pyright: ignore[reportMissingTypeStubs]
from transformers import AutoTokenizer as AnyTokenizer

from ...openai_api_modules import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from ..abstract_tool import ToolParser, ToolParserManager
from ..utils import stable_json_stream_prefix

logger = get_logger(__name__)

_SCRATCH_PAD_START = "<scratch_pad>"
_SCRATCH_PAD_END = "</scratch_pad>"


def _trailing_prefix_len(text: str, start: int, marker: str) -> int:
    """Return length of the longest proper prefix of *marker* at the end of ``text[start:]``."""
    avail = len(text) - start
    max_len = min(len(marker) - 1, avail)
    for size in range(max_len, 0, -1):
        if text.endswith(marker[:size]):
            return size
    return 0


@ToolParserManager.register_module("hermes")  # pyright: ignore[reportUntypedClassDecorator]
class HermesToolParser(ToolParser):
    """Tool call parser for NousResearch Hermes models.

    Handles one or more JSON function calls wrapped in ``<tool_call>`` /
    ``</tool_call>``. Partial JSON arguments are emitted incrementally, while
    reasoning inside complete or still-open scratch pads remains content.

    Attributes:
        current_tool_name_sent: Whether the active tool's name was emitted.
        prev_tool_call_arr: Previously parsed calls, indexed by tool index.
        current_tool_id: Active tool index, or -1 before any tool starts.
        streamed_args_for_tool: JSON argument prefixes already emitted per tool.
        tool_call_start_token: Opening tool delimiter.
        tool_call_end_token: Closing tool delimiter.
        tool_call_regex: Compatibility regex matching full or trailing calls.
        scratch_pad_regex: Matches complete or still-open reasoning scratch pads.
        tool_call_start_token_ids: Tokenized opening delimiter.
        tool_call_end_token_ids: Tokenized closing delimiter.
        tool_call_start_token_array: Per-token decoded opening delimiter pieces.
        tool_call_end_token_array: Per-token decoded closing delimiter pieces.
        buffered_delta_text: Buffer used by ``tool_call_delta_buffer``.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the parser and its delimiter metadata.

        Args:
            tokenizer: Tokenizer providing encode and decode operations.

        Raises:
            ValueError: If the tokenizer is missing.
        """
        super().__init__(tokenizer)
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []
        self._streamed_content_length = 0
        self._pending_content_suffix = ""
        self._streaming_initialized = False
        self._region_to_emitted_index: dict[int, int] = {}
        self._scan_pos: int = 0
        self._in_scratch_pad: bool = False
        self._emitted_source_pos: int = 0
        self._has_open_region: bool = False

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
        self.scratch_pad_regex = re.compile(r"<scratch_pad>.*?(?:</scratch_pad>|\Z)", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")
        self.tool_call_start_token_ids = self.model_tokenizer.encode(
            self.tool_call_start_token, add_special_tokens=False
        )
        self.tool_call_end_token_ids = self.model_tokenizer.encode(self.tool_call_end_token, add_special_tokens=False)
        self.tool_call_start_token_array = [
            self.model_tokenizer.decode([token_id]) for token_id in self.tool_call_start_token_ids
        ]
        self.tool_call_end_token_array = [
            self.model_tokenizer.decode([token_id]) for token_id in self.tool_call_end_token_ids
        ]
        self.buffered_delta_text = ""

    def tool_call_delta_buffer(self, delta_text: str) -> str:
        """Buffer per-token fragments of a tool delimiter.

        Args:
            delta_text: Newly decoded token text.

        Returns:
            Buffered text when a delimiter completes, or text that is not a delimiter.
        """
        if delta_text in self.tool_call_start_token_array or delta_text in self.tool_call_end_token_array:
            if delta_text == self.tool_call_start_token_array[-1] or delta_text == self.tool_call_end_token_array[-1]:
                buffered_text = self.buffered_delta_text
                self.buffered_delta_text = ""
                return buffered_text + delta_text
            candidate = self.buffered_delta_text + delta_text
            if self.tool_call_start_token.startswith(candidate) or self.tool_call_end_token.startswith(candidate):
                self.buffered_delta_text = candidate
                return ""
            buffered_text = self.buffered_delta_text
            if self.tool_call_start_token.startswith(delta_text) or self.tool_call_end_token.startswith(delta_text):
                self.buffered_delta_text = delta_text
                return buffered_text
            self.buffered_delta_text = ""
            return buffered_text + delta_text
        if self.buffered_delta_text:
            buffered_text = self.buffered_delta_text
            self.buffered_delta_text = ""
            return buffered_text + delta_text
        return delta_text

    @staticmethod
    def _find_tool_call_end(text: str, body_start: int, end_token: str) -> int | None:
        """Locate *end_token* at or after *body_start* outside JSON double-quoted strings."""
        position = body_start
        n = len(text)
        in_string = False
        while position < n:
            if not in_string:
                quote_idx = text.find('"', position)
                end_idx = text.find(end_token, position)
                if end_idx != -1 and (quote_idx == -1 or end_idx < quote_idx):
                    return end_idx
                if quote_idx == -1:
                    return None
                in_string = True
                position = quote_idx + 1
            else:
                quote_idx = text.find('"', position)
                if quote_idx == -1:
                    return None
                bs_idx = text.find("\\", position, quote_idx)
                if bs_idx == -1:
                    in_string = False
                    position = quote_idx + 1
                else:
                    position = bs_idx + 2
        return None

    def _tool_call_regions(self, text: str) -> list[tuple[int, str, int | None]]:
        """Locate calls without interpreting reasoning or quoted markup as protocol.

        Args:
            text: Complete or cumulative model output.

        Returns:
            Tuples of opening offset, unmodified JSON body and end offset.
            The end offset is None for a still-open call.
        """
        regions: list[tuple[int, str, int | None]] = []
        start_token = self.tool_call_start_token
        end_token = self.tool_call_end_token
        start_len = len(start_token)
        end_len = len(end_token)
        scratch_start_len = len(_SCRATCH_PAD_START)
        scratch_end_len = len(_SCRATCH_PAD_END)
        position = 0
        n = len(text)
        no_more_scratch = False

        while position < n:
            tc_idx = text.find(start_token, position)
            if tc_idx == -1:
                break
            if not no_more_scratch:
                sp_idx = text.find(_SCRATCH_PAD_START, position, tc_idx)
                if sp_idx != -1:
                    sp_end = text.find(_SCRATCH_PAD_END, sp_idx + scratch_start_len)
                    if sp_end == -1:
                        break
                    position = sp_end + scratch_end_len
                    continue
                if text.find(_SCRATCH_PAD_START, tc_idx + start_len) == -1:
                    no_more_scratch = True

            body_start = tc_idx + start_len
            call_end = self._find_tool_call_end(text, body_start, end_token)
            if call_end is None:
                regions.append((tc_idx, text[body_start:], None))
                break
            region_end = call_end + end_len
            regions.append((tc_idx, text[body_start:call_end], region_end))
            position = region_end

        return regions

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Report whether a real tool-call region or trailing start-marker prefix is buffered.

        Respects ``<scratch_pad>...</scratch_pad>`` blocks and JSON strings inside
        tool calls, using dynamic ``self.tool_call_start_token`` / ``self.tool_call_end_token``
        so subclasses such as :class:`LongcatFlashToolParser` inherit exact behavior.
        """
        del delta_text
        if self._streaming_initialized:
            return self._has_open_region or bool(self._pending_content_suffix)

        start_token = self.tool_call_start_token
        end_token = self.tool_call_end_token
        start_len = len(start_token)
        end_len = len(end_token)
        scratch_start_len = len(_SCRATCH_PAD_START)
        scratch_end_len = len(_SCRATCH_PAD_END)
        position = 0
        n = len(current_text)

        while position < n:
            sp_idx = current_text.find(_SCRATCH_PAD_START, position)
            tc_idx = current_text.find(start_token, position)
            if sp_idx != -1 and (tc_idx == -1 or sp_idx < tc_idx):
                sp_end = current_text.find(_SCRATCH_PAD_END, sp_idx + scratch_start_len)
                if sp_end == -1:
                    return False
                position = sp_end + scratch_end_len
                continue
            if tc_idx == -1:
                return self._has_trailing_marker_prefix(current_text[position:], (start_token,))
            body_start = tc_idx + start_len
            call_end = self._find_tool_call_end(current_text, body_start, end_token)
            if call_end is None:
                return True
            position = call_end + end_len
        return False

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls, ignoring examples inside reasoning scratch pads.

        Args:
            model_output: Complete model output that may contain tool calls.
            request: Original chat completion request, retained for API parity.

        Returns:
            Parsed calls and leading content, or the original content on failure.
        """
        try:
            regions = self._tool_call_regions(model_output)
            if not regions:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            tool_calls: list[ToolCall] = []
            content_parts: list[str] = []
            content_start = 0
            for start, body, end in regions:
                content_parts.append(model_output[content_start:start])
                content_start = end if end is not None else len(model_output)
                try:
                    call = json.loads(body)
                except (json.JSONDecodeError, ValueError, RecursionError):
                    continue
                if not isinstance(call, dict):
                    continue
                name = call.get("name")
                if not isinstance(name, str) or not name:
                    continue
                raw_args = call.get("arguments", {})
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=name,
                            arguments=json.dumps(raw_args, ensure_ascii=False),
                        ),
                    )
                )

            if not tool_calls:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            content_parts.append(model_output[content_start:])
            content = "".join(content_parts)
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content or None,
            )
        except Exception:
            logger.exception("Failed to extract Hermes tool calls from model output")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Emit stable JSON argument prefixes and flush each call before advancing.

        Cumulative text is scanned directly so arbitrary token/chunk boundaries
        and chunks containing both closing and opening tags follow the same path.
        Names are withheld until their JSON string is complete.

        Args:
            previous_text: Cumulative text before this update.
            current_text: Cumulative text including this update.
            delta_text: Newly decoded text, unused because cumulative text is authoritative.
            previous_token_ids: Previous token IDs, retained for API parity.
            current_token_ids: Current token IDs, retained for API parity.
            delta_token_ids: New token IDs, retained for API parity.
            request: Original chat completion request, retained for API parity.

        Returns:
            New content and/or tool deltas, or None while waiting for more input.
        """
        if (
            not previous_text
            or not self._streaming_initialized
            or self._scan_pos > len(current_text)
            or self._emitted_source_pos > len(current_text)
        ):
            self.current_tool_id = -1
            self.current_tool_name_sent = False
            self.prev_tool_call_arr = []
            self.streamed_args_for_tool = []
            # DelegatingParser may pass ordinary text through before engaging
            # this parser at a marker hint. That previous text is already
            # visible, unlike suffixes withheld on our own subsequent calls.
            self._streamed_content_length = len(previous_text)
            self._pending_content_suffix = ""
            self.buffered_delta_text = ""
            self._streaming_initialized = True
            self._region_to_emitted_index = {}
            self._scan_pos = 0
            self._in_scratch_pad = False
            self._emitted_source_pos = len(previous_text)
            self._has_open_region = False

        try:
            start_token = self.tool_call_start_token
            end_token = self.tool_call_end_token
            start_len = len(start_token)
            end_len = len(end_token)
            scratch_start_len = len(_SCRATCH_PAD_START)
            scratch_end_len = len(_SCRATCH_PAD_END)

            pos = self._scan_pos
            in_scratch = self._in_scratch_pad
            n = len(current_text)
            visible_spans: list[tuple[int, int]] = []
            regions: list[tuple[int, str, int | None]] = []
            pending_suffix = ""
            has_open_region = False

            while pos < n:
                if in_scratch:
                    sp_end = current_text.find(_SCRATCH_PAD_END, pos)
                    if sp_end == -1:
                        visible_spans.append((pos, n))
                        hold_sp = _trailing_prefix_len(current_text, pos, _SCRATCH_PAD_END)
                        pos = n - hold_sp
                        break
                    close_end = sp_end + scratch_end_len
                    visible_spans.append((pos, close_end))
                    pos = close_end
                    in_scratch = False
                    continue

                sp_idx = current_text.find(_SCRATCH_PAD_START, pos)
                tc_idx = current_text.find(start_token, pos)

                if sp_idx != -1 and (tc_idx == -1 or sp_idx < tc_idx):
                    sp_body = sp_idx + scratch_start_len
                    visible_spans.append((pos, sp_body))
                    pos = sp_body
                    in_scratch = True
                    continue

                if tc_idx == -1:
                    hold_tc = _trailing_prefix_len(current_text, pos, start_token)
                    hold_sp = _trailing_prefix_len(current_text, pos, _SCRATCH_PAD_START)
                    visible_end = n - hold_tc
                    if visible_end > pos:
                        visible_spans.append((pos, visible_end))
                    if hold_tc > 0:
                        pending_suffix = current_text[visible_end:]
                    pos = n - max(hold_tc, hold_sp)
                    break

                if tc_idx > pos:
                    visible_spans.append((pos, tc_idx))
                body_start = tc_idx + start_len
                call_end = self._find_tool_call_end(current_text, body_start, end_token)
                if call_end is None:
                    regions.append((tc_idx, current_text[body_start:], None))
                    has_open_region = True
                    pos = tc_idx
                    break
                region_end = call_end + end_len
                regions.append((tc_idx, current_text[body_start:call_end], region_end))
                pos = region_end

            deltas: list[DeltaToolCall] = []
            for start, body, end in regions:
                assigned_index = self._region_to_emitted_index.get(start)
                announced = assigned_index is not None and bool(
                    assigned_index < len(self.prev_tool_call_arr) and self.prev_tool_call_arr[assigned_index].get("name")
                )
                try:
                    if end is not None:
                        call = json.loads(body)
                    else:
                        # A closing delimiter can itself be split across updates.
                        for length in range(len(end_token) - 1, 0, -1):
                            if body.endswith(end_token[:length]):
                                body = body[:-length]
                                break
                        flags = Allow.ALL if announced else Allow.ALL & ~Allow.STR
                        call = partial_json_parser.loads(body, flags)
                except (
                    partial_json_parser.core.exceptions.MalformedJSON,
                    ValueError,
                    AssertionError,
                    IndexError,
                    RecursionError,
                ):
                    continue
                if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not call.get("name"):
                    continue

                if assigned_index is None:
                    assigned_index = len(self._region_to_emitted_index)
                    self._region_to_emitted_index[start] = assigned_index

                index = assigned_index
                while len(self.prev_tool_call_arr) <= index:
                    self.prev_tool_call_arr.append({})
                    self.streamed_args_for_tool.append("")
                self.current_tool_id = index
                self.current_tool_name_sent = announced

                prefix = self.streamed_args_for_tool[index]
                arguments = ""
                has_arguments = "arguments" in call
                if has_arguments or end is not None:
                    raw_args = call.get("arguments", {})
                    # Always use exactly the batch JSON serialization, including
                    # string-valued arguments. Raw strings are not JSON prefixes.
                    serialized = json.dumps(raw_args, ensure_ascii=False)
                    if end is not None:
                        target = serialized
                    elif isinstance(raw_args, (int, float)) and not isinstance(raw_args, bool):
                        # Scalar numbers have no leading container delimiter for
                        # stable_json_stream_prefix to recognize. Withhold the
                        # entire value: e.g. 1 can still become 0.01 or 1e+20.
                        target = ""
                    else:
                        target = stable_json_stream_prefix(serialized)
                    if target.startswith(prefix):
                        arguments = target[len(prefix) :]
                        self.streamed_args_for_tool[index] = target
                    else:
                        # Isolate non-prefix updates (e.g. duplicate JSON keys) to
                        # this tool call instead of aborting subsequent calls/content.
                        logger.warning(
                            "Hermes argument update for tool index %d is not a prefix extension; skipping diff.",
                            index,
                        )
                if not announced or arguments:
                    deltas.append(
                        DeltaToolCall(
                            index=index,
                            type="function" if not announced else None,
                            id=f"chatcmpl-tool-{uuid4()}" if not announced else None,
                            function=DeltaFunctionCall(
                                name=call["name"] if not announced else None,
                                arguments=arguments or None,
                            ),
                        )
                    )
                self.prev_tool_call_arr[index] = call
                self.current_tool_name_sent = True

            emitted_pos = self._emitted_source_pos
            content_parts: list[str] = []
            for seg_start, seg_end in visible_spans:
                if seg_end > emitted_pos:
                    content_parts.append(current_text[max(seg_start, emitted_pos) : seg_end])
                    emitted_pos = seg_end

            self._emitted_source_pos = emitted_pos
            self._scan_pos = pos
            self._in_scratch_pad = in_scratch
            self._has_open_region = has_open_region
            self._pending_content_suffix = pending_suffix

            content_delta = "".join(content_parts)
            self._streamed_content_length += len(content_delta)
            if deltas or content_delta or self._pending_content_suffix:
                # An explicit empty content delta tells orchestrators that we
                # intentionally withheld a suffix; None may trigger raw-text
                # pass-through and then duplicate it when it is disambiguated.
                return DeltaMessage(tool_calls=deltas, content=content_delta)
            return None
        except Exception:
            logger.exception("Error trying to handle Hermes streaming tool call")
            return None

    def flush_streaming_content(self) -> str:
        """Release a possible opening-marker prefix at the end of a stream.

        The streaming path withholds an ambiguous trailing ``<tool_call>``
        prefix until more input disambiguates it. At end of stream it is
        ordinary content. ``ToolCallingMixin`` calls this hook on its terminal
        update; direct parser consumers should call it after their final update.
        In-progress tool JSON and completed tool markup are never returned.

        Returns:
            The withheld content suffix, or an empty string when there is none.
            Repeated calls without further input return an empty string.
        """
        suffix, self._pending_content_suffix = self._pending_content_suffix, ""
        self.buffered_delta_text = ""
        self._streamed_content_length += len(suffix)
        self._emitted_source_pos += len(suffix)
        return suffix
