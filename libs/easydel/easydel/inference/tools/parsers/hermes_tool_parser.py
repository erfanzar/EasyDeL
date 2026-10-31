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
            else:
                self.buffered_delta_text = self.buffered_delta_text + delta_text
                return ""
        else:
            if self.buffered_delta_text:
                buffered_text = self.buffered_delta_text
                self.buffered_delta_text = ""
                return buffered_text + delta_text
            else:
                return delta_text

    def _tool_call_regions(self, text: str) -> list[tuple[int, str, int | None]]:
        """Locate calls without interpreting reasoning or quoted markup as protocol.

        Args:
            text: Complete or cumulative model output.

        Returns:
            Tuples of opening offset, unmodified JSON body and end offset.
            The end offset is None for a still-open call.
        """
        regions = []
        position = 0
        while position < len(text):
            # Only recognize scratch pads outside tool JSON. Never mask or edit
            # argument bytes, even if a string contains a complete reasoning block.
            scratch_pad = self.scratch_pad_regex.match(text, position)
            if scratch_pad:
                position = scratch_pad.end()
                continue
            if not text.startswith(self.tool_call_start_token, position):
                position += 1
                continue
            start = position
            position += len(self.tool_call_start_token)
            body_start = position
            in_string = False
            escaped = False
            while position < len(text):
                char = text[position]
                if not in_string and text.startswith(self.tool_call_end_token, position):
                    body = text[body_start:position]
                    position += len(self.tool_call_end_token)
                    regions.append((start, body, position))
                    break
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                elif char == '"':
                    in_string = True
                position += 1
            else:
                regions.append((start, text[body_start:], None))
        return regions

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
            raw_calls = [json.loads(body) for _, body, _ in regions]
            tool_calls = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=call["name"],
                        arguments=json.dumps(call["arguments"], ensure_ascii=False),
                    ),
                )
                for call in raw_calls
            ]
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=model_output[: regions[0][0]] or None,
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
        if not previous_text or not self._streaming_initialized:
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

        try:
            regions = self._tool_call_regions(current_text)
            deltas = []
            content_parts = []
            content_start = 0
            for index, (start, body, end) in enumerate(regions):
                content_parts.append(current_text[content_start:start])
                content_start = end if end is not None else len(current_text)
                while len(self.prev_tool_call_arr) <= index:
                    self.prev_tool_call_arr.append({})
                    self.streamed_args_for_tool.append("")
                announced = bool(self.prev_tool_call_arr[index].get("name"))
                self.current_tool_id = index
                self.current_tool_name_sent = announced
                try:
                    if end is not None:
                        call = json.loads(body)
                    else:
                        # A closing delimiter can itself be split across updates.
                        for length in range(len(self.tool_call_end_token) - 1, 0, -1):
                            if body.endswith(self.tool_call_end_token[:length]):
                                body = body[:-length]
                                break
                        flags = Allow.ALL if announced else Allow.ALL & ~Allow.STR
                        call = partial_json_parser.loads(body, flags)
                except (partial_json_parser.core.exceptions.MalformedJSON, json.JSONDecodeError):
                    continue
                if not isinstance(call, dict) or not call.get("name"):
                    continue

                prefix = self.streamed_args_for_tool[index]
                arguments = ""
                if "arguments" in call:
                    # Always use exactly the batch JSON serialization, including
                    # string-valued arguments. Raw strings are not JSON prefixes.
                    serialized = json.dumps(call["arguments"], ensure_ascii=False)
                    if end is not None:
                        target = serialized
                    elif isinstance(call["arguments"], (int, float)) and not isinstance(call["arguments"], bool):
                        # Scalar numbers have no leading container delimiter for
                        # stable_json_stream_prefix to recognize. Withhold the
                        # entire value: e.g. 1 can still become 0.01 or 1e+20.
                        target = ""
                    else:
                        target = stable_json_stream_prefix(serialized)
                    if not target.startswith(prefix):
                        raise ValueError("Hermes argument update is not a prefix extension")
                    arguments = target[len(prefix) :]
                    self.streamed_args_for_tool[index] = target
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

            # Only the trailing source segment can still become an opening tag.
            # A prefix before a recognized call is already unambiguous content.
            tail = current_text[content_start:]
            self._pending_content_suffix = ""
            for length in range(len(self.tool_call_start_token) - 1, 0, -1):
                if tail.endswith(self.tool_call_start_token[:length]):
                    self._pending_content_suffix = tail[-length:]
                    tail = tail[:-length]
                    break
            content_parts.append(tail)
            content = "".join(content_parts)
            content_delta = content[self._streamed_content_length :]
            self._streamed_content_length = len(content)
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
        self._streamed_content_length += len(suffix)
        return suffix
