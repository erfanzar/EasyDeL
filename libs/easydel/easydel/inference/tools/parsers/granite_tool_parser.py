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

"""Granite tool parser module for parsing tool calls from Granite 3.0 model outputs.

This module provides the GraniteToolParser class which handles the JSON array
format used by Granite 3.0 models. The format uses a simple JSON array
structure with optional bot token markers.

Example tool call format:
    <|tool_call|>[{"name": "get_weather", "arguments": {"location": "NYC"}}]

Or without token marker:
    <tool_call>[{"name": "get_weather", "arguments": {"location": "NYC"}}]
"""

from __future__ import annotations

import json
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
from ..utils import consume_space, is_complete_json, partial_json_loads, stable_json_stream_prefix

logger = get_logger(__name__)


def _trailing_prefix_len_any(text: str, markers: Sequence[str]) -> int:
    """Return length of the longest proper prefix of any marker matching the end of *text*."""
    best = 0
    for marker in markers:
        max_len = min(len(text), len(marker) - 1)
        for size in range(max_len, best, -1):
            if text.endswith(marker[:size]):
                best = size
                break
    return best


@ToolParserManager.register_module("granite")  # pyright: ignore[reportUntypedClassDecorator]
class GraniteToolParser(ToolParser):
    """Tool call parser for Granite 3.0 models.

    Intended for use with the examples/tool_chat_template_granite.jinja
    template. Handles JSON array format with optional bot tokens.

    Features:
    - Supports <|tool_call|> and <tool_call> token markers
    - Parses JSON array of tool calls
    - Handles partial JSON for streaming using partial_json_parser
    - Tracks argument completion state for proper streaming

    Format:
        <|tool_call|>[{"name": "func", "arguments": {...}}, ...]

    Used when --enable-auto-tool-choice --tool-call-parser granite
    are all set.

    Attributes:
        bot_token (str): Primary bot token marker '<|tool_call|>'.
        bot_string (str): Alternative token marker '<tool_call>'.
        current_tool_name_sent (bool): Whether tool name was sent (inherited).
        prev_tool_call_arr (list): Previous tool calls for streaming (inherited).
        current_tool_id (int): Current tool index (inherited).
        streamed_args_for_tool (list[str]): Streamed args per tool (inherited).

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("granite-3.0")
        >>> parser = GraniteToolParser(tokenizer)
        >>> result = parser.extract_tool_calls(model_output, request)
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the Granite tool parser.

        Sets up the bot token markers for identifying tool call regions.

        Args:
            tokenizer: The tokenizer associated with the Granite model.
        """
        super().__init__(tokenizer)
        self.bot_token = "<|tool_call|>"
        self.bot_string = "<tool_call>"
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []
        self._tool_name_sent: list[bool] = []
        self._tool_call_ids: list[str] = []
        self._streaming_initialized: bool = False
        self._cursor: int = 0
        self._accumulated_text: str = ""
        self._buffering_tool_call: bool = False
        self._emitted_tool_call: bool = False
        self._bare_decided_plain: bool = False
        self._in_tool_mode: bool = False
        self._has_marker: bool = False
        self._marker_start: int = -1
        self._payload_start: int = -1
        self._tool_block_done: bool = False
        self._ws_scan_pos: int = 0
        self._post_tool_ws_pos: int = 0

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Return True while withholding a potential bare array or Granite marker block."""
        del delta_text
        if self._streaming_initialized:
            return self._buffering_tool_call
        stripped = current_text.lstrip()
        if stripped.startswith("["):
            return not is_complete_json(stripped)
        if self.bot_token in current_text or self.bot_string in current_text:
            return True
        return _trailing_prefix_len_any(current_text, (self.bot_token, self.bot_string)) > 0

    def flush_streaming_content(self) -> str | None:
        """Flush any withheld prefix that turned out to be ordinary text at EOS."""
        if self._in_tool_mode and (self._emitted_tool_call or self._has_marker):
            self._accumulated_text = ""
            return None
        flushed: str | None = None
        if self._cursor < len(self._accumulated_text):
            flushed = self._accumulated_text[self._cursor :] or None
            self._cursor = len(self._accumulated_text)
            self._buffering_tool_call = False
            self._in_tool_mode = False
        self._accumulated_text = ""
        return flushed

    def get_tool_machinery_hints(self) -> Sequence[str]:
        """Include the bare-JSON array prefix: this parser accepts markerless calls.

        Both extraction paths accept a plain ``[...]`` array with neither
        ``<|tool_call|>`` nor ``<tool_call>`` present, so the delegating
        parser's machinery gate must engage on a bare ``[``.
        """
        hints = list(super().get_tool_machinery_hints())
        if "[" not in hints:
            hints.append("[")
        return tuple(hints)

    @staticmethod
    def _build_tool_calls(raw_function_calls: object) -> list[ToolCall] | None:
        """Validate a decoded JSON list of Granite tool call objects."""
        if not isinstance(raw_function_calls, list) or not raw_function_calls:
            return None
        tool_calls: list[ToolCall] = []
        for function_call in raw_function_calls:
            if (
                not isinstance(function_call, dict)
                or not isinstance(function_call.get("name"), str)
                or not function_call["name"]
                or "arguments" not in function_call
            ):
                return None
            raw_args = function_call["arguments"]
            tool_calls.append(
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=function_call["name"],
                        arguments=raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False),
                    ),
                )
            )
        return tool_calls

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output.

        Parses a JSON array of tool call objects from the model output,
        supporting optional preamble text before ``<|tool_call|>`` /
        ``<tool_call>`` markers as well as bare JSON arrays.
        """
        del request
        decoder = json.JSONDecoder()
        markers = (self.bot_token, self.bot_string)

        # Path 1: Explicit <|tool_call|> or <tool_call> marker anywhere in output.
        pos = 0
        while pos < len(model_output):
            candidates = [(idx, m) for m in markers if (idx := model_output.find(m, pos)) != -1]
            if not candidates:
                break
            m_idx, marker = min(candidates, key=lambda pair: pair[0])
            arr_start = consume_space(m_idx + len(marker), model_output)
            if marker == self.bot_token and model_output.startswith(self.bot_string, arr_start):
                arr_start = consume_space(arr_start + len(self.bot_string), model_output)
            if arr_start < len(model_output) and model_output[arr_start] == "[":
                try:
                    raw_function_calls, end_pos = decoder.raw_decode(model_output, arr_start)
                except (json.JSONDecodeError, ValueError, RecursionError):
                    raw_function_calls = None
                    end_pos = arr_start
                tool_calls = self._build_tool_calls(raw_function_calls)
                if tool_calls:
                    preamble = model_output[:m_idx]
                    trailing = model_output[end_pos:]
                    combined = preamble + (trailing if trailing.strip() else "")
                    content = combined if combined.strip() else None
                    return ExtractedToolCallInformation(
                        tools_called=True,
                        tool_calls=tool_calls,
                        content=content,
                    )
            pos = m_idx + len(marker)

        # Path 2: Bare JSON array covering the full output (ignoring outer whitespace).
        stripped = model_output.strip()
        if stripped.startswith("["):
            try:
                raw_function_calls = json.loads(stripped)
            except Exception as e:
                logger.debug("Ignoring non-JSON bare bracket content in Granite parser: %s", e)
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
            tool_calls = self._build_tool_calls(raw_function_calls)
            if tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=None,
                )

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
        """Extract tool calls from streaming model output."""
        del delta_text, previous_token_ids, current_token_ids, delta_token_ids, request
        if not previous_text:
            self._reset_streaming_state()
            self._streaming_initialized = True
        elif not self._streaming_initialized:
            self._reset_streaming_state()
            self._cursor = len(previous_text)
            if previous_text.strip():
                self._bare_decided_plain = True
            else:
                self._ws_scan_pos = len(previous_text)
            self._streaming_initialized = True

        self._accumulated_text = current_text
        if not current_text:
            return None

        content_parts: list[str] = []
        deltas: list[DeltaToolCall] = []
        markers = (self.bot_token, self.bot_string)

        # Step 1: Bare leading '[' check before settling into prose/marker scanning.
        if not self._bare_decided_plain and not self._in_tool_mode and not self._tool_block_done:
            ws_end = consume_space(self._ws_scan_pos, current_text)
            self._ws_scan_pos = ws_end
            if ws_end >= len(current_text):
                self._buffering_tool_call = True
                return None
            if current_text[ws_end] == "[":
                self._in_tool_mode = True
                self._has_marker = False
                self._marker_start = 0
                self._payload_start = ws_end
                self._buffering_tool_call = True
            else:
                self._bare_decided_plain = True

        while True:
            # Post-tool trailing content after a completed tool array.
            if self._tool_block_done:
                ws_end = consume_space(self._post_tool_ws_pos, current_text)
                self._post_tool_ws_pos = ws_end
                if ws_end < len(current_text):
                    content_parts.append(current_text[self._cursor :])
                    self._cursor = len(current_text)
                    self._post_tool_ws_pos = self._cursor
                self._buffering_tool_call = False
                break

            # Step 2: Prose scanning for explicit <|tool_call|> / <tool_call> markers.
            if not self._in_tool_mode:
                candidates = [(idx, m) for m in markers if (idx := current_text.find(m, self._cursor)) != -1]
                if candidates:
                    m_idx, _marker = min(candidates, key=lambda pair: pair[0])
                    if m_idx > self._cursor:
                        content_parts.append(current_text[self._cursor : m_idx])
                    self._cursor = m_idx
                    self._in_tool_mode = True
                    self._has_marker = True
                    self._marker_start = m_idx
                    self._payload_start = -1
                    self._buffering_tool_call = True
                    continue

                rem = current_text[self._cursor :]
                hold = _trailing_prefix_len_any(rem, markers)
                safe_end = len(current_text) - hold
                if safe_end > self._cursor:
                    content_parts.append(current_text[self._cursor : safe_end])
                    self._cursor = safe_end
                self._buffering_tool_call = hold > 0
                break

            # Step 3: Inside marker or bare array tool mode.
            if self._payload_start == -1:
                marker_len = (
                    len(self.bot_token)
                    if current_text.startswith(self.bot_token, self._marker_start)
                    else len(self.bot_string)
                )
                after = self._marker_start + marker_len
                idx = consume_space(after, current_text)
                if marker_len == len(self.bot_token):
                    if current_text.startswith(self.bot_string, idx):
                        idx = consume_space(idx + len(self.bot_string), current_text)
                    elif idx < len(current_text) and self.bot_string.startswith(current_text[idx:]):
                        self._buffering_tool_call = True
                        break
                if idx >= len(current_text):
                    self._buffering_tool_call = True
                    break
                if current_text[idx] != "[":
                    # Marker not followed by '[' -> release marker as prose and continue scanning.
                    content_parts.append(current_text[self._cursor : after])
                    self._cursor = after
                    self._in_tool_mode = False
                    self._has_marker = False
                    self._buffering_tool_call = False
                    continue
                self._payload_start = idx

            payload = current_text[self._payload_start :]
            try:
                try:
                    tool_calls_all, end_idx = partial_json_loads(payload, Allow.ALL)
                except (
                    partial_json_parser.core.exceptions.MalformedJSON,
                    ValueError,
                    AssertionError,
                    RecursionError,
                ):
                    if not self._emitted_tool_call:
                        if not self._has_marker:
                            self._bare_decided_plain = True
                            emit_to = self._payload_start + 1
                        else:
                            emit_to = self._payload_start + 1
                        content_parts.append(current_text[self._cursor : emit_to])
                        self._cursor = emit_to
                        self._in_tool_mode = False
                        self._has_marker = False
                        self._buffering_tool_call = False
                        continue
                    self._buffering_tool_call = True
                    break

                if type(tool_calls_all) is not list or any(not isinstance(item, dict) for item in tool_calls_all):
                    if not self._emitted_tool_call:
                        if not self._has_marker:
                            self._bare_decided_plain = True
                        emit_to = self._payload_start + 1
                        content_parts.append(current_text[self._cursor : emit_to])
                        self._cursor = emit_to
                        self._in_tool_mode = False
                        self._has_marker = False
                        self._buffering_tool_call = False
                        continue
                    self._buffering_tool_call = True
                    break

                array_complete = is_complete_json(payload[:end_idx])
                if not tool_calls_all:
                    if array_complete:
                        if not self._has_marker:
                            self._bare_decided_plain = True
                        end_abs = self._payload_start + end_idx
                        content_parts.append(current_text[self._cursor : end_abs])
                        self._cursor = end_abs
                        self._in_tool_mode = False
                        self._has_marker = False
                        self._buffering_tool_call = False
                        continue
                    self._buffering_tool_call = True
                    break

                is_complete = [True] * len(tool_calls_all)
                if not array_complete:
                    is_complete[-1] = False

                if not self._emitted_tool_call:
                    invalid_complete = any(
                        (is_complete[i] or i < len(tool_calls_all) - 1)
                        and not (
                            isinstance(item.get("name"), str)
                            and item.get("name")
                            and "arguments" in item
                        )
                        for i, item in enumerate(tool_calls_all)
                    )
                    if invalid_complete:
                        if not self._has_marker:
                            self._bare_decided_plain = True
                        end_abs = self._payload_start + end_idx if array_complete else self._payload_start + 1
                        content_parts.append(current_text[self._cursor : end_abs])
                        self._cursor = end_abs
                        self._in_tool_mode = False
                        self._has_marker = False
                        self._buffering_tool_call = False
                        continue

                try:
                    tool_calls_safe_names, _ = partial_json_loads(payload, Allow.ALL & ~Allow.STR)
                    if not isinstance(tool_calls_safe_names, list):
                        tool_calls_safe_names = []
                except (
                    partial_json_parser.core.exceptions.MalformedJSON,
                    ValueError,
                    AssertionError,
                    RecursionError,
                ):
                    tool_calls_safe_names = []

                for idx, call in enumerate(tool_calls_all):
                    while len(self.streamed_args_for_tool) <= idx:
                        self.streamed_args_for_tool.append("")
                    while len(self._tool_name_sent) <= idx:
                        self._tool_name_sent.append(False)
                    while len(self._tool_call_ids) <= idx:
                        self._tool_call_ids.append("")

                    if self._has_marker and idx > self.current_tool_id:
                        self.current_tool_id = idx
                        self.current_tool_name_sent = self._tool_name_sent[idx]

                    call_complete = is_complete[idx] or idx < len(tool_calls_all) - 1
                    if self._tool_name_sent[idx] or call_complete:
                        func_name = call.get("name")
                    else:
                        safe_call = (
                            tool_calls_safe_names[idx]
                            if idx < len(tool_calls_safe_names) and isinstance(tool_calls_safe_names[idx], dict)
                            else {}
                        )
                        func_name = safe_call.get("name")

                    if not isinstance(func_name, str) or not func_name or "arguments" not in call:
                        break

                    if idx > self.current_tool_id:
                        self.current_tool_id = idx
                        self.current_tool_name_sent = self._tool_name_sent[idx]

                    first = not self._tool_name_sent[idx]
                    if first:
                        self._tool_call_ids[idx] = f"chatcmpl-tool-{uuid4()}"
                        self._tool_name_sent[idx] = True
                    if idx >= self.current_tool_id:
                        self.current_tool_id = idx
                        self.current_tool_name_sent = True

                    cur_arguments = call.get("arguments")
                    argument_diff = ""
                    if cur_arguments is not None:
                        cur_args_json = (
                            cur_arguments
                            if isinstance(cur_arguments, str)
                            else json.dumps(cur_arguments, ensure_ascii=False)
                        )
                        safe_json = cur_args_json if call_complete else stable_json_stream_prefix(cur_args_json)
                        sent = self.streamed_args_for_tool[idx]
                        if safe_json.startswith(sent):
                            argument_diff = safe_json[len(sent) :]
                            if argument_diff:
                                self.streamed_args_for_tool[idx] = sent + argument_diff

                    if first or argument_diff:
                        self._emitted_tool_call = True
                        deltas.append(
                            DeltaToolCall(
                                index=idx,
                                type="function" if first else None,
                                id=self._tool_call_ids[idx] if first else None,
                                function=DeltaFunctionCall(
                                    name=func_name if first else None,
                                    arguments=argument_diff if argument_diff else None,
                                ).model_dump(exclude_none=True),
                            )
                        )

                self.prev_tool_call_arr = tool_calls_all
                if array_complete and self._emitted_tool_call:
                    self._tool_block_done = True
                    self._cursor = self._payload_start + end_idx
                    self._post_tool_ws_pos = self._cursor
                    self._buffering_tool_call = False
                    continue
                self._buffering_tool_call = not array_complete
                break

            except Exception as e:
                logger.error("Error trying to handle streaming tool call: %s", e)
                if not self._emitted_tool_call:
                    if not self._has_marker:
                        self._bare_decided_plain = True
                    emit_to = min(len(current_text), self._payload_start + 1)
                    if emit_to > self._cursor:
                        content_parts.append(current_text[self._cursor : emit_to])
                        self._cursor = emit_to
                    self._in_tool_mode = False
                    self._has_marker = False
                    self._buffering_tool_call = False
                    continue
                self._buffering_tool_call = True
                break

        content = "".join(content_parts)
        if content or deltas:
            return DeltaMessage(content=content or None, tool_calls=deltas or None)
        return None
