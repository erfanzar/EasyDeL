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

"""InternLM2 tool parser for action/plugin style tool calls.

This module provides a tool call parser designed for InternLM2 models that
generate tool calls using special action tokens. The parser handles the
<|action_start|><|plugin|> and <|action_end|> delimiters and extracts
JSON-formatted function calls.

Example format:
    <|action_start|><|plugin|>{"name": "func", "parameters": {...}}<|action_end|>

Features:
    - Position-based streaming parser
    - Supports both 'parameters' and 'arguments' fields
    - Adjusts request settings for special tokens
    - Handles partial JSON with incremental diff extraction
    - Maintains cursor position for stream tracking
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
from ..utils import consume_space, partial_json_loads, stable_json_stream_prefix

logger = get_logger(__name__)


def _strip_trailing_partial_marker(text: str, marker: str) -> str:
    """Withhold any trailing non-empty proper prefix of *marker* from *text*."""
    max_len = min(len(marker) - 1, len(text))
    for k in range(max_len, 0, -1):
        if text.endswith(marker[:k]):
            return text[:-k]
    return text


def _has_open_marker_or_trailing_prefix(text: str, start_token: str, end_token: str) -> bool:
    """Return True if *text* has an unclosed block or ends with a partial start marker."""
    pos = 0
    n = len(text)
    start_len = len(start_token)
    end_len = len(end_token)
    while pos < n:
        start_idx = text.find(start_token, pos)
        if start_idx == -1:
            return ToolParser._has_trailing_marker_prefix(text[pos:], (start_token,))
        end_idx = text.find(end_token, start_idx + start_len)
        if end_idx == -1:
            return True
        pos = end_idx + end_len
    return False


@ToolParserManager.register_module(["internlm"])  # pyright: ignore[reportUntypedClassDecorator]
class Internlm2ToolParser(ToolParser):
    """Tool parser for InternLM2 models.

    Handles action/plugin calls with special tokens:
    <|action_start|><|plugin|>{...}<|action_end|>

    Features:
        - Position-based streaming parser
        - Supports both 'parameters' and 'arguments' fields
        - Adjusts request settings for special tokens
        - Handles partial JSON with incremental diff extraction

    The parser maintains a cursor position to track progress through
    the output stream and properly handle action boundaries.

    Attributes:
        position: Current cursor position in the text stream.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the Internlm2ToolParser.

        Args:
            tokenizer: The tokenizer associated with the InternLM2 model.
                Used for token-level operations during streaming.
        """
        super().__init__(tokenizer)
        self.position = 0
        self.tool_call_start_token = "<|action_start|><|plugin|>"
        self.tool_call_end_token = "<|action_end|>"
        self._tool_name_sent: list[bool] = []
        self._tool_call_ids: list[str] = []
        self._emitted_content_pos: int | None = None
        self._pending_tail_start: int | None = None
        self._last_current_text: str = ""
        self._streaming_initialized: bool = False
        self._in_open_block: bool = False
        self._scan_pos: int = 0
        self._closed_emitted_idx: int = 0
        self._closed_tracked_calls: list[dict] = []

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Report whether an action block or partial start marker is buffered."""
        del delta_text
        if self._streaming_initialized:
            return self._in_open_block or self._pending_tail_start is not None
        return _has_open_marker_or_trailing_prefix(
            current_text,
            self.tool_call_start_token,
            self.tool_call_end_token,
        )

    def _reset_streaming_state(self, emitted_pos: int = 0) -> None:
        self.position = emitted_pos
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []
        self._tool_name_sent = []
        self._tool_call_ids = []
        self._emitted_content_pos = emitted_pos
        self._pending_tail_start = None
        self._last_current_text = ""
        self._streaming_initialized = False
        self._in_open_block = False
        self._scan_pos = 0
        self._closed_emitted_idx = 0
        self._closed_tracked_calls = []

    def flush_streaming_content(self) -> str | None:
        """Flush any trailing partial ``<|action_start|><|plugin|>`` prefix at EOS."""
        flushed: str | None = None
        if self._pending_tail_start is not None and self._pending_tail_start < len(self._last_current_text):
            flushed = self._last_current_text[self._pending_tail_start :] or None
            self._emitted_content_pos = len(self._last_current_text)
            self.position = self._emitted_content_pos
            self._pending_tail_start = None
        self._last_current_text = ""
        return flushed

    @staticmethod
    def _allowed_tool_names(request: ChatCompletionRequest | None) -> set[str] | None:
        """Return declared tool names when ``request.tools`` is provided, or empty set if none."""
        tools = getattr(request, "tools", None) if request is not None else None
        if not tools:
            return set()
        return {tool.function.name for tool in tools if hasattr(tool, "function") and hasattr(tool.function, "name")}

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Adjust the request settings for InternLM2 tool parsing.

        Modifies the request to ensure special tokens are not skipped
        when tools are enabled, as the action tokens are required for
        proper tool call detection.

        Args:
            request: The chat completion request to adjust.

        Returns:
            The modified request with skip_special_tokens set to False
            when tools are enabled and tool_choice is not 'none'.
        """
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def get_arguments(self, obj: dict) -> dict | None:
        """Extract arguments from a tool call object.

        InternLM2 models may use either 'parameters' or 'arguments' field
        for function arguments. This method handles both cases.

        Args:
            obj: Dictionary containing tool call data with either
                'parameters' or 'arguments' field.

        Returns:
            The arguments dictionary if found, None otherwise.
        """
        if "parameters" in obj:
            return obj.get("parameters")
        elif "arguments" in obj:
            return obj.get("arguments")
        return None

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
        """Extract tool calls incrementally during streaming.

        Processes streaming output using position-based tracking to identify
        action boundaries and extract JSON tool call data incrementally.

        Args:
            previous_text: Text accumulated before this delta.
            current_text: Complete text including the current delta.
            delta_text: The new text added in this streaming chunk.
            previous_token_ids: Token IDs before this delta.
            current_token_ids: All token IDs including current delta.
            delta_token_ids: Token IDs in the current delta.
            request: The chat completion request for context.

        Returns:
            DeltaMessage containing either:
                - content: Regular text content if no action detected
                - tool_calls: Tool call deltas if action is being parsed
                - Empty content ("") if tool call is complete
            Returns None if more tokens are needed.
        """
        if not previous_text:
            self._reset_streaming_state(0)
        elif self._emitted_content_pos is None:
            self._reset_streaming_state(len(previous_text))
        elif self._scan_pos > len(current_text):
            self._reset_streaming_state(0)

        self._streaming_initialized = True
        self._last_current_text = current_text
        self._pending_tail_start = None
        self._in_open_block = False

        allowed_names = self._allowed_tool_names(request)

        try:
            dec = json.JSONDecoder()
            pos = self._scan_pos
            next_scan_pos = pos
            visible_parts: list[tuple[int, int]] = []
            # list of (call_dict, safe_name_dict, call_complete, block_closed)
            parsed_actions: list[tuple[dict, dict, bool, bool]] = []

            while pos < len(current_text):
                start = current_text.find(self.tool_call_start_token, pos)
                if start == -1:
                    tail = current_text[pos:]
                    safe_tail = _strip_trailing_partial_marker(tail, self.tool_call_start_token)
                    safe_end = pos + len(safe_tail)
                    if safe_tail:
                        visible_parts.append((pos, safe_end))
                    if len(safe_tail) < len(tail):
                        self._pending_tail_start = safe_end
                    next_scan_pos = safe_end
                    break

                if start > pos:
                    visible_parts.append((pos, start))

                payload_start = consume_space(start + len(self.tool_call_start_token), current_text)
                if payload_start >= len(current_text):
                    self._in_open_block = True
                    next_scan_pos = start
                    break

                # String-aware JSON decode first so literal <|action_end|> inside JSON strings
                # never truncates the payload.
                decoded: object = None
                end_pos = -1
                try:
                    decoded, end_pos = dec.raw_decode(current_text, payload_start)
                except (json.JSONDecodeError, ValueError, RecursionError):
                    decoded = None

                if decoded is not None:
                    after_json = consume_space(end_pos, current_text)
                    has_end_marker = current_text.startswith(self.tool_call_end_token, after_json)
                    if has_end_marker:
                        if isinstance(decoded, dict):
                            parsed_actions.append((decoded, decoded, True, True))
                        pos = after_json + len(self.tool_call_end_token)
                        next_scan_pos = pos
                        continue
                    if after_json == len(current_text) or self.tool_call_end_token.startswith(current_text[after_json:]):
                        if isinstance(decoded, dict):
                            parsed_actions.append((decoded, decoded, True, False))
                        self._in_open_block = True
                        next_scan_pos = start
                        break
                    end_idx = current_text.find(self.tool_call_end_token, end_pos)
                    next_start = current_text.find(self.tool_call_start_token, end_pos)
                    if end_idx != -1 and (next_start == -1 or end_idx < next_start):
                        if isinstance(decoded, dict):
                            parsed_actions.append((decoded, decoded, True, True))
                        pos = end_idx + len(self.tool_call_end_token)
                        next_scan_pos = pos
                        continue
                    if next_start != -1:
                        if isinstance(decoded, dict):
                            parsed_actions.append((decoded, decoded, True, True))
                        pos = next_start
                        next_scan_pos = pos
                        continue
                    if isinstance(decoded, dict):
                        parsed_actions.append((decoded, decoded, True, False))
                    self._in_open_block = True
                    next_scan_pos = start
                    break

                # Incomplete JSON object: parse partially without splitting on <|action_end|>.
                slice_text = current_text[payload_start:]
                try:
                    part_all, _ = partial_json_loads(slice_text, Allow.ALL)
                except (
                    partial_json_parser.core.exceptions.MalformedJSON,
                    ValueError,
                    AssertionError,
                    RecursionError,
                ):
                    logger.debug("not enough tokens to parse into JSON yet")
                    end_idx = current_text.find(self.tool_call_end_token, payload_start)
                    next_start = current_text.find(self.tool_call_start_token, payload_start)
                    if end_idx != -1 and (next_start == -1 or end_idx < next_start):
                        pos = end_idx + len(self.tool_call_end_token)
                        next_scan_pos = pos
                        continue
                    if next_start != -1:
                        pos = next_start
                        next_scan_pos = pos
                        continue
                    self._in_open_block = True
                    next_scan_pos = start
                    break

                if isinstance(part_all, dict):
                    try:
                        part_safe, _ = partial_json_loads(slice_text, Allow.ALL & ~Allow.STR)
                        if not isinstance(part_safe, dict):
                            part_safe = {}
                    except (
                        partial_json_parser.core.exceptions.MalformedJSON,
                        ValueError,
                        AssertionError,
                        RecursionError,
                    ):
                        part_safe = {}
                    parsed_actions.append((part_all, part_safe, False, False))
                self._in_open_block = True
                next_scan_pos = start
                break

            content_chunks: list[str] = []
            emitted_pos = self._emitted_content_pos or 0
            for seg_start, seg_end in visible_parts:
                if seg_end > emitted_pos:
                    content_chunks.append(current_text[max(seg_start, emitted_pos) : seg_end])
                    emitted_pos = seg_end
            self._emitted_content_pos = emitted_pos
            self.position = emitted_pos
            content_out = "".join(content_chunks)

            deltas: list[DeltaToolCall] = []
            emitted_idx = self._closed_emitted_idx
            next_closed_emitted_idx = self._closed_emitted_idx
            new_closed_records: list[dict] = []
            open_tracked_records: list[dict] = []
            for call, safe_call, call_complete, block_closed in parsed_actions:
                already_sent = emitted_idx < len(self._tool_name_sent) and self._tool_name_sent[emitted_idx]
                if already_sent or call_complete:
                    func_name = call.get("name")
                else:
                    func_name = safe_call.get("name")

                if not isinstance(func_name, str) or not func_name:
                    continue

                # Align tool-name filtering with batch extract_tool_calls.
                if not allowed_names or func_name not in allowed_names:
                    continue

                idx = emitted_idx
                emitted_idx += 1
                if block_closed:
                    next_closed_emitted_idx = emitted_idx
                while len(self.streamed_args_for_tool) <= idx:
                    self.streamed_args_for_tool.append("")
                while len(self._tool_name_sent) <= idx:
                    self._tool_name_sent.append(False)
                while len(self._tool_call_ids) <= idx:
                    self._tool_call_ids.append("")

                first = not self._tool_name_sent[idx]
                if first:
                    self._tool_call_ids[idx] = f"chatcmpl-tool-{uuid4()}"
                    self._tool_name_sent[idx] = True
                if idx >= self.current_tool_id:
                    self.current_tool_id = idx
                    self.current_tool_name_sent = True

                cur_arguments = self.get_arguments(call)
                if cur_arguments is None and call_complete:
                    cur_arguments = {}

                arguments_delta = ""
                if cur_arguments is not None:
                    cur_args_json = (
                        cur_arguments
                        if isinstance(cur_arguments, str)
                        else json.dumps(cur_arguments, ensure_ascii=False)
                    )
                    safe_json = cur_args_json if call_complete else stable_json_stream_prefix(cur_args_json)
                    sent = self.streamed_args_for_tool[idx]
                    if safe_json.startswith(sent):
                        arguments_delta = safe_json[len(sent) :]
                        if arguments_delta:
                            self.streamed_args_for_tool[idx] = sent + arguments_delta

                call_record = dict(call)
                call_record["arguments"] = cur_arguments
                if block_closed:
                    new_closed_records.append(call_record)
                else:
                    open_tracked_records.append(call_record)

                if first or arguments_delta:
                    deltas.append(
                        DeltaToolCall(
                            index=idx,
                            type="function" if first else None,
                            id=self._tool_call_ids[idx] if first else None,
                            function=DeltaFunctionCall(
                                name=func_name if first else None,
                                arguments=arguments_delta if arguments_delta else None,
                            ).model_dump(exclude_none=True),
                        )
                    )

            if new_closed_records:
                self._closed_tracked_calls.extend(new_closed_records)
            combined_tracked = (
                [*self._closed_tracked_calls, *open_tracked_records]
                if open_tracked_records
                else self._closed_tracked_calls
            )
            if combined_tracked:
                self.prev_tool_call_arr = list(combined_tracked)

            self._scan_pos = next_scan_pos
            self._closed_emitted_idx = next_closed_emitted_idx

            if content_out and deltas:
                return DeltaMessage(content=content_out, tool_calls=deltas)
            if deltas:
                return DeltaMessage(tool_calls=deltas)
            if content_out:
                return DeltaMessage(content=content_out)
            return None
        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug("Skipping chunk as a result of tool streaming extraction error")
            return None

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model response.

        Parses the model output to identify and extract all tool calls wrapped
        in ``<|action_start|><|plugin|>`` and ``<|action_end|>`` tokens using
        string-aware JSON decoding so embedded markers inside string arguments
        cannot truncate the payload.

        Args:
            model_output: The complete text output from the model.
            request: The chat completion request containing tools definition.

        Returns:
            ExtractedToolCallInformation containing:
                - tools_called: True if valid allowed tool calls were found
                - tool_calls: List of ToolCall objects extracted
                - content: Text content before the first valid action, or full
                    ``model_output`` if no valid tools were called
        """
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        allowed_names = self._allowed_tool_names(request)
        dec = json.JSONDecoder()
        tool_calls: list[ToolCall] = []
        first_action_pos: int | None = None
        pos = 0

        while pos < len(model_output):
            start = model_output.find(self.tool_call_start_token, pos)
            if start == -1:
                break
            payload_start = consume_space(start + len(self.tool_call_start_token), model_output)
            if payload_start >= len(model_output):
                break
            try:
                action_dict, end_pos = dec.raw_decode(model_output, payload_start)
            except (json.JSONDecodeError, ValueError, RecursionError):
                logger.exception("Failed to parse InternLM2 action as a tool call")
                pos = start + len(self.tool_call_start_token)
                continue

            after_json = consume_space(end_pos, model_output)
            if model_output.startswith(self.tool_call_end_token, after_json):
                pos = after_json + len(self.tool_call_end_token)
            else:
                pos = end_pos

            if not isinstance(action_dict, dict):
                continue
            name = action_dict.get("name")
            if not isinstance(name, str) or not name:
                continue
            if not allowed_names or name not in allowed_names:
                continue

            raw_params = self.get_arguments(action_dict)
            if raw_params is None:
                raw_params = {}
            parameters = raw_params if isinstance(raw_params, str) else json.dumps(raw_params, ensure_ascii=False)
            if first_action_pos is None:
                first_action_pos = start
            tool_calls.append(ToolCall(function=FunctionCall(name=name, arguments=parameters)))

        if not tool_calls:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        content = model_output[:first_action_pos] if first_action_pos is not None else ""
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content if len(content) > 0 else None,
        )
