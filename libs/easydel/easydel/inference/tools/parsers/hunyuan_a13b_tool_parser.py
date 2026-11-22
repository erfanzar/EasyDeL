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

"""Tool parser implementation for Hunyuan A13B models.

This module provides a tool parser specifically designed for Hunyuan A13B model outputs.
It handles tool calls wrapped in <tool_calls> XML-style tags with support for filtering
out tool calls that appear within thinking sections.

The parser supports Chinese language tokens (e.g., "assistant:" in Chinese) and handles
nested JSON objects in function arguments.

Example format:
    <tool_calls>[{"name": "function_name", "arguments": {"param": "value"}}]</tool_calls>
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import partial_json_parser
from eformer.loggings import get_logger
from partial_json_parser.core.options import Allow
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

_THINK_START = "<think>"
_THINK_END = "</think>"


def _trailing_prefix_len(text: str, start: int, marker: str) -> int:
    """Return length of the longest proper prefix of *marker* at the end of ``text[start:]``."""
    avail = len(text) - start
    max_len = min(len(marker) - 1, avail)
    for size in range(max_len, 0, -1):
        if text.endswith(marker[:size]):
            return size
    return 0


@ToolParserManager.register_module("hunyuan_a13b")  # pyright: ignore[reportUntypedClassDecorator]
class HunyuanA13BToolParser(ToolParser):
    """Tool parser for Hunyuan A13B model outputs.

    This parser handles the extraction of function/tool calls from Hunyuan A13B model
    outputs. The model uses XML-style <tool_calls> tags containing a JSON array of
    function call objects.

    The parser includes special handling for:
    - Filtering tool calls that appear within <think>...</think> sections
    - Chinese language tokens (removes "assistant:" prefix in Chinese)
    - Nested JSON objects within function arguments
    - Both empty and non-empty argument patterns

    Attributes:
        prev_tool_calls: List of previously parsed tool call dictionaries.
        current_tools_sent: Flags indicating which tool names have been sent.
        current_tool_id: Index of the current tool being processed in streaming mode.
        current_tool_name_sent: Flag indicating if the current tool name has been sent.
        streamed_args: List of argument strings streamed for each tool.
        prev_tool_call_arr: Array storing parsed tool call data.
        answer_tool_calls_pattern: Regex pattern for extracting tool calls content.
        tool_name_reg: Regex pattern for extracting function names.
        tool_empty_arg_reg: Regex pattern for empty arguments.
        tool_non_empty_arg_reg: Regex pattern for non-empty arguments.
        bot_string: The tag that indicates the start of tool calls.
        streaming_state: Dictionary maintaining state during streaming extraction.

    Example:
        >>> parser = HunyuanA13BToolParser(tokenizer)
        >>> result = parser.extract_tool_calls(model_output, request)
        >>> if result.tools_called:
        ...     for tool_call in result.tool_calls:
        ...         print(f"Function: {tool_call.function.name}")
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the Hunyuan A13B tool parser.

        Sets up regex patterns for parsing tool calls and initializes streaming state.

        Args:
            tokenizer: A HuggingFace tokenizer instance used for token processing.
                This tokenizer should be compatible with the Hunyuan A13B model.
        """
        super().__init__(tokenizer)

        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args: list[str] = []

        self.prev_tool_call_arr = []
        self.prev_tool_calls: list[dict] = []
        self.current_tools_sent: list[bool] = []

        self.answer_tool_calls_pattern = re.compile(r"<tool_calls>([\s\S]*?)</tool_calls>", re.DOTALL)

        self.tool_name_reg = re.compile(r'"name"\s*:\s*"([^"]+)"')

        self.tool_empty_arg_reg = re.compile(r'"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{\s*\}')

        # Compatibility regex attributes; streaming uses structural JSON parsing
        # below so nested argument fields cannot become executable calls.
        self.tool_non_empty_arg_reg = re.compile(
            r'"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\})'
        )

        self.bot_string = "<tool_calls>"
        self.eot_string = "</tool_calls>"

        self.streaming_state: dict[str, Any] = {
            "current_tool_index": -1,
            "tool_ids": [],
            "sent_tools": [],
        }
        self._emitted_content_pos: int | None = None
        self._pending_tail_start: int | None = None
        self._last_current_text: str = ""
        self._streaming_initialized: bool = False
        self._in_open_block: bool = False
        self._scan_pos: int = 0
        self._in_think: bool = False
        self._closed_emitted_idx: int = 0
        self._closed_complete_calls: list[dict] = []

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Report whether a ``<tool_calls>`` block outside ``<think>`` or a partial prefix is buffered."""
        del delta_text
        if self._streaming_initialized:
            return self._in_open_block or self._pending_tail_start is not None

        pos = 0
        n = len(current_text)
        think_start_len = len(_THINK_START)
        think_end_len = len(_THINK_END)
        bot_len = len(self.bot_string)
        eot_len = len(self.eot_string)
        while pos < n:
            t_idx = current_text.find(_THINK_START, pos)
            b_idx = current_text.find(self.bot_string, pos)
            if t_idx != -1 and (b_idx == -1 or t_idx < b_idx):
                t_end = current_text.find(_THINK_END, t_idx + think_start_len)
                if t_end == -1:
                    return False
                pos = t_end + think_end_len
                continue
            if b_idx == -1:
                return self._has_trailing_marker_prefix(current_text[pos:], (self.bot_string,))
            e_idx = current_text.find(self.eot_string, b_idx + bot_len)
            if e_idx == -1:
                return True
            pos = e_idx + eot_len
        return False

    def _reset_streaming_state(self, emitted_pos: int = 0) -> None:
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args = []
        self.streamed_args_for_tool = []
        self.prev_tool_call_arr = []
        self.prev_tool_calls = []
        self.current_tools_sent = []
        self.streaming_state = {
            "current_tool_index": -1,
            "tool_ids": [],
            "sent_tools": [],
        }
        self._emitted_content_pos = emitted_pos
        self._pending_tail_start = None
        self._last_current_text = ""
        self._streaming_initialized = False
        self._in_open_block = False
        self._scan_pos = 0
        self._in_think = False
        self._closed_emitted_idx = 0
        self._closed_complete_calls = []

    def flush_streaming_content(self) -> str | None:
        """Flush any trailing partial ``<tool_calls>`` prefix withheld at EOS."""
        flushed: str | None = None
        if self._pending_tail_start is not None and self._pending_tail_start < len(self._last_current_text):
            flushed = self._last_current_text[self._pending_tail_start :] or None
            self._emitted_content_pos = len(self._last_current_text)
            self._pending_tail_start = None
        self._last_current_text = ""
        return flushed

    def preprocess_model_output(self, model_output: str) -> tuple[str | None, str | None]:
        """Preprocess model output to extract tool calls content.

        Scans the model output for <tool_calls> tags and extracts the content,
        while filtering out any tool calls that appear within <think> sections.
        This ensures only actual function calls (not reasoning about function calls)
        are extracted.

        Args:
            model_output: The raw model output string to preprocess.

        Returns:
            A tuple of (content, tool_calls_content) where:
                - content: The text before the tool calls section, or the full
                    output if no valid tool calls were found.
                - tool_calls_content: The JSON string inside the tool_calls tags,
                    or None if no valid tool calls were found.

        Example:
            >>> content, tools = parser.preprocess_model_output(
            ...     "Hello<tool_calls>[{...}]</tool_calls>"
            ... )
            >>> content
            'Hello'
        """
        dec = json.JSONDecoder()
        merged_calls: list[Any] = []
        first_valid_start: int | None = None
        first_raw_payload: str | None = None
        first_non_list_block: tuple[int, str] | None = None
        valid_block_count = 0

        pos = 0
        n = len(model_output)
        think_start_len = len(_THINK_START)
        think_end_len = len(_THINK_END)
        while pos < n:
            think_idx = model_output.find(_THINK_START, pos)
            start = model_output.find(self.bot_string, pos)
            if think_idx != -1 and (start == -1 or think_idx < start):
                think_end = model_output.find(_THINK_END, think_idx + think_start_len)
                if think_end == -1:
                    break
                pos = think_end + think_end_len
                continue

            if start == -1:
                break

            payload_start = consume_space(start + len(self.bot_string), model_output)
            if payload_start >= n:
                break
            try:
                parsed, end_pos = dec.raw_decode(model_output, payload_start)
            except (json.JSONDecodeError, ValueError, RecursionError):
                pos = start + len(self.bot_string)
                continue

            after_json = consume_space(end_pos, model_output)
            if not model_output.startswith(self.eot_string, after_json):
                pos = end_pos
                continue

            pos = after_json + len(self.eot_string)
            raw_payload = model_output[payload_start:end_pos].strip()
            if isinstance(parsed, list):
                if first_valid_start is None:
                    first_valid_start = start
                    first_raw_payload = raw_payload
                valid_block_count += 1
                merged_calls.extend(parsed)
            elif first_non_list_block is None:
                # Preserve non-list payloads as fallback when no valid list block follows.
                first_non_list_block = (start, raw_payload)

        if first_valid_start is not None:
            content = model_output[:first_valid_start]
            if valid_block_count == 1 and first_raw_payload is not None:
                return content, first_raw_payload
            return content, json.dumps(merged_calls, ensure_ascii=False)
        if first_non_list_block is not None:
            nl_start, nl_payload = first_non_list_block
            return model_output[:nl_start], nl_payload
        return model_output, None

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model output.

        Parses the model output to find and extract function calls in Hunyuan A13B format.
        The function looks for <tool_calls> tags containing a JSON array of function
        call objects with "name" and "arguments" fields.

        Args:
            model_output: The complete text output from the model to parse.
            request: The chat completion request that triggered this response.

        Returns:
            ExtractedToolCallInformation: An object containing:
                - tools_called: True if valid tool calls were found, False otherwise.
                - tool_calls: List of ToolCall objects representing parsed function calls.
                - content: Any text content before the tool calls section, or the full
                    output if no valid tool calls were found.

        Note:
            This method removes the Chinese "assistant:" prefix ("assistant:") from content
            if present at the beginning of the output.
        """
        try:
            content, potential_tool_calls = self.preprocess_model_output(model_output)

            if not potential_tool_calls:
                if content:
                    content = content.replace("助手：", "", 1)
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=content)

            tool_calls_data = json.loads(potential_tool_calls)

            if not isinstance(tool_calls_data, list):
                logger.debug("Tool calls data is not an array")
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )

            tool_calls: list[ToolCall] = []

            for _, call in enumerate(tool_calls_data):
                if (
                    not isinstance(call, dict)
                    or not isinstance(call.get("name"), str)
                    or not call.get("name")
                    or "arguments" not in call
                ):
                    continue

                raw_args = call["arguments"]
                tool_call = ToolCall(
                    id=f"call_{uuid4()}",
                    type="function",
                    function=FunctionCall(
                        name=call["name"],
                        arguments=raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False),
                    ),
                )
                tool_calls.append(tool_call)

            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )

            if not content or len(content.strip()) == 0:
                content = None

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content,
            )

        except Exception:
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
        """Extract tool calls incrementally during streaming generation.

        Processes tokens as they are generated to progressively extract and emit
        tool call information. Supports multiple tool calls in a single response.

        Args:
            previous_text: The accumulated text from all previous tokens.
            current_text: The complete text including the current delta.
            delta_text: The new text added in this streaming step.
            previous_token_ids: Token IDs from all previous generation steps.
            current_token_ids: All token IDs including the current step.
            delta_token_ids: The new token IDs added in this step.
            request: The chat completion request for context.

        Returns:
            DeltaMessage or None: A delta message containing either:
                - Content text if no tool call is detected.
                - Tool call information (name or argument delta) if parsing functions.
                - None if waiting for more tokens to complete parsing.
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

        dec = json.JSONDecoder()
        pos = self._scan_pos
        in_think = self._in_think
        n = len(current_text)
        think_start_len = len(_THINK_START)
        think_end_len = len(_THINK_END)
        visible_parts: list[tuple[int, int]] = []
        # list of (call_dict, safe_call_dict, call_complete, block_closed)
        all_calls: list[tuple[dict, dict, bool, bool]] = []
        new_closed_complete_calls: list[dict] = []
        open_complete_calls: list[dict] = []

        while pos < n:
            if in_think:
                t_end = current_text.find(_THINK_END, pos)
                if t_end == -1:
                    visible_parts.append((pos, n))
                    hold_t = _trailing_prefix_len(current_text, pos, _THINK_END)
                    pos = n - hold_t
                    break
                close_end = t_end + think_end_len
                visible_parts.append((pos, close_end))
                pos = close_end
                in_think = False
                continue

            t_idx = current_text.find(_THINK_START, pos)
            sec_start = current_text.find(self.bot_string, pos)
            if t_idx != -1 and (sec_start == -1 or t_idx < sec_start):
                t_body = t_idx + think_start_len
                visible_parts.append((pos, t_body))
                pos = t_body
                in_think = True
                continue

            if sec_start == -1:
                hold_bot = _trailing_prefix_len(current_text, pos, self.bot_string)
                hold_think = _trailing_prefix_len(current_text, pos, _THINK_START)
                safe_end = n - hold_bot
                if safe_end > pos:
                    visible_parts.append((pos, safe_end))
                if hold_bot > 0:
                    self._pending_tail_start = safe_end
                pos = n - max(hold_bot, hold_think)
                break

            if sec_start > pos:
                visible_parts.append((pos, sec_start))

            payload_start = consume_space(sec_start + len(self.bot_string), current_text)
            if payload_start >= n:
                self._in_open_block = True
                pos = sec_start
                break

            parsed: object = None
            end_pos = -1
            try:
                parsed, end_pos = dec.raw_decode(current_text, payload_start)
            except (json.JSONDecodeError, ValueError, RecursionError):
                parsed = None

            if parsed is not None:
                after_json = consume_space(end_pos, current_text)
                if current_text.startswith(self.eot_string, after_json):
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict):
                                all_calls.append((item, item, True, True))
                                new_closed_complete_calls.append(item)
                    pos = after_json + len(self.eot_string)
                    continue
                if after_json == n or self.eot_string.startswith(current_text[after_json:]):
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict):
                                all_calls.append((item, item, True, False))
                                open_complete_calls.append(item)
                    self._in_open_block = True
                    pos = sec_start
                    break
                eot_idx = current_text.find(self.eot_string, end_pos)
                if eot_idx != -1:
                    pos = eot_idx + len(self.eot_string)
                    continue
                self._in_open_block = True
                pos = sec_start
                break

            slice_text = current_text[payload_start:]
            try:
                part_all, _ = partial_json_loads(slice_text, Allow.ALL)
            except (
                partial_json_parser.core.exceptions.MalformedJSON,
                ValueError,
                AssertionError,
                RecursionError,
            ):
                eot_idx = current_text.find(self.eot_string, payload_start)
                if eot_idx != -1:
                    pos = eot_idx + len(self.eot_string)
                    continue
                self._in_open_block = True
                pos = sec_start
                break

            if isinstance(part_all, list):
                try:
                    part_safe, _ = partial_json_loads(slice_text, Allow.ALL & ~Allow.STR)
                    if not isinstance(part_safe, list):
                        part_safe = []
                except (
                    partial_json_parser.core.exceptions.MalformedJSON,
                    ValueError,
                    AssertionError,
                    RecursionError,
                ):
                    part_safe = []

                for i, item in enumerate(part_all):
                    if not isinstance(item, dict):
                        continue
                    safe_item = part_safe[i] if i < len(part_safe) and isinstance(part_safe[i], dict) else {}
                    all_calls.append((item, safe_item, i < len(part_all) - 1, False))
            self._in_open_block = True
            pos = sec_start
            break

        if new_closed_complete_calls:
            self._closed_complete_calls.extend(new_closed_complete_calls)
        combined_complete = (
            [*self._closed_complete_calls, *open_complete_calls]
            if open_complete_calls
            else self._closed_complete_calls
        )
        if combined_complete:
            self.prev_tool_call_arr = list(combined_complete)
            self.prev_tool_calls = list(combined_complete)

        content_chunks: list[str] = []
        emitted_pos = self._emitted_content_pos or 0
        for seg_start, seg_end in visible_parts:
            if seg_end > emitted_pos:
                content_chunks.append(current_text[max(seg_start, emitted_pos) : seg_end])
                emitted_pos = seg_end
        self._emitted_content_pos = emitted_pos
        content_out = "".join(content_chunks)

        deltas: list[DeltaToolCall] = []
        emitted_idx = self._closed_emitted_idx
        next_closed_emitted_idx = self._closed_emitted_idx

        for call, safe_call, call_complete, block_closed in all_calls:
            already_sent = emitted_idx < len(self.current_tools_sent) and self.current_tools_sent[emitted_idx]
            if already_sent or call_complete:
                tool_name = call.get("name")
            else:
                tool_name = safe_call.get("name")

            if not isinstance(tool_name, str) or not tool_name or "arguments" not in call:
                continue

            idx = emitted_idx
            emitted_idx += 1
            if block_closed:
                next_closed_emitted_idx = emitted_idx
            self._ensure_state_arrays(idx + 1)
            while len(self.streamed_args) <= idx:
                self.streamed_args.append("")
            while len(self.streamed_args_for_tool) <= idx:
                self.streamed_args_for_tool.append("")

            first = not self.current_tools_sent[idx]
            if first:
                tool_id = f"call_{idx}_{uuid4()}"
                self.streaming_state["tool_ids"][idx] = tool_id
                self.streaming_state["sent_tools"][idx]["sent_name"] = True
                self.streaming_state["current_tool_index"] = idx
                self.current_tools_sent[idx] = True
                self.current_tool_id = idx
                self.current_tool_name_sent = True

            args_diff = ""
            if "arguments" in call and call["arguments"] is not None:
                arguments = call["arguments"]
                args_text = json.dumps(arguments, ensure_ascii=False) if not isinstance(arguments, str) else arguments
                safe_args = args_text if call_complete else stable_json_stream_prefix(args_text)
                state = self.streaming_state["sent_tools"][idx]
                sent_args = state["sent_arguments"]
                if safe_args.startswith(sent_args):
                    args_diff = safe_args[len(sent_args) :]
                    if args_diff:
                        state["sent_arguments_prefix"] = True
                        state["sent_arguments"] = safe_args
                        self.streamed_args[idx] = safe_args
                        self.streamed_args_for_tool[idx] = safe_args

            if first or args_diff:
                deltas.append(
                    DeltaToolCall(
                        index=idx,
                        type="function" if first else None,
                        id=self.streaming_state["tool_ids"][idx] if first else None,
                        function=DeltaFunctionCall(
                            name=tool_name if first else None,
                            arguments=args_diff if args_diff else None,
                        ).model_dump(exclude_none=True),
                    )
                )

        self._scan_pos = pos
        self._in_think = in_think
        self._closed_emitted_idx = next_closed_emitted_idx

        if content_out and deltas:
            return DeltaMessage(content=content_out, tool_calls=deltas)
        if deltas:
            return DeltaMessage(tool_calls=deltas)
        if content_out:
            return DeltaMessage(content=content_out)
        return None

    def _try_parse_json_tools(self, current_text: str):
        """Attempt to parse the current text as a JSON array of tools.

        Tries to parse the text as complete JSON and stores the result in
        prev_tool_call_arr if successful. This is used to maintain state
        about fully parsed tool calls.

        Args:
            current_text: The text to attempt parsing, expected to start with '['.
        """
        try:
            parsed_tools, _ = json.JSONDecoder().raw_decode(current_text)
            if isinstance(parsed_tools, list):
                self.prev_tool_call_arr = parsed_tools
                self.prev_tool_calls = parsed_tools
        except (json.JSONDecodeError, ValueError, RecursionError):
            pass

    def _ensure_state_arrays(self, tool_count: int):
        """Ensure streaming state arrays have sufficient capacity.

        Expands the sent_tools and tool_ids arrays to accommodate the
        specified number of tools.

        Args:
            tool_count: The number of tools that need to be tracked.
        """
        while len(self.streaming_state["sent_tools"]) < tool_count:
            self.streaming_state["sent_tools"].append(
                {
                    "sent_name": False,
                    "sent_arguments_prefix": False,
                    "sent_arguments": "",
                }
            )
        while len(self.streaming_state["tool_ids"]) < tool_count:
            self.streaming_state["tool_ids"].append(None)
        while len(self.current_tools_sent) < tool_count:
            self.current_tools_sent.append(False)

    def _handle_tool_name_streaming(self, current_idx: int, tool_count: int, name_matches):
        """Handle streaming of tool names.

        Detects when a new tool name should be sent and emits the appropriate
        delta message with the tool name and ID.

        Args:
            current_idx: The current tool index being processed.
            tool_count: Total number of tools detected so far.
            name_matches: Top-level call dictionaries with complete tool names.

        Returns:
            DeltaMessage or None: A delta message with the tool name if a new
                tool is detected, None otherwise.
        """
        if 0 <= current_idx < tool_count:
            tool_name = name_matches[current_idx].get("name")
            if isinstance(tool_name, str) and tool_name and not self.current_tools_sent[current_idx]:
                self.streaming_state["current_tool_index"] = current_idx
                self.current_tool_id = current_idx
                tool_id = f"call_{current_idx}_{uuid4()}"
                self.streaming_state["tool_ids"][current_idx] = tool_id
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=current_idx,
                            type="function",
                            id=tool_id,
                            function=DeltaFunctionCall(name=tool_name).model_dump(exclude_none=True),
                        )
                    ]
                )
                self.streaming_state["sent_tools"][current_idx]["sent_name"] = True
                self.current_tools_sent[current_idx] = True
                self.current_tool_name_sent = True
                while len(self.streamed_args) <= current_idx:
                    self.streamed_args.append("")
                return delta
        return None

    def _handle_tool_args_streaming(self, current_text: str, current_idx: int, tool_count: int):
        """Handle streaming of tool arguments.

        Processes and emits argument deltas for the current tool being streamed.
        Handles both empty arguments ({}) and non-empty argument objects.

        Args:
            current_text: One complete or partial top-level call object.
            current_idx: The current tool index being processed.
            tool_count: Total number of tools detected so far.

        Returns:
            DeltaMessage or None: A delta message with argument content if new
                arguments are detected, None otherwise.
        """
        if not (0 <= current_idx < tool_count) or not self.current_tools_sent[current_idx]:
            return None
        complete = False
        try:
            call = json.loads(current_text)
            complete = True
        except (ValueError, RecursionError):
            try:
                call = partial_json_parser.loads(current_text, Allow.ALL)
            except (
                ValueError,
                partial_json_parser.MalformedJSON,
                AssertionError,
                IndexError,
                RecursionError,
            ):
                return None
        if not isinstance(call, dict) or "arguments" not in call:
            return None
        arguments = call["arguments"]
        args_text = json.dumps(arguments) if not isinstance(arguments, str) else arguments
        if not complete:
            args_text = stable_json_stream_prefix(args_text)
        state = self.streaming_state["sent_tools"][current_idx]
        sent_args = state["sent_arguments"]
        if not args_text.startswith(sent_args):
            return None
        args_diff = args_text[len(sent_args) :]
        if not args_diff:
            return None
        state["sent_arguments_prefix"] = True
        state["sent_arguments"] = args_text
        self.streamed_args[current_idx] = args_text
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=current_idx,
                    function=DeltaFunctionCall(arguments=args_diff).model_dump(exclude_none=True),
                )
            ]
        )
