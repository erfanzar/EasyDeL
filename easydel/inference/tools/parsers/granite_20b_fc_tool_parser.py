# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Granite 20B function calling tool parser module.

This module provides the Granite20bFCToolParser class which handles the
specific tool call format used by the Granite 20B function calling model.
The format uses <function_call> markers with JSON objects.

Example tool call format:
    <function_call>{"name": "get_weather", "arguments": {"location": "NYC"}}
    <function_call>{"name": "get_time", "arguments": {"timezone": "EST"}}
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from json import JSONDecoder
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
from ..utils import consume_space, find_common_prefix, is_complete_json, partial_json_loads

logger = get_logger(__name__)


@ToolParserManager.register_module("granite-20b-fc")
class Granite20bFCToolParser(ToolParser):
    """Tool call parser for Granite 20B function calling model.

    Designed for the granite-20b-functioncalling model and intended
    for use with the examples/tool_chat_template_granite20b_fc.jinja
    template.

    Features:
    - Handles <function_call> token markers
    - Supports multiple function calls in sequence (not array)
    - Uses partial JSON parsing for streaming
    - Tracks completion state for proper argument streaming

    Format:
        <function_call>{"name": "func", "arguments": {...}}<function_call>{...}

    Unlike the standard Granite parser, this format uses individual
    <function_call> markers before each JSON object rather than a
    JSON array.

    Used when --enable-auto-tool-choice --tool-call-parser granite-20b-fc
    are all set.

    Attributes:
        bot_token (str): Bot token marker '<function_call>'.
        tool_start_token (str): Alias for bot_token.
        tool_call_regex (re.Pattern): Regex for finding tool call markers.
        current_tool_name_sent (bool): Whether tool name was sent (inherited).
        prev_tool_call_arr (list): Previous tool calls for streaming (inherited).
        current_tool_id (int): Current tool index (inherited).
        streamed_args_for_tool (list[str]): Streamed args per tool (inherited).

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("granite-20b-fc")
        >>> parser = Granite20bFCToolParser(tokenizer)
        >>> result = parser.extract_tool_calls(model_output, request)
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the Granite 20B FC tool parser.

        Sets up the function_call token marker and regex pattern for
        identifying tool call regions in the output.

        Args:
            tokenizer: The tokenizer associated with the Granite 20B FC model.
        """
        super().__init__(tokenizer)

        self.bot_token = "<function_call>"
        self.tool_start_token = self.bot_token
        self.tool_call_regex = re.compile(r"<function_call>\s*")

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output.

        Parses multiple JSON objects following <function_call> markers.
        Uses JSONDecoder.raw_decode to handle consecutive JSON objects
        without a separating array structure.

        Args:
            model_output: Complete text output from the model containing
                potential tool calls with <function_call> markers.
            request: Original chat completion request (unused but required
                for interface compatibility).

        Returns:
            ExtractedToolCallInformation: Contains:
                - tools_called (bool): True if tool call markers found.
                - tool_calls (list[ToolCall]): List of parsed tool calls.
                - content (str | None): Text before first tool call marker,
                  or None if output starts with tool call.

        Example:
            >>> output = '<function_call>{"name": "test", "arguments": {"x": 1}}'
            >>> result = parser.extract_tool_calls(output, request)
            >>> result.tools_called
            True
        """
        if self.tool_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        dec = JSONDecoder()
        try:
            matches = list(self.tool_call_regex.finditer(model_output))
            logger.debug("Found %d tool call matches", len(matches))

            raw_function_calls = []

            for i, match in enumerate(matches):
                start_of_json = match.end()
                next_function_call_start = matches[i + 1].start() if i + 1 < len(matches) else None

                raw_function_calls.append(dec.raw_decode(model_output[start_of_json:next_function_call_start])[0])

            logger.debug("Extracted %d tool calls", len(raw_function_calls))
            tool_calls = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=function_call["name"],
                        arguments=json.dumps(function_call["arguments"], ensure_ascii=False),
                    ),
                )
                for function_call in raw_function_calls
            ]

            content = model_output[: model_output.find(self.bot_token)]
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception as e:
            logger.error("Error in extracting tool call from response %s", e)
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
        """Extract tool calls from streaming model output.

        Handles incremental parsing of <function_call> format during
        streaming. Parses consecutive JSON objects separated by
        <function_call> markers and tracks streaming state.

        The method processes text incrementally:
        - Waits if current text might become bot token
        - Passes through content not starting with bot token
        - Parses JSON objects after each <function_call> marker
        - Emits tool names and argument deltas progressively

        Args:
            previous_text: Previously generated text (unused).
            current_text: All text generated so far.
            delta_text: New text added in this chunk.
            previous_token_ids: Previous token IDs (unused).
            current_token_ids: All token IDs (unused).
            delta_token_ids: New token IDs (unused).
            request: Original chat completion request (unused).

        Returns:
            DeltaMessage | None: A delta message containing:
                - Tool call name when a new tool starts
                - Argument fragments as JSON is parsed
                - Content text if not starting with bot token
                - None if waiting for more data or parsing fails

        Note:
            Uses partial_json_parser to handle incomplete JSON during
            streaming. The parser tracks which arguments have been
            sent to emit only new content in each delta.
        """
        if len(current_text) < len(self.bot_token) and self.bot_token.startswith(current_text):
            return None

        if not current_text.startswith(self.bot_token):
            return DeltaMessage(content=delta_text)

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            tool_call_arr = []
            is_complete = []
            try:
                start_idx = len(self.bot_token)
                start_idx = consume_space(start_idx, current_text)

                while start_idx < len(current_text):
                    (obj, end_idx) = partial_json_loads(current_text[start_idx:], flags)
                    is_complete.append(is_complete_json(current_text[start_idx : start_idx + end_idx]))
                    start_idx += end_idx
                    start_idx = consume_space(start_idx, current_text)
                    start_idx += len(self.bot_token)
                    start_idx = consume_space(start_idx, current_text)
                    tool_call_arr.append(obj)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug("not enough tokens to parse into JSON yet")
                return None

            current_tool_call: dict = tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}

            if len(tool_call_arr) == 0:
                return None

            elif len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1:
                if self.current_tool_id >= 0:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                        sent = len(self.streamed_args_for_tool[self.current_tool_id])
                        argument_diff = cur_args_json[sent:]

                        logger.debug("got arguments diff: %s", argument_diff)
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(arguments=argument_diff).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += argument_diff
                    else:
                        delta = None
                else:
                    delta = None
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("starting on new tool %d", self.current_tool_id)
                return delta

            elif not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=f"chatcmpl-tool-{uuid4()}",
                                function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.current_tool_name_sent = True
                else:
                    delta = None

            else:
                cur_arguments = current_tool_call.get("arguments")
                delta = None

                if cur_arguments:
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get("arguments")

                    argument_diff = None
                    if is_complete[self.current_tool_id]:
                        argument_diff = cur_args_json[sent:]
                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                        if cur_args_json != prev_args_json:
                            prefix = find_common_prefix(prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]

                    if argument_diff is not None:
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(arguments=argument_diff).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += argument_diff

            self.prev_tool_call_arr = tool_call_arr
            return delta

        except Exception as e:
            logger.error("Error trying to handle streaming tool call: %s", e)
            logger.debug("Skipping chunk as a result of tool streaming extraction error")
            return None
