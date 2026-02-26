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
from ..utils import consume_space, find_common_prefix, is_complete_json, partial_json_loads

logger = get_logger(__name__)


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

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output.

        Parses a JSON array of tool call objects from the model output.
        Removes bot token markers before parsing the JSON content.

        Args:
            model_output: Complete text output from the model containing
                potential tool calls in JSON array format.
            request: Original chat completion request (unused but required
                for interface compatibility).

        Returns:
            ExtractedToolCallInformation: Contains:
                - tools_called (bool): True if valid JSON array was parsed.
                - tool_calls (list[ToolCall]): List of parsed tool calls.
                - content (str | None): None for this parser (all output
                  is assumed to be tool calls if format matches).

        Example:
            >>> output = '<|tool_call|>[{"name": "test", "arguments": {"x": 1}}]'
            >>> result = parser.extract_tool_calls(output, request)
            >>> result.tools_called
            True
            >>> result.tool_calls[0].function.name
            'test'
        """
        stripped = model_output.strip().removeprefix(self.bot_token).removeprefix(self.bot_string).lstrip()
        if not stripped or stripped[0] != "[":
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        try:
            raw_function_calls = json.loads(stripped)
            if not isinstance(raw_function_calls, list):
                raise Exception(f"Expected dict or list, got {type(raw_function_calls)}")

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

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=None,
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

        Handles incremental parsing of JSON array format during streaming.
        Uses partial_json_parser to handle incomplete JSON and tracks
        which portions have been streamed to emit proper deltas.

        The method handles several scenarios:
        - Text content before tool call markers (passed through)
        - New tool calls starting (emits tool name)
        - Argument streaming (emits argument deltas)
        - Multiple tool calls in array (tracks transitions)

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
                - Tool call name when a new tool is detected
                - Argument fragments as they become available
                - Content text if not in tool call format
                - None if waiting for more data

        Note:
            Uses partial_json_parser with Allow.ALL or Allow.ALL & ~Allow.STR
            flags depending on whether the tool name has been sent, to
            control what partial values are acceptable.
        """
        start_idx = consume_space(0, current_text)
        if current_text[start_idx:].startswith(self.bot_token):
            start_idx = consume_space(start_idx + len(self.bot_token), current_text)
        if current_text[start_idx:].startswith(self.bot_string):
            start_idx = consume_space(start_idx + len(self.bot_string), current_text)
        if not current_text or start_idx >= len(current_text) or current_text[start_idx] != "[":
            return DeltaMessage(content=delta_text)

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            tool_call_arr = None
            is_complete = None
            try:
                tool_calls, end_idx = partial_json_loads(current_text[start_idx:], flags)
                if type(tool_calls) is list:
                    tool_call_arr = tool_calls
                else:
                    return DeltaMessage(content=delta_text)

                is_complete = [True] * len(tool_calls)
                if not is_complete_json(current_text[start_idx : start_idx + end_idx]):
                    is_complete[-1] = False
            except partial_json_parser.core.exceptions.MalformedJSON:
                return None

            if not tool_call_arr:
                return None

            current_tool_call: dict = tool_call_arr[self.current_tool_id]

            delta = None
            if len(tool_call_arr) > self.current_tool_id + 1:
                if self.current_tool_id >= 0:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                        sent = len(self.streamed_args_for_tool[self.current_tool_id])
                        argument_diff = cur_args_json[sent:]

                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(arguments=argument_diff).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += argument_diff

                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
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
                cur_arguments = current_tool_call.get("arguments")

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
            return None
