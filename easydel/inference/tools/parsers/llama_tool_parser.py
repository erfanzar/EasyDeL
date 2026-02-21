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

"""Llama 3.x and 4 JSON-format tool call parser.

This module provides a tool call parser specifically designed for Meta's Llama
3.x and 4 model families when using JSON-formatted function calling. The parser
handles the specific output format used by Llama models with the
tool_chat_template_llama.jinja template.

The parser supports both single and multiple tool calls in a single response,
with tool calls separated by semicolons. It also handles the optional
<|python_tag|> token that may prefix tool call sections.

Example:
    >>> from easydel.inference.tools.parsers.llama_tool_parser import Llama3JsonToolParser
    >>> parser = Llama3JsonToolParser(tokenizer)
    >>> result = parser.extract_tool_calls(
    ...     '{"name": "get_weather", "arguments": {"city": "NYC"}}',
    ...     request
    ... )
    >>> result.tools_called
    True

See Also:
    - llama4_pythonic_tool_parser: For Llama 4 pythonic-style tool calls
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from uuid import uuid4

import partial_json_parser
from eformer.loggings import get_logger
from partial_json_parser.core.options import Allow
from transformers import PreTrainedTokenizerBase

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
from ..utils import find_common_prefix, is_complete_json, partial_json_loads

logger = get_logger(__name__)


@ToolParserManager.register_module("llama3_json")
@ToolParserManager.register_module("llama4_json")
class Llama3JsonToolParser(ToolParser):
    """Tool call parser for Llama 3.x and 4 models with JSON format.

    This parser is designed for use with Meta's Llama 3.x and 4 model families
    when configured with the tool_chat_template_llama.jinja template. It handles
    JSON-formatted tool calls with support for both single and multiple tool
    invocations separated by semicolons.

    Supported Formats:
        - Single tool call: `{"name": "func", "arguments": {...}}`
        - Multiple tool calls: `{"name": "func1", ...}; {"name": "func2", ...}`
        - With python tag: `<|python_tag|>{"name": "func", "arguments": {...}}`
        - Parameters alias: Supports both "arguments" and "parameters" field names

    Attributes:
        bot_token (str): The special token `<|python_tag|>` that optionally marks
            the beginning of tool call sections.
        bot_token_id (int): The token ID for the bot_token in the tokenizer.
        tool_call_regex (re.Pattern): Compiled regex pattern for extracting JSON
            tool calls from text, including nested objects.
        prev_tool_call_arr (list[dict]): Previous tool calls stored for streaming
            comparison to calculate argument diffs.
        current_tool_id (int): Index of the current tool being processed in
            streaming mode (-1 if none started).
        current_tool_name_sent (bool): Whether the function name has been sent
            in the current streaming sequence.
        streamed_args_for_tool (list[str]): Accumulated argument strings for
            each tool in streaming mode.

    Example:
        >>> parser = Llama3JsonToolParser(tokenizer)
        >>> # Single tool call
        >>> result = parser.extract_tool_calls(
        ...     '{"name": "search", "arguments": {"query": "weather"}}',
        ...     request
        ... )
        >>> result.tools_called
        True
        >>> # Multiple tool calls
        >>> result = parser.extract_tool_calls(
        ...     '{"name": "search", "arguments": {}}; {"name": "calc", "parameters": {}}',
        ...     request
        ... )
        >>> len(result.tool_calls)
        2

    Note:
        This parser is registered with both "llama3_json" and "llama4_json" names
        for compatibility with different model configurations. Use with
        --enable-auto-tool-choice --tool-call-parser llama3_json (or llama4_json).
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        """Initialize the Llama JSON tool parser.

        Args:
            tokenizer (PreTrainedTokenizerBase): The HuggingFace tokenizer
                instance for the Llama model. Used to encode the bot_token
                for token-level detection.
        """
        super().__init__(tokenizer)

        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
        self.bot_token = "<|python_tag|>"
        self.bot_token_id = tokenizer.encode(self.bot_token, add_special_tokens=False)[0]
        self.tool_call_regex = re.compile(
            r"{[^{}]*(?:{[^{}]*}[^{}]*)*}(?:\s*;\s*{[^{}]*(?:{[^{}]*}[^{}]*)*})*", re.DOTALL
        )

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete Llama model response.

        Parses the model output to identify and extract JSON-formatted tool calls.
        The method extracts JSON content while ignoring surrounding plain text,
        and supports both single JSON objects and multiple JSONs separated by
        semicolons.

        Args:
            model_output (str): The complete text output from the Llama model.
                May contain the <|python_tag|> token and/or raw JSON.
            request (ChatCompletionRequest): The original chat completion request.
                Currently unused but included for API consistency.

        Returns:
            ExtractedToolCallInformation: An object containing:
                - tools_called (bool): True if valid tool calls were extracted.
                - tool_calls (list[ToolCall]): List of ToolCall objects with
                  function names and JSON-serialized arguments.
                - content (str | None): Non-tool-call text content, or None if
                  the entire output was tool calls.

        Note:
            The parser accepts both "arguments" and "parameters" as field names
            for function arguments, providing compatibility with different model
            output styles.

        Example:
            >>> result = parser.extract_tool_calls(
            ...     '<|python_tag|>{"name": "search", "arguments": {"q": "test"}}',
            ...     request
            ... )
            >>> result.tools_called
            True
            >>> result.tool_calls[0].function.name
            'search'
        """
        if not (self.bot_token in model_output or "{" in model_output):
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        match = self.tool_call_regex.search(model_output)
        if not match:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            json_str = match.group(0)
            json_objects = [obj.strip() for obj in json_str.split(";")]

            tool_calls: list[ToolCall] = []
            for json_obj in json_objects:
                if not json_obj:
                    continue
                obj = json.loads(json_obj)
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=obj["name"],
                            arguments=json.dumps(
                                obj["arguments"] if "arguments" in obj else obj["parameters"], ensure_ascii=False
                            ),
                        ),
                    )
                )

            return ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls, content=None)

        except Exception:
            pass
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

        Processes streaming Llama model output to extract tool calls as they
        are being generated. Maintains internal state to track partial tool
        calls and emit incremental updates for function names and arguments.

        Args:
            previous_text (str): The accumulated text before this delta.
            current_text (str): The accumulated text including this delta.
            delta_text (str): The new text generated in this streaming chunk.
            previous_token_ids (Sequence[int]): Token IDs before this delta.
            current_token_ids (Sequence[int]): Token IDs including this delta.
            delta_token_ids (Sequence[int]): New token IDs in this chunk.
            request (ChatCompletionRequest): The original chat completion request.

        Returns:
            DeltaMessage | None: A delta message containing either:
                - Content text if no tool call is detected (text before
                  <|python_tag|> or opening brace).
                - Tool call name update when a new function name is parsed.
                - Tool call arguments update with incremental argument text.
                - None if more tokens are needed or parsing failed.

        Note:
            This method maintains state across calls via instance attributes.
            It supports multiple tool calls separated by semicolons and handles
            the transition between tools by emitting remaining arguments before
            starting a new tool.

        Warning:
            The parser converts "parameters" to "arguments" internally, and will
            raise an assertion error if a model generates both fields in a single
            tool call object.
        """
        if not (current_text.startswith(self.bot_token) or current_text.startswith("{")):
            return DeltaMessage(content=delta_text)

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            tool_call_arr = []
            is_complete = []
            try:
                start_idx = len(self.bot_token) if current_text.startswith(self.bot_token) else 0
                while start_idx < len(current_text):
                    (obj, end_idx) = partial_json_loads(current_text[start_idx:], flags)
                    is_complete.append(is_complete_json(current_text[start_idx : start_idx + end_idx]))
                    start_idx += end_idx + len("; ")
                    if "parameters" in obj:
                        assert "arguments" not in obj, "model generated both parameters and arguments"
                        obj["arguments"] = obj["parameters"]
                    tool_call_arr.append(obj)
            except partial_json_parser.core.exceptions.MalformedJSON:
                pass
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

        except Exception:
            pass
            return None
