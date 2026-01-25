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

"""Tool parser implementation for ERNIE 4.5 models.

This module provides a tool parser specifically designed for ERNIE 4.5 (Baidu) model outputs.
ERNIE 4.5 is a thinking model that uses a specific format with </think> tags followed by
<tool_call> tags containing JSON function call data.

The parser handles the complex token structure including thinking sections, response tags,
and tool call delimiters, with special handling for newline tokens and streaming state.

Example format:
    reasoning content</think>

    <tool_call>
    {"name": "function_name", "arguments": {"param": "value"}}
    </tool_call>
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from eformer.loggings import get_logger
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

logger = get_logger(__name__)


@ToolParserManager.register_module("ernie45")
class Ernie45ToolParser(ToolParser):
    """Tool parser for ERNIE 4.5 (Baidu) model outputs.

    This parser handles the extraction of function/tool calls from ERNIE 4.5 model
    outputs. ERNIE 4.5 is a thinking model that outputs reasoning in a thinking
    section followed by tool calls in XML-style tags.

    The parser handles several special tokens and structures:
    - </think> tag marking the end of the thinking section
    - <response>...</response> tags for regular responses
    - <tool_call>...</tool_call> tags for function calls
    - Special newline token handling (<0x0A>)

    Attributes:
        current_tool_name_sent: Flag indicating if the current tool name has been sent.
        prev_tool_call_arr: List of previously parsed tool call dictionaries.
        current_tool_id: Index of the current tool being processed (-1 means none).
        streamed_args_for_tool: List of argument strings streamed for each tool.
        think_end_token: Token marking end of thinking section.
        response_start_token: Token marking start of response.
        response_end_token: Token marking end of response.
        tool_call_start_token: Token marking start of tool call.
        tool_call_end_token: Token marking end of tool call.
        tool_calls_start_token: Alias for tool_call_start_token.
        newline_token: Special newline token representation.
        tool_call_regex: Compiled regex for extracting tool call JSON.
        think_end_token_id: Token ID for think_end_token.
        response_start_token_id: Token ID for response_start_token.
        response_end_token_id: Token ID for response_end_token.
        tool_call_start_token_id: Token ID for tool_call_start_token.
        tool_call_end_token_id: Token ID for tool_call_end_token.
        newline_token_id: Token ID for newline_token.
        parser_token_ids: List of special parser token IDs.

    Raises:
        ValueError: If the tokenizer is not provided during initialization.

    Example:
        >>> parser = Ernie45ToolParser(tokenizer)
        >>> result = parser.extract_tool_calls(model_output, request)
        >>> if result.tools_called:
        ...     for tool_call in result.tool_calls:
        ...         print(f"Function: {tool_call.function.name}")
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the ERNIE 4.5 tool parser.

        Sets up token mappings and regex patterns for parsing ERNIE 4.5 format
        tool calls. Validates that a tokenizer is provided and maps special
        tokens to their IDs.

        Args:
            tokenizer: A HuggingFace tokenizer instance used for token processing.
                This tokenizer should be compatible with the ERNIE 4.5 model and
                must contain the special tokens used by this parser.

        Raises:
            ValueError: If the tokenizer is not provided (is None or falsy).

        Note:
            The parser expects the tokenizer vocabulary to contain special tokens
            like </think>, <response>, </response>, <tool_call>, </tool_call>,
            and <0x0A> (newline). Missing tokens will result in None values for
            the corresponding token IDs.
        """
        super().__init__(tokenizer)

        self.current_tool_name_sent = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id = -1
        self.streamed_args_for_tool: list[str] = []

        self.think_end_token = "</think>"
        self.response_start_token: str = "<response>"
        self.response_end_token: str = "</response>"
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_calls_start_token = self.tool_call_start_token
        self.newline_token: str = "<0x0A>"

        self.tool_call_regex = re.compile(r"<tool_call>\s*(?P<json>\{.*?\})\s*</tool_call>", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")

        self.think_end_token_id = self.vocab.get(self.think_end_token)
        self.response_start_token_id = self.vocab.get(self.response_start_token)
        self.response_end_token_id = self.vocab.get(self.response_end_token)
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        self.newline_token_id = self.vocab.get(self.newline_token)
        self.parser_token_ids = [
            self.think_end_token_id,
            self.response_start_token_id,
            self.response_end_token_id,
        ]

        self._buffer = ""

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model output.

        Parses the model output to find and extract function calls in ERNIE 4.5 format.
        Searches for <tool_call>...</tool_call> tags containing JSON function data.

        Args:
            model_output: The complete text output from the model to parse.
            request: The chat completion request that triggered this response.

        Returns:
            ExtractedToolCallInformation: An object containing:
                - tools_called: True if valid tool calls were found, False otherwise.
                - tool_calls: List of ToolCall objects representing parsed function calls.
                - content: Any text content before the tool call section, or the full
                    output if no valid tool calls were found.

        Note:
            If no <tool_call> start token is found, returns immediately with
            tools_called=False and the full output as content.
        """
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            tool_call_json_list = self.tool_call_regex.findall(model_output)

            tool_calls: list[ToolCall] = []
            for tool_call_json in tool_call_json_list:
                tool_call_dict = json.loads(tool_call_json)
                args_str = json.dumps(tool_call_dict.get("arguments", {}), ensure_ascii=False)
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(name=tool_call_dict.get("name", ""), arguments=args_str),
                    )
                )

            content = model_output[: model_output.find(self.tool_calls_start_token)].rstrip("\n")
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
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
        tool call information. Handles the complex ERNIE 4.5 format including
        thinking sections, response tags, and tool call delimiters.

        This method buffers content between tool call start and end tokens,
        then extracts and emits the complete tool call when the end token is found.

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
                - Content text if outside tool call section.
                - Tool call information when a complete tool call is parsed.
                - None if buffering content within a tool call.

        Note:
            The method handles several special cases:
            - Strips leading newlines after </think> or response tokens
            - Handles <response>...</response> tags for non-tool responses
            - Buffers content between <tool_call> and </tool_call> tags
            - Supports multiple sequential tool calls
        """
        self._buffer += delta_text
        cur_text = self._buffer

        start_idx = cur_text.find(self.tool_call_start_token)
        if start_idx == -1:
            self._buffer = ""
            # At least one toolcall has been completed.
            if self.current_tool_id > 0:
                cur_text = ""

            if (
                self.current_tool_id == -1
                and self.newline_token_id is not None
                and all(token_id == self.newline_token_id for token_id in previous_token_ids)
            ):
                cur_text = cur_text.strip("\n")

            # handle <response> </response> when tool_call is not triggered
            content = cur_text
            if self.response_start_token_id is not None and self.response_start_token_id in delta_token_ids:
                content = content.lstrip("\n")
                response_start_idx = content.find(self.response_start_token)
                content = content[response_start_idx + len(self.response_start_token) :]
                response_end_idx = content.rfind(self.response_end_token)
                if response_end_idx != -1:
                    content = content[:response_end_idx]
            elif self.response_end_token_id is not None and self.response_end_token_id in delta_token_ids:
                response_end_idx = content.rfind(self.response_end_token)
                content = content[:response_end_idx]

            # remove \\n after </think> or <response> or </response>
            if (
                len(previous_token_ids) > 0
                and previous_token_ids[-1] in self.parser_token_ids
                and self.newline_token_id is not None
            ) and (len(delta_token_ids) > 0 and delta_token_ids[0] == self.newline_token_id):
                content = content.lstrip("\n")

            return DeltaMessage(content=content if content else None)

        logger.debug("cur_text = %s", cur_text)
        end_idx = cur_text.find(self.tool_call_end_token)
        if end_idx != -1:
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = []

            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

            extracted_tool_calls = self.extract_tool_calls(cur_text[: end_idx + len(self.tool_call_end_token)], request)

            if len(extracted_tool_calls.tool_calls) == 0:
                logger.warning("Failed to extract any tool calls.")
                return None

            tool_call = extracted_tool_calls.tool_calls[0]
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments),
            }
            self.streamed_args_for_tool[self.current_tool_id] = tool_call.function.arguments

            delta = DeltaMessage(
                content=extracted_tool_calls.content,
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        id=tool_call.id,
                        type=tool_call.type,
                        function=DeltaFunctionCall(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
                ],
            )
            self.current_tool_id += 1
            self._buffer = cur_text[end_idx + len(self.tool_call_end_token) :]
            return delta

        # Keep buffering from the tool call start
        self._buffer = cur_text[start_idx:]
        content = cur_text[:start_idx].rstrip("\n")
        return DeltaMessage(content=content if content else None)
