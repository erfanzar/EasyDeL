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
from ..utils import extract_intermediate_diff

logger = get_logger(__name__)


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
        if "<|action_start|>" not in current_text:
            self.position = len(current_text)
            return DeltaMessage(content=delta_text)
        if self.current_tool_id > 0:
            return DeltaMessage(content="")

        last_pos = self.position
        if "<|action_start|><|plugin|>" not in current_text[last_pos:]:
            return None

        new_delta = current_text[last_pos:]
        text, action = new_delta.split("<|action_start|><|plugin|>")

        if len(text) > 0:
            self.position = self.position + len(text)
            return DeltaMessage(content=text)

        action = action.strip()
        action = action.split("<|action_end|>".strip())[0]

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

        try:
            delta: DeltaMessage | None = None
            parsable_arr = action

            try:
                tool_call_arr: dict = partial_json_parser.loads(parsable_arr, flags)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug("not enough tokens to parse into JSON yet")
                return None

            if not self.current_tool_name_sent:
                function_name = tool_call_arr.get("name")
                if function_name:
                    self.current_tool_id = self.current_tool_id + 1
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
                    self.streamed_args_for_tool.append("")
                else:
                    delta = None
            else:
                prev_arguments = self.get_arguments(self.prev_tool_call_arr[self.current_tool_id])
                cur_arguments = self.get_arguments(tool_call_arr)

                if not cur_arguments and not prev_arguments:
                    delta = None
                elif not cur_arguments and prev_arguments:
                    logger.error("INVARIANT - impossible to have arguments reset mid-arguments")
                    delta = None
                elif cur_arguments and not prev_arguments:
                    cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)

                    arguments_delta = cur_arguments_json[: cur_arguments_json.index(delta_text) + len(delta_text)]
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=arguments_delta).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += arguments_delta
                elif cur_arguments and prev_arguments:
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)

                    argument_diff = extract_intermediate_diff(cur_args_json, prev_args_json)

                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=argument_diff).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff

            tool_call_arr["arguments"] = self.get_arguments(tool_call_arr)
            self.prev_tool_call_arr = [tool_call_arr]
            return delta
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

        Parses the model output to identify and extract tool calls wrapped
        in <|action_start|><|plugin|> and <|action_end|> tokens.

        Args:
            model_output: The complete text output from the model.
            request: The chat completion request containing tools definition.

        Returns:
            ExtractedToolCallInformation containing:
                - tools_called: True if valid tool calls were found
                - tool_calls: List of ToolCall objects extracted
                - content: Text content before the action, or full content
                    if no tools were called
        """
        text = model_output
        tools = request.tools
        if "<|action_start|><|plugin|>" in text:
            text, action = text.split("<|action_start|><|plugin|>")
            action = action.split("<|action_end|>".strip())[0]
            action = action[action.find("{") :]
            action_dict = json.loads(action)
            name, parameters = (
                action_dict["name"],
                json.dumps(action_dict.get("parameters", action_dict.get("arguments", {})), ensure_ascii=False),
            )

            if not tools or name not in [t.function.name for t in tools]:
                ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=text)

            tool_calls = [ToolCall(function=FunctionCall(name=name, arguments=parameters))]
            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=text if len(text) > 0 else None
            )

        return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=text)
