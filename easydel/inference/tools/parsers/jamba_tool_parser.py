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

"""Jamba tool parser for tool calls with XML-style tokens.

This module provides a tool call parser designed for Jamba models that
generate tool calls wrapped in <tool_calls> and </tool_calls> tokens.
The parser handles JSON arrays of function calls within these delimiters.

Example format:
    <tool_calls>[{"name": "func", "arguments": {...}}]</tool_calls>

Features:
    - Token-based boundary detection
    - JSON array parsing with regex fallback
    - Streaming with partial JSON support
    - Automatic special token configuration
    - Validates tokenizer compatibility (rejects MistralTokenizer)
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
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
from ..utils import extract_intermediate_diff

try:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
except ImportError:
    MistralTokenizer = None
logger = get_logger(__name__)


@ToolParserManager.register_module("jamba")
class JambaToolParser(ToolParser):
    """Tool parser for Jamba models.

    Handles tool calls wrapped in <tool_calls> and </tool_calls> tokens.
    Validates tokenizer compatibility (not Mistral) and parses JSON arrays
    of function calls.

    Features:
        - Token-based boundary detection
        - JSON array parsing with regex fallback
        - Streaming with partial JSON support
        - Automatic special token configuration

    Format:
        <tool_calls>[{"name": "func", "arguments": {...}}]</tool_calls>

    Attributes:
        tool_calls_start_token: The opening token for tool calls section.
        tool_calls_end_token: The closing token for tool calls section.
        tool_calls_regex: Compiled regex for extracting tool call content.
        tool_calls_start_token_id: Token ID for the start token.
        tool_calls_end_token_id: Token ID for the end token.

    Raises:
        ValueError: If a MistralTokenizer is detected or tokenizer is None.
        RuntimeError: If start/end tokens cannot be found in vocabulary.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the JambaToolParser.

        Sets up the parser with tool call tokens and validates the tokenizer
        is compatible (not a MistralTokenizer).

        Args:
            tokenizer: The tokenizer associated with the Jamba model.
                Must not be a MistralTokenizer.

        Raises:
            ValueError: If tokenizer is a MistralTokenizer or is None.
            RuntimeError: If tool call start/end tokens are not found
                in the tokenizer vocabulary.
        """
        super().__init__(tokenizer)
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        if isinstance(self.model_tokenizer, MistralTokenizer):
            raise ValueError("Detected a MistralTokenizer tokenizer when using a Jamba model")

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        self.tool_calls_start_token: str = "<tool_calls>"
        self.tool_calls_end_token: str = "</tool_calls>"

        self.tool_calls_regex = re.compile(rf"{self.tool_calls_start_token}(.*?){self.tool_calls_end_token}", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")
        self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)
        if self.tool_calls_start_token_id is None or self.tool_calls_end_token_id is None:
            raise RuntimeError("Jamba Tool parser could not locate tool calls start/end tokens in the tokenizer!")

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Adjust the request settings for Jamba tool parsing.

        Modifies the request to ensure special tokens are not skipped
        when tools are enabled, as the tool_calls tokens are required
        for proper tool call detection.

        Args:
            request: The chat completion request to adjust.

        Returns:
            The modified request with skip_special_tokens set to False
            when tools are enabled and tool_choice is not 'none'.
        """
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model response.

        Parses the model output to identify and extract tool calls wrapped
        in <tool_calls> and </tool_calls> tokens. Expects a JSON array
        of tool call objects within the delimiters.

        Args:
            model_output: The complete text output from the model.
            request: The chat completion request for context.

        Returns:
            ExtractedToolCallInformation containing:
                - tools_called: True if valid tool calls were found
                - tool_calls: List of ToolCall objects extracted
                - content: Text content before the tool calls, or full
                    content if no tools were called
        """
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        else:
            try:
                function_calls = self.tool_calls_regex.findall(model_output)[0]

                raw_function_calls = json.loads(function_calls)
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

                content = model_output[: model_output.find(self.tool_calls_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if (len(content) > 0 and content != " ") else None,
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
        """Extract tool calls incrementally during streaming.

        Processes streaming output token by token, parsing partial JSON
        to extract tool call information as it becomes available. Handles
        multi-tool scenarios and tracks argument deltas.

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
                - content: Regular text content if no tool call detected
                - tool_calls: Tool call deltas if parsing tool calls
            Returns None if more tokens are needed or if the start token
            is the only content in the delta.
        """
        if self.tool_calls_start_token not in current_text:
            return DeltaMessage(content=delta_text)

        if self.tool_calls_start_token_id in delta_token_ids and len(delta_token_ids) == 1:
            return None

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            parsable_arr = current_text.split(self.tool_calls_start_token)[-1].split(self.tool_calls_end_token)[0]

            try:
                tool_call_arr: list[dict] = partial_json_parser.loads(parsable_arr, flags)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug("not enough tokens to parse into JSON yet")
                return None

            current_tool_call: dict = tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}

            if len(tool_call_arr) == 0:
                return None

            elif len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1:
                if self.current_tool_id >= 0:
                    diff: str | None = current_tool_call.get("arguments")

                    if diff:
                        diff = json.dumps(diff, ensure_ascii=False).replace(
                            self.streamed_args_for_tool[self.current_tool_id], ""
                        )
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(arguments=diff).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += diff
                    else:
                        delta = None
                else:
                    delta = None
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("starting on new tool %d", self.current_tool_id)
                return delta

            if not self.current_tool_name_sent:
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
                prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                cur_arguments = current_tool_call.get("arguments")

                new_text = delta_text.replace("'", '"')

                if not cur_arguments and not prev_arguments:
                    delta = None
                elif not cur_arguments and prev_arguments:
                    logger.error("INVARIANT - impossible to have arguments reset mid-arguments")
                    delta = None
                elif cur_arguments and not prev_arguments:
                    cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)
                    logger.debug("finding %s in %s", new_text, cur_arguments_json)

                    arguments_delta = cur_arguments_json[: cur_arguments_json.index(new_text) + len(new_text)]
                    logger.debug("First tokens in arguments received: %s", arguments_delta)
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
                    logger.debug("Searching for diff between \n%s\n%s", cur_args_json, prev_args_json)

                    argument_diff = extract_intermediate_diff(cur_args_json, prev_args_json)
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

            self.prev_tool_call_arr = tool_call_arr
            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug("Skipping chunk as a result of tool streaming extraction error")
            return None
