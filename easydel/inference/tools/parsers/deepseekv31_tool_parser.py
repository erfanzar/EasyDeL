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

"""DeepSeek V3.1 tool parser module for parsing tool calls from DeepSeek V3.1 model outputs.

This module provides the DeepSeekV31ToolParser class which handles the updated
tool call format used by DeepSeek V3.1 models. The format is similar to V3 but
uses a simplified structure without the tool type field.

Example tool call format:
    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "Beijing"}<｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from uuid import uuid4

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


@ToolParserManager.register_module("deepseek_v31")
class DeepSeekV31ToolParser(ToolParser):
    """Tool parser for DeepSeek V3.1 models.

    This parser handles the updated tool call format used by DeepSeek V3.1 models.
    Unlike V3, this format directly uses function name and arguments without
    an explicit tool type field.

    The format structure:
    - Tool calls wrapped in <｜tool▁calls▁begin｜> and <｜tool▁calls▁end｜>
    - Individual calls wrapped in <｜tool▁call▁begin｜> and <｜tool▁call▁end｜>
    - Function name and arguments separated by <｜tool▁sep｜>

    Attributes:
        current_tool_name_sent (bool): Tracks if tool name has been sent in streaming.
        prev_tool_call_arr (list[dict]): Previous tool calls for comparison in streaming.
        current_tool_id (int): Index of current tool being processed.
        streamed_args_for_tool (list[str]): Arguments streamed so far for each tool.
        tool_calls_start_token (str): Token marking start of tool calls section.
        tool_calls_end_token (str): Token marking end of tool calls section.
        tool_call_start_token (str): Token marking start of individual tool call.
        tool_call_end_token (str): Token marking end of individual tool call.
        tool_call_regex (re.Pattern): Regex pattern for parsing complete tool calls.
        stream_tool_call_portion_regex (re.Pattern): Regex for parsing streaming portions.
        stream_tool_call_name_regex (re.Pattern): Regex for extracting tool names.
        tool_calls_start_token_id (int | None): Token ID for tool calls start marker.
        tool_calls_end_token_id (int | None): Token ID for tool calls end marker.
        tool_call_start_token_id (int | None): Token ID for individual call start.
        tool_call_end_token_id (int | None): Token ID for individual call end.

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-v3.1")
        >>> parser = DeepSeekV31ToolParser(tokenizer)
        >>> result = parser.extract_tool_calls(model_output, request)
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the DeepSeek V3.1 tool parser.

        Sets up the token markers, regex patterns, and token IDs required for
        parsing DeepSeek V3.1 tool call format.

        Args:
            tokenizer: The tokenizer associated with the DeepSeek V3.1 model.
                Must contain the special tool call tokens in its vocabulary.

        Raises:
            ValueError: If the tokenizer is not provided.
            RuntimeError: If the tool call start/end tokens cannot be found
                in the tokenizer vocabulary.
        """
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        self.tool_calls_start_token: str = "<｜tool▁calls▁begin｜>"
        self.tool_calls_end_token: str = "<｜tool▁calls▁end｜>"

        self.tool_call_start_token: str = "<｜tool▁call▁begin｜>"
        self.tool_call_end_token: str = "<｜tool▁call▁end｜>"

        self.tool_call_regex = re.compile(
            r"<｜tool▁call▁begin｜>(?P<function_name>.*?)<｜tool▁sep｜>(?P<function_arguments>.*?)<｜tool▁call▁end｜>",
            re.DOTALL,
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"(?P<function_name>.*)<｜tool▁sep｜>(?P<function_arguments>.*)",
            re.DOTALL,
        )

        self.stream_tool_call_name_regex = re.compile(r"(?P<function_name>.*)<｜tool▁sep｜>", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")
        self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_calls_start_token_id is None or self.tool_calls_end_token_id is None:
            raise RuntimeError("DeepSeek-V3.1 Tool parser could not locate tool call start/end tokens in the tokenizer!")

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output.

        Parses the DeepSeek V3.1 format to extract function names and arguments
        from the model's response. Uses regex pattern matching to identify
        tool call boundaries and extract the function details.

        Args:
            model_output: Complete text output from the model containing
                potential tool calls in DeepSeek V3.1 format.
            request: Original chat completion request with tool definitions.
                Used for context but not directly accessed in parsing.

        Returns:
            ExtractedToolCallInformation: Contains the following fields:
                - tools_called (bool): True if any tools were invoked.
                - tool_calls (list[ToolCall]): List of parsed tool calls with
                  function names and argument strings.
                - content (str | None): Text content before tool calls, or
                  None if no content precedes the tool calls.

        Example:
            >>> output = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"NYC\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
            >>> result = parser.extract_tool_calls(output, request)
            >>> result.tools_called
            True
        """
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            function_call_tuples = self.tool_call_regex.findall(model_output)

            tool_calls: list[ToolCall] = []
            for function_name, function_args in function_call_tuples:
                tool_calls.append(
                    ToolCall(type="function", function=FunctionCall(name=function_name, arguments=function_args))
                )

            content = model_output[: model_output.find(self.tool_calls_start_token)]
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
        """Extract tool calls from streaming model output.

        Handles incremental parsing of DeepSeek V3.1 tool call format during
        streaming generation. Tracks state across chunks to identify tool
        boundaries and progressively emit argument content.

        The method handles several streaming scenarios:
        - Starting a new tool call (detecting start token)
        - Updating an existing tool call (accumulating arguments)
        - Closing a tool call (detecting end token)
        - Passing through regular content tokens

        Args:
            previous_text: Text generated up to the previous chunk.
            current_text: All text generated so far including current chunk.
            delta_text: New text added in this chunk only.
            previous_token_ids: Sequence of token IDs up to previous chunk.
            current_token_ids: Sequence of all token IDs generated so far.
            delta_token_ids: Sequence of new token IDs in this chunk.
            request: Original chat completion request with tool definitions.

        Returns:
            DeltaMessage | None: A delta message containing:
                - Tool call information (name, arguments) for streaming updates
                - Content text if not in a tool call
                - None if more data is needed before emitting

        Note:
            This method maintains internal state across calls including
            current_tool_id, current_tool_name_sent, prev_tool_call_arr,
            and streamed_args_for_tool.
        """
        logger.debug("delta_text: %s", delta_text)
        logger.debug("delta_token_ids: %s", delta_token_ids)

        if self.tool_calls_start_token_id not in current_token_ids:
            logger.debug("No tool call tokens found!")
            return DeltaMessage(content=delta_text)

        delta_text = delta_text.replace(self.tool_calls_start_token, "").replace(self.tool_calls_end_token, "")

        try:
            prev_tool_start_count = previous_token_ids.count(self.tool_call_start_token_id)
            prev_tool_end_count = previous_token_ids.count(self.tool_call_end_token_id)
            cur_tool_start_count = current_token_ids.count(self.tool_call_start_token_id)
            cur_tool_end_count = current_token_ids.count(self.tool_call_end_token_id)
            tool_call_portion = None
            text_portion = None

            if (
                cur_tool_start_count == cur_tool_end_count
                and prev_tool_end_count == cur_tool_end_count
                and self.tool_call_end_token not in delta_text
            ):
                logger.debug("Generating text content! skipping tool parsing.")
                return DeltaMessage(content=delta_text)

            if self.tool_call_end_token in delta_text:
                logger.debug("tool_call_end_token in delta_text")
                full_text = current_text + delta_text
                tool_call_portion = (
                    full_text.split(self.tool_call_start_token)[-1].split(self.tool_call_end_token)[0].rstrip()
                )
                delta_text = delta_text.split(self.tool_call_end_token)[0].rstrip()
                text_portion = delta_text.split(self.tool_call_end_token)[-1].lstrip()

            # Starting a new tool call
            if cur_tool_start_count > cur_tool_end_count and cur_tool_start_count > prev_tool_start_count:
                if len(delta_token_ids) > 1:
                    tool_call_portion = current_text.split(self.tool_call_start_token)[-1]
                else:
                    tool_call_portion = None
                    delta = None

                text_portion = None

                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("Starting on a new tool %s", self.current_tool_id)

            # Updating existing tool call
            elif cur_tool_start_count > cur_tool_end_count and cur_tool_start_count == prev_tool_start_count:
                tool_call_portion = current_text.split(self.tool_call_start_token)[-1]
                text_portion = None

            # Closing tool call
            elif cur_tool_start_count == cur_tool_end_count and cur_tool_end_count >= prev_tool_end_count:
                if self.prev_tool_call_arr is None or len(self.prev_tool_call_arr) == 0:
                    logger.debug("attempting to close tool call, but no tool call")
                    return None
                diff = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                if diff:
                    if isinstance(diff, str):
                        diff = diff.encode("utf-8").decode("unicode_escape")
                    if '"}' not in delta_text:
                        return None
                    end_loc = delta_text.rindex('"}')
                    diff = delta_text[:end_loc] + '"}'
                    logger.debug("Finishing tool and found diff that had not been streamed yet: %s", diff)
                    self.streamed_args_for_tool[self.current_tool_id] += diff
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=diff).model_dump(exclude_none=True),
                            )
                        ]
                    )

            # Otherwise, regular content tokens (strip tool call tokens if they leaked)
            else:
                text = delta_text.replace(self.tool_call_start_token, "")
                text = text.replace(self.tool_call_end_token, "")
                return DeltaMessage(tool_calls=[], content=text)

            current_tool_call: dict[str, str] = {}
            if tool_call_portion:
                matches = self.stream_tool_call_portion_regex.match(tool_call_portion)
                if matches:
                    tool_name, tool_args = matches.groups()
                    current_tool_call["name"] = tool_name
                    current_tool_call["arguments"] = tool_args
                else:
                    name_matches = self.stream_tool_call_name_regex.match(tool_call_portion)
                    if name_matches:
                        (tool_name,) = name_matches.groups()
                        current_tool_call["name"] = tool_name
                        current_tool_call["arguments"] = ""
                    else:
                        logger.debug("Not enough token")
                        return None

            # Send tool name first
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:
                    self.current_tool_name_sent = True
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=f"chatcmpl-tool-{uuid4()}",
                                function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True),
                            )
                        ]
                    )
                return None

            # If we don't have tool portion yet, pass through any text after tool closure
            if tool_call_portion is None:
                return DeltaMessage(content=delta_text) if text_portion is not None else None

            logger.debug("Trying to parse current tool call with ID %s", self.current_tool_id)

            if len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
            cur_arguments = current_tool_call.get("arguments")

            if not cur_arguments and not prev_arguments:
                delta = None
            elif not cur_arguments and prev_arguments:
                logger.error("should be impossible to have arguments reset mid-call. skipping streaming anything.")
                delta = None
            elif cur_arguments and not prev_arguments:
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(arguments=cur_arguments).model_dump(exclude_none=True),
                        )
                    ]
                )
                self.streamed_args_for_tool[self.current_tool_id] = cur_arguments
            elif cur_arguments and prev_arguments:
                if (
                    isinstance(delta_text, str)
                    and cur_arguments != prev_arguments
                    and len(cur_arguments) > len(prev_arguments)
                    and cur_arguments.startswith(prev_arguments)
                ):
                    delta_arguments = cur_arguments[len(prev_arguments) :]
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=delta_arguments).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] = cur_arguments
                else:
                    delta = None

            if self.current_tool_id == len(self.prev_tool_call_arr) - 1:
                self.prev_tool_call_arr[self.current_tool_id] = current_tool_call
            else:
                self.prev_tool_call_arr.append(current_tool_call)

            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None
