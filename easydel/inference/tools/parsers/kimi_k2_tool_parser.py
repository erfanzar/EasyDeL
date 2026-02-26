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

"""Tool parser implementation for Kimi K2 models.

This module provides a tool parser specifically designed for Kimi K2 model outputs.
Kimi K2 uses a hierarchical token-based structure with section and individual call
delimiters, and encodes function identity in a namespace.function:id format.

The parser handles nested token structures with distinct section-level and
call-level markers, making it suitable for complex multi-tool scenarios.

Example format:
    <|tool_calls_section_begin|>
    <|tool_call_begin|>namespace.function_name:123<|tool_call_argument_begin|>{"arg": "value"}<|tool_call_end|>
    <|tool_calls_section_end|>
"""

from __future__ import annotations

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


@ToolParserManager.register_module(["kimi_k2"])  # pyright: ignore[reportUntypedClassDecorator]
class KimiK2ToolParser(ToolParser):
    """Tool parser for Kimi K2 model outputs.

    This parser handles the extraction of function/tool calls from Kimi K2 model
    outputs. Kimi K2 uses a hierarchical token structure with section-level and
    call-level delimiters, and encodes tool identity in namespace.function:id format.

    The parser handles two levels of nesting:
    - Section level: <|tool_calls_section_begin|> ... <|tool_calls_section_end|>
    - Call level: <|tool_call_begin|> ... <|tool_call_end|>

    Tool IDs follow the format: namespace.function_name:numeric_id

    Attributes:
        current_tool_name_sent: Flag indicating if the current tool name has been sent.
        prev_tool_call_arr: List of previously parsed tool call dictionaries.
        current_tool_id: Index of the current tool being processed (-1 means none).
        streamed_args_for_tool: List of argument strings streamed for each tool.
        tool_calls_start_token: Section start marker.
        tool_calls_end_token: Section end marker.
        tool_call_start_token: Individual call start marker.
        tool_call_end_token: Individual call end marker.
        tool_call_regex: Regex for extracting complete tool calls.
        stream_tool_call_portion_regex: Regex for partial tool call with arguments.
        stream_tool_call_name_regex: Regex for tool name/id extraction.
        tool_calls_start_token_id: Token ID for section start.
        tool_calls_end_token_id: Token ID for section end.
        tool_call_start_token_id: Token ID for call start.
        tool_call_end_token_id: Token ID for call end.

    Raises:
        ValueError: If the tokenizer is not provided.
        RuntimeError: If required tool call tokens are not in the tokenizer.

    Example:
        >>> parser = KimiK2ToolParser(tokenizer)
        >>> result = parser.extract_tool_calls(model_output, request)
        >>> if result.tools_called:
        ...     for tool_call in result.tool_calls:
        ...         print(f"Function: {tool_call.function.name}")
        ...         print(f"ID: {tool_call.id}")
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the Kimi K2 tool parser.

        Sets up token mappings and regex patterns for parsing Kimi K2 format
        tool calls. Validates that required section and call delimiter tokens
        exist in the tokenizer vocabulary.

        Args:
            tokenizer: A HuggingFace tokenizer instance used for token processing.
                This tokenizer should be compatible with the Kimi K2 model and
                must contain the section and call delimiter tokens.

        Raises:
            ValueError: If the tokenizer is not provided (is None or falsy).
            RuntimeError: If the tokenizer doesn't contain required tool call
                section start/end tokens.
        """
        super().__init__(tokenizer)
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] | None = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        self.tool_calls_start_token: str = "<|tool_calls_section_begin|>"
        self.tool_calls_end_token: str = "<|tool_calls_section_end|>"

        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"

        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>.+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*?)\s*<\|tool_call_end\|>"
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"(?P<tool_call_id>.+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*)"
        )

        self.stream_tool_call_name_regex = re.compile(r"(?P<tool_call_id>.+:\d+)\s*")

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")
        self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_calls_start_token_id is None or self.tool_calls_end_token_id is None:
            raise RuntimeError("Kimi-K2 Tool parser could not locate tool call start/end tokens in the tokenizer!")

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model output.

        Parses the model output to find and extract function calls in Kimi K2 format.
        Extracts tool ID (in namespace.function:id format) and JSON arguments from
        each tool call block.

        Args:
            model_output: The complete text output from the model to parse.
            request: The chat completion request that triggered this response.

        Returns:
            ExtractedToolCallInformation: An object containing:
                - tools_called: True if valid tool calls were found, False otherwise.
                - tool_calls: List of ToolCall objects with id, function name, and arguments.
                - content: Any text content before the tool calls section, or the full
                    output if no valid tool calls were found.

        Note:
            The function name is extracted from the tool_call_id by parsing the
            namespace.function:id format and extracting the function portion.
        """
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        else:
            try:
                function_call_tuples = self.tool_call_regex.findall(model_output)

                logger.debug("function_call_tuples: %s", function_call_tuples)

                tool_calls = []
                for match in function_call_tuples:
                    function_id, function_args = match
                    function_name = function_id.split(".")[1].split(":")[0]
                    tool_calls.append(
                        ToolCall(
                            id=function_id,
                            type="function",
                            function=FunctionCall(name=function_name, arguments=function_args),
                        )
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
        """Extract tool calls incrementally during streaming generation.

        Processes tokens as they are generated to progressively extract and emit
        tool call information. Handles the hierarchical Kimi K2 format with
        section and call level delimiters.

        The method tracks the number of tool call start/end tokens to determine
        when new tools begin and when arguments should be streamed.

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
                - Tool call name/id when a new tool starts.
                - Argument deltas as JSON content is streamed.
                - None if waiting for more tokens or parsing state is ambiguous.

        Note:
            The streaming logic handles several state transitions:
            - Detecting entry into tool calls section
            - Recognizing new tool calls by counting start/end tokens
            - Incrementally streaming arguments as they appear
            - Properly closing tool calls when end tokens are seen
        """
        logger.debug("delta_text: %s", delta_text)
        logger.debug("delta_token_ids: %s", delta_token_ids)
        if self.tool_calls_start_token_id not in current_token_ids:
            logger.debug("No tool call tokens found!")
            return DeltaMessage(content=delta_text)
        delta_text = delta_text.replace(self.tool_calls_start_token, "").replace(self.tool_calls_end_token, "")
        try:
            delta: DeltaMessage | None = None
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

            elif cur_tool_start_count > cur_tool_end_count and cur_tool_start_count == prev_tool_start_count:
                tool_call_portion = current_text.split(self.tool_call_start_token)[-1]
                text_portion = None

            elif cur_tool_start_count == cur_tool_end_count and cur_tool_end_count >= prev_tool_end_count:
                if self.prev_tool_call_arr is None or len(self.prev_tool_call_arr) == 0:
                    logger.debug("attempting to close tool call, but no tool call")
                    return None
                diff = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                if diff:
                    diff = diff.encode("utf-8").decode("unicode_escape") if diff is str else diff
                    if '"}' not in delta_text:
                        return None
                    end_loc = delta_text.rindex('"}')
                    diff = delta_text[:end_loc] + '"}'
                    logger.debug(
                        "Finishing tool and found diff that had not been streamed yet: %s",
                        diff,
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += diff
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=diff).model_dump(exclude_none=True),
                            )
                        ]
                    )

            else:
                text = delta_text.replace(self.tool_call_start_token, "")
                text = text.replace(self.tool_call_end_token, "")
                delta = DeltaMessage(tool_calls=[], content=text)
                return delta

            current_tool_call: dict | None = None
            if tool_call_portion:
                current_tool_call = dict()
                current_tool_call_matches = self.stream_tool_call_portion_regex.match(tool_call_portion)
                if current_tool_call_matches:
                    tool_id, tool_args = current_tool_call_matches.groups()
                    tool_name = tool_id.split(".")[1].split(":")[0]
                    current_tool_call["id"] = tool_id
                    current_tool_call["name"] = tool_name
                    current_tool_call["arguments"] = tool_args
                else:
                    current_tool_call_name_matches = self.stream_tool_call_name_regex.match(tool_call_portion)
                    if current_tool_call_name_matches:
                        (tool_id_str,) = current_tool_call_name_matches.groups()
                        tool_name = tool_id_str.split(".")[1].split(":")[0]
                        current_tool_call["id"] = tool_id_str
                        current_tool_call["name"] = tool_name
                        current_tool_call["arguments"] = ""
                    else:
                        logger.debug("Not enough token")
                        return None

            if not self.current_tool_name_sent:
                if current_tool_call is None:
                    return None
                function_name: str | None = current_tool_call.get("name")
                tool_id = current_tool_call.get("id")
                if function_name:
                    self.current_tool_name_sent = True
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=tool_id,
                                function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True),
                            )
                        ]
                    )
                else:
                    return None

            if tool_call_portion is None:
                delta = DeltaMessage(content=delta_text) if text_portion is not None else None
                return delta

            logger.debug("Trying to parse current tool call with ID %s", self.current_tool_id)

            assert self.prev_tool_call_arr is not None

            if len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
            cur_arguments = current_tool_call.get("arguments")

            logger.debug("diffing old arguments: %s", prev_arguments)
            logger.debug("against new ones: %s", cur_arguments)

            if not cur_arguments and not prev_arguments:
                logger.debug("Skipping text %s - no arguments", delta_text)
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
                    logger.debug("got diff %s", delta_text)

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
