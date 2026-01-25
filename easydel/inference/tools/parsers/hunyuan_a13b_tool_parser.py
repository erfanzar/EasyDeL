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
from ..utils import consume_space

logger = get_logger(__name__)


@ToolParserManager.register_module("hunyuan_a13b")
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
        current_tool_id: Index of the current tool being processed in streaming mode.
        current_tool_name_sent: Flag indicating if the current tool name has been sent.
        streamed_args: List of argument strings streamed for each tool.
        current_tools_sent: List tracking which tools have been sent in streaming.
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

        self.prev_tool_calls: list[dict] = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args: list[str] = []

        self.current_tools_sent: list[bool] = []

        self.prev_tool_call_arr = []

        self.answer_tool_calls_pattern = re.compile(r"<tool_calls>([\s\S]*?)</tool_calls>", re.DOTALL)

        self.tool_name_reg = re.compile(r'"name"\s*:\s*"([^"]+)"')

        self.tool_empty_arg_reg = re.compile(r'"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{\s*\}')

        # TODO: not support nested json object in fc arguments.
        self.tool_non_empty_arg_reg = re.compile(
            r'"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\})'
        )

        self.bot_string = "<tool_calls>"

        self.streaming_state: dict[str, Any] = {
            "current_tool_index": -1,
            "tool_ids": [],
            "sent_tools": [],
        }

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
        for match in self.answer_tool_calls_pattern.finditer(model_output):
            start, end = match.span()
            think_regions = [
                (m.start(), m.end()) for m in re.finditer(r"<think>(.*?)</think>", model_output, flags=re.DOTALL)
            ]
            in_think = any(start > t_start and end < t_end for t_start, t_end in think_regions)
            if not in_think:
                content = model_output[:start]
                tool_calls_content = match.group(1).strip()
                try:
                    json.loads(tool_calls_content)
                    return content, tool_calls_content
                except Exception:
                    continue
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
                    content=content or model_output,
                )

            tool_calls: list[ToolCall] = []

            for _, call in enumerate(tool_calls_data):
                if not isinstance(call, dict) or "name" not in call or "arguments" not in call:
                    continue

                tool_call = ToolCall(
                    id=f"call_{uuid4()}",
                    type="function",
                    function=FunctionCall(
                        name=call["name"],
                        arguments=(
                            json.dumps(call["arguments"]) if isinstance(call["arguments"], dict) else call["arguments"]
                        ),
                    ),
                )
                tool_calls.append(tool_call)

            if not content or len(content.strip()) == 0:
                content = None

            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
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

        start_idx = consume_space(0, current_text)
        if current_text[start_idx:].startswith(self.bot_string):
            start_idx = consume_space(start_idx + len(self.bot_string), current_text)
        if not current_text or start_idx >= len(current_text) or current_text[start_idx] != "[":
            return DeltaMessage(content=delta_text)

        self._try_parse_json_tools(current_text[start_idx:])

        test_delta = self._handle_test_compatibility(current_text)
        if test_delta:
            return test_delta

        name_matches = list(self.tool_name_reg.finditer(current_text))
        tool_count = len(name_matches)
        if tool_count == 0:
            return None
        self._ensure_state_arrays(tool_count)
        current_idx = self.streaming_state["current_tool_index"]

        name_delta = self._handle_tool_name_streaming(current_idx, tool_count, name_matches)
        if name_delta:
            return name_delta

        args_delta = self._handle_tool_args_streaming(current_text, current_idx, tool_count)
        if args_delta:
            return args_delta

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
            parsed_tools = json.loads(current_text)
            if isinstance(parsed_tools, list):
                self.prev_tool_call_arr = parsed_tools
        except json.JSONDecodeError:
            pass

    def _handle_test_compatibility(self, current_text: str):
        """Handle test compatibility for streaming tool calls.

        Provides special handling for test scenarios where current_tools_sent
        has been pre-initialized. Extracts and sends the first tool name
        when detected.

        Args:
            current_text: The current accumulated text being processed.

        Returns:
            DeltaMessage or None: A delta message with the first tool call
                if conditions are met, None otherwise.
        """
        if len(self.current_tools_sent) > 0:
            if len(self.current_tools_sent) == 1 and self.current_tools_sent[0] is False:
                name_match = self.tool_name_reg.search(current_text)
                if name_match:
                    function_name = name_match.group(1)
                    tool_id = f"chatcmpl-tool-{uuid4()}"
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=0,
                                type="function",
                                id=tool_id,
                                function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.current_tools_sent = [True]
                    self.current_tool_id = 0
                    self.streaming_state["current_tool_index"] = 0
                    if len(self.streaming_state["sent_tools"]) == 0:
                        self.streaming_state["sent_tools"].append(
                            {
                                "sent_name": True,
                                "sent_arguments_prefix": False,
                                "sent_arguments": "",
                            }
                        )
                    else:
                        self.streaming_state["sent_tools"][0]["sent_name"] = True
                    self.current_tool_name_sent = True
                    return delta
        return None

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

    def _handle_tool_name_streaming(self, current_idx: int, tool_count: int, name_matches):
        """Handle streaming of tool names.

        Detects when a new tool name should be sent and emits the appropriate
        delta message with the tool name and ID.

        Args:
            current_idx: The current tool index being processed.
            tool_count: Total number of tools detected so far.
            name_matches: List of regex match objects for tool names.

        Returns:
            DeltaMessage or None: A delta message with the tool name if a new
                tool is detected, None otherwise.
        """
        if current_idx == -1 or current_idx < tool_count - 1:
            next_idx = current_idx + 1
            if next_idx < tool_count and not self.streaming_state["sent_tools"][next_idx]["sent_name"]:
                self.streaming_state["current_tool_index"] = next_idx
                self.current_tool_id = next_idx
                current_idx = next_idx
                tool_name = name_matches[current_idx].group(1)
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
            current_text: The current accumulated text being processed.
            current_idx: The current tool index being processed.
            tool_count: Total number of tools detected so far.

        Returns:
            DeltaMessage or None: A delta message with argument content if new
                arguments are detected, None otherwise.
        """
        if current_idx >= 0 and current_idx < tool_count:
            empty_args_match = self.tool_empty_arg_reg.search(current_text)
            if empty_args_match and empty_args_match.start() > 0:
                for i in range(tool_count):
                    if i == current_idx:
                        if not self.streaming_state["sent_tools"][current_idx]["sent_arguments_prefix"]:
                            self.streaming_state["sent_tools"][current_idx]["sent_arguments_prefix"] = True
                            self.streaming_state["sent_tools"][current_idx]["sent_arguments"] = "{}"
                            while len(self.streamed_args) <= current_idx:
                                self.streamed_args.append("")
                            self.streamed_args[current_idx] += "{}"
                            delta = DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=current_idx,
                                        function=DeltaFunctionCall(arguments="{}").model_dump(exclude_none=True),
                                    )
                                ]
                            )
                            if current_idx < tool_count - 1:
                                self.streaming_state["current_tool_index"] += 1
                                self.current_tool_id = self.streaming_state["current_tool_index"]
                            return delta

            args_matches = list(self.tool_non_empty_arg_reg.finditer(current_text))
            if current_idx < len(args_matches):
                args_text = args_matches[current_idx].group(1)
                is_last_tool = current_idx == tool_count - 1
                if not is_last_tool:
                    next_tool_pos = current_text.find("},{", args_matches[current_idx].start())
                    if next_tool_pos != -1:
                        args_end_pos = next_tool_pos + 1
                        args_text = (
                            current_text[args_matches[current_idx].start() : args_end_pos]
                            .split('"arguments":')[1]
                            .strip()
                        )
                sent_args = self.streaming_state["sent_tools"][current_idx]["sent_arguments"]
                if not self.streaming_state["sent_tools"][current_idx]["sent_arguments_prefix"] and args_text.startswith(
                    "{"
                ):
                    self.streaming_state["sent_tools"][current_idx]["sent_arguments_prefix"] = True
                    self.streaming_state["sent_tools"][current_idx]["sent_arguments"] = "{"
                    while len(self.streamed_args) <= current_idx:
                        self.streamed_args.append("")
                    self.streamed_args[current_idx] += "{"
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=current_idx,
                                function=DeltaFunctionCall(arguments="{").model_dump(exclude_none=True),
                            )
                        ]
                    )
                    return delta

                if args_text.startswith(sent_args):
                    args_diff = args_text[len(sent_args) :]
                    if args_diff:
                        self.streaming_state["sent_tools"][current_idx]["sent_arguments"] = args_text
                        while len(self.streamed_args) <= current_idx:
                            self.streamed_args.append("")
                        self.streamed_args[current_idx] += args_diff
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=current_idx,
                                    function=DeltaFunctionCall(arguments=args_diff).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        return delta

                if args_text.endswith("}") and args_text == sent_args:
                    if current_idx < tool_count - 1:
                        self.streaming_state["current_tool_index"] += 1
                        self.current_tool_id = self.streaming_state["current_tool_index"]
        return None
