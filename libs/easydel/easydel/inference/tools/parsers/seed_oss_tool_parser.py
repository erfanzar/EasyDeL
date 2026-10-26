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

"""Tool parser implementation for Seed OSS models.

This module provides a tool parser specifically designed for Seed OSS model outputs.
Seed OSS uses an XML-style format with seed: namespace prefixes for tool calls,
similar to Qwen3Coder but with different tag names.

The parser supports thinking tags for chain-of-thought reasoning and handles
type-aware parameter conversion based on tool definitions.

Example format:
    <seed:think>reasoning content</seed:think>
    <seed:tool_call>
    <function=function_name>
    <parameter=param_name>value</parameter>
    </function>
    </seed:tool_call>
"""

from __future__ import annotations

import json
import re
import uuid
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
    ToolDefinition,
)
from ..abstract_tool import ToolParser, ToolParserManager
from ._schema_params import convert_param_value, get_arguments_config

logger = get_logger(__name__)


@ToolParserManager.register_module("seed_oss")  # pyright: ignore[reportUntypedClassDecorator]
class SeedOssToolParser(ToolParser):
    """Tool parser for Seed OSS model outputs.

    This parser handles the extraction of function/tool calls from Seed OSS model
    outputs. Seed OSS uses an XML-style format with seed: namespace prefixes,
    supporting thinking sections and type-aware parameter conversion.

    The parser handles several XML structures:
    - <seed:think>...</seed:think> for reasoning content
    - <seed:tool_call>...</seed:tool_call> wrapper for tool calls
    - <function=name>...</function> for function definitions
    - <parameter=name>value</parameter> for parameters

    Attributes:
        TOOL_CALL_START: Class constant for tool call start tag.
        TOOL_CALL_END: Class constant for tool call end tag.
        prev_tool_call_arr: List of previously parsed tool call dictionaries.
        tool_call_start_token: Token marking start of tool call.
        tool_call_end_token: Token marking end of tool call.
        tool_call_prefix: Prefix for function declarations.
        function_end_token: Token marking end of function.
        parameter_prefix: Prefix for parameter declarations.
        parameter_end_token: Token marking end of parameter.
        think_start_token: Token marking start of thinking section.
        think_end_token: Token marking end of thinking section.
        is_tool_call_started: Flag indicating tool call parsing has started.
        is_thinking_end: Flag indicating thinking section has ended.
        failed_count: Counter for parsing failures.
        tool_call_start_token_id: Token ID for tool call start.
        tool_call_end_token_id: Token ID for tool call end.
        think_end_token_id: Token ID for think end.
        tool_call_complete_regex: Regex for complete tool calls.
        tool_call_regex: Regex for partial/complete tool calls.
        tool_call_function_regex: Regex for function extraction.
        tool_call_parameter_regex: Regex for parameter extraction.

    Raises:
        RuntimeError: If the tokenizer doesn't contain required tool call tokens.

    Example:
        >>> parser = SeedOssToolParser(tokenizer)
        >>> result = parser.extract_tool_calls(model_output, request)
        >>> if result.tools_called:
        ...     for tool_call in result.tool_calls:
        ...         print(f"Function: {tool_call.function.name}")
    """

    TOOL_CALL_START = "<seed:tool_call>"
    TOOL_CALL_END = "</seed:tool_call>"

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the Seed OSS tool parser.

        Sets up token mappings, regex patterns, and streaming state for parsing
        Seed OSS format tool calls. Validates that required tokens exist in the
        tokenizer vocabulary.

        Args:
            tokenizer: A HuggingFace tokenizer instance used for token processing.
                This tokenizer should be compatible with the Seed OSS model and
                must contain <seed:tool_call> and </seed:tool_call> tokens.

        Raises:
            RuntimeError: If the tokenizer doesn't contain the required
                <seed:tool_call> or </seed:tool_call> tokens.
        """
        super().__init__(tokenizer)

        self.prev_tool_call_arr: list[dict] = []

        self.tool_call_start_token: str = self.TOOL_CALL_START
        self.tool_call_end_token: str = self.TOOL_CALL_END
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"
        self.think_start_token: str = "<seed:think>"
        self.think_end_token: str = "</seed:think>"
        self.is_tool_call_started: bool = False
        self.is_thinking_end: bool = False
        self.failed_count: int = 0
        self._reset_streaming_state()

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        self.think_end_token_id = self.vocab.get(self.think_end_token)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError("Seed_Oss XML parser: tokenizer did not include <seed:tool_call> or its closing tag.")

        tool_start_re = re.escape(self.tool_call_start_token)
        tool_end_re = re.escape(self.tool_call_end_token)

        self.tool_call_complete_regex = re.compile(rf"{tool_start_re}(.*?){tool_end_re}", re.DOTALL)
        self.tool_call_regex = re.compile(rf"{tool_start_re}(.*?){tool_end_re}|{tool_start_re}(.*?)$", re.DOTALL)

        self.tool_call_function_regex = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
        self.tool_call_parameter_regex = re.compile(r"<parameter=(.*?)</parameter>|<parameter=(.*?)$", re.DOTALL)

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID.

        Creates a unique identifier for a tool call using UUID4, truncated
        to 24 characters for compatibility.

        Returns:
            str: A unique tool call ID in the format "call_<24-char-hex>".
        """
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self):
        """Reset all streaming state variables.

        Clears all state variables used during streaming extraction to prepare
        for a new streaming session or after completing a tool call.
        """
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = -1
        self.current_function_name = None
        self.current_param_name = None
        self.param_count = 0
        self.in_function = False
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool = []
        self._stream_calls = []
        self._stream_content_sent = 0
        self.is_thinking_end = False

    def _parse_xml_function_call(
        self,
        function_call_str: str,
        tools: list[ToolDefinition] | None,
    ) -> ToolCall | None:
        """Parse an XML-formatted function call string into a ToolCall object.

        Extracts the function name and parameters from the XML format, performing
        type conversion based on the tool definitions when available.

        Args:
            function_call_str: The XML string containing the function call,
                starting after "<function=" and including the function name
                and parameter tags.
            tools: Optional list of tool definitions for type-aware parameter
                conversion. If provided, parameters will be converted to their
                declared types (int, float, bool, object, etc.).

        Returns:
            ToolCall or None: A ToolCall object with the parsed function name
                and JSON-encoded arguments, or None if parsing fails.

        Note:
            Parameter type conversion supports:
            - string/str/text/varchar/char/enum: kept as string
            - int/uint/long/short/unsigned: converted to integer
            - num/float: converted to float (or int if whole number)
            - boolean/bool/binary: converted to bool (true/false)
            - object/dict: parsed as JSON, falls back to ast.literal_eval
        """

        end_index = function_call_str.index(">")
        function_name = function_call_str[:end_index]
        param_config = get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        for match in self.tool_call_parameter_regex.findall(parameters):
            match_text = match[0] if match[0] else match[1]
            idx = match_text.index(">")
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1 :])
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = convert_param_value(param_value, param_name, param_config, function_name)
        return ToolCall(
            type="function",
            function=FunctionCall(name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False)),
        )

    def _get_function_calls(self, model_output: str) -> list[str]:
        """Extract raw function call strings from model output.

        Parses the model output to find all function declarations within
        tool call tags.

        Args:
            model_output: The model output containing tool calls.

        Returns:
            list[str]: List of raw function call strings, each starting
                after "<function=" and containing the function content.
        """
        matched_ranges = self.tool_call_regex.findall(model_output)
        raw_tool_calls = [match[0] if match[0] else match[1] for match in matched_ranges]

        if len(raw_tool_calls) == 0:
            raw_tool_calls = [model_output]

        raw_function_calls = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(self.tool_call_function_regex.findall(tool_call))

        function_calls = [match[0] if match[0] else match[1] for match in raw_function_calls]
        return function_calls

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model output.

        Parses the model output to find and extract function calls in Seed OSS
        XML format. Handles thinking sections by preserving them in the content.

        Args:
            model_output: The complete text output from the model to parse.
            request: The chat completion request containing tool definitions
                for type-aware parameter conversion.

        Returns:
            ExtractedToolCallInformation: An object containing:
                - tools_called: True if valid tool calls were found, False otherwise.
                - tool_calls: List of ToolCall objects representing parsed function calls.
                - content: Thinking content plus any text before tool calls, or the
                    full output if no valid tool calls were found.
        """
        if self.tool_call_prefix not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        result_content = model_output
        thinking_content = ""
        if self.think_start_token in model_output and self.think_end_token in model_output:
            think_end_index = model_output.find(self.think_end_token) + len(self.think_end_token)
            result_content = model_output[think_end_index:]
            thinking_content = model_output[:think_end_index]

        try:
            function_calls = self._get_function_calls(result_content)
            if len(function_calls) == 0:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            tool_calls = [
                self._parse_xml_function_call(function_call_str, request.tools) for function_call_str in function_calls
            ]

            self.prev_tool_call_arr.clear()
            for tool_call in tool_calls:
                if tool_call:
                    self.prev_tool_call_arr.append(
                        {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    )

            tool_call_start_index = result_content.find(self.tool_call_start_token)
            tool_call_start_index = (
                tool_call_start_index if tool_call_start_index >= 0 else result_content.find(self.tool_call_prefix)
            )
            content = thinking_content + result_content[:tool_call_start_index]

            return ExtractedToolCallInformation(
                tools_called=(len(tool_calls) > 0),
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    def flush_streaming_content(self) -> str:
        """Release a withheld, unfinished opening marker when no tool call followed."""
        if self.is_tool_call_started:
            return ""
        pending = self.accumulated_text[self._stream_content_sent :]
        self._stream_content_sent = len(self.accumulated_text)
        return pending

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
        tool call information. Handles the XML format with nested function and
        parameter tags, emitting JSON fragments as parameters are parsed.

        Args:
            previous_text: The accumulated text from all previous tokens.
            current_text: The complete text including the current delta.
            delta_text: The new text added in this streaming step.
            previous_token_ids: Token IDs from all previous generation steps.
            current_token_ids: All token IDs including the current step.
            delta_token_ids: The new token IDs added in this step.
            request: The chat completion request containing tool definitions.

        Returns:
            DeltaMessage or None: A delta message containing either:
                - Content text if in thinking section or no tool call detected.
                - Tool call name when function header is parsed.
                - JSON argument fragments as parameters are extracted.
                - None if waiting for more tokens.

        Note:
            The streaming output builds valid JSON incrementally:
            1. Emits "{" when entering function body
            2. Emits '"param_name": "value"' for each parameter
            3. Emits "}" when function ends
        """
        if not previous_text:
            self._reset_streaming_state()
        self.accumulated_text = current_text
        # Thinking is optional. Ignore tool-shaped text inside an actual thinking
        # section, but never require an end marker in a tool-only response.
        think_end = current_text.rfind(self.think_end_token)
        think_start = current_text.rfind(self.think_start_token)
        self.is_thinking_end = think_end >= 0
        result_start = think_end + len(self.think_end_token) if think_end >= 0 else 0
        active_thinking = think_start > think_end
        result_text = current_text[result_start:]
        tool_start = -1 if active_thinking else result_text.find(self.tool_call_start_token)
        visible = current_text if tool_start < 0 else current_text[: result_start + tool_start]
        if tool_start < 0 and not active_thinking:
            # Hold an incomplete opening marker instead of leaking XML to content.
            for size in range(min(len(visible), len(self.tool_call_start_token) - 1), 0, -1):
                if visible.endswith(self.tool_call_start_token[:size]):
                    visible = visible[:-size]
                    break
        content = visible[self._stream_content_sent :]
        self._stream_content_sent = len(visible)
        if tool_start < 0:
            return DeltaMessage(content=content) if content else None

        self.is_tool_call_started = True
        deltas = []
        functions = []
        for envelope in self.tool_call_regex.findall(result_text):
            body = envelope[0] or envelope[1]
            functions.extend(re.finditer(r"<function=([^>]+)>(.*?)(</function>|$)", body, re.DOTALL))
        for index, match in enumerate(functions):
            name, body, end = match.groups()
            first = index >= len(self._stream_calls)
            if first:
                self._stream_calls.append({"id": self._generate_tool_call_id(), "count": 0, "closed": False})
                self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr.append({"name": name, "arguments": "{}"})
            state = self._stream_calls[index]
            fragment = "{" if first else ""
            parameters = list(re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", body, re.DOTALL))
            config = get_arguments_config(name, request.tools if request else None)
            for parameter in parameters[state["count"] :]:
                key, value = parameter.groups()
                if value.startswith("\n"):
                    value = value[1:]
                if value.endswith("\n"):
                    value = value[:-1]
                converted = convert_param_value(value, key, config, name)
                fragment += (", " if state["count"] else "") + json.dumps(key, ensure_ascii=False)
                fragment += ": " + json.dumps(converted, ensure_ascii=False)
                state["count"] += 1
            if end and not state["closed"]:
                fragment += "}"
                state["closed"] = True
            if first or fragment:
                self.streamed_args_for_tool[index] += fragment
                deltas.append(
                    DeltaToolCall(
                        index=index,
                        id=state["id"] if first else None,
                        type="function" if first else None,
                        function=DeltaFunctionCall(name=name if first else None, arguments=fragment or None),
                    )
                )
            if end:
                self.prev_tool_call_arr[index]["arguments"] = self.streamed_args_for_tool[index]
            self.current_tool_index = index
            self.current_function_name = name
            self.current_tool_id = state["id"]
            self.header_sent = True
            self.json_started = True
            self.json_closed = state["closed"]
            self.in_function = not state["closed"]
            self.param_count = state["count"]
        return DeltaMessage(content=content or None, tool_calls=deltas or None) if content or deltas else None
