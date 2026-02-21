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

"""DeepSeek V3.2 tool parser module for parsing tool calls from DeepSeek V3.2 model outputs.

This module provides the DeepSeekV32ToolParser class which handles the DSML
(DeepSeek Markup Language) format used by DeepSeek V3.2 models. This format
uses XML-like tags with explicit parameter typing.

Example tool call format:
    <｜DSML｜function_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>
    <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

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


@ToolParserManager.register_module("deepseek_v32")
class DeepSeekV32ToolParser(ToolParser):
    """Tool parser for DeepSeek V3.2 models using DSML format.

    This parser handles the DSML (DeepSeek Markup Language) tool call format
    which uses XML-like tags with explicit parameter typing. The format provides
    structured representation of function calls with typed parameters.

    Format structure:
        <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param" string="true">value</｜DSML｜parameter>
        </｜DSML｜invoke>
        </｜DSML｜function_calls>

    Attributes:
        prev_tool_call_arr (list[dict]): Previous tool calls for streaming state.
        dsml_token (str): Base DSML token identifier.
        dsml_start_check (str): Start pattern for DSML detection.
        tool_call_start_token (str): Token marking function_calls start.
        tool_call_end_token (str): Token marking function_calls end.
        invoke_start_prefix (str): Prefix for invoke tag start.
        invoke_end_token (str): Token marking invoke end.
        parameter_prefix (str): Prefix for parameter tag.
        parameter_end_token (str): Token marking parameter end.
        current_tool_name_sent (bool): Whether current tool name was sent.
        current_tool_id (str | None): Current tool call ID.
        streamed_args_for_tool (list[str]): Streamed arguments per tool.
        is_tool_call_started (bool): Whether tool call parsing has started.
        failed_count (int): Count of failed parsing attempts.
        current_tool_index (int): Index of current tool being processed.
        invoke_index (int): Index of current invoke block.
        header_sent (bool): Whether the tool header was sent.
        current_function_name (str | None): Name of current function.
        current_param_name (str | None): Name of current parameter.
        current_param_value (str): Value of current parameter.
        param_count (int): Count of parameters processed.
        in_param (bool): Whether currently inside a parameter tag.
        in_function (bool): Whether currently inside a function invoke.
        json_started (bool): Whether JSON output has started.
        json_closed (bool): Whether JSON output has closed.
        accumulated_params (dict): Accumulated parameter key-value pairs.
        streaming_request (ChatCompletionRequest | None): Current streaming request.
        tool_call_complete_regex (re.Pattern): Regex for complete tool calls.
        invoke_complete_regex (re.Pattern): Regex for complete invoke blocks.
        parameter_complete_regex (re.Pattern): Regex for complete parameters.

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-v3.2")
        >>> parser = DeepSeekV32ToolParser(tokenizer)
        >>> result = parser.extract_tool_calls(model_output, request)
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the DeepSeek V3.2 tool parser.

        Sets up the DSML token markers, regex patterns, and streaming state
        variables required for parsing DeepSeek V3.2 tool call format.

        Args:
            tokenizer: The tokenizer associated with the DeepSeek V3.2 model.
                Must be a valid tokenizer instance.

        Raises:
            ValueError: If the tokenizer is not provided.
        """
        super().__init__(tokenizer)

        self.prev_tool_call_arr: list[dict] = []

        # Sentinel tokens
        self.dsml_token: str = "｜DSML｜"
        self.dsml_start_check: str = "<" + self.dsml_token
        self.tool_call_start_token: str = "<｜DSML｜function_calls>"
        self.tool_call_end_token: str = "</｜DSML｜function_calls>"
        self.invoke_start_prefix: str = "<｜DSML｜invoke name="
        self.invoke_end_token: str = "</｜DSML｜invoke>"
        self.parameter_prefix: str = "<｜DSML｜parameter name="
        self.parameter_end_token: str = "</｜DSML｜parameter>"

        # Streaming state variables
        self.current_tool_name_sent: bool = False
        # Override base class type - we use string IDs for tool calls
        self.current_tool_id: str | None = None  # type: ignore[assignment]
        self.streamed_args_for_tool: list[str] = []
        self.is_tool_call_started: bool = False
        self.failed_count: int = 0

        # Initialize streaming state variables
        self.current_tool_index: int = 0
        self.invoke_index: int = 0
        self.header_sent: bool = False
        self.current_function_name: str | None = None
        self.current_param_name: str | None = None
        self.current_param_value: str = ""
        self.param_count: int = 0
        self.in_param: bool = False
        self.in_function: bool = False
        self.json_started: bool = False
        self.json_closed: bool = False
        self.accumulated_params: dict = {}
        self.streaming_request: ChatCompletionRequest | None = None

        # Enhanced streaming state - reset for each new message
        self._reset_streaming_state()

        # Regex patterns for complete parsing
        self.tool_call_complete_regex = re.compile(r"<｜DSML｜function_calls>(.*?)</｜DSML｜function_calls>", re.DOTALL)
        self.invoke_complete_regex = re.compile(r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>', re.DOTALL)
        self.parameter_complete_regex = re.compile(
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(?:true|false)"\s*>(.*?)</｜DSML｜parameter>',
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID.

        Creates a unique identifier for a tool call using UUID4,
        formatted with a 'call_' prefix and 24-character hex suffix.

        Returns:
            str: A unique tool call ID in format 'call_<24-char-hex>'.
        """
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state variables.

        Clears all internal state used for tracking streaming tool call
        parsing progress. Should be called at the start of each new
        streaming session.
        """
        self.current_tool_index = 0
        self.invoke_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = None
        self.current_function_name = None
        self.current_param_name = None
        self.current_param_value = ""
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.json_started = False
        self.json_closed = False
        self.accumulated_params = {}
        self.streaming_request = None
        self.prev_tool_call_arr.clear()

    def _parse_invoke_params(self, invoke_str: str) -> dict | None:
        """Parse parameters from an invoke block string.

        Extracts all parameter name-value pairs from the content of an
        invoke block using regex pattern matching.

        Args:
            invoke_str: The string content inside an invoke block,
                containing parameter tags.

        Returns:
            dict | None: Dictionary mapping parameter names to their
                string values, or None if parsing fails.
        """
        param_dict: dict[str, str] = {}
        for param_name, param_val in self.parameter_complete_regex.findall(invoke_str):
            param_dict[param_name] = param_val
        return param_dict

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output (non-streaming).

        Parses the DSML format to extract function invocations and their
        parameters from the model's response. Converts the XML-like
        structure into standard tool call format.

        Args:
            model_output: Complete text output from the model containing
                potential tool calls in DSML format.
            request: Original chat completion request with tool definitions.
                Used for context but not directly accessed in this method.

        Returns:
            ExtractedToolCallInformation: Contains:
                - tools_called (bool): True if any tools were found.
                - tool_calls (list[ToolCall]): List of parsed tool calls
                  with function names and JSON-serialized arguments.
                - content (str | None): Text before tool calls, or None.

        Example:
            >>> output = '<｜DSML｜function_calls><｜DSML｜invoke name="test"><｜DSML｜parameter name="x" string="true">1</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>'
            >>> result = parser.extract_tool_calls(output, request)
            >>> result.tools_called
            True
        """
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            tool_calls: list[ToolCall] = []

            for tool_call_match in self.tool_call_complete_regex.findall(model_output):
                for invoke_name, invoke_content in self.invoke_complete_regex.findall(tool_call_match):
                    param_dict = self._parse_invoke_params(invoke_content)
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=invoke_name, arguments=json.dumps(param_dict, ensure_ascii=False)
                            ),
                        )
                    )

            if not tool_calls:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            first_tool_idx = model_output.find(self.tool_call_start_token)
            content = model_output[:first_tool_idx] if first_tool_idx > 0 else None

            return ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls, content=content)

        except Exception:
            logger.exception("Error extracting tool calls")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    def _extract_name(self, name_str: str) -> str:
        """Extract name from a potentially quoted string.

        Removes surrounding quotes (single or double) from a string
        if present, returning the unquoted content.

        Args:
            name_str: The string potentially wrapped in quotes.

        Returns:
            str: The string with surrounding quotes removed, or
                the original string if no quotes present.
        """
        name_str = name_str.strip()
        if (name_str.startswith('"') and name_str.endswith('"')) or (
            name_str.startswith("'") and name_str.endswith("'")
        ):
            return name_str[1:-1]
        return name_str

    def _extract_param_name(self, input_str: str) -> str:
        """Extract parameter name from an attribute string.

        Finds the parameter name within a string that may contain
        attribute definitions like 'name="param_name"'.

        Args:
            input_str: String containing parameter name definition.

        Returns:
            str: The extracted parameter name, or the original
                string if extraction fails.
        """
        start = input_str.find('"') + 1
        end = input_str.find('"', start)
        return input_str[start:end] if start > 0 and end > start else input_str

    def _convert_param_value(self, value: str, param_type: str) -> Any:
        """Convert parameter value to the correct Python type.

        Converts a string value to the appropriate type based on the
        parameter type specification from the tool definition.

        Args:
            value: The string value to convert.
            param_type: The target type (e.g., "string", "integer",
                "number", "boolean", "object", "array").

        Returns:
            Any: The converted value in the appropriate Python type.
                Falls back to string if conversion fails.
        """
        if value.lower() == "null":
            return None

        param_type = param_type.lower()
        if param_type in ["string", "str", "text"]:
            return value
        if param_type in ["integer", "int"]:
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        if param_type in ["number", "float"]:
            try:
                val = float(value)
                return val if val != int(val) else int(val)
            except (ValueError, TypeError):
                return value
        if param_type in ["boolean", "bool"]:
            return value.lower() in ["true", "1"]
        if param_type in ["object", "array"]:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

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

        Handles incremental parsing of DSML format during streaming
        generation. Maintains extensive state to track position within
        the structured DSML format and emit incremental updates.

        The method processes:
        - Detection of DSML block start
        - Function invoke header parsing
        - Parameter extraction and type conversion
        - JSON fragment streaming
        - Tool call completion

        Args:
            previous_text: Text generated up to the previous chunk.
            current_text: All text generated so far including current chunk.
            delta_text: New text added in this chunk only.
            previous_token_ids: Token IDs up to previous chunk (unused).
            current_token_ids: All token IDs so far (unused).
            delta_token_ids: New token IDs in this chunk.
            request: Original chat completion request with tool definitions.
                Used for parameter type lookups.

        Returns:
            DeltaMessage | None: A delta message containing:
                - Tool call with function name on first detection
                - JSON argument fragments as parameters are parsed
                - Content text if outside tool call region
                - None if more data is needed

        Note:
            This method maintains extensive internal state and should
            only be called sequentially with proper text accumulation.
        """

        if not previous_text:
            self._reset_streaming_state()
            self.streaming_request = request

        if not delta_text:
            if delta_token_ids:
                complete_calls = len(self.tool_call_complete_regex.findall(current_text))
                if complete_calls > 0 and len(self.prev_tool_call_arr) > 0:
                    open_calls = current_text.count(self.tool_call_start_token) - current_text.count(
                        self.tool_call_end_token
                    )
                    if open_calls == 0:
                        return DeltaMessage(content="")
                elif not self.is_tool_call_started and current_text:
                    return DeltaMessage(content="")
            return None

        if self.json_closed and not self.in_function:
            invoke_ends = current_text.count(self.invoke_end_token)
            if invoke_ends > self.current_tool_index:
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.in_function = False
                self.accumulated_params = {}
                return None

        if not self.is_tool_call_started:
            if self.dsml_token in current_text:
                self.is_tool_call_started = True
                if self.dsml_start_check in delta_text:
                    content_before = delta_text[: delta_text.index(self.dsml_start_check)]
                    if content_before:
                        return DeltaMessage(content=content_before)
                return None

            if current_text.rstrip().endswith(self.tool_call_end_token) and delta_text.strip() == "":
                return None

            if delta_text.endswith("<"):
                return DeltaMessage(content=delta_text[:-1])
            if previous_text and previous_text.endswith("<"):
                return DeltaMessage(content="<" + delta_text)
            return DeltaMessage(content=delta_text)

        invoke_starts_count = current_text.count(self.invoke_start_prefix)
        if self.current_tool_index >= invoke_starts_count:
            return None

        invoke_start_positions: list[int] = []
        idx = 0
        while True:
            idx = current_text.find(self.invoke_start_prefix, idx)
            if idx == -1:
                break
            invoke_start_positions.append(idx)
            idx += len(self.invoke_start_prefix)

        if self.current_tool_index >= len(invoke_start_positions):
            return None

        invoke_start_idx = invoke_start_positions[self.current_tool_index]
        invoke_end_idx = current_text.find(self.invoke_end_token, invoke_start_idx)
        if invoke_end_idx == -1:
            tool_text = current_text[invoke_start_idx:]
        else:
            tool_text = current_text[invoke_start_idx : invoke_end_idx + len(self.invoke_end_token)]

        if not self.header_sent:
            if self.invoke_start_prefix in tool_text:
                func_start = tool_text.find(self.invoke_start_prefix) + len(self.invoke_start_prefix)
                func_end = tool_text.find(">", func_start)
                if func_end != -1:
                    function_name_raw = tool_text[func_start:func_end]
                    self.current_function_name = self._extract_name(function_name_raw)
                    self.current_tool_id = self._generate_tool_call_id()
                    self.header_sent = True
                    self.in_function = True

                    if len(self.prev_tool_call_arr) <= self.current_tool_index:
                        self.prev_tool_call_arr.append({"name": self.current_function_name, "arguments": "{}"})

                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_index,
                                id=self.current_tool_id,
                                function=DeltaFunctionCall(name=self.current_function_name, arguments=""),
                                type="function",
                            )
                        ]
                    )
            return None

        if self.in_function:
            if self.in_function and not self.json_started:
                self.json_started = True
                return DeltaMessage(
                    tool_calls=[DeltaToolCall(index=self.current_tool_index, function=DeltaFunctionCall(arguments="{"))]
                )

            if not self.json_started:
                self.json_started = True

            if not self.json_closed and self.invoke_end_token in tool_text:
                total_param_count = tool_text.count(self.parameter_prefix)
                if self.param_count >= total_param_count:
                    self.json_closed = True

                    invoke_start = tool_text.find(self.invoke_start_prefix) + len(self.invoke_start_prefix)
                    invoke_content_end = tool_text.find(self.invoke_end_token, invoke_start)
                    if invoke_content_end != -1:
                        invoke_content = tool_text[invoke_start:invoke_content_end]
                        try:
                            invoke_params = self._parse_invoke_params(invoke_content)
                            if invoke_params and self.current_tool_index < len(self.prev_tool_call_arr):
                                self.prev_tool_call_arr[self.current_tool_index]["arguments"] = json.dumps(
                                    invoke_params, ensure_ascii=False
                                )
                        except Exception:
                            pass

                    result = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(index=self.current_tool_index, function=DeltaFunctionCall(arguments="}"))
                        ]
                    )

                    self.json_closed = True
                    self.in_function = False
                    self.accumulated_params = {}
                    return result

                return None

            param_starts: list[int] = []
            idx = 0
            while True:
                idx = tool_text.find(self.parameter_prefix, idx)
                if idx == -1:
                    break
                param_starts.append(idx)
                idx += len(self.parameter_prefix)

            if not self.in_param and self.param_count < len(param_starts) and len(param_starts) > self.param_count:
                param_idx = param_starts[self.param_count]
                param_start = param_idx + len(self.parameter_prefix)
                remaining = tool_text[param_start:]

                if ">" in remaining:
                    name_end = remaining.find(">")
                    param_name_raw = remaining[:name_end]
                    self.current_param_name = self._extract_param_name(param_name_raw)

                    value_start = param_start + name_end + 1
                    value_text = tool_text[value_start:]
                    if value_text.startswith("\n"):
                        value_text = value_text[1:]

                    param_end_idx = value_text.find(self.parameter_end_token)
                    if param_end_idx == -1:
                        next_param_idx = value_text.find(self.parameter_prefix)
                        func_end_idx = value_text.find(self.invoke_end_token)

                        if next_param_idx != -1 and (func_end_idx == -1 or next_param_idx < func_end_idx):
                            param_end_idx = next_param_idx
                        elif func_end_idx != -1:
                            param_end_idx = func_end_idx
                        else:
                            if self.invoke_end_token in tool_text:
                                param_end_idx = len(value_text)
                            else:
                                return None

                    if param_end_idx != -1:
                        param_value = value_text[:param_end_idx]
                        if param_value.endswith("\n"):
                            param_value = param_value[:-1]

                        self.accumulated_params[self.current_param_name] = param_value

                        param_config: dict[str, Any] = {}
                        if self.streaming_request and self.streaming_request.tools:
                            for tool in self.streaming_request.tools:
                                if (
                                    hasattr(tool, "function")
                                    and tool.function.name == self.current_function_name
                                    and hasattr(tool.function, "parameters")
                                ):
                                    params = tool.function.parameters
                                    if isinstance(params, dict) and "properties" in params:
                                        param_config = params["properties"]
                                    break

                        param_type = "string"
                        if (
                            self.current_param_name in param_config
                            and isinstance(param_config[self.current_param_name], dict)
                            and "type" in param_config[self.current_param_name]
                        ):
                            param_type = param_config[self.current_param_name]["type"]

                        converted_value = self._convert_param_value(param_value, param_type)
                        serialized_value = json.dumps(converted_value, ensure_ascii=False)

                        if self.param_count == 0:
                            json_fragment = f'"{self.current_param_name}": {serialized_value}'
                        else:
                            json_fragment = f', "{self.current_param_name}": {serialized_value}'

                        self.param_count += 1

                        return DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_index,
                                    function=DeltaFunctionCall(arguments=json_fragment),
                                )
                            ]
                        )

        return None
