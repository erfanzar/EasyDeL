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

"""Minimax M2 tool parser with XML-style invoke and parameter tags.

This module provides a tool call parser for Minimax M2 models that use
a structured XML-like format with invoke and parameter tags. Unlike
JSON-based parsers, this parser handles named parameter elements.

Example format:
    <minimax:tool_call>
    <invoke name="function_name">
    <parameter name="param1">value1</parameter>
    <parameter name="param2">value2</parameter>
    </invoke>
    </minimax:tool_call>

Features:
    - XML-style invoke and parameter parsing
    - Schema-aware type conversion for parameters
    - Streaming support with incremental JSON generation
    - Multi-tool support with index tracking
    - Parameter type inference from tool definitions
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


@ToolParserManager.register_module("minimax_m2")  # pyright: ignore[reportUntypedClassDecorator]
class MinimaxM2ToolParser(ToolParser):
    """Tool parser for Minimax M2 models with XML-style tool calls.

    Parses tool calls that use <minimax:tool_call>, <invoke>, and
    <parameter> tags instead of JSON format. Supports schema-aware
    type conversion for parameters.

    Features:
        - XML-style invoke and parameter parsing
        - Schema-aware type conversion (int, float, bool, object, array)
        - Streaming support with incremental JSON generation
        - Multi-tool support with index tracking
        - Parameter type inference from tool definitions

    Format:
        <minimax:tool_call>
        <invoke name="function_name">
        <parameter name="param1">value1</parameter>
        </invoke>
        </minimax:tool_call>

    Attributes:
        tool_call_start_token: Opening delimiter '<minimax:tool_call>'.
        tool_call_end_token: Closing delimiter '</minimax:tool_call>'.
        invoke_start_prefix: Prefix for invoke tags '<invoke name='.
        invoke_end_token: Closing invoke tag '</invoke>'.
        parameter_prefix: Prefix for parameter tags '<parameter name='.
        parameter_end_token: Closing parameter tag '</parameter>'.

    Raises:
        ValueError: If tokenizer is None.
        RuntimeError: If start/end tokens are not found in vocabulary.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the MinimaxM2ToolParser.

        Sets up the parser with XML-style token patterns and initializes
        streaming state. Validates tokenizer presence and token vocabulary.

        Args:
            tokenizer: The tokenizer associated with the Minimax M2 model.
                Used for token-level operations during streaming.

        Raises:
            ValueError: If tokenizer is None.
            RuntimeError: If tool call start/end tokens are not found
                in the tokenizer vocabulary.
        """
        super().__init__(tokenizer)

        self.prev_tool_call_arr: list[dict] = []

        self.tool_call_start_token: str = "<minimax:tool_call>"
        self.tool_call_end_token: str = "</minimax:tool_call>"
        self.invoke_start_prefix: str = "<invoke name="
        self.invoke_end_token: str = "</invoke>"
        self.parameter_prefix: str = "<parameter name="
        self.parameter_end_token: str = "</parameter>"

        self.current_tool_name_sent: bool = False
        self.current_tool_id: str | None = None
        self.streamed_args_for_tool: list[str] = []
        self.is_tool_call_started: bool = False
        self.failed_count: int = 0

        self.current_tool_index: int = 0
        self.invoke_index: int = 0
        self.header_sent: bool = False
        self.current_function_name: str | None = None
        self.current_param_name: str | None = None
        self.current_param_value: str = ""
        self.param_count: int = 0
        self.in_param: bool = False
        self.in_function: bool = False
        self.accumulated_text: str = ""
        self.json_started: bool = False
        self.json_closed: bool = False
        self.accumulated_params: dict = {}
        self.streaming_request: ChatCompletionRequest | None = None

        self._reset_streaming_state()

        self.tool_call_complete_regex = re.compile(r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL)
        self.invoke_complete_regex = re.compile(r"<invoke name=(.*?)</invoke>", re.DOTALL)
        self.parameter_complete_regex = re.compile(r"<parameter name=(.*?)</parameter>", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError("MiniMax M2 Tool parser could not locate tool call start/end tokens in the tokenizer!")

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID.

        Creates a unique identifier for a tool call using UUID.

        Returns:
            A unique string ID in format 'call_<24-char-hex>'.
        """
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state to initial values.

        Clears all state variables used during streaming to prepare
        for a new tool call sequence.
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
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False
        self.accumulated_params = {}
        self.streaming_request = None
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()

    def _extract_name(self, name_str: str) -> str:
        """Extract a name from a quoted or unquoted string.

        Removes surrounding quotes (single or double) from the name string.

        Args:
            name_str: The name string, possibly wrapped in quotes.

        Returns:
            The extracted name without surrounding quotes.
        """
        name_str = name_str.strip()
        if (name_str.startswith('"') and name_str.endswith('"')) or (
            name_str.startswith("'") and name_str.endswith("'")
        ):
            return name_str[1:-1]
        return name_str

    def _convert_param_value(self, value: str, param_type: str) -> Any:
        """Convert a parameter value based on a single type.

        Convenience wrapper around _convert_param_value_with_types
        for single type conversion.

        Args:
            value: The string value to convert.
            param_type: The target type name.

        Returns:
            The converted value.
        """
        return self._convert_param_value_with_types(value, [param_type])

    def _extract_types_from_schema(self, schema: Any) -> list[str]:
        """Extract possible types from a JSON schema definition.

        Analyzes a JSON schema to determine all possible types for a
        parameter, including types from anyOf, oneOf, allOf constructs
        and enum values.

        Args:
            schema: JSON schema dictionary or None.

        Returns:
            List of type strings (e.g., ['string', 'integer', 'null']).
            Defaults to ['string'] if no types can be determined.
        """
        if schema is None or not isinstance(schema, dict):
            return ["string"]

        types: set[str] = set()

        if "type" in schema:
            type_value = schema["type"]
            if isinstance(type_value, str):
                types.add(type_value)
            elif isinstance(type_value, list):
                for t in type_value:
                    if isinstance(t, str):
                        types.add(t)

        if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
            for value in schema["enum"]:
                if value is None:
                    types.add("null")
                elif isinstance(value, bool):
                    types.add("boolean")
                elif isinstance(value, int):
                    types.add("integer")
                elif isinstance(value, float):
                    types.add("number")
                elif isinstance(value, str):
                    types.add("string")
                elif isinstance(value, list):
                    types.add("array")
                elif isinstance(value, dict):
                    types.add("object")

        for choice_field in ("anyOf", "oneOf", "allOf"):
            if choice_field in schema and isinstance(schema[choice_field], list):
                for choice in schema[choice_field]:
                    types.update(self._extract_types_from_schema(choice))

        if not types:
            return ["string"]
        return list(types)

    def _convert_param_value_with_types(self, value: str, param_types: list[str]) -> Any:
        """Convert a parameter value based on possible types.

        Attempts to convert the string value to one of the specified
        types in priority order: integer, number, boolean, object,
        array, string.

        Args:
            value: The string value to convert.
            param_types: List of possible type names.

        Returns:
            The converted value in the appropriate Python type.
            Falls back to string if no conversion succeeds.
        """
        if value.lower() == "null":
            return None

        normalized_types = [t.lower() for t in param_types]
        if "null" in normalized_types or value.lower() in ("null", "none", "nil"):
            return None

        type_priority = [
            "integer",
            "int",
            "number",
            "float",
            "boolean",
            "bool",
            "object",
            "array",
            "string",
            "str",
            "text",
        ]

        for param_type in type_priority:
            if param_type not in normalized_types:
                continue

            if param_type in ["string", "str", "text"]:
                return value
            if param_type in ["integer", "int"]:
                try:
                    return int(value)
                except (ValueError, TypeError):
                    continue
            if param_type in ["number", "float"]:
                try:
                    val = float(value)
                    return val if val != int(val) else int(val)
                except (ValueError, TypeError):
                    continue
            if param_type in ["boolean", "bool"]:
                lower_val = value.lower().strip()
                if lower_val in ["true", "1", "yes", "on"]:
                    return True
                if lower_val in ["false", "0", "no", "off"]:
                    return False
                continue
            if param_type in ["object", "array"]:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    continue

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _get_param_types_from_config(self, param_name: str, param_config: dict) -> list[str]:
        """Get possible types for a parameter from tool configuration.

        Looks up the parameter schema in the tool configuration and
        extracts possible types.

        Args:
            param_name: Name of the parameter.
            param_config: Dictionary mapping parameter names to schemas.

        Returns:
            List of possible type names for the parameter.
            Defaults to ['string'] if not found.
        """
        if param_name not in param_config:
            return ["string"]
        param_schema = param_config[param_name]
        if not isinstance(param_schema, dict):
            return ["string"]
        return self._extract_types_from_schema(param_schema)

    def _parse_single_invoke(self, invoke_str: str, tools: list | None) -> ToolCall | None:
        """Parse a single invoke element into a ToolCall.

        Extracts the function name and parameters from an invoke string,
        converts parameter values based on tool schema, and creates
        a ToolCall object.

        Args:
            invoke_str: The content of an invoke element (after 'name=').
            tools: List of available tool definitions for type inference.

        Returns:
            ToolCall object or None if parsing fails.
        """
        name_match = re.search(r"^([^>]+)", invoke_str)
        if not name_match:
            return None

        function_name = self._extract_name(name_match.group(1))

        param_config: dict[str, Any] = {}
        if tools:
            for tool in tools:
                if (
                    hasattr(tool, "function")
                    and tool.function.name == function_name
                    and hasattr(tool.function, "parameters")
                ):
                    params = tool.function.parameters
                    if isinstance(params, dict) and "properties" in params:
                        param_config = params["properties"]
                    break

        param_dict: dict[str, Any] = {}
        for match in self.parameter_complete_regex.findall(invoke_str):
            param_match = re.search(r"^([^>]+)>(.*)", match, re.DOTALL)
            if not param_match:
                continue
            param_name = self._extract_name(param_match.group(1))
            param_value = param_match.group(2).strip()
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_type = self._get_param_types_from_config(param_name, param_config)
            param_dict[param_name] = self._convert_param_value_with_types(param_value, param_type)

        return ToolCall(
            type="function",
            function=FunctionCall(name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False)),
        )

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model response.

        Parses all tool call blocks in the model output and extracts
        invoke elements with their parameters.

        Args:
            model_output: Complete model output string.
            request: Chat completion request containing tool definitions.

        Returns:
            ExtractedToolCallInformation containing:
                - tools_called: True if valid tool calls were found
                - tool_calls: List of ToolCall objects extracted
                - content: Text content before tool calls, None if none
        """
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            tool_calls: list[ToolCall] = []

            for tool_call_match in self.tool_call_complete_regex.findall(model_output):
                for invoke_match in self.invoke_complete_regex.findall(tool_call_match):
                    tool_call = self._parse_single_invoke(invoke_match, request.tools if request else None)
                    if tool_call:
                        tool_calls.append(tool_call)

            if not tool_calls:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            self.prev_tool_call_arr.clear()
            for tool_call in tool_calls:
                self.prev_tool_call_arr.append(
                    {"name": tool_call.function.name, "arguments": tool_call.function.arguments}
                )

            first_tool_idx = model_output.find(self.tool_call_start_token)
            content = model_output[:first_tool_idx] if first_tool_idx > 0 else None
            return ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls, content=content)

        except Exception:
            logger.exception("Error extracting tool calls")
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

        Processes streaming output to detect invoke and parameter tags,
        generating JSON argument fragments as they become available.

        Args:
            previous_text: Text accumulated before this delta.
            current_text: Complete text including the current delta.
            delta_text: The new text added in this streaming chunk.
            previous_token_ids: Token IDs before this delta.
            current_token_ids: All token IDs including current delta.
            delta_token_ids: Token IDs in the current delta.
            request: The chat completion request with tool definitions.

        Returns:
            DeltaMessage containing either:
                - content: Regular text content before tool calls
                - tool_calls: Tool call deltas with function name or arguments
            Returns None if more tokens are needed.
        """
        if not previous_text or self.tool_call_start_token in delta_text:
            self._reset_streaming_state()
            self.streaming_request = request

        if not delta_text:
            if delta_token_ids and self.tool_call_end_token_id not in delta_token_ids:
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

        self.accumulated_text = current_text

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
            if (self.tool_call_start_token_id in delta_token_ids) or (self.tool_call_start_token in delta_text):
                self.is_tool_call_started = True
                if self.tool_call_start_token in delta_text:
                    content_before = delta_text[: delta_text.index(self.tool_call_start_token)]
                    if content_before:
                        return DeltaMessage(content=content_before)
                return None

            if current_text.rstrip().endswith(self.tool_call_end_token) and delta_text.strip() == "":
                return None
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
                            invoke_params = {}
                            for match in self.parameter_complete_regex.findall(invoke_content):
                                param_match = re.search(r"^([^>]+)>(.*)", match, re.DOTALL)
                                if param_match:
                                    param_name = self._extract_name(param_match.group(1))
                                    param_value = param_match.group(2).strip()
                                    if param_value.startswith("\n"):
                                        param_value = param_value[1:]
                                    if param_value.endswith("\n"):
                                        param_value = param_value[:-1]

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

                                    param_type = self._get_param_types_from_config(param_name, param_config)
                                    invoke_params[param_name] = self._convert_param_value_with_types(
                                        param_value, param_type
                                    )

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
                    self.current_param_name = self._extract_name(param_name_raw)

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

                        param_type = self._get_param_types_from_config(self.current_param_name, param_config)
                        converted_value = self._convert_param_value_with_types(param_value, param_type)
                        serialized_value = json.dumps(converted_value, ensure_ascii=False)

                        if self.param_count == 0:
                            json_fragment = f'"{self.current_param_name}": {serialized_value}'
                        else:
                            json_fragment = f', "{self.current_param_name}": {serialized_value}'

                        self.param_count += 1
                        if self.current_tool_index < len(self.streamed_args_for_tool):
                            self.streamed_args_for_tool[self.current_tool_index] += json_fragment

                        return DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_index, function=DeltaFunctionCall(arguments=json_fragment)
                                )
                            ]
                        )

        return None
