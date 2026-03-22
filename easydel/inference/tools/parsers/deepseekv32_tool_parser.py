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


@ToolParserManager.register_module("deepseek_v32")  # pyright: ignore[reportUntypedClassDecorator]
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
        self.current_tool_id: str | None = None
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

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Adjust the chat completion request for DSML tool call parsing.

        Disables special token skipping when tools are enabled so that DSML
        sentinel tokens are preserved in the model output for parsing.

        Args:
            request: The incoming chat completion request to adjust.

        Returns:
            The modified request with ``skip_special_tokens`` set to False
            when tools are active.
        """
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID.

        Creates a unique identifier for a tool call using UUID4,
        formatted with a 'call_' prefix and 24-character hex suffix.

        Returns:
            str: A unique tool call ID in format 'call_<24-char-hex>'.
        """
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state for a new message.

        Clears the tool index counter, tool-call-started flag, previously
        accumulated tool call records, and per-tool streamed argument buffers.
        Called automatically at the start of each new streaming response.
        """
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()

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

    def _convert_params_with_schema(
        self,
        function_name: str,
        param_dict: dict[str, str],
        request: ChatCompletionRequest | None,
    ) -> dict[str, Any]:
        """Convert raw string parameter values using the tool schema types.

        Looks up each parameter's expected type from the tool definition in
        the request, then delegates to ``_convert_param_value`` for type
        coercion. Parameters without a matching schema entry default to string.

        Args:
            function_name: Name of the function whose schema to look up.
            param_dict: Raw parameter name-to-value mapping (all strings).
            request: The chat completion request containing tool definitions,
                or None if unavailable.

        Returns:
            Dictionary mapping parameter names to their type-converted values.
        """
        param_config: dict[str, Any] = {}
        if request and request.tools:
            for tool in request.tools:
                if (
                    hasattr(tool, "function")
                    and tool.function.name == function_name
                    and hasattr(tool.function, "parameters")
                ):
                    schema = tool.function.parameters
                    if isinstance(schema, dict) and "properties" in schema:
                        param_config = schema["properties"]
                    break

        converted: dict[str, Any] = {}
        for name, value in param_dict.items():
            param_type = "string"
            if name in param_config and isinstance(param_config[name], dict):
                param_type = param_config[name].get("type", "string")
            converted[name] = self._convert_param_value(value, param_type)
        return converted

    def _extract_delta_tool_calls(
        self,
        current_text: str,
        request: ChatCompletionRequest | None,
    ) -> list[DeltaToolCall]:
        """Extract DeltaToolCalls from newly completed invoke blocks.

        Scans ``current_text`` for complete ``<invoke>...</invoke>`` blocks
        that have not yet been emitted, parses their parameters, converts
        values using the tool schema, and returns one ``DeltaToolCall`` per
        newly completed invoke.

        Args:
            current_text: The full accumulated model output so far.
            request: The chat completion request with tool definitions for
                schema-based type conversion, or None.

        Returns:
            List of ``DeltaToolCall`` objects for each newly completed invoke
            block. Empty if no new blocks were found.
        """
        complete_invokes = self.invoke_complete_regex.findall(current_text)
        delta_tool_calls: list[DeltaToolCall] = []

        while len(complete_invokes) > self.current_tool_index:
            invoke_name, invoke_body = complete_invokes[self.current_tool_index]
            param_dict = self._parse_invoke_params(invoke_body) or {}
            converted = self._convert_params_with_schema(invoke_name, param_dict, request)
            args_json = json.dumps(converted, ensure_ascii=False)
            idx = self.current_tool_index
            self.current_tool_index += 1
            self.prev_tool_call_arr.append({"name": invoke_name, "arguments": converted})
            self.streamed_args_for_tool.append(args_json)
            delta_tool_calls.append(
                DeltaToolCall(
                    index=idx,
                    id=self._generate_tool_call_id(),
                    function=DeltaFunctionCall(name=invoke_name, arguments=args_json),
                    type="function",
                )
            )

        return delta_tool_calls

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

        Uses a buffer-until-complete-invoke strategy: tokens are accumulated
        until a complete ``<invoke>...</invoke>`` block is available, then
        the block is parsed and emitted in one shot as a ``DeltaToolCall``.
        Content tokens that appear before the first DSML block are forwarded
        as regular content deltas.

        Args:
            previous_text: All text generated before this step.
            current_text: All text generated up to and including this step.
            delta_text: The new text added in this step.
            previous_token_ids: Token IDs for ``previous_text``.
            current_token_ids: Token IDs for ``current_text``.
            delta_token_ids: Token IDs for ``delta_text``.
            request: The chat completion request with tool definitions.

        Returns:
            A ``DeltaMessage`` containing either content text, tool call
            deltas, or both. Returns None when there is nothing new to emit.
        """
        if not previous_text:
            self._reset_streaming_state()
        content_before = None
        if self.is_tool_call_started:
            pass
        elif self.tool_call_start_token in current_text:
            self.is_tool_call_started = True
            start_idx = current_text.index(self.tool_call_start_token)
            content_before = current_text[len(previous_text) : start_idx] or None
        else:
            return DeltaMessage(content=delta_text) if delta_text else None

        delta_tool_calls = self._extract_delta_tool_calls(current_text, request)

        if delta_tool_calls or content_before:
            return DeltaMessage(content=content_before, tool_calls=delta_tool_calls)
        if not delta_text and delta_token_ids and self.prev_tool_call_arr:
            return DeltaMessage(content="")
        return None
