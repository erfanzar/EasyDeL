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
        dsml_token (str): Base DSML token identifier.
        dsml_start_check (str): Prefix used to recognize DSML markup.
        invoke_start_prefix (str): Prefix of a named invocation tag.
        current_tool_id (int): Zero-based index of the latest emitted tool call.
        failed_count (int): Failed parsing or conversion attempts in this response.
        invoke_index (int): Index of the latest emitted invocation.
        header_sent (bool): Whether any tool header has been emitted.
        current_function_name (str | None): Function in the latest invocation.
        current_param_name (str | None): Parameter in the latest parameter tag.
        current_param_value (str): Raw value in the latest parameter tag.
        param_count (int): Number of parameters in the latest invocation.
        in_param (bool): Whether a parameter tag is currently open.
        in_function (bool): Whether an invocation is currently open.
        json_started (bool): Whether a function-call section has started.
        json_closed (bool): Whether the function-call section has closed.
        accumulated_params (dict): Parameters of the latest completed invocation.
        streaming_request (ChatCompletionRequest | None): Active streaming request.
        prev_tool_call_arr (list[dict]): Previous tool calls for streaming state.
        tool_call_start_token (str): Token marking function_calls start.
        tool_call_end_token (str): Token marking function_calls end.
        invoke_end_token (str): Token marking invoke end.
        parameter_prefix (str): Prefix for parameter tag.
        parameter_end_token (str): Token marking parameter end.
        current_tool_name_sent (bool): Whether current tool name was sent.
        streamed_args_for_tool (list[str]): Streamed arguments per tool.
        is_tool_call_started (bool): Whether tool call parsing has started.
        current_tool_index (int): Index of current tool being processed.
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

        self.dsml_token = "｜DSML｜"
        self.dsml_start_check = "<" + self.dsml_token
        self.invoke_start_prefix = self.dsml_start_check + "invoke name="
        self.tool_call_start_token: str = self.dsml_start_check + "function_calls>"
        self.tool_call_end_token: str = "</" + self.dsml_token + "function_calls>"
        self.invoke_end_token: str = "</｜DSML｜invoke>"
        self.parameter_prefix: str = "<｜DSML｜parameter name="
        self.parameter_end_token: str = "</｜DSML｜parameter>"

        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
        self.is_tool_call_started: bool = False

        self.current_tool_index: int = 0

        self._reset_streaming_state()

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
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.failed_count = 0
        self.invoke_index = 0
        self.header_sent = False
        self.current_function_name: str | None = None
        self.current_param_name: str | None = None
        self.current_param_value = ""
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.json_started = False
        self.json_closed = False
        self.accumulated_params: dict = {}
        self.streaming_request: ChatCompletionRequest | None = None

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
                Used to coerce parameter values to their schema types.

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
                    param_dict = self._parse_invoke_params(invoke_content) or {}
                    # Same schema-driven typing as the streaming path so both modes agree.
                    converted = self._convert_params_with_schema(invoke_name, param_dict, request, invoke_content)
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(name=invoke_name, arguments=json.dumps(converted, ensure_ascii=False)),
                        )
                    )

            if not tool_calls:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            first_tool_idx = model_output.find(self.tool_call_start_token)
            content = model_output[:first_tool_idx] if first_tool_idx > 0 else None

            return ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls, content=content)

        except Exception:
            self.failed_count += 1
            logger.exception("Error extracting tool calls")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

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
        invoke_body: str = "",
    ) -> dict[str, Any]:
        """Convert raw string parameter values using the tool schema types.

        Looks up each parameter's expected type from the tool definition in
        the request, then delegates to ``_convert_param_value`` for type
        coercion. Without a matching schema type, the DSML ``string`` attribute
        determines whether a value remains a string or is decoded as JSON.

        Args:
            function_name: Name of the function whose schema to look up.
            param_dict: Raw parameter name-to-value mapping (all strings).
            request: The chat completion request containing tool definitions,
                or None if unavailable.
            invoke_body: Original parameter markup. Without a schema, ``string="false"``
                values are decoded as JSON and ``string="true"`` values remain strings.

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

        string_flags = dict(re.findall(r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>', invoke_body))
        converted: dict[str, Any] = {}
        for name, value in param_dict.items():
            schema = param_config.get(name)
            param_type = schema.get("type") if isinstance(schema, dict) else None
            if isinstance(param_type, list):
                # Do not discard union nullability: DSML distinguishes JSON null
                # from the literal string "null" with its explicit string flag.
                if "null" in param_type and string_flags.get(name) == "false" and value.strip() == "null":
                    converted[name] = None
                    continue
                param_type = next((kind for kind in param_type if kind != "null"), None)
            if isinstance(param_type, str):
                # A schema-typed string, including the literal "null", stays a string.
                converted[name] = value if param_type == "string" else self._convert_param_value(value, param_type)
            elif string_flags.get(name) == "false":
                try:
                    converted[name] = json.loads(value)
                except (ValueError, TypeError):
                    converted[name] = value
            else:
                converted[name] = value
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
            converted = self._convert_params_with_schema(invoke_name, param_dict, request, invoke_body)
            args_json = json.dumps(converted, ensure_ascii=False)
            idx = self.current_tool_index
            self.current_tool_index += 1
            self.prev_tool_call_arr.append({"name": invoke_name, "arguments": converted})
            self.streamed_args_for_tool.append(args_json)
            self.current_tool_id = idx
            self.invoke_index = idx
            self.header_sent = True
            self.current_tool_name_sent = True
            self.accumulated_params = converted
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
        self.streaming_request = request
        content_before = None
        if self.is_tool_call_started:
            pass
        elif self.tool_call_start_token in current_text:
            self.is_tool_call_started = True
            start_idx = current_text.index(self.tool_call_start_token)
            content_before = current_text[len(previous_text) : start_idx] or None
        else:
            return DeltaMessage(content=delta_text) if delta_text else None

        self.json_started = self.is_tool_call_started
        self.json_closed = self.tool_call_end_token in current_text
        invoke_start = current_text.rfind(self.dsml_start_check + "invoke")
        self.in_function = invoke_start > current_text.rfind(self.invoke_end_token)
        if invoke_start >= 0:
            invoke_text = current_text[invoke_start:]
            name_match = re.match(re.escape(self.dsml_start_check) + r'invoke\s+name="([^"]+)"', invoke_text)
            self.current_function_name = name_match.group(1) if name_match else None
            self.param_count = len(self.parameter_complete_regex.findall(invoke_text))
            param_start = invoke_text.rfind(self.parameter_prefix)
            self.in_param = param_start > invoke_text.rfind(self.parameter_end_token)
            self.current_param_name = None
            self.current_param_value = ""
            if param_start >= 0:
                parameter_text = invoke_text[param_start:]
                param_match = re.match(re.escape(self.parameter_prefix) + r'"([^"]+)"', parameter_text)
                self.current_param_name = param_match.group(1) if param_match else None
                _, separator, value_text = parameter_text.partition(">")
                if separator:
                    self.current_param_value = value_text.split(self.parameter_end_token, 1)[0]
                self.param_count += int(self.in_param)
        try:
            delta_tool_calls = self._extract_delta_tool_calls(current_text, request)
        except Exception:
            self.failed_count += 1
            logger.exception("Error extracting streaming DSML tool calls")
            return None

        if delta_tool_calls or content_before:
            return DeltaMessage(content=content_before, tool_calls=delta_tool_calls)
        if not delta_text and delta_token_ids and self.prev_tool_call_arr:
            return DeltaMessage(content="")
        return None
