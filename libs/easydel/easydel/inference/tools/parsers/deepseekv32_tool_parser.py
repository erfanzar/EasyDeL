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
import math
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

_STRING_SCHEMA_TYPES = frozenset({"string", "str", "text"})
_INVOKE_OPEN_RE = re.compile(r'<｜DSML｜invoke\s+name="([^"]+)"\s*>')
_PARAM_OPEN_RE = re.compile(r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>')


def _strip_trailing_partial_marker(text: str, marker: str) -> str:
    """Withhold any trailing non-empty proper prefix of *marker* from *text*."""
    max_len = min(len(marker) - 1, len(text))
    for k in range(max_len, 0, -1):
        if text.endswith(marker[:k]):
            return text[:-k]
    return text


def _has_open_marker_or_trailing_prefix(text: str, start_token: str, end_token: str) -> bool:
    """Return True if *text* has an unclosed outer section or ends with a partial start marker."""
    pos = 0
    n = len(text)
    start_len = len(start_token)
    end_len = len(end_token)
    while pos < n:
        start_idx = text.find(start_token, pos)
        if start_idx == -1:
            return ToolParser._has_trailing_marker_prefix(text[pos:], (start_token,))
        end_idx = text.find(end_token, start_idx + start_len)
        if end_idx == -1:
            return True
        pos = end_idx + end_len
    return False


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
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</｜DSML｜parameter>',
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")

    def get_streaming_buffer_marker_pairs(self) -> Sequence[tuple[str, str | None]]:
        """Scope protocol buffering strictly to ``<｜DSML｜function_calls>`` envelopes."""
        return ((self.tool_call_start_token, self.tool_call_end_token),)

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Report whether a DSML function_calls envelope or partial start marker is buffered."""
        del delta_text
        if self._streaming_initialized:
            return self.is_tool_call_started or self._pending_tail_start is not None
        return _has_open_marker_or_trailing_prefix(
            current_text,
            self.tool_call_start_token,
            self.tool_call_end_token,
        )

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
        self._emitted_content_pos: int | None = None
        self._pending_tail_start: int | None = None
        self._last_current_text: str = ""
        self._streaming_initialized: bool = False

    def flush_streaming_content(self) -> str | None:
        """Flush any trailing partial DSML start marker withheld at end-of-stream."""
        flushed: str | None = None
        if self._pending_tail_start is not None and self._pending_tail_start < len(self._last_current_text):
            flushed = self._last_current_text[self._pending_tail_start :] or None
            self._emitted_content_pos = len(self._last_current_text)
            self._pending_tail_start = None
        self._last_current_text = ""
        self.streaming_request = None
        return flushed

    def _find_complete_invokes(self, text: str) -> list[tuple[str, str]]:
        """Linearly scan complete ``<｜DSML｜invoke ...>...</｜DSML｜invoke>`` blocks."""
        invokes: list[tuple[str, str]] = []
        invoke_prefix = self.dsml_start_check + "invoke"
        end_token = self.invoke_end_token
        end_len = len(end_token)
        pos = 0
        n = len(text)
        while pos < n:
            start_idx = text.find(invoke_prefix, pos)
            if start_idx == -1:
                break
            open_match = _INVOKE_OPEN_RE.match(text, start_idx)
            if open_match is None:
                pos = start_idx + len(invoke_prefix)
                continue
            body_start = open_match.end()
            end_idx = text.find(end_token, body_start)
            if end_idx == -1:
                break
            invokes.append((open_match.group(1), text[body_start:end_idx]))
            pos = end_idx + end_len
        return invokes

    def _parse_invoke_params_with_flags(self, invoke_str: str) -> tuple[dict[str, str], dict[str, str]]:
        """Parse parameter values and their DSML ``string`` attributes from an invoke body."""
        param_dict: dict[str, str] = {}
        string_flags: dict[str, str] = {}
        param_prefix = self.dsml_start_check + "parameter"
        end_token = self.parameter_end_token
        end_len = len(end_token)
        pos = 0
        n = len(invoke_str)
        while pos < n:
            start_idx = invoke_str.find(param_prefix, pos)
            if start_idx == -1:
                break
            open_match = _PARAM_OPEN_RE.match(invoke_str, start_idx)
            if open_match is None:
                pos = start_idx + len(param_prefix)
                continue
            body_start = open_match.end()
            end_idx = invoke_str.find(end_token, body_start)
            if end_idx == -1:
                break
            param_name = open_match.group(1)
            param_dict[param_name] = invoke_str[body_start:end_idx]
            string_flags[param_name] = open_match.group(2)
            pos = end_idx + end_len
        return param_dict, string_flags

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
        param_dict, _ = self._parse_invoke_params_with_flags(invoke_str)
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

            pos = 0
            n = len(model_output)
            start_len = len(self.tool_call_start_token)
            end_len = len(self.tool_call_end_token)
            while pos < n:
                sec_start = model_output.find(self.tool_call_start_token, pos)
                if sec_start == -1:
                    break
                body_start = sec_start + start_len
                sec_end = model_output.find(self.tool_call_end_token, body_start)
                if sec_end == -1:
                    break
                tool_call_match = model_output[body_start:sec_end]
                pos = sec_end + end_len
                for invoke_name, invoke_content in self._find_complete_invokes(tool_call_match):
                    param_dict, string_flags = self._parse_invoke_params_with_flags(invoke_content)
                    # Same schema-driven typing as the streaming path so both modes agree.
                    converted = self._convert_params_with_schema(
                        invoke_name,
                        param_dict,
                        request,
                        invoke_content,
                        string_flags=string_flags,
                    )
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
            logger.exception("Error extracting tool calls")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    def _convert_param_value(self, value: str, param_type: str, *, allow_null: bool = True) -> Any:
        """Convert parameter value to the correct Python type.

        Converts a string value to the appropriate type based on the
        parameter type specification from the tool definition.

        Args:
            value: The string value to convert.
            param_type: The target type (e.g., "string", "integer",
                "number", "boolean", "object", "array").
            allow_null: Whether literal ``"null"`` may be coerced to ``None`` for
                non-string types.

        Returns:
            Any: The converted value in the appropriate Python type.
                Falls back to string if conversion fails.
        """
        param_type = param_type.lower()
        if value.strip().lower() == "null":
            if allow_null and param_type not in _STRING_SCHEMA_TYPES:
                return None
            return value

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
                if not math.isfinite(val):
                    return value
                return val if val != int(val) else int(val)
            except (ValueError, TypeError, OverflowError):
                return value
        if param_type in ["boolean", "bool"]:
            return value.lower() in ["true", "1"]
        if param_type == "object":
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else value
            except (json.JSONDecodeError, ValueError, RecursionError):
                return value
        if param_type == "array":
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else value
            except (json.JSONDecodeError, ValueError, RecursionError):
                return value

        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError, RecursionError):
            return value

    def _convert_params_with_schema(
        self,
        function_name: str,
        param_dict: dict[str, str],
        request: ChatCompletionRequest | None,
        invoke_body: str = "",
        *,
        string_flags: dict[str, str] | None = None,
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
            string_flags: Optional pre-parsed mapping of parameter name to its DSML
                ``string`` attribute (``"true"`` or ``"false"``).

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
                        props = schema["properties"]
                        if isinstance(props, dict):
                            param_config = props
                    break

        if string_flags is None:
            _, string_flags = self._parse_invoke_params_with_flags(invoke_body)
        converted: dict[str, Any] = {}
        for name, value in param_dict.items():
            schema = param_config.get(name)
            param_type = schema.get("type") if isinstance(schema, dict) else None
            string_flag = string_flags.get(name)

            if isinstance(param_type, list):
                lower_types = [kind.strip().lower() for kind in param_type if isinstance(kind, str)]
                has_string = any(kind in _STRING_SCHEMA_TYPES for kind in lower_types)
                allows_null = "null" in lower_types or (isinstance(schema, dict) and schema.get("nullable") is True)
                if string_flag == "true" and has_string:
                    converted[name] = value
                    continue
                if string_flag == "false":
                    if allows_null and value.strip().lower() == "null":
                        converted[name] = None
                        continue
                    non_string_types = [
                        kind for kind in lower_types if kind != "null" and kind not in _STRING_SCHEMA_TYPES
                    ]
                    if non_string_types:
                        coerced: Any = value
                        for candidate_type in non_string_types:
                            if candidate_type in {"boolean", "bool"} and value.strip().lower() not in {
                                "true",
                                "false",
                                "1",
                                "0",
                            }:
                                continue
                            candidate = self._convert_param_value(value, candidate_type, allow_null=False)
                            if not isinstance(candidate, str) or candidate_type in _STRING_SCHEMA_TYPES:
                                coerced = candidate
                                break
                        converted[name] = coerced
                        continue
                    if has_string:
                        converted[name] = value
                        continue
                if allows_null and value.strip().lower() == "null":
                    converted[name] = None
                    continue
                param_type = next((kind for kind in lower_types if kind != "null"), None)
                if isinstance(param_type, str):
                    normalized_type = param_type.strip().lower()
                    converted[name] = (
                        value
                        if normalized_type in _STRING_SCHEMA_TYPES
                        else self._convert_param_value(value, param_type, allow_null=allows_null)
                    )
                    continue
                if lower_types:
                    converted[name] = value
                    continue

            if isinstance(param_type, str):
                # A schema-typed string, including the literal "null", stays a string.
                normalized_type = param_type.strip().lower()
                converted[name] = (
                    value if normalized_type in _STRING_SCHEMA_TYPES else self._convert_param_value(value, param_type)
                )
            elif string_flag == "false":
                try:
                    converted[name] = json.loads(value)
                except (ValueError, TypeError, RecursionError):
                    converted[name] = value
            else:
                converted[name] = value
        return converted

    def _collect_section_bodies(self, current_text: str) -> tuple[list[tuple[int, int, bool]], list[tuple[int, int]]]:
        """Partition *current_text* into DSML function_calls sections and visible content spans."""
        sections: list[tuple[int, int, bool]] = []
        visible_parts: list[tuple[int, int]] = []
        pos = 0
        while pos < len(current_text):
            sec_start = current_text.find(self.tool_call_start_token, pos)
            if sec_start == -1:
                tail = current_text[pos:]
                safe_tail = _strip_trailing_partial_marker(tail, self.tool_call_start_token)
                if safe_tail:
                    visible_parts.append((pos, pos + len(safe_tail)))
                if len(safe_tail) < len(tail):
                    self._pending_tail_start = pos + len(safe_tail)
                break
            if sec_start > pos:
                visible_parts.append((pos, sec_start))
            body_start = sec_start + len(self.tool_call_start_token)
            sec_end = current_text.find(self.tool_call_end_token, body_start)
            if sec_end == -1:
                sections.append((body_start, len(current_text), False))
                break
            sections.append((body_start, sec_end, True))
            pos = sec_end + len(self.tool_call_end_token)
        return sections, visible_parts

    def _extract_delta_tool_calls(
        self,
        current_text: str,
        request: ChatCompletionRequest | None,
        *,
        sections: list[tuple[int, int, bool]] | None = None,
    ) -> list[DeltaToolCall]:
        """Extract DeltaToolCalls from newly completed invoke blocks inside DSML sections.

        Scans ``<｜DSML｜function_calls>`` blocks in ``current_text`` for complete
        ``<invoke>...</invoke>`` blocks that have not yet been emitted, parses their
        parameters, converts values using the tool schema, and returns one
        ``DeltaToolCall`` per newly completed invoke.

        Args:
            current_text: The full accumulated model output so far.
            request: The chat completion request with tool definitions for
                schema-based type conversion, or None.
            sections: Optional precomputed list of ``(body_start, body_end, is_closed)``
                spans for DSML function_calls blocks.

        Returns:
            List of ``DeltaToolCall`` objects for each newly completed invoke
            block. Empty if no new blocks were found.
        """
        if sections is None:
            sections, _ = self._collect_section_bodies(current_text)

        complete_invokes: list[tuple[str, str]] = []
        for body_start, body_end, _is_closed in sections:
            complete_invokes.extend(self._find_complete_invokes(current_text[body_start:body_end]))

        delta_tool_calls: list[DeltaToolCall] = []

        while len(complete_invokes) > self.current_tool_index:
            invoke_name, invoke_body = complete_invokes[self.current_tool_index]
            param_dict, string_flags = self._parse_invoke_params_with_flags(invoke_body)
            converted = self._convert_params_with_schema(
                invoke_name,
                param_dict,
                request,
                invoke_body,
                string_flags=string_flags,
            )
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
        until a complete ``<invoke>...</invoke>`` block is available inside a
        ``<｜DSML｜function_calls>`` section, then the block is parsed and
        emitted in one shot as a ``DeltaToolCall``. Content tokens outside DSML
        sections are forwarded as regular content deltas while withholding
        partial ``<｜DSML｜function_calls>`` marker prefixes.

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
            self._emitted_content_pos = 0
        elif self._emitted_content_pos is None:
            self._emitted_content_pos = len(previous_text)

        self._streaming_initialized = True
        self.streaming_request = request
        self._last_current_text = current_text
        self._pending_tail_start = None

        sections, visible_parts = self._collect_section_bodies(current_text)

        content_chunks: list[str] = []
        emitted_pos = self._emitted_content_pos or 0
        for seg_start, seg_end in visible_parts:
            if seg_end > emitted_pos:
                content_chunks.append(current_text[max(seg_start, emitted_pos) : seg_end])
                emitted_pos = seg_end
        self._emitted_content_pos = emitted_pos
        content_out = "".join(content_chunks) or None

        has_sections = bool(sections)
        is_open_section = has_sections and not sections[-1][2]
        self.is_tool_call_started = is_open_section
        self.json_started = has_sections
        self.json_closed = has_sections and not is_open_section

        if not has_sections:
            return DeltaMessage(content=content_out) if content_out else None

        last_body_start, last_body_end, _ = sections[-1]
        active_section_text = current_text[last_body_start:last_body_end]
        invoke_start = active_section_text.rfind(self.dsml_start_check + "invoke")
        self.in_function = invoke_start > active_section_text.rfind(self.invoke_end_token)
        if invoke_start >= 0:
            invoke_text = active_section_text[invoke_start:]
            name_match = re.match(re.escape(self.dsml_start_check) + r'invoke\s+name="([^"]+)"', invoke_text)
            self.current_function_name = name_match.group(1) if name_match else None
            self.param_count = len(self._parse_invoke_params_with_flags(invoke_text)[0])
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
        else:
            self.in_param = False

        try:
            delta_tool_calls = self._extract_delta_tool_calls(current_text, request, sections=sections)
        except Exception:
            self.failed_count += 1
            logger.exception("Error extracting streaming DSML tool calls")
            return None

        if delta_tool_calls or content_out:
            return DeltaMessage(content=content_out, tool_calls=delta_tool_calls)
        if not delta_text and delta_token_ids and self.prev_tool_call_arr:
            return DeltaMessage(content="")
        return None
