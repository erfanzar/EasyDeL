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
from ._schema_params import lookup_tool_properties

logger = get_logger(__name__)


def _strip_param_newlines(value: str) -> str:
    """Strip at most one leading and one trailing newline, preserving indentation."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _trailing_prefix_len(text: str, marker: str) -> int:
    """Return length of the longest proper prefix of *marker* matching the end of *text*."""
    max_len = min(len(text), len(marker) - 1)
    for size in range(max_len, 0, -1):
        if text.endswith(marker[:size]):
            return size
    return 0


@ToolParserManager.register_module("minimax_m2")  # pyright: ignore[reportUntypedClassDecorator]
class MinimaxM2ToolParser(ToolParser):
    """Tool parser for Minimax M2 models with XML-style tool calls.

    Parses tool calls that use ``<minimax:tool_call>``, ``<invoke>``, and
    ``<parameter>`` tags instead of JSON format. Supports schema-aware
    type conversion for parameters in both batch and streaming modes.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the MinimaxM2ToolParser.

        Args:
            tokenizer: The tokenizer associated with the Minimax M2 model.

        Raises:
            ValueError: If tokenizer is None.
            RuntimeError: If tool call start/end tokens are not found in vocab.
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
        self.current_tool_id: str | None = None  # type: ignore[assignment]
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
        self.in_function: bool = False
        self.accumulated_text: str = ""
        self.json_started: bool = False
        self.json_closed: bool = False
        self.accumulated_params: dict[str, Any] = {}
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

    def get_streaming_buffer_marker_pairs(self) -> Sequence[tuple[str, str | None]]:
        """Scope protocol buffering strictly to ``<minimax:tool_call>`` envelopes."""
        return ((self.tool_call_start_token, self.tool_call_end_token),)

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Return True while inside an active ``<minimax:tool_call>`` block or trailing prefix."""
        if self._streaming_initialized:
            if self._in_tool_block or self.in_function or self.is_tool_call_started:
                return True
            return self._has_trailing_marker_prefix(
                current_text[self._cursor :],
                (self.tool_call_start_token,),
            )
        return super().is_buffering_protocol(current_text=current_text, delta_text=delta_text)

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state to initial values."""
        self.current_tool_index = 0
        self.invoke_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = None
        self.current_tool_name_sent = False
        self.current_function_name = None
        self.current_param_name = None
        self.current_param_value = ""
        self.param_count = 0
        self.in_function = False
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False
        self.accumulated_params = {}
        self.streaming_request = None
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()
        self._streaming_initialized = False
        self._cursor = 0
        self._in_tool_block = False
        self._block_start = -1
        self._block_invoke_cursor = 0
        self._invoke_body_start = -1
        self._invoke_param_cursor = -1

    def flush_streaming_content(self) -> str:
        """Release withheld opening-tag prefix when the stream ends outside a tool block."""
        self.streaming_request = None
        if self._in_tool_block:
            self.accumulated_text = ""
            self._cursor = 0
            return ""
        pending = self.accumulated_text[self._cursor :]
        self.accumulated_text = ""
        self._cursor = 0
        return pending

    def _extract_name(self, name_str: str) -> str:
        """Extract a name from a quoted or unquoted string."""
        name_str = name_str.strip()
        if (name_str.startswith('"') and name_str.endswith('"')) or (
            name_str.startswith("'") and name_str.endswith("'")
        ):
            return name_str[1:-1]
        return name_str

    def _record_streamed_args(self, tool_index: int, fragment: str) -> None:
        """Append an emitted argument fragment to *tool_index*'s streamed-argument buffer."""
        while len(self.streamed_args_for_tool) <= tool_index:
            self.streamed_args_for_tool.append("")
        self.streamed_args_for_tool[tool_index] += fragment

    def _extract_types_from_schema(self, schema: Any) -> list[str]:
        """Extract possible types from a JSON schema definition."""
        if schema is None or not isinstance(schema, dict):
            return ["string"]

        types: set[str] = set()

        if schema.get("nullable") is True:
            types.add("null")

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
        """Convert a parameter value based on possible schema types."""
        normalized_types = [t.strip().lower() for t in param_types]
        stripped = value.strip()

        if stripped.lower() == "null":
            if "null" in normalized_types:
                return None
            if any(t in normalized_types for t in ("string", "str", "text")):
                return value
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
                    return int(stripped)
                except (ValueError, TypeError):
                    continue
            if param_type in ["number", "float"]:
                try:
                    val = float(stripped)
                    if not math.isfinite(val):
                        continue
                    return val if val != int(val) else int(val)
                except (ValueError, TypeError, OverflowError):
                    continue
            if param_type in ["boolean", "bool"]:
                lower_val = stripped.lower()
                if lower_val in ["true", "1", "yes", "on"]:
                    return True
                if lower_val in ["false", "0", "no", "off"]:
                    return False
                continue
            if param_type == "object":
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, ValueError, RecursionError):
                    pass
                continue
            if param_type == "array":
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return parsed
                except (json.JSONDecodeError, ValueError, RecursionError):
                    pass
                continue

        return value

    def _get_param_types_from_config(self, param_name: str, param_config: dict) -> list[str]:
        """Get possible types for a parameter from tool configuration."""
        if param_name not in param_config:
            return ["string"]
        param_schema = param_config[param_name]
        if not isinstance(param_schema, dict):
            return ["string"]
        return self._extract_types_from_schema(param_schema)

    def _parse_single_invoke(self, invoke_str: str, tools: list | None) -> ToolCall | None:
        """Parse a single invoke element into a ToolCall."""
        name_match = re.search(r"^([^>]+)", invoke_str)
        if not name_match:
            return None

        function_name = self._extract_name(name_match.group(1))
        param_config = lookup_tool_properties(function_name, tools)

        param_dict: dict[str, Any] = {}
        for match in self.parameter_complete_regex.findall(invoke_str):
            param_match = re.search(r"^([^>]+)>(.*)", match, re.DOTALL)
            if not param_match:
                continue
            param_name = self._extract_name(param_match.group(1))
            param_value = _strip_param_newlines(param_match.group(2))

            param_type = self._get_param_types_from_config(param_name, param_config)
            param_dict[param_name] = self._convert_param_value_with_types(param_value, param_type)

        return ToolCall(
            type="function",
            function=FunctionCall(name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False)),
        )

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model response (stateless)."""
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
        """Extract tool calls incrementally during streaming."""
        del previous_token_ids, current_token_ids
        if not previous_text:
            self._reset_streaming_state()
            self._streaming_initialized = True
        elif not self._streaming_initialized:
            self._reset_streaming_state()
            self._cursor = len(previous_text)
            self._streaming_initialized = True

        if request is not None:
            self.streaming_request = request

        self.accumulated_text = current_text

        if not delta_text and self._cursor >= len(current_text) and not self._in_tool_block:
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

        content_parts: list[str] = []
        tool_deltas: list[DeltaToolCall] = []

        while True:
            if not self._in_tool_block:
                start_idx = current_text.find(self.tool_call_start_token, self._cursor)
                if start_idx != -1:
                    if start_idx > self._cursor:
                        content_parts.append(current_text[self._cursor : start_idx])
                    self._in_tool_block = True
                    self.is_tool_call_started = True
                    self._block_start = start_idx + len(self.tool_call_start_token)
                    self._block_invoke_cursor = self._block_start
                    self._cursor = self._block_start
                    continue

                rem = current_text[self._cursor :]
                hold = _trailing_prefix_len(rem, self.tool_call_start_token)
                safe_end = len(current_text) - hold
                if safe_end > self._cursor:
                    content_parts.append(current_text[self._cursor : safe_end])
                    self._cursor = safe_end
                break

            block_end = current_text.find(self.tool_call_end_token, self._block_invoke_cursor)
            block_limit = block_end if block_end != -1 else len(current_text)

            if not self.in_function:
                inv_idx = current_text.find(self.invoke_start_prefix, self._block_invoke_cursor, block_limit)
                if inv_idx != -1:
                    name_start = inv_idx + len(self.invoke_start_prefix)
                    name_end = current_text.find(">", name_start, block_limit)
                    if name_end == -1:
                        if block_end != -1:
                            self._in_tool_block = False
                            self.is_tool_call_started = False
                            self._cursor = block_end + len(self.tool_call_end_token)
                            self._block_invoke_cursor = self._cursor
                            continue
                        break
                    self.current_function_name = self._extract_name(current_text[name_start:name_end])
                    self.current_tool_id = self._generate_tool_call_id()
                    self.header_sent = True
                    self.current_tool_name_sent = True
                    self.in_function = True
                    self.json_started = False
                    self.json_closed = False
                    self.param_count = 0
                    self.accumulated_params = {}
                    self._invoke_body_start = name_end + 1
                    self._invoke_param_cursor = name_end + 1
                    self.invoke_index = self.current_tool_index

                    while len(self.prev_tool_call_arr) <= self.current_tool_index:
                        self.prev_tool_call_arr.append({"name": self.current_function_name, "arguments": "{}"})
                    self.prev_tool_call_arr[self.current_tool_index] = {
                        "name": self.current_function_name,
                        "arguments": "{}",
                    }
                    while len(self.streamed_args_for_tool) <= self.current_tool_index:
                        self.streamed_args_for_tool.append("")

                    tool_deltas.append(
                        DeltaToolCall(
                            index=self.current_tool_index,
                            id=self.current_tool_id,
                            function=DeltaFunctionCall(name=self.current_function_name, arguments=""),
                            type="function",
                        )
                    )
                    continue

                if block_end != -1:
                    self._in_tool_block = False
                    self.is_tool_call_started = False
                    self._cursor = block_end + len(self.tool_call_end_token)
                    continue
                break

            # Inside an active <invoke name=...>: drain parameters BEFORE checking </invoke> or </minimax:tool_call>.
            inv_end = current_text.find(self.invoke_end_token, self._invoke_body_start, block_limit)
            closed_by_block = inv_end == -1 and block_end != -1
            is_closed = inv_end != -1 or closed_by_block
            inv_limit = inv_end if inv_end != -1 else block_limit

            arg_parts: list[str] = []
            if not self.json_started:
                self.json_started = True
                arg_parts.append("{")

            tools_def = self.streaming_request.tools if self.streaming_request else None
            param_config = lookup_tool_properties(self.current_function_name, tools_def)

            while True:
                p_start = current_text.find(self.parameter_prefix, self._invoke_param_cursor, inv_limit)
                if p_start == -1:
                    break
                p_name_start = p_start + len(self.parameter_prefix)
                p_name_end = current_text.find(">", p_name_start, inv_limit)
                if p_name_end == -1:
                    break
                val_start = p_name_end + 1
                p_end = current_text.find(self.parameter_end_token, val_start, inv_limit)
                if p_end != -1:
                    next_cursor = p_end + len(self.parameter_end_token)
                else:
                    next_p = current_text.find(self.parameter_prefix, val_start, inv_limit)
                    if next_p != -1:
                        p_end = next_p
                        next_cursor = next_p
                    elif is_closed:
                        p_end = inv_limit
                        next_cursor = inv_limit
                    else:
                        break

                param_name = self._extract_name(current_text[p_name_start:p_name_end])
                param_value = _strip_param_newlines(current_text[val_start:p_end])
                self.current_param_name = param_name
                self.current_param_value = param_value

                param_type = self._get_param_types_from_config(param_name, param_config)
                converted_value = self._convert_param_value_with_types(param_value, param_type)
                self.accumulated_params[param_name] = converted_value

                encoded_key = json.dumps(param_name, ensure_ascii=False)
                encoded_val = json.dumps(converted_value, ensure_ascii=False)
                if self.param_count == 0:
                    arg_parts.append(f"{encoded_key}: {encoded_val}")
                else:
                    arg_parts.append(f", {encoded_key}: {encoded_val}")

                self.param_count += 1
                self._invoke_param_cursor = next_cursor

            if is_closed:
                arg_parts.append("}")
                combined = "".join(arg_parts)
                tool_idx = self.current_tool_index
                self._record_streamed_args(tool_idx, combined)
                if tool_idx < len(self.prev_tool_call_arr):
                    self.prev_tool_call_arr[tool_idx]["arguments"] = json.dumps(
                        self.accumulated_params, ensure_ascii=False
                    )
                if tool_deltas and tool_deltas[-1].index == tool_idx and tool_deltas[-1].function is not None:
                    prev_arg = tool_deltas[-1].function.arguments or ""
                    tool_deltas[-1].function.arguments = prev_arg + combined
                else:
                    tool_deltas.append(
                        DeltaToolCall(index=tool_idx, function=DeltaFunctionCall(arguments=combined))
                    )

                self.json_closed = True
                self.in_function = False
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.accumulated_params = {}
                if inv_end != -1:
                    self._block_invoke_cursor = inv_end + len(self.invoke_end_token)
                else:
                    self._in_tool_block = False
                    self.is_tool_call_started = False
                    self._cursor = block_end + len(self.tool_call_end_token)
                    self._block_invoke_cursor = self._cursor
                continue

            if arg_parts:
                combined = "".join(arg_parts)
                tool_idx = self.current_tool_index
                self._record_streamed_args(tool_idx, combined)
                if tool_deltas and tool_deltas[-1].index == tool_idx and tool_deltas[-1].function is not None:
                    prev_arg = tool_deltas[-1].function.arguments or ""
                    tool_deltas[-1].function.arguments = prev_arg + combined
                else:
                    tool_deltas.append(
                        DeltaToolCall(index=tool_idx, function=DeltaFunctionCall(arguments=combined))
                    )
            break

        content = "".join(content_parts)
        if content or tool_deltas:
            return DeltaMessage(content=content or None, tool_calls=tool_deltas or None)
        return None
