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

"""Qwen3 Coder tool parser aligned with vLLM."""

from __future__ import annotations

import ast
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
    ToolDefinition,
)
from ..abstract_tool import ToolParser, ToolParserManager

logger = get_logger(__name__)


@ToolParserManager.register_module(["qwen3_coder"])  # pyright: ignore[reportUntypedClassDecorator]
class Qwen3CoderToolParser(ToolParser):
    """Tool parser for Qwen3 Coder XML tool calls.

    Parses tool calls emitted in ``<tool_call><function=name>...</function></tool_call>``
    XML format with typed ``<parameter=name>value</parameter>`` tags. Supports
    both complete (non-streaming) and incremental (streaming) extraction with
    schema-aware type conversion of parameter values.

    Attributes:
        tool_call_start_token: Opening tag that wraps a tool call block.
        tool_call_end_token: Closing tag that wraps a tool call block.
        tool_call_prefix: Prefix identifying a function invocation.
        function_end_token: Closing tag for a function invocation.
        parameter_prefix: Prefix identifying a parameter definition.
        parameter_end_token: Closing tag for a parameter definition.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the Qwen3 Coder tool parser.

        Sets up XML token markers, regex patterns for matching tool call
        structures, and validates that start/end tokens exist in the
        tokenizer vocabulary.

        Args:
            tokenizer: The tokenizer associated with the Qwen3 Coder model.

        Raises:
            ValueError: If the tokenizer is not provided.
            RuntimeError: If tool call start/end tokens are not found in
                the tokenizer vocabulary.
        """
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        # Override the base class type: vLLM uses string call IDs here.
        self.current_tool_id: str | None = None  # type: ignore[assignment]
        self.streamed_args_for_tool: list[str] = []

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"
        self.is_tool_call_started: bool = False
        self.failed_count: int = 0

        self._reset_streaming_state()

        self.tool_call_complete_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL)
        self.tool_call_function_regex = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError("Qwen3 XML Tool parser could not locate tool call start/end tokens in the tokenizer!")

        logger.debug("vLLM Successfully import tool parser %s !", self.__class__.__name__)

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self):
        """Reset all streaming state for a new message.

        Clears tool indices, function tracking flags, accumulated text and
        parameter buffers, and JSON emission state. Called at the start of
        each new streaming response.
        """
        self.current_tool_index = 0
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
        self.accumulated_params: dict[str, str] = {}
        self.streaming_request: ChatCompletionRequest | None = None

    def _get_arguments_config(self, func_name: str, tools: list[ToolDefinition] | None) -> dict:
        """Extract the parameter properties schema for a function.

        Looks up ``func_name`` in the provided tool definitions and returns
        the ``properties`` dict from its JSON Schema parameters, which maps
        parameter names to their type metadata.

        Args:
            func_name: The name of the function to look up.
            tools: List of tool definitions from the request, or None.

        Returns:
            A dict mapping parameter names to their schema definitions,
            or an empty dict if the function is not found or has no schema.
        """
        if tools is None:
            return {}
        for config in tools:
            if not hasattr(config, "type") or not (hasattr(config, "function") and hasattr(config.function, "name")):
                continue
            if config.type == "function" and config.function.name == func_name:
                if not hasattr(config.function, "parameters"):
                    return {}
                params = config.function.parameters
                if isinstance(params, dict) and "properties" in params:
                    return params["properties"]
                elif isinstance(params, dict):
                    return params
                else:
                    return {}
        logger.debug("Tool '%s' is not defined in the tools list.", func_name)
        return {}

    def _convert_param_value(self, param_value: str, param_name: str, param_config: dict, func_name: str) -> Any:
        """Convert a raw string parameter value to its schema-defined type.

        Uses the parameter's type from ``param_config`` to coerce the string
        into the appropriate Python type (int, float, bool, dict, list, etc.).
        Falls back to the raw string or ``ast.literal_eval`` when standard
        conversions fail.

        Args:
            param_value: The raw string value to convert.
            param_name: Name of the parameter (for schema lookup and logging).
            param_config: Parameter properties schema from the tool definition.
            func_name: Name of the parent function (for logging).

        Returns:
            The converted value in the appropriate Python type, or the
            original string if conversion is not possible.
        """
        if param_value.lower() == "null":
            return None
        if param_name not in param_config:
            if param_config != {}:
                logger.debug(
                    "Parsed parameter '%s' is not defined in the tool parameters for tool '%s', directly "
                    "returning the string value.",
                    param_name,
                    func_name,
                )
            return param_value
        if isinstance(param_config[param_name], dict) and "type" in param_config[param_name]:
            param_type = str(param_config[param_name]["type"]).strip().lower()
        elif isinstance(param_config[param_name], dict) and "anyOf" in param_config[param_name]:
            param_type = "object"
        else:
            param_type = "string"
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                return int(param_value)
            except (ValueError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not an integer in tool '%s', degenerating to string.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_param_value = float(param_value)
                return float_param_value if float_param_value - int(float_param_value) != 0 else int(float_param_value)
            except (ValueError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not a float in tool '%s', degenerating to string.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not a boolean (`true` or `false`) in tool '%s', "
                    "degenerating to false.",
                    param_value,
                    param_name,
                    func_name,
                )
            return param_value == "true"
        else:
            if (
                param_type in ["object", "array", "arr"]
                or param_type.startswith("dict")
                or param_type.startswith("list")
            ):
                try:
                    return json.loads(param_value)
                except (json.JSONDecodeError, TypeError, ValueError):
                    logger.debug(
                        "Parsed value '%s' of parameter '%s' cannot be parsed with json.loads in tool '%s', "
                        "will try other methods to parse it.",
                        param_value,
                        param_name,
                        func_name,
                    )
            try:
                param_value = ast.literal_eval(param_value)
            except (ValueError, SyntaxError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' cannot be converted via Python ast.literal_eval() "
                    "in tool '%s', degenerating to string.",
                    param_value,
                    param_name,
                    func_name,
                )
            return param_value

    def _parse_xml_function_call(self, function_call_str: str, tools: list[ToolDefinition] | None) -> ToolCall | None:
        """Parse a single XML function call string into a ToolCall object.

        Extracts the function name and all ``<parameter=...>`` tags from
        ``function_call_str``, converts parameter values using the tool
        schema, and returns a structured ``ToolCall``.

        Args:
            function_call_str: The inner content of a ``<function=...>``
                block (everything after ``<function=``).
            tools: Tool definitions for schema-based type conversion.

        Returns:
            A ``ToolCall`` with the function name and JSON-serialized
            arguments, or None if the function name cannot be extracted.
        """
        end_index = function_call_str.find(">")
        if end_index == -1:
            return None
        function_name = function_call_str[:end_index]
        param_config = self._get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1 :]
        param_dict: dict[str, Any] = {}
        for match_text in self.tool_call_parameter_regex.findall(parameters):
            idx = match_text.index(">")
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1 :])
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]
            param_dict[param_name] = self._convert_param_value(param_value, param_name, param_config, function_name)
        return ToolCall(
            type="function",
            function=FunctionCall(name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False)),
        )

    def _get_function_calls(self, model_output: str) -> list[str]:
        """Extract raw function-call body strings from model output.

        Finds all ``<tool_call>...</tool_call>`` regions, then extracts
        ``<function=...>`` blocks from within each. Falls back to treating
        the entire output as a single tool call region if no wrappers are
        found.

        Args:
            model_output: The complete model output text.

        Returns:
            List of raw function body strings (content after ``<function=``).
        """
        matched_ranges = self.tool_call_regex.findall(model_output)
        raw_tool_calls = [match[0] if match[0] else match[1] for match in matched_ranges]

        if len(raw_tool_calls) == 0:
            raw_tool_calls = [model_output]

        raw_function_calls = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(self.tool_call_function_regex.findall(tool_call))

        return [match[0] if match[0] else match[1] for match in raw_function_calls]

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete (non-streaming) model output.

        Searches for ``<function=...>`` blocks within the output, parses
        each into a ``ToolCall``, and separates any leading content text.

        Args:
            model_output: The full model output text.
            request: The chat completion request containing tool definitions.

        Returns:
            An ``ExtractedToolCallInformation`` with parsed tool calls and
            any content preceding the first tool call block.
        """
        if self.tool_call_prefix not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            function_calls = self._get_function_calls(model_output)
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

            content_index = model_output.find(self.tool_call_start_token)
            idx = model_output.find(self.tool_call_prefix)
            content_index = content_index if content_index >= 0 else idx
            content = model_output[:content_index]
            valid_tool_calls = [tool_call for tool_call in tool_calls if tool_call is not None]

            return ExtractedToolCallInformation(
                tools_called=(len(valid_tool_calls) > 0),
                tool_calls=valid_tool_calls,
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
        """Extract tool calls incrementally from streaming model output.

        Tracks streaming state across calls to detect when tool call tokens
        appear, parses function headers and parameters as they complete, and
        emits JSON argument fragments progressively. Content tokens before
        the first tool call are forwarded as regular text deltas.

        Args:
            previous_text: All text generated before this step.
            current_text: All text generated up to and including this step.
            delta_text: The new text added in this step.
            previous_token_ids: Token IDs for ``previous_text``.
            current_token_ids: Token IDs for ``current_text``.
            delta_token_ids: Token IDs for ``delta_text``.
            request: The chat completion request with tool definitions.

        Returns:
            A ``DeltaMessage`` with content text, tool call deltas, or both.
            Returns None when there is nothing new to emit.
        """
        if not previous_text:
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
            tool_ends = current_text.count(self.tool_call_end_token)
            if tool_ends > self.current_tool_index:
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.accumulated_params = {}
                tool_starts = current_text.count(self.tool_call_start_token)
                if self.current_tool_index >= tool_starts:
                    self.is_tool_call_started = False
                return None

        if not self.is_tool_call_started:
            if self.tool_call_start_token_id in delta_token_ids or self.tool_call_start_token in delta_text:
                self.is_tool_call_started = True
                if self.tool_call_start_token in delta_text:
                    content_before = delta_text[: delta_text.index(self.tool_call_start_token)]
                    if content_before:
                        return DeltaMessage(content=content_before)
                return None
            else:
                if current_text.rstrip().endswith(self.tool_call_end_token) and delta_text.strip() == "":
                    return None
                return DeltaMessage(content=delta_text)

        tool_starts_count = current_text.count(self.tool_call_start_token)
        if self.current_tool_index >= tool_starts_count:
            return None

        tool_start_positions: list[int] = []
        idx = 0
        while True:
            idx = current_text.find(self.tool_call_start_token, idx)
            if idx == -1:
                break
            tool_start_positions.append(idx)
            idx += len(self.tool_call_start_token)

        if self.current_tool_index >= len(tool_start_positions):
            return None

        tool_start_idx = tool_start_positions[self.current_tool_index]
        tool_end_idx = current_text.find(self.tool_call_end_token, tool_start_idx)
        if tool_end_idx == -1:
            tool_text = current_text[tool_start_idx:]
        else:
            tool_text = current_text[tool_start_idx : tool_end_idx + len(self.tool_call_end_token)]

        if not self.header_sent:
            if self.tool_call_prefix in tool_text:
                func_start = tool_text.find(self.tool_call_prefix) + len(self.tool_call_prefix)
                func_end = tool_text.find(">", func_start)
                if func_end != -1:
                    self.current_function_name = tool_text[func_start:func_end]
                    self.current_tool_id = self._generate_tool_call_id()
                    self.header_sent = True
                    self.in_function = True
                    self.prev_tool_call_arr.append(
                        {
                            "name": self.current_function_name,
                            "arguments": "{}",
                        }
                    )
                    self.streamed_args_for_tool.append("")
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
            if not self.json_started:
                self.json_started = True
                if self.current_tool_index < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[self.current_tool_index] += "{"
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments="{"),
                        )
                    ]
                )

            param_starts: list[int] = []
            search_idx = 0
            while True:
                search_idx = tool_text.find(self.parameter_prefix, search_idx)
                if search_idx == -1:
                    break
                param_starts.append(search_idx)
                search_idx += len(self.parameter_prefix)

            json_fragments: list[str] = []
            while not self.in_param and self.param_count < len(param_starts):
                param_idx = param_starts[self.param_count]
                param_start = param_idx + len(self.parameter_prefix)
                remaining = tool_text[param_start:]
                if ">" not in remaining:
                    break

                name_end = remaining.find(">")
                current_param_name = remaining[:name_end]

                value_start = param_start + name_end + 1
                value_text = tool_text[value_start:]
                if value_text.startswith("\n"):
                    value_text = value_text[1:]

                param_end_idx = value_text.find(self.parameter_end_token)
                if param_end_idx == -1:
                    next_param_idx = value_text.find(self.parameter_prefix)
                    func_end_idx = value_text.find(self.function_end_token)
                    if next_param_idx != -1 and (func_end_idx == -1 or next_param_idx < func_end_idx):
                        param_end_idx = next_param_idx
                    elif func_end_idx != -1:
                        param_end_idx = func_end_idx
                    else:
                        tool_end_in_value = value_text.find(self.tool_call_end_token)
                        if tool_end_in_value != -1:
                            param_end_idx = tool_end_in_value
                        else:
                            break
                if param_end_idx == -1:
                    break

                param_value = value_text[:param_end_idx]
                if param_value.endswith("\n"):
                    param_value = param_value[:-1]

                self.current_param_name = current_param_name
                self.accumulated_params[current_param_name] = param_value
                param_config = self._get_arguments_config(
                    self.current_function_name or "",
                    self.streaming_request.tools if self.streaming_request else None,
                )
                converted_value = self._convert_param_value(
                    param_value,
                    current_param_name,
                    param_config,
                    self.current_function_name or "",
                )
                serialized_value = json.dumps(converted_value, ensure_ascii=False)

                if self.param_count == 0:
                    json_fragment = f'"{current_param_name}": {serialized_value}'
                else:
                    json_fragment = f', "{current_param_name}": {serialized_value}'

                self.param_count += 1
                json_fragments.append(json_fragment)

            if json_fragments:
                combined = "".join(json_fragments)
                if self.current_tool_index < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[self.current_tool_index] += combined
                else:
                    logger.warning(
                        "streamed_args_for_tool out of sync: index=%d len=%d",
                        self.current_tool_index,
                        len(self.streamed_args_for_tool),
                    )
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments=combined),
                        )
                    ]
                )

            if not self.json_closed and self.function_end_token in tool_text:
                self.json_closed = True
                func_start = tool_text.find(self.tool_call_prefix) + len(self.tool_call_prefix)
                func_content_end = tool_text.find(self.function_end_token, func_start)
                if func_content_end != -1:
                    func_content = tool_text[func_start:func_content_end]
                    try:
                        parsed_tool = self._parse_xml_function_call(
                            func_content,
                            self.streaming_request.tools if self.streaming_request else None,
                        )
                        if parsed_tool and self.current_tool_index < len(self.prev_tool_call_arr):
                            self.prev_tool_call_arr[self.current_tool_index]["arguments"] = (
                                parsed_tool.function.arguments
                            )
                    except Exception:
                        logger.debug("Failed to parse tool call during streaming: %s", tool_text, exc_info=True)

                if self.current_tool_index < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[self.current_tool_index] += "}"
                else:
                    logger.warning(
                        "streamed_args_for_tool out of sync: index=%d len=%d",
                        self.current_tool_index,
                        len(self.streamed_args_for_tool),
                    )

                result = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments="}"),
                        )
                    ]
                )

                self.in_function = False
                self.json_closed = True
                self.accumulated_params = {}
                return result

        return None
