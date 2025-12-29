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
    """
    Example tool call content:

    <｜DSML｜function_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
    </｜DSML｜invoke>
    ...
    </｜DSML｜function_calls>
    """

    def __init__(self, tokenizer: AnyTokenizer):
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
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self):
        """Reset all streaming state."""
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
        param_dict: dict[str, str] = {}
        for param_name, param_val in self.parameter_complete_regex.findall(invoke_str):
            param_dict[param_name] = param_val
        return param_dict

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output (non-streaming)."""
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
        """Extract name from quoted string."""
        name_str = name_str.strip()
        if (name_str.startswith('"') and name_str.endswith('"')) or (
            name_str.startswith("'") and name_str.endswith("'")
        ):
            return name_str[1:-1]
        return name_str

    def _extract_param_name(self, input_str: str) -> str:
        """Extract param name."""
        start = input_str.find('"') + 1
        end = input_str.find('"', start)
        return input_str[start:end] if start > 0 and end > start else input_str

    def _convert_param_value(self, value: str, param_type: str) -> Any:
        """Convert parameter value to the correct type."""
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
        """Extract tool calls from streaming model output."""

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
