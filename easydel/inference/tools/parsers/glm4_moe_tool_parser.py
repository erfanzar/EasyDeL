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

import ast
import json
import re
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


@ToolParserManager.register_module("glm-4.5")
class Glm4MoeModelToolParser(ToolParser):
    """
    Tool parser for GLM-4 MoE (Mixture of Experts) models.

    Handles the GLM-4 specific tool call format which uses XML-like tags:
    - Tool calls wrapped in <tool_call> and </tool_call>
    - Arguments wrapped in <arg_key> and <arg_value> tags
    - Supports automatic type conversion based on tool parameter definitions

    The parser maintains streaming state and can handle incremental
    generation of tool calls during streaming responses.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.current_tool_name_sent = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id = -1
        self.streamed_args_for_tool: list[str] = []
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"

        self.tool_calls_start_token = self.tool_call_start_token

        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
        self.func_detail_regex = re.compile(r"<tool_call>([^\n]*)\n(.*)</tool_call>", re.DOTALL)
        self.func_arg_regex = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        self._buffer = ""

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from complete GLM-4 model output.

        Parses XML-like structured format with tool names and key-value
        argument pairs. Automatically deserializes argument values based
        on tool parameter type definitions.

        Args:
            model_output: Complete text from the model
            request: Chat request containing tool definitions

        Returns:
            Extracted tool call information with parsed functions
        """

        def _is_string_type(tool_name: str, arg_name: str, tools: list[ToolDefinition] | None) -> bool:
            if tools is None:
                return False
            for tool in tools:
                if tool.function.name == tool_name:
                    if tool.function.parameters is None:
                        return False
                    arg_type = tool.function.parameters.get("properties", {}).get(arg_name, {}).get("type", None)
                    return arg_type == "string"
            return False

        def _deserialize(value: str) -> Any:
            try:
                return json.loads(value)
            except Exception:
                pass

            try:
                return ast.literal_eval(value)
            except Exception:
                pass
            return value

        matched_tool_calls = self.func_call_regex.findall(model_output)
        try:
            tool_calls = []
            for match in matched_tool_calls:
                tc_detail = self.func_detail_regex.search(match)
                tc_name = tc_detail.group(1)
                tc_args = tc_detail.group(2)
                pairs = self.func_arg_regex.findall(tc_args)
                arg_dct = {}
                for key, value in pairs:
                    arg_key = key.strip()
                    arg_val = value.strip()
                    if not _is_string_type(tc_name, arg_key, request.tools):
                        arg_val = _deserialize(arg_val)
                    arg_dct[arg_key] = arg_val
                tool_calls.append(
                    ToolCall(type="function", function=FunctionCall(name=tc_name, arguments=json.dumps(arg_dct)))
                )
        except Exception:
            logger.exception("Failed to extract tool call spec")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        else:
            if len(tool_calls) > 0:
                content = model_output[: model_output.find(self.tool_calls_start_token)]
                return ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls, content=content)
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
        """
        Handle streaming extraction of GLM-4 tool calls.

        Uses a buffer-based approach to accumulate partial tool calls
        and emit complete tool information when boundaries are detected.

        Args:
            previous_text: Previously generated text
            current_text: All text generated so far
            delta_text: New text in this chunk
            previous_token_ids: Previous token IDs
            current_token_ids: All token IDs
            delta_token_ids: New token IDs
            request: Original request

        Returns:
            Delta message with tool updates or None
        """
        self._buffer += delta_text
        cur_text = self._buffer
        start_idx = cur_text.find(self.tool_call_start_token)
        if start_idx == -1:
            self._buffer = ""
            if self.current_tool_id > 0:
                cur_text = ""
            return DeltaMessage(content=cur_text)
        end_idx = cur_text.find(self.tool_call_end_token)
        if end_idx != -1:
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = []
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

            extracted_tool_calls = self.extract_tool_calls(cur_text[: end_idx + len(self.tool_call_end_token)], request)

            if len(extracted_tool_calls.tool_calls) == 0:
                return None
            tool_call = extracted_tool_calls.tool_calls[0]
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments),
            }
            self.streamed_args_for_tool[self.current_tool_id] = tool_call.function.arguments
            delta = DeltaMessage(
                content=extracted_tool_calls.content,
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        id=tool_call.id,
                        type=tool_call.type,
                        function=DeltaFunctionCall(name=tool_call.function.name, arguments=tool_call.function.arguments),
                    )
                ],
            )
            self.current_tool_id += 1
            self._buffer = cur_text[end_idx + len(self.tool_call_end_token) :]
            return delta

        self._buffer = cur_text[start_idx:]
        return DeltaMessage(content=cur_text[:start_idx])
