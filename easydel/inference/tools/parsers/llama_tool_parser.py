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
from collections.abc import Sequence
from uuid import uuid4

import partial_json_parser
from eformer.loggings import get_logger
from partial_json_parser.core.options import Allow
from transformers import PreTrainedTokenizerBase

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
from ..utils import find_common_prefix, is_complete_json, partial_json_loads

logger = get_logger(__name__)


@ToolParserManager.register_module("llama3_json")
@ToolParserManager.register_module("llama4_json")
class Llama3JsonToolParser(ToolParser):
    """
    Tool call parser for Llama 3.x and 4 models with JSON format.

    Intended for use with the examples/tool_chat_template_llama.jinja template.
    Handles JSON-formatted tool calls with support for both single and multiple
    tool invocations separated by semicolons.

    Format supported:
    - Single: {"name": "func", "arguments": {...}}
    - Multiple: {"name": "func1", ...}; {"name": "func2", ...}
    - Uses <|python_tag|> token as optional marker
    - Supports both "arguments" and "parameters" field names

    Used when --enable-auto-tool-choice --tool-call-parser llama3_json or
    llama4_json are set.

    Attributes:
        bot_token (str): Special token marking tool calls
        tool_call_regex (re.Pattern): Pattern for extracting JSON tool calls
        prev_tool_call_arr (list): Previous tool calls for streaming comparison
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
        self.bot_token = "<|python_tag|>"
        self.bot_token_id = tokenizer.encode(self.bot_token, add_special_tokens=False)[0]
        self.tool_call_regex = re.compile(
            r"{[^{}]*(?:{[^{}]*}[^{}]*)*}(?:\s*;\s*{[^{}]*(?:{[^{}]*}[^{}]*)*})*", re.DOTALL
        )

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Extract tool calls from complete Llama model response.

        Extracts JSON content and ignores surrounding plain text.
        Supports both single JSON and multiple JSONs separated by semicolons.
        Handles both "arguments" and "parameters" field names for compatibility.

        Args:
            model_output: Complete model output text
            request: Original request (unused)

        Returns:
            ExtractedToolCallInformation with parsed JSON tool calls
        """
        if not (self.bot_token in model_output or "{" in model_output):
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        match = self.tool_call_regex.search(model_output)
        if not match:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            json_str = match.group(0)
            json_objects = [obj.strip() for obj in json_str.split(";")]

            tool_calls: list[ToolCall] = []
            for json_obj in json_objects:
                if not json_obj:
                    continue
                obj = json.loads(json_obj)
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=obj["name"],
                            arguments=json.dumps(
                                obj["arguments"] if "arguments" in obj else obj["parameters"], ensure_ascii=False
                            ),
                        ),
                    )
                )

            return ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls, content=None)

        except Exception:
            pass
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
        if not (current_text.startswith(self.bot_token) or current_text.startswith("{")):
            return DeltaMessage(content=delta_text)

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            tool_call_arr = []
            is_complete = []
            try:
                start_idx = len(self.bot_token) if current_text.startswith(self.bot_token) else 0
                while start_idx < len(current_text):
                    (obj, end_idx) = partial_json_loads(current_text[start_idx:], flags)
                    is_complete.append(is_complete_json(current_text[start_idx : start_idx + end_idx]))
                    start_idx += end_idx + len("; ")
                    if "parameters" in obj:
                        assert "arguments" not in obj, "model generated both parameters and arguments"
                        obj["arguments"] = obj["parameters"]
                    tool_call_arr.append(obj)
            except partial_json_parser.core.exceptions.MalformedJSON:
                pass
                return None

            current_tool_call: dict = tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}

            if len(tool_call_arr) == 0:
                return None

            elif len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1:
                if self.current_tool_id >= 0:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                        sent = len(self.streamed_args_for_tool[self.current_tool_id])
                        argument_diff = cur_args_json[sent:]

                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(arguments=argument_diff).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += argument_diff
                    else:
                        delta = None
                else:
                    delta = None
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                return delta

            elif not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=f"chatcmpl-tool-{uuid4()}",
                                function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.current_tool_name_sent = True
                else:
                    delta = None

            else:
                cur_arguments = current_tool_call.get("arguments")
                delta = None

                if cur_arguments:
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get("arguments")

                    argument_diff = None
                    if is_complete[self.current_tool_id]:
                        argument_diff = cur_args_json[sent:]
                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                        if cur_args_json != prev_args_json:
                            prefix = find_common_prefix(prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]

                    if argument_diff is not None:
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(arguments=argument_diff).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += argument_diff

            self.prev_tool_call_arr = tool_call_arr
            return delta

        except Exception:
            pass
            return None
