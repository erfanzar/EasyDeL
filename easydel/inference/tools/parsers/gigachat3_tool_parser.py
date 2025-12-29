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

REGEX_FUNCTION_CALL = re.compile(r"function call(?:<\|role_sep\|>\n)?(\{.*)", re.DOTALL)
NAME_REGEX = re.compile(r'"name"\s*:\s*"([^"]*)"', re.DOTALL)
ARGS_REGEX = re.compile(r'"arguments"\s*:\s*(.*)', re.DOTALL)


@ToolParserManager.register_module("gigachat3")
class GigaChat3ToolParser(ToolParser):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.tool_started: bool = False
        self.tool_name_sent: bool = False
        self.tool_id: str | None = None
        self.prev_tool_call_arr: list[dict] = []
        self.content_buffer: str = ""
        self.trigger_start = "function call{"

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        match = REGEX_FUNCTION_CALL.search(model_output)
        if not match:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        json_candidate = match.group(1).strip()
        try:
            data = json.loads(json_candidate)
        except json.JSONDecodeError:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        if not (isinstance(data, dict) and "name" in data and "arguments" in data):
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        name = data["name"]
        args = data["arguments"]
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)

        tool_calls = [ToolCall(type="function", function=FunctionCall(name=name, arguments=args))]
        prefix = model_output[: match.start()]
        content = prefix.rstrip() if prefix and prefix.strip() else None
        return ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls, content=content)

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
        func_name = None
        cur_args = None

        if not self.tool_started:
            match = REGEX_FUNCTION_CALL.search(current_text)
            if match:
                self.tool_started = True
                self.content_buffer = ""
            else:
                self.content_buffer += delta_text
                clean_buffer = self.content_buffer.lstrip()
                is_prefix = self.trigger_start.startswith(clean_buffer)
                starts_with_trigger = clean_buffer.startswith(self.trigger_start)
                if is_prefix or starts_with_trigger:
                    return None
                flush_text = self.content_buffer
                self.content_buffer = ""
                return DeltaMessage(content=flush_text)

        match = REGEX_FUNCTION_CALL.search(current_text)
        if not match:
            return None

        json_tail = match.group(1).strip()

        name_match = NAME_REGEX.search(json_tail)
        if name_match:
            func_name = name_match.group(1)

        args_match = ARGS_REGEX.search(json_tail)
        if args_match:
            cur_args = args_match.group(1).strip()
            if cur_args.endswith("}"):
                try:
                    candidate = cur_args[:-1].strip()
                    json.loads(candidate)
                    cur_args = candidate
                except json.JSONDecodeError:
                    pass

        if not self.prev_tool_call_arr:
            self.prev_tool_call_arr.append({})

        if not self.tool_name_sent:
            if not func_name:
                return None
            self.tool_name_sent = True
            self.tool_id = f"chatcmpl-tool-{uuid4()}"
            self.prev_tool_call_arr[0]["name"] = func_name
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0,
                        id=self.tool_id,
                        type="function",
                        function=DeltaFunctionCall(name=func_name).model_dump(exclude_none=True),
                    )
                ],
                content=None,
            )

        if cur_args is None:
            return None

        prev_args = self.prev_tool_call_arr[0].get("arguments", "")
        if not prev_args:
            delta_args = cur_args
        elif cur_args.startswith(prev_args):
            delta_args = cur_args[len(prev_args) :]
        else:
            return None

        if not delta_args:
            return None

        self.prev_tool_call_arr[0]["arguments"] = cur_args
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    function=DeltaFunctionCall(arguments=delta_args).model_dump(exclude_none=True),
                )
            ],
            content=None,
        )
