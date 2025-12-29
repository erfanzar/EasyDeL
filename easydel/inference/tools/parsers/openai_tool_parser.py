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
from ..utils import find_common_prefix, is_complete_json, partial_json_loads

logger = get_logger(__name__)


@ToolParserManager.register_module("openai")
class OpenAIToolParser(ToolParser):
    """Best-effort OpenAI-style tool call parser for local model outputs.

    Unlike OpenAI's hosted API (where tool calls are out-of-band), local models
    often emit tool calls as JSON in the generated text. This parser supports:
    - A JSON list of calls: `[{"name": ..., "arguments": ...}, ...]`
    - A JSON object: `{"name": ..., "arguments": ...}`
    - `{"tool_calls": [{"function": {"name": ..., "arguments": ...}}, ...]}`
    - Code-fenced JSON: ```json ... ```
    """

    _json_block_re = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.MULTILINE)

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

    @staticmethod
    def _extract_json_candidate(text: str) -> tuple[str | None, str | None]:
        """Return (content_without_json, json_candidate)."""
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return None, stripped

        match = OpenAIToolParser._json_block_re.search(text)
        if match:
            json_part = match.group(1).strip()
            content = (text[: match.start()] + text[match.end() :]).strip()
            return (content or None), json_part

        return text, None

    @staticmethod
    def _normalize_tool_call_objects(obj: object) -> list[dict]:
        if isinstance(obj, dict):
            if isinstance(obj.get("tool_calls"), list):
                tool_calls: list[dict] = []
                for item in obj["tool_calls"]:
                    if not isinstance(item, dict):
                        continue
                    fn = item.get("function")
                    if isinstance(fn, dict):
                        tool_calls.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
                    else:
                        tool_calls.append(item)
                return tool_calls
            if "name" in obj and "arguments" in obj:
                return [obj]
            return []
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        return []

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        content, json_candidate = self._extract_json_candidate(model_output)
        if not json_candidate:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=content)

        try:
            parsed = json.loads(json_candidate)
        except json.JSONDecodeError:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        calls = self._normalize_tool_call_objects(parsed)
        tool_calls: list[ToolCall] = []
        for call in calls:
            name = call.get("name")
            if not isinstance(name, str) or not name:
                continue
            args = call.get("arguments", {})
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            tool_calls.append(ToolCall(type="function", function=FunctionCall(name=name, arguments=args)))

        return ExtractedToolCallInformation(tools_called=bool(tool_calls), tool_calls=tool_calls, content=content)

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
        _content, json_candidate = self._extract_json_candidate(current_text)
        if json_candidate is None:
            return DeltaMessage(content=delta_text)

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            (obj, _end_idx) = partial_json_loads(json_candidate, flags)
            is_complete = is_complete_json(json_candidate)
            tool_call_arr = self._normalize_tool_call_objects(obj)
        except partial_json_parser.core.exceptions.MalformedJSON:
            return None
        except Exception:
            logger.exception("Error parsing streaming OpenAI tool call JSON")
            return None

        if not tool_call_arr:
            return None

        current_tool_call: dict = tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}

        # Starting a new tool call (or moving to next)
        if len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1:
            if self.current_tool_id >= 0:
                cur_arguments = current_tool_call.get("arguments")
                if cur_arguments:
                    cur_args_json = (
                        cur_arguments
                        if isinstance(cur_arguments, str)
                        else json.dumps(cur_arguments, ensure_ascii=False)
                    )
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    argument_diff = cur_args_json[sent:]
                    delta = (
                        DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(arguments=argument_diff).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        if argument_diff
                        else None
                    )
                    if argument_diff:
                        self.streamed_args_for_tool[self.current_tool_id] += argument_diff
                else:
                    delta = None
            else:
                delta = None

            self.current_tool_id = len(tool_call_arr) - 1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool.append("")
            return delta

        # Emit function name
        if not self.current_tool_name_sent:
            function_name = current_tool_call.get("name")
            if isinstance(function_name, str) and function_name:
                self.current_tool_name_sent = True
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            type="function",
                            id=f"chatcmpl-tool-{uuid4()}",
                            function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True),
                        )
                    ]
                )
            return None

        # Emit argument diffs
        cur_arguments = current_tool_call.get("arguments")
        if cur_arguments is None:
            return None

        cur_args_json = (
            cur_arguments if isinstance(cur_arguments, str) else json.dumps(cur_arguments, ensure_ascii=False)
        )
        sent = len(self.streamed_args_for_tool[self.current_tool_id])

        prev_arguments = (
            self.prev_tool_call_arr[self.current_tool_id].get("arguments") if self.prev_tool_call_arr else None
        )
        argument_diff: str | None = None

        if is_complete:
            argument_diff = cur_args_json[sent:]
        elif prev_arguments is not None:
            prev_args_json = (
                prev_arguments if isinstance(prev_arguments, str) else json.dumps(prev_arguments, ensure_ascii=False)
            )
            if cur_args_json != prev_args_json:
                prefix = find_common_prefix(prev_args_json, cur_args_json)
                argument_diff = prefix[sent:]

        delta = None
        if argument_diff:
            delta = DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        function=DeltaFunctionCall(arguments=argument_diff).model_dump(exclude_none=True),
                    )
                ]
            )
            self.streamed_args_for_tool[self.current_tool_id] += argument_diff

        # Save state for next diff
        if len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({"arguments": cur_arguments})
        else:
            self.prev_tool_call_arr[self.current_tool_id] = {"arguments": cur_arguments}

        return delta
