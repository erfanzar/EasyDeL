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
from typing import Any
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


def _random_uuid() -> str:
    return uuid4().hex


def _make_tool_call_id() -> str:
    return f"chatcmpl-tool-{uuid4()}"


@ToolParserManager.register_module("xlam")
class xLAMToolParser(ToolParser):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        self.prev_tool_calls: list[dict] = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args: list[str] = []

        # For backward compatibility with tests / serving code
        self.current_tools_sent: list[bool] = []
        self.prev_tool_call_arr: list[dict] = []

        self.json_code_block_patterns = [
            r"```(?:json)?\s*([\s\S]*?)```",
            r"\[TOOL_CALLS\]([\s\S]*?)(?=\n|$)",
            r"<tool_call>([\s\S]*?)</tool_call>",
        ]
        self.thinking_tag_pattern = r"</think>([\s\S]*)"

        self.streaming_state: dict[str, Any] = {
            "current_tool_index": -1,
            "tool_ids": [],
            "sent_tools": [],
        }

    def preprocess_model_output(self, model_output: str) -> tuple[str | None, str | None]:
        thinking_match = re.search(self.thinking_tag_pattern, model_output)
        if thinking_match:
            content = model_output[: thinking_match.start() + len("</think>")].strip()
            thinking_content = thinking_match.group(1).strip()

            try:
                json.loads(thinking_content)
                return content, thinking_content
            except json.JSONDecodeError:
                for json_pattern in self.json_code_block_patterns:
                    json_matches = re.findall(json_pattern, thinking_content)
                    if json_matches:
                        for json_str in json_matches:
                            try:
                                json.loads(json_str)
                                return content, json_str
                            except json.JSONDecodeError:
                                continue

        for json_pattern in self.json_code_block_patterns:
            json_matches = re.findall(json_pattern, model_output)
            if json_matches:
                for json_str in json_matches:
                    try:
                        json.loads(json_str)
                        content = re.sub(json_pattern, "", model_output).strip()
                        return content, json_str
                    except json.JSONDecodeError:
                        continue

        if model_output.strip().startswith("["):
            try:
                json.loads(model_output)
                return None, model_output
            except json.JSONDecodeError:
                if "{" in model_output and "name" in model_output and "arguments" in model_output:
                    return None, model_output

        return model_output, None

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        try:
            content, potential_tool_calls = self.preprocess_model_output(model_output)
            if not potential_tool_calls:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=content)

            tool_calls_data = json.loads(potential_tool_calls)
            if not isinstance(tool_calls_data, list):
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=content or model_output)

            tool_calls: list[ToolCall] = []
            for idx, call in enumerate(tool_calls_data):
                if not isinstance(call, dict) or "name" not in call or "arguments" not in call:
                    continue

                tool_calls.append(
                    ToolCall(
                        id=f"call_{idx}_{_random_uuid()}",
                        type="function",
                        function=FunctionCall(
                            name=call["name"],
                            arguments=json.dumps(call["arguments"])
                            if isinstance(call["arguments"], dict)
                            else call["arguments"],
                        ),
                    )
                )

            return ExtractedToolCallInformation(tools_called=len(tool_calls) > 0, tool_calls=tool_calls, content=content)

        except Exception as e:
            logger.exception("Error extracting tool calls: %s", str(e))
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
        stripped_text = current_text.strip()
        _, preprocessed_tool_calls = self.preprocess_model_output(current_text)

        has_potential_json_block = (
            "```json" in current_text
            or "```\n[" in current_text
            or "[TOOL_CALLS]" in current_text
            or "<tool_call>" in current_text
        )

        is_tool_call_block = (
            stripped_text.startswith("[")
            or stripped_text.startswith("<tool_call>")
            or stripped_text.startswith("[TOOL_CALLS]")
            or ("</think>[" in current_text)
            or preprocessed_tool_calls is not None
            or (has_potential_json_block and '"name"' in current_text and '"arguments"' in current_text)
        )

        if not is_tool_call_block:
            return DeltaMessage(content=delta_text)

        try:
            if not hasattr(self, "streaming_state"):
                self.streaming_state = {"current_tool_index": -1, "tool_ids": [], "sent_tools": []}

            try:
                tool_calls_text = preprocessed_tool_calls if preprocessed_tool_calls else current_text
                parsed_tools = json.loads(tool_calls_text)
                if isinstance(parsed_tools, list):
                    self.prev_tool_call_arr = parsed_tools
            except json.JSONDecodeError:
                pass

            if hasattr(self, "current_tools_sent") and len(self.current_tools_sent) > 0:
                if len(self.current_tools_sent) == 1 and self.current_tools_sent[0] is False:
                    name_pattern = r'"name"\s*:\s*"([^"]+)"'
                    name_match = re.search(name_pattern, current_text)
                    if name_match:
                        function_name = name_match.group(1)
                        tool_id = _make_tool_call_id()
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=0,
                                    type="function",
                                    id=tool_id,
                                    function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        self.current_tools_sent = [True]
                        self.current_tool_id = 0
                        self.streaming_state["current_tool_index"] = 0
                        if len(self.streaming_state["sent_tools"]) == 0:
                            self.streaming_state["sent_tools"].append(
                                {"sent_name": True, "sent_arguments_prefix": False, "sent_arguments": ""}
                            )
                        else:
                            self.streaming_state["sent_tools"][0]["sent_name"] = True
                        self.current_tool_name_sent = True
                        return delta

            search_text = preprocessed_tool_calls if preprocessed_tool_calls else current_text
            if not preprocessed_tool_calls and has_potential_json_block:
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)(?:```|$)", current_text)
                if json_match:
                    potential_json = json_match.group(1).strip()
                    if potential_json.startswith("[") and (
                        '"name"' in potential_json and '"arguments"' in potential_json
                    ):
                        search_text = potential_json

            name_pattern = r'"name"\s*:\s*"([^"]+)"'
            name_matches = list(re.finditer(name_pattern, search_text))
            tool_count = len(name_matches)

            if tool_count == 0:
                partial_name_pattern = r'"name"\s*:\s*"([^"]*)'
                partial_matches = list(re.finditer(partial_name_pattern, search_text))
                if partial_matches:
                    return None
                return DeltaMessage(content=delta_text)

            while len(self.streaming_state["tool_ids"]) < tool_count:
                self.streaming_state["tool_ids"].append(None)
            while len(self.streaming_state["sent_tools"]) < tool_count:
                self.streaming_state["sent_tools"].append(
                    {"sent_name": False, "sent_arguments_prefix": False, "sent_arguments": ""}
                )

            if self.streaming_state["current_tool_index"] == -1:
                self.streaming_state["current_tool_index"] = 0
                self.current_tool_id = 0

            current_idx = self.streaming_state["current_tool_index"]
            if current_idx >= tool_count:
                return None

            if not self.streaming_state["sent_tools"][current_idx]["sent_name"]:
                function_name = name_matches[current_idx].group(1)
                tool_id = f"call_{current_idx}_{_random_uuid()}"
                self.streaming_state["tool_ids"][current_idx] = tool_id
                self.streaming_state["sent_tools"][current_idx]["sent_name"] = True
                self.streaming_state["sent_tools"][current_idx]["sent_arguments_prefix"] = False
                self.streaming_state["sent_tools"][current_idx]["sent_arguments"] = ""

                while len(self.streamed_args) <= current_idx:
                    self.streamed_args.append("")

                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=current_idx,
                            type="function",
                            id=tool_id,
                            function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True),
                        )
                    ]
                )

            empty_args_pattern = r'"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{\s*\}'
            empty_args_matches = list(re.finditer(empty_args_pattern, search_text))
            if current_idx < len(empty_args_matches):
                if not self.streaming_state["sent_tools"][current_idx]["sent_arguments_prefix"]:
                    self.streaming_state["sent_tools"][current_idx]["sent_arguments_prefix"] = True
                    self.streaming_state["sent_tools"][current_idx]["sent_arguments"] = "{"
                    while len(self.streamed_args) <= current_idx:
                        self.streamed_args.append("")
                    self.streamed_args[current_idx] += "{"
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=current_idx,
                                function=DeltaFunctionCall(arguments="{").model_dump(exclude_none=True),
                            )
                        ]
                    )

                if self.streaming_state["sent_tools"][current_idx]["sent_arguments"] == "{":
                    self.streaming_state["sent_tools"][current_idx]["sent_arguments"] = "{}"
                    self.streamed_args[current_idx] += "}"
                    self.streaming_state["sent_tools"][current_idx]["sent_arguments_prefix"] = True
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=current_idx,
                                function=DeltaFunctionCall(arguments="}").model_dump(exclude_none=True),
                            )
                        ]
                    )
                    if current_idx < tool_count - 1:
                        self.streaming_state["current_tool_index"] += 1
                        self.current_tool_id = self.streaming_state["current_tool_index"]
                    return delta

            args_pattern = r'"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\})'
            args_matches = list(re.finditer(args_pattern, search_text))
            if current_idx >= len(args_matches):
                return None

            args_text = args_matches[current_idx].group(1)
            if tool_count > 1:
                try:
                    parsed_tools = json.loads(search_text)
                    if isinstance(parsed_tools, list) and current_idx < len(parsed_tools):
                        current_tool = parsed_tools[current_idx]
                        if isinstance(current_tool.get("arguments"), dict):
                            args_text = json.dumps(current_tool["arguments"])
                        else:
                            args_text = str(current_tool.get("arguments", "{}"))
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass

            sent_args = self.streaming_state["sent_tools"][current_idx]["sent_arguments"]
            if not self.streaming_state["sent_tools"][current_idx]["sent_arguments_prefix"] and args_text.startswith(
                "{"
            ):
                self.streaming_state["sent_tools"][current_idx]["sent_arguments_prefix"] = True
                self.streaming_state["sent_tools"][current_idx]["sent_arguments"] = "{"
                while len(self.streamed_args) <= current_idx:
                    self.streamed_args.append("")
                self.streamed_args[current_idx] += "{"
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=current_idx,
                            function=DeltaFunctionCall(arguments="{").model_dump(exclude_none=True),
                        )
                    ]
                )

            if args_text.startswith(sent_args):
                args_diff = args_text[len(sent_args) :]
                if args_diff:
                    self.streaming_state["sent_tools"][current_idx]["sent_arguments"] = args_text
                    while len(self.streamed_args) <= current_idx:
                        self.streamed_args.append("")
                    self.streamed_args[current_idx] += args_diff
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=current_idx,
                                function=DeltaFunctionCall(arguments=args_diff).model_dump(exclude_none=True),
                            )
                        ]
                    )

            if current_idx < tool_count - 1 and args_text.endswith("}"):
                self.streaming_state["current_tool_index"] += 1
                self.current_tool_id = self.streaming_state["current_tool_index"]
                return None

            return None

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None
