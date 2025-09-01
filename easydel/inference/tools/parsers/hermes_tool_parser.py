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


@ToolParserManager.register_module("hermes")
class HermesToolParser(ToolParser):
    """
    Tool call parser for Hermes models.

    Handles tool calls wrapped in <tool_call> XML-style tags with JSON content.
    Designed for NousResearch Hermes models and similar architectures that use
    XML-style delimiters for function calling.

    Format:
        <tool_call>{"name": "function_name", "arguments": {...}}</tool_call>

    Features:
        - XML-style token boundary detection (<tool_call> and </tool_call>)
        - Token-level buffering for accurate boundary detection
        - Supports multiple tool calls in a single response
        - Handles partial JSON parsing for streaming
        - Scratch pad support for intermediate reasoning

    Attributes:
        current_tool_name_sent: Tracks if function name was sent in stream
        prev_tool_call_arr: Previous tool calls for streaming comparison
        current_tool_id: Index of current tool being processed
        streamed_args_for_tool: Arguments sent so far for each tool
        tool_call_start_token: Opening delimiter for tool calls
        tool_call_end_token: Closing delimiter for tool calls
        buffered_delta_text: Buffer for multi-token delimiter detection
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """
        Initialize the Hermes tool parser.

        Args:
            tokenizer: The model tokenizer for encoding/decoding tokens

        Raises:
            ValueError: If tokenizer is not provided
        """
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
        self.scratch_pad_regex = re.compile(r"<scratch_pad>(.*?)</scratch_pad>", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")
        self.tool_call_start_token_ids = self.model_tokenizer.encode(
            self.tool_call_start_token, add_special_tokens=False
        )
        self.tool_call_end_token_ids = self.model_tokenizer.encode(self.tool_call_end_token, add_special_tokens=False)

        self.tool_call_start_token_array = [
            self.model_tokenizer.decode([token_id]) for token_id in self.tool_call_start_token_ids
        ]

        self.tool_call_end_token_array = [
            self.model_tokenizer.decode([token_id]) for token_id in self.tool_call_end_token_ids
        ]

        self.buffered_delta_text = ""

    def tool_call_delta_buffer(self, delta_text: str) -> str:
        """
        Buffer delta text to handle multi-token delimiters.

        This method accumulates partial tokens that might form tool call
        delimiters, ensuring accurate boundary detection when delimiters
        span multiple tokens.

        Args:
            delta_text: The new text delta from streaming

        Returns:
            Processed text with complete delimiters or empty string if buffering
        """
        if delta_text in self.tool_call_start_token_array or delta_text in self.tool_call_end_token_array:
            if delta_text == self.tool_call_start_token_array[-1] or delta_text == self.tool_call_end_token_array[-1]:
                buffered_text = self.buffered_delta_text
                self.buffered_delta_text = ""
                return buffered_text + delta_text
            else:
                self.buffered_delta_text = self.buffered_delta_text + delta_text
                return ""
        else:
            if self.buffered_delta_text:
                buffered_text = self.buffered_delta_text
                self.buffered_delta_text = ""
                return buffered_text + delta_text
            else:
                return delta_text

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from complete model response.

        Parses XML-style tool call tags and extracts JSON function calls.
        Supports multiple tool calls and returns remaining content.

        Args:
            model_output: Complete model output containing tool calls
            request: Original chat completion request (unused)

        Returns:
            ExtractedToolCallInformation with:
                - tools_called: Whether tool calls were found
                - tool_calls: List of ToolCall objects
                - content: Text content before tool calls (if any)

        Example:
            Input: "Let me help. <tool_call>{"name": "search", "arguments": {"q": "weather"}}</tool_call>"
            Output: tools_called=True, tool_calls=[ToolCall(...)], content="Let me help. "
        """
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        else:
            try:
                function_call_tuples = self.tool_call_regex.findall(model_output)

                raw_function_calls = [json.loads(match[0] if match[0] else match[1]) for match in function_call_tuples]
                tool_calls = [
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=function_call["name"],
                            arguments=json.dumps(function_call["arguments"], ensure_ascii=False),
                        ),
                    )
                    for function_call in raw_function_calls
                ]

                content = model_output[: model_output.find(self.tool_call_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True, tool_calls=tool_calls, content=content if content else None
                )

            except Exception:
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
        Extract tool calls from streaming model output.

        Handles incremental parsing of tool calls during streaming generation.
        Maintains state across calls to track partial tool calls and arguments.
        Uses buffering to handle multi-token delimiters correctly.

        Args:
            previous_text: Text generated before this delta
            current_text: Text including this delta
            delta_text: New text in this streaming chunk
            previous_token_ids: Token IDs before this delta
            current_token_ids: Token IDs including this delta
            delta_token_ids: New token IDs in this chunk
            request: Original chat completion request

        Returns:
            DeltaMessage with incremental tool call updates or content,
            or None if more tokens needed for parsing

        State Management:
            - Tracks tool call boundaries with start/end token counts
            - Maintains current tool ID for multi-tool responses
            - Buffers partial arguments until complete
            - Handles transition between content and tool calls
        """
        delta_text = self.tool_call_delta_buffer(delta_text)

        if (
            len(previous_text) >= len(self.buffered_delta_text)
            and previous_text[-len(self.buffered_delta_text) :] == self.buffered_delta_text
        ):
            previous_text = previous_text[: -len(self.buffered_delta_text)]
            current_text = previous_text + delta_text
        if self.tool_call_start_token not in current_text:
            return DeltaMessage(content=delta_text)

        try:
            prev_tool_start_count = previous_text.count(self.tool_call_start_token)
            prev_tool_end_count = previous_text.count(self.tool_call_end_token)
            cur_tool_start_count = current_text.count(self.tool_call_start_token)
            cur_tool_end_count = current_text.count(self.tool_call_end_token)
            tool_call_portion = None
            text_portion = None

            if (
                cur_tool_start_count == cur_tool_end_count
                and prev_tool_end_count == cur_tool_end_count
                and self.tool_call_end_token not in delta_text
            ):
                return DeltaMessage(content=delta_text)

            if self.tool_call_end_token in delta_text:
                full_text = current_text + delta_text
                tool_call_portion = (
                    full_text.split(self.tool_call_start_token)[-1].split(self.tool_call_end_token)[0].rstrip()
                )
                delta_text = delta_text.split(self.tool_call_end_token)[0].rstrip()
                text_portion = delta_text.split(self.tool_call_end_token)[-1].lstrip()

            flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

            if cur_tool_start_count > cur_tool_end_count and cur_tool_start_count > prev_tool_start_count:
                if len(delta_token_ids) > 1:
                    tool_call_portion = current_text.split(self.tool_call_start_token)[-1]
                else:
                    tool_call_portion = None
                    delta = None

                text_portion = None

                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")

            elif cur_tool_start_count > cur_tool_end_count and cur_tool_start_count == prev_tool_start_count:
                tool_call_portion = current_text.split(self.tool_call_start_token)[-1]
                text_portion = None

            elif cur_tool_start_count == cur_tool_end_count and cur_tool_end_count >= prev_tool_end_count:
                if self.prev_tool_call_arr is None or len(self.prev_tool_call_arr) == 0:
                    return None
                diff = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                if diff:
                    diff = diff.encode("utf-8").decode("unicode_escape") if diff is str else diff
                    if '"}' not in delta_text:
                        return None
                    end_loc = delta_text.rindex('"}')
                    diff = delta_text[:end_loc] + '"}'
                    self.streamed_args_for_tool[self.current_tool_id] += diff
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=diff).model_dump(exclude_none=True),
                            )
                        ]
                    )

            else:
                text = delta_text.replace(self.tool_call_start_token, "")
                text = text.replace(self.tool_call_end_token, "")
                delta = DeltaMessage(tool_calls=[], content=text)
                return delta

            try:
                current_tool_call = (
                    partial_json_parser.loads(tool_call_portion or "{}", flags) if tool_call_portion else None
                )
            except partial_json_parser.core.exceptions.MalformedJSON:
                return None
            except json.decoder.JSONDecodeError:
                return None

            if not self.current_tool_name_sent:
                if current_tool_call is None:
                    return None
                function_name: str | None = current_tool_call.get("name")
                if function_name:
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
                else:
                    return None

            if tool_call_portion is None:
                delta = DeltaMessage(content=delta_text) if text_portion is not None else None
                return delta

            if len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
            cur_arguments = current_tool_call.get("arguments")

            if not cur_arguments and not prev_arguments:
                delta = None

            elif not cur_arguments and prev_arguments:
                delta = None

            elif cur_arguments and not prev_arguments:
                cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)
                if delta_text not in cur_arguments_json[:-2]:
                    return None
                args_delta_start_loc = cur_arguments_json[:-2].rindex(delta_text) + len(delta_text)

                arguments_delta = cur_arguments_json[:args_delta_start_loc]
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(arguments=arguments_delta).model_dump(exclude_none=True),
                        )
                    ]
                )
                self.streamed_args_for_tool[self.current_tool_id] += arguments_delta

            elif cur_arguments and prev_arguments:
                if isinstance(delta_text, str) and len(delta_text.rstrip()) >= 1 and delta_text.rstrip()[-1] == "}":
                    delta_text = delta_text.rstrip()[:-1]

                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(arguments=delta_text).model_dump(exclude_none=True),
                        )
                    ]
                )
                self.streamed_args_for_tool[self.current_tool_id] += delta_text

            if self.current_tool_id == len(self.prev_tool_call_arr) - 1:
                self.prev_tool_call_arr[self.current_tool_id] = current_tool_call
            else:
                self.prev_tool_call_arr.append(current_tool_call)

            return delta

        except Exception:
            return None
