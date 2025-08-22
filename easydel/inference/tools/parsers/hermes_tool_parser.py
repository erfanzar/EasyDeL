# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from transformers import AutoTokenizer as AnyTokenizer

from ...openai_api_modules import (
    ChatCompletionRequest,
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from ..abstract_tool import ToolParser, ToolParserManager


@ToolParserManager.register_module("hermes")
class HermesToolParser(ToolParser):
    """
    Tool parser for Hermes-style tool calls.

    Handles extraction of tool calls in the format:
    <tool_call>
    {"name": "function_name", "arguments": {...}}
    </tool_call>

    Features:
    - Simple JSON-based tool call format
    - Supports streaming with buffer-based parsing
    - Handles multiple tool calls in sequence
    - Automatic JSON validation and error recovery

    Attributes:
        current_tool_call_buffer (str): Buffer for accumulating partial tool calls
        in_tool_call (bool): Flag indicating if currently parsing a tool call
        tool_call_pattern (re.Pattern): Regex for extracting complete tool calls
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.current_tool_call_buffer = ""
        self.in_tool_call = False
        self.tool_call_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Extract complete tool calls from Hermes model output.

        Searches for tool calls wrapped in <tool_call> tags and parses
        the JSON content within. Handles multiple tool calls and returns
        remaining content after extraction.

        Args:
            model_output: Complete model output text
            request: Original chat request (unused but required by interface)

        Returns:
            ExtractedToolCallInformation with parsed tool calls and content
        """
        tool_calls = []
        content = model_output
        matches = self.tool_call_pattern.findall(model_output)

        for match in matches:
            try:
                tool_data = json.loads(match.strip())
                tool_call = ToolCall(
                    id=f"call_{len(tool_calls)}",
                    type="function",
                    function=FunctionCall(
                        name=tool_data.get("name", ""),
                        arguments=json.dumps(tool_data.get("arguments", {}))
                        if isinstance(tool_data.get("arguments"), dict)
                        else tool_data.get("arguments", "{}"),
                    ),
                )
                tool_calls.append(tool_call)
                content = content.replace(f"<tool_call>{match}</tool_call>", "")
            except json.JSONDecodeError:
                print(f"Failed to parse tool call JSON: {match}")
                continue

        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            content=content.strip() if content.strip() else None,
        )

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
        Extract tool calls from streaming Hermes output.

        Maintains state across streaming chunks to detect tool call
        boundaries and progressively parse JSON content. Emits tool
        names and arguments as they become available.

        Args:
            previous_text: Text from previous chunks
            current_text: All text generated so far
            delta_text: New text in current chunk
            previous_token_ids: Previous token IDs (unused)
            current_token_ids: All token IDs (unused)
            delta_token_ids: New token IDs (unused)
            request: Original request (unused)

        Returns:
            DeltaMessage with incremental tool information or content
        """

        if "<tool_call>" in delta_text and not self.in_tool_call:
            self.in_tool_call = True
            self.current_tool_call_buffer = ""
            parts = delta_text.split("<tool_call>", 1)
            if len(parts) > 1 and parts[1]:
                self.current_tool_call_buffer = parts[1]

            if parts[0]:
                return DeltaMessage(content=parts[0])
            self.current_tool_id += 1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool.append("")
            return DeltaMessage(
                tool_calls=[
                    {
                        "index": self.current_tool_id,
                        "id": f"call_{self.current_tool_id}",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                ]
            )

        elif "</tool_call>" in delta_text and self.in_tool_call:
            parts = delta_text.split("</tool_call>", 1)
            if parts[0]:
                self.current_tool_call_buffer += parts[0]

            try:
                tool_data = json.loads(self.current_tool_call_buffer.strip())

                function_name = tool_data.get("name", "")
                arguments = tool_data.get("arguments", {})
                arguments_str = json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)

                if not self.current_tool_name_sent:
                    self.current_tool_name_sent = True
                    delta_msg = DeltaMessage(
                        tool_calls=[
                            {
                                "index": self.current_tool_id,
                                "function": {"name": function_name},
                            }
                        ]
                    )

                    self.streamed_args_for_tool[self.current_tool_id] = arguments_str
                    return delta_msg

                return DeltaMessage(
                    tool_calls=[
                        {
                            "index": self.current_tool_id,
                            "function": {"arguments": arguments_str},
                        }
                    ]
                )

            except json.JSONDecodeError:
                pass
            finally:
                self.in_tool_call = False
                self.current_tool_call_buffer = ""

                if len(parts) > 1 and parts[1]:
                    return DeltaMessage(content=parts[1])  # noqa: B012

        elif self.in_tool_call:
            self.current_tool_call_buffer += delta_text

            if not self.current_tool_name_sent and '{"name"' in self.current_tool_call_buffer:
                try:
                    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', self.current_tool_call_buffer)
                    if name_match:
                        function_name = name_match.group(1)
                        self.current_tool_name_sent = True
                        return DeltaMessage(
                            tool_calls=[{"index": self.current_tool_id, "function": {"name": function_name}}]
                        )
                except Exception:
                    pass

            if self.current_tool_name_sent and '"arguments"' in self.current_tool_call_buffer:
                try:
                    args_match = re.search(r'"arguments"\s*:\s*({.*}|\[.*\]|"[^"]*")', self.current_tool_call_buffer)
                    if args_match:
                        args_str = args_match.group(1)
                        prev_args = self.streamed_args_for_tool[self.current_tool_id]
                        if len(args_str) > len(prev_args):
                            delta_args = args_str[len(prev_args) :]
                            self.streamed_args_for_tool[self.current_tool_id] = args_str
                            return DeltaMessage(
                                tool_calls=[
                                    {
                                        "index": self.current_tool_id,
                                        "function": {"arguments": delta_args},
                                    }
                                ]
                            )
                except Exception:
                    pass

            return None
        else:
            if delta_text:
                return DeltaMessage(content=delta_text)

        return None
