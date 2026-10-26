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

"""xLAM tool parser module.

This module provides the xLAMToolParser class for parsing tool calls in various
formats used by xLAM and similar models. It supports multiple tool call formats
including JSON code blocks, [TOOL_CALLS] markers, and XML-style tool_call tags.

Supported formats:
    - ```json [...] ```
    - [TOOL_CALLS] [...]
    - <tool_call>...</tool_call>
    - </think>[...]  (thinking mode)
    - Raw JSON arrays starting with [
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import partial_json_parser
from eformer.loggings import get_logger
from partial_json_parser.core.options import Allow
from transformers import AutoTokenizer as AnyTokenizer

from ...openai_api_modules import (
    ChatCompletionRequest,
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from ..abstract_tool import ToolParser, ToolParserManager
from .utils import is_complete_json, partial_json_loads, stream_json_calls

logger = get_logger(__name__)


def _random_uuid() -> str:
    """Generate a random UUID hex string.

    Returns:
        A 32-character hexadecimal string from a UUID4.
    """
    return uuid4().hex


@ToolParserManager.register_module("xlam")  # pyright: ignore[reportUntypedClassDecorator]
class xLAMToolParser(ToolParser):
    """Tool parser for xLAM and similar models with flexible JSON formats.

    Handles multiple tool call formats including JSON code blocks, TOOL_CALLS
    markers, XML-style tags, and thinking mode output. Supports both complete
    and streaming extraction.

    Features:
        - Multiple format detection (JSON blocks, markers, tags)
        - Thinking mode support (</think> prefix)
        - Stateful streaming with argument diffing
        - Backward compatibility with various serving implementations

    Supported formats:
        - ```json [...] ``` - JSON code blocks
        - [TOOL_CALLS] [...] - TOOL_CALLS marker
        - <tool_call>...</tool_call> - XML-style tags
        - </think>[...] - Thinking mode output
        - [...] - Raw JSON arrays

    Attributes:
        prev_tool_calls: List of previously parsed tool calls.
        current_tool_id: Index of the current tool being processed.
        current_tool_name_sent: Whether the current tool name has been sent.
        streamed_args: List of streamed arguments per tool.
        current_tools_sent: Backward compatibility list for sent tool tracking.
        prev_tool_call_arr: Backward compatibility list for tool calls.
        json_code_block_patterns: Regex patterns for detecting JSON blocks.
        thinking_tag_pattern: Regex pattern for thinking mode output.
        streaming_state: State dictionary for streaming extraction.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the xLAM tool parser.

        Sets up regex patterns for various tool call formats and initializes
        tracking state for both complete and streaming extraction.

        Args:
            tokenizer: The tokenizer instance used for encoding/decoding tokens.
        """
        super().__init__(tokenizer)

        self.prev_tool_calls: list[dict] = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args: list[str] = []

        # For backward compatibility with tests / serving code
        self.current_tools_sent: list[bool] = []
        self.prev_tool_call_arr: list[dict] = []
        self.tool_call_start_token = "["
        self.tool_call_end_token = "]"

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

    def get_tool_machinery_hints(self) -> Sequence[str]:
        """Include the bare-JSON dict prefix alongside the ``[`` start token.

        ``preprocess_model_output`` accepts dict-form payloads (fenced
        ``\\`\\`\\`json`` blocks, ``<tool_call>``-wrapped dicts, malformed
        arrays containing ``{"name": ...}``) that carry no ``[``; the
        delegating parser's machinery gate must engage on ``{`` too.
        """
        hints = list(super().get_tool_machinery_hints())
        if "{" not in hints:
            hints.append("{")
        return tuple(hints)

    def preprocess_model_output(self, model_output: str) -> tuple[str | None, str | None]:
        """Preprocess model output to extract potential tool calls.

        Examines the model output for various tool call formats and separates
        regular content from tool call JSON.

        Args:
            model_output: The raw model output string to preprocess.

        Returns:
            A tuple of (content, tool_calls_json) where:
                - content: Text content outside of tool calls (may be None)
                - tool_calls_json: JSON string of tool calls if found (may be None)
        """
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
        """Extract tool calls from a complete model output.

        Preprocesses the output to find tool call JSON, then parses each
        tool call into ToolCall objects.

        Args:
            model_output: The complete model output string to parse.
            request: The chat completion request (unused but required by interface).

        Returns:
            ExtractedToolCallInformation containing:
                - tools_called: True if valid tool calls were found
                - tool_calls: List of parsed ToolCall objects
                - content: Text content if found, else original output on failure
        """
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
        """Extract tool calls from streaming model output.

        Processes incremental output to detect tool calls and stream
        function names and arguments as they become available. Uses
        stateful tracking to send only new content.

        Args:
            previous_text: The accumulated text from previous chunks.
            current_text: The current accumulated text including the new delta.
            delta_text: The new text in this chunk.
            previous_token_ids: Token IDs from previous chunks.
            current_token_ids: All token IDs including the new chunk.
            delta_token_ids: Token IDs for just the new chunk.
            request: The chat completion request.

        Returns:
            A DeltaMessage containing either:
                - content: Regular text if no tool call detected
                - tool_calls: Delta with function name or argument fragments
                - None if more data needed or no new content to send
        """
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

        if not previous_text:
            self.prev_tool_call_arr.clear()
            self.streamed_args.clear()
            self.streaming_state = {"current_tool_index": -1, "tool_ids": [], "sent_tools": []}
        search_text = preprocessed_tool_calls or current_text
        if not preprocessed_tool_calls:
            # Strip protocol wrappers, not JSON structure. The partial JSON parser
            # identifies top-level objects, including arbitrarily nested arguments.
            if "</think>" in search_text:
                search_text = search_text.split("</think>", 1)[1]
            for marker in ("[TOOL_CALLS]", "<tool_call>"):
                if marker in search_text:
                    search_text = search_text.split(marker, 1)[1]
            match = re.search(r"```(?:json)?\s*([\s\S]*?)(?:```|$)", search_text)
            if match:
                search_text = match.group(1)
        try:
            parsed, _ = partial_json_loads(search_text.strip(), Allow.ALL & ~Allow.STR)
        except (ValueError, partial_json_parser.core.exceptions.MalformedJSON):
            return None
        calls = parsed if isinstance(parsed, list) else [parsed]
        if not all(isinstance(call, dict) for call in calls):
            return None
        try:
            partial, _ = partial_json_loads(search_text.strip(), Allow.ALL)
            partial_calls = partial if isinstance(partial, list) else [partial]
            for index, call in enumerate(calls):
                if index < len(partial_calls) and isinstance(partial_calls[index], dict) and "name" in call:
                    call["arguments"] = partial_calls[index].get("arguments")
        except (ValueError, partial_json_parser.core.exceptions.MalformedJSON):
            pass
        deltas = stream_json_calls(
            calls,
            self.prev_tool_call_arr,
            self.streamed_args,
            self.streaming_state["tool_ids"],
            is_complete_json(search_text.strip()),
        )
        self.prev_tool_calls = self.prev_tool_call_arr
        self.current_tools_sent = [bool(tool_id) for tool_id in self.streaming_state["tool_ids"]]
        self.current_tool_id = len(calls) - 1
        self.current_tool_name_sent = bool(self.current_tools_sent and self.current_tools_sent[-1])
        self.streaming_state["current_tool_index"] = self.current_tool_id
        self.streaming_state["sent_tools"] = [
            {"sent_name": sent, "sent_arguments_prefix": bool(args), "sent_arguments": args}
            for sent, args in zip(self.current_tools_sent, self.streamed_args, strict=True)
        ]
        return DeltaMessage(tool_calls=deltas) if deltas else None
