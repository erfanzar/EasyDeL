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

"""OpenAI-style tool call parser for local model outputs.

This module provides a best-effort parser for extracting tool calls from local
model outputs that follow OpenAI-style JSON formatting conventions. Unlike
OpenAI's hosted API where tool calls are transmitted out-of-band, local models
often emit tool calls directly as JSON in the generated text.

The parser supports multiple JSON formats commonly used by models trained to
emulate OpenAI's function calling behavior, including raw JSON objects, JSON
arrays, and code-fenced JSON blocks.

Example:
    >>> from easydel.inference.tools.parsers.openai_tool_parser import OpenAIToolParser
    >>> parser = OpenAIToolParser(tokenizer)
    >>> result = parser.extract_tool_calls(
    ...     '{"name": "get_weather", "arguments": {"city": "NYC"}}',
    ...     request
    ... )
    >>> result.tools_called
    True
"""

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

    This parser handles the extraction of tool/function calls from LLM outputs
    that follow OpenAI-style JSON formatting. Unlike OpenAI's hosted API where
    tool calls are transmitted out-of-band, local models often emit tool calls
    directly as JSON in the generated text.

    Supported Formats:
        - JSON list of calls: `[{"name": ..., "arguments": ...}, ...]`
        - Single JSON object: `{"name": ..., "arguments": ...}`
        - Nested tool_calls: `{"tool_calls": [{"function": {"name": ..., "arguments": ...}}, ...]}`
        - Code-fenced JSON: ```json ... ```

    Attributes:
        prev_tool_call_arr (list[dict]): Previous tool calls for streaming diff calculation.
        current_tool_id (int): Index of the current tool being processed (-1 if none).
        current_tool_name_sent (bool): Whether the function name has been sent in streaming.
        streamed_args_for_tool (list[str]): Accumulated arguments for each tool in streaming.

    Example:
        >>> parser = OpenAIToolParser(tokenizer)
        >>> # Single tool call
        >>> result = parser.extract_tool_calls(
        ...     '{"name": "search", "arguments": {"query": "weather"}}',
        ...     request
        ... )
        >>> # Multiple tool calls
        >>> result = parser.extract_tool_calls(
        ...     '[{"name": "search", "arguments": {}}, {"name": "calculate", "arguments": {}}]',
        ...     request
        ... )
    """

    _json_block_re = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.MULTILINE)

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the OpenAI tool parser.

        Args:
            tokenizer (AnyTokenizer): The tokenizer instance used for encoding
                and decoding tokens. Can be any HuggingFace-compatible tokenizer.
        """
        super().__init__(tokenizer)

        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

    @staticmethod
    def _extract_json_candidate(text: str) -> tuple[str | None, str | None]:
        """Extract JSON content from model output text.

        Attempts to identify and extract JSON tool call data from the model's
        output, handling both raw JSON and code-fenced JSON blocks.

        Args:
            text (str): The raw model output text to parse.

        Returns:
            tuple[str | None, str | None]: A tuple of (content_without_json, json_candidate)
                where:
                - content_without_json: Text content excluding the JSON portion,
                  or None if the entire text is JSON.
                - json_candidate: The extracted JSON string candidate, or None
                  if no JSON was found.

        Example:
            >>> OpenAIToolParser._extract_json_candidate('{"name": "test", "arguments": {}}')
            (None, '{"name": "test", "arguments": {}}')
            >>> OpenAIToolParser._extract_json_candidate('Hello ```json{"name": "test"}``` world')
            ('Hello  world', '{"name": "test"}')
            >>> OpenAIToolParser._extract_json_candidate('Just plain text')
            ('Just plain text', None)
        """
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
        """Normalize various JSON structures into a standard tool call format.

        Handles multiple JSON formats that models may produce and converts them
        into a uniform list of dictionaries with 'name' and 'arguments' keys.

        Args:
            obj (object): The parsed JSON object, which can be:
                - A dict with 'name' and 'arguments' keys (single tool call)
                - A dict with 'tool_calls' list containing function objects
                - A list of tool call dictionaries

        Returns:
            list[dict]: A list of normalized tool call dictionaries, each containing
                'name' and 'arguments' keys. Returns empty list if input doesn't
                match any supported format.

        Example:
            >>> # Single tool call
            >>> OpenAIToolParser._normalize_tool_call_objects(
            ...     {"name": "search", "arguments": {"q": "test"}}
            ... )
            [{"name": "search", "arguments": {"q": "test"}}]
            >>> # Nested format
            >>> OpenAIToolParser._normalize_tool_call_objects({
            ...     "tool_calls": [{"function": {"name": "search", "arguments": {}}}]
            ... })
            [{"name": "search", "arguments": {}}]
        """
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
        """Extract tool calls from a complete model response.

        Parses the model output to identify and extract any JSON-formatted tool
        calls. Handles multiple tool call formats and returns structured tool
        call information.

        Args:
            model_output (str): The complete text output from the model.
            request (ChatCompletionRequest): The original chat completion request.
                Currently unused but included for API consistency.

        Returns:
            ExtractedToolCallInformation: An object containing:
                - tools_called (bool): True if valid tool calls were found.
                - tool_calls (list[ToolCall]): List of extracted ToolCall objects.
                - content (str | None): Non-JSON text content, if any.

        Example:
            >>> parser = OpenAIToolParser(tokenizer)
            >>> result = parser.extract_tool_calls(
            ...     'Here is the result: {"name": "search", "arguments": {"q": "test"}}',
            ...     request
            ... )
            >>> result.tools_called
            True
            >>> result.tool_calls[0].function.name
            'search'
        """
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
        """Extract tool calls incrementally during streaming generation.

        Processes streaming model output to extract tool calls as they are
        being generated. Maintains internal state to track partial tool calls
        and emit incremental updates.

        Args:
            previous_text (str): The accumulated text before this delta.
            current_text (str): The accumulated text including this delta.
            delta_text (str): The new text generated in this streaming chunk.
            previous_token_ids (Sequence[int]): Token IDs before this delta.
            current_token_ids (Sequence[int]): Token IDs including this delta.
            delta_token_ids (Sequence[int]): New token IDs in this chunk.
            request (ChatCompletionRequest): The original chat completion request.

        Returns:
            DeltaMessage | None: A delta message containing either:
                - Content text if no tool call is being processed.
                - Tool call updates (name or argument increments).
                - None if more tokens are needed to parse.

        Note:
            This method maintains state across calls via instance attributes:
            - prev_tool_call_arr: Tracks previous tool call state for diff calculation.
            - current_tool_id: Index of current tool being streamed.
            - current_tool_name_sent: Whether function name has been emitted.
            - streamed_args_for_tool: Accumulated arguments per tool.
        """
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
