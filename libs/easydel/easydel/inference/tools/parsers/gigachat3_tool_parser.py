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

"""Tool parser implementation for GigaChat3 models.

This module provides a tool parser specifically designed for GigaChat3 model outputs.
It handles the parsing of function calls in the format used by GigaChat3, which uses
a "function call" prefix followed by JSON data containing the function name and arguments.

The parser supports both complete extraction and streaming extraction modes, making it
suitable for real-time inference scenarios.

Example format:
    function call<|role_sep|>
    {"name": "function_name", "arguments": {"param": "value"}}
"""

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

REGEX_FUNCTION_CALL = re.compile(r"function call(?:<\|role_sep\|>)?\n?(\{.*)", re.DOTALL)
_TRIGGER_WORD = "function call"
# Every spelling of the separator that may sit between ``function call`` and the opening ``{``
# (the ``<|role_sep|>`` special token may be stripped by the detokenizer).
_TRIGGER_TAILS = ("{", "\n{", "<|role_sep|>{", "<|role_sep|>\n{")
NAME_REGEX = re.compile(r'"name"\s*:\s*"([^"]*)"', re.DOTALL)
ARGS_REGEX = re.compile(r'"arguments"\s*:\s*(.*)', re.DOTALL)


@ToolParserManager.register_module("gigachat3")  # pyright: ignore[reportUntypedClassDecorator]
class GigaChat3ToolParser(ToolParser):
    """Tool parser for GigaChat3 model outputs.

    This parser handles the extraction of function/tool calls from GigaChat3 model
    outputs. GigaChat3 uses a specific format where function calls are prefixed with
    "function call" followed by optional role separator tokens and JSON data.

    The parser maintains streaming state to handle incremental token generation,
    buffering content until a complete function call can be identified and parsed.

    Attributes:
        tool_started: Flag indicating if a tool call has been detected in the stream.
        tool_name_sent: Flag indicating if the tool name has been sent in streaming mode.
        tool_id: The unique identifier for the current tool call being processed.
        prev_tool_call_arr: List of previously parsed tool call dictionaries.
        content_buffer: Buffer for accumulating content during streaming.
        trigger_start: The prefix string that triggers tool call detection.

    Example:
        >>> parser = GigaChat3ToolParser(tokenizer)
        >>> result = parser.extract_tool_calls(model_output, request)
        >>> if result.tools_called:
        ...     for tool_call in result.tool_calls:
        ...         print(f"Function: {tool_call.function.name}")
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the GigaChat3 tool parser.

        Args:
            tokenizer: A HuggingFace tokenizer instance used for token processing.
                This tokenizer should be compatible with the GigaChat3 model.
        """
        super().__init__(tokenizer)
        self.tool_started: bool = False
        self.tool_name_sent: bool = False
        self.tool_id: str | None = None
        self.prev_tool_call_arr: list[dict] = []
        self.content_buffer: str = ""
        self.trigger_start = "function call{"

    @staticmethod
    def _may_become_function_call(buffer: str) -> bool:
        """Report whether ``buffer`` can still grow into a ``function call`` header.

        Accepts any prefix of ``function call`` and, once that word is complete, any
        prefix of the allowed separator spellings (``{``, ``\\n{``, ``<|role_sep|>{``,
        ``<|role_sep|>\\n{``) so the protocol prefix is never leaked as content.

        Args:
            buffer: Left-stripped text buffered so far, before a full header matched.

        Returns:
            True while the buffer must keep being withheld.
        """
        if len(buffer) <= len(_TRIGGER_WORD):
            return _TRIGGER_WORD.startswith(buffer)
        if not buffer.startswith(_TRIGGER_WORD):
            return False
        rest = buffer[len(_TRIGGER_WORD) :]
        return any(tail.startswith(rest) for tail in _TRIGGER_TAILS)

    def get_tool_machinery_hints(self) -> Sequence[str]:
        """Engage orchestration for every supported header separator spelling.

        Returns:
            The separator-independent trigger and existing compatibility hints.
        """
        return (_TRIGGER_WORD, *super().get_tool_machinery_hints())

    def is_buffering_protocol(self, *, current_text: str, delta_text: str) -> bool:
        """Expose pending headers even when ordinary content precedes them.

        Args:
            current_text: Cumulative visible model output.
            delta_text: Newly decoded text.

        Returns:
            Whether content is withheld or the stream is parsing a tool call.
        """
        return (
            self.tool_started
            or bool(self.content_buffer)
            or super().is_buffering_protocol(current_text=current_text, delta_text=delta_text)
        )

    def flush_streaming_content(self) -> str:
        """Release ambiguous buffered text when the stream ends without a call.

        ``ToolCallingMixin`` invokes this hook on its terminal update. An active
        function call has already consumed the buffer, and its JSON must never
        be surfaced as ordinary content.

        Returns:
            Unemitted plain text, or an empty string for an active call or after
            a previous flush. Repeated flushes without new input are idempotent.
        """
        content, self.content_buffer = self.content_buffer, ""
        return "" if self.tool_started else content

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model output.

        Parses the model output to find and extract function calls in GigaChat3 format.
        The function looks for the "function call" pattern followed by JSON data
        containing "name" and "arguments" fields.

        Args:
            model_output: The complete text output from the model to parse.
            request: The chat completion request that triggered this response.
                Used for context but not directly in parsing.

        Returns:
            ExtractedToolCallInformation: An object containing:
                - tools_called: True if valid tool calls were found, False otherwise.
                - tool_calls: List of ToolCall objects representing parsed function calls.
                - content: Any text content before the function call, or the full
                    output if no valid function call was found.

        Example:
            >>> output = 'function call{"name": "search", "arguments": {"query": "test"}}'
            >>> result = parser.extract_tool_calls(output, request)
            >>> result.tools_called
            True
            >>> result.tool_calls[0].function.name
            'search'
        """
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
        """Extract tool calls incrementally during streaming generation.

        Processes tokens as they are generated to progressively extract and emit
        tool call information. This method maintains internal state to track the
        progress of tool call parsing across multiple invocations.

        The method buffers content and checks for the function call trigger pattern.
        Once a function call is detected, it extracts the function name first, then
        incrementally streams the arguments as they become available.

        Args:
            previous_text: The accumulated text from all previous tokens.
            current_text: The complete text including the current delta.
            delta_text: The new text added in this streaming step.
            previous_token_ids: Token IDs from all previous generation steps.
            current_token_ids: All token IDs including the current step.
            delta_token_ids: The new token IDs added in this step.
            request: The chat completion request for context.

        Returns:
            DeltaMessage or None: A delta message containing either:
                - Content text if no tool call is in progress.
                - Tool call information (name or argument delta) if parsing a function.
                - None if buffering or waiting for more tokens.

        Note:
            This method modifies internal state (tool_started, tool_name_sent,
            tool_id, prev_tool_call_arr, content_buffer) and should be called
            sequentially for each streaming step.
        """
        func_name = None
        cur_args = None
        content = None

        if not self.tool_started:
            self.content_buffer += delta_text
            match = REGEX_FUNCTION_CALL.search(current_text)
            if match:
                # Only the buffered suffix is still un-emitted. Preserve its
                # ordinary prefix even when the header and JSON share a chunk.
                prefix_length = max(0, len(self.content_buffer) - (len(current_text) - match.start()))
                content = self.content_buffer[:prefix_length] or None
                self.tool_started = True
                self.content_buffer = ""
            else:
                clean_buffer = self.content_buffer.lstrip()
                if self._may_become_function_call(clean_buffer):
                    return None
                # A header can start anywhere, not just at the beginning of
                # the content buffer. Emit ordinary text but retain the longest
                # suffix that a future chunk could complete into a header.
                pending_length = 0
                max_header_length = len(_TRIGGER_WORD) + max(map(len, _TRIGGER_TAILS))
                for length in range(min(len(self.content_buffer), max_header_length - 1), 0, -1):
                    if self._may_become_function_call(self.content_buffer[-length:]):
                        pending_length = length
                        break
                split = len(self.content_buffer) - pending_length
                flush_text, self.content_buffer = self.content_buffer[:split], self.content_buffer[split:]
                return DeltaMessage(content=flush_text) if flush_text else None

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
                return DeltaMessage(content=content) if content else None
            self.tool_name_sent = True
            self.tool_id = f"chatcmpl-tool-{uuid4()}"
            self.prev_tool_call_arr[0]["name"] = func_name
            self.prev_tool_call_arr[0]["arguments"] = cur_args or ""
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0,
                        id=self.tool_id,
                        type="function",
                        function=DeltaFunctionCall(name=func_name, arguments=cur_args or None).model_dump(
                            exclude_none=True
                        ),
                    )
                ],
                content=content,
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
