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

import partial_json_parser  # pyright: ignore[reportMissingTypeStubs]
from eformer.loggings import get_logger
from partial_json_parser.core.options import Allow  # pyright: ignore[reportMissingTypeStubs]
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
from ..utils import partial_json_loads, stable_json_stream_prefix

logger = get_logger(__name__)

REGEX_FUNCTION_CALL = re.compile(r"function call(?:<\|role_sep\|>)?\n?(\{.*)", re.DOTALL)
_REGEX_FUNCTION_CALL_HEADER = re.compile(r"function call(?:<\|role_sep\|>)?\n?(?=\{)")
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
        self._active_header_start: int | None = None
        self._active_json_start: int | None = None
        self._cursor: int = 0
        self._accumulated_text: str = ""
        self._streaming_initialized: bool = False
        self._tool_completed: bool = False

    def _reset_streaming_state(self, cursor: int = 0) -> None:
        self.tool_started = False
        self.tool_name_sent = False
        self.tool_id = None
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.content_buffer = ""
        self._active_header_start = None
        self._active_json_start = None
        self._cursor = cursor
        self._accumulated_text = ""
        self._streaming_initialized = False
        self._tool_completed = False

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
        if not buffer:
            return False
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
        if self._streaming_initialized:
            return self.tool_started or bool(self.content_buffer)
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
        if self.tool_name_sent:
            self.content_buffer = ""
            self._accumulated_text = ""
            return ""
        self.tool_started = False
        self._active_header_start = None
        self._active_json_start = None
        content, self.content_buffer = self.content_buffer, ""
        if self._streaming_initialized:
            self._cursor = len(self._accumulated_text)
        self._accumulated_text = ""
        return content

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
        dec = json.JSONDecoder()
        for match in _REGEX_FUNCTION_CALL_HEADER.finditer(model_output):
            try:
                data, _ = dec.raw_decode(model_output, match.end())
            except (json.JSONDecodeError, ValueError, RecursionError):
                continue

            if not (isinstance(data, dict) and isinstance(data.get("name"), str) and data.get("name") and "arguments" in data):
                continue

            name = data["name"]
            args = data["arguments"]
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)

            tool_calls = [ToolCall(type="function", function=FunctionCall(name=name, arguments=args))]
            prefix = model_output[: match.start()]
            content = prefix.rstrip() if prefix and prefix.strip() else None
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
        del delta_text, previous_token_ids, current_token_ids, delta_token_ids, request
        if not previous_text:
            self._reset_streaming_state(0)
            self._streaming_initialized = True
        elif not self._streaming_initialized:
            self._reset_streaming_state(len(previous_text))
            self._streaming_initialized = True

        self._accumulated_text = current_text
        dec = json.JSONDecoder()
        content_parts: list[str] = []

        if self.tool_name_sent:
            if self._tool_completed:
                return None

            json_start = self._active_json_start
            if json_start is None:
                match = _REGEX_FUNCTION_CALL_HEADER.search(current_text)
                if not match:
                    return None
                json_start = match.end()
                self._active_json_start = json_start

            is_complete = False
            json_end = json_start
            data: object = None
            try:
                data, json_end = dec.raw_decode(current_text, json_start)
                is_complete = True
            except (json.JSONDecodeError, ValueError, RecursionError):
                json_tail = current_text[json_start:]
                try:
                    data, _ = partial_json_loads(json_tail, Allow.ALL)
                except (
                    partial_json_parser.core.exceptions.MalformedJSON,
                    ValueError,
                    AssertionError,
                    RecursionError,
                ):
                    return None

            if is_complete:
                self.tool_started = False
                self._tool_completed = True
                self._cursor = json_end

            if not isinstance(data, dict):
                return None

            raw_args = data.get("arguments")
            if raw_args is None:
                return None
            serialized = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
            safe_args = serialized if is_complete else stable_json_stream_prefix(serialized)

            prev_args = self.streamed_args_for_tool[0]
            if not safe_args.startswith(prev_args):
                return None

            delta_args = safe_args[len(prev_args) :]
            if not delta_args:
                return None

            self.prev_tool_call_arr[0]["arguments"] = safe_args
            self.streamed_args_for_tool[0] = safe_args
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0,
                        function=DeltaFunctionCall(arguments=delta_args).model_dump(exclude_none=True),
                    )
                ],
                content=None,
            )

        while True:
            if not self.tool_started:
                matched_header = _REGEX_FUNCTION_CALL_HEADER.search(current_text, self._cursor)

                if matched_header is not None:
                    if matched_header.start() > self._cursor:
                        content_parts.append(current_text[self._cursor : matched_header.start()])
                    self.tool_started = True
                    self._active_header_start = matched_header.start()
                    self._active_json_start = matched_header.end()
                    self._cursor = matched_header.start()
                    self.content_buffer = current_text[matched_header.start() :]
                    continue

                rem = current_text[self._cursor :]
                clean_buffer = rem.lstrip()
                leading_ws_len = len(rem) - len(clean_buffer)
                if leading_ws_len > 0:
                    content_parts.append(rem[:leading_ws_len])
                    self._cursor += leading_ws_len
                    rem = clean_buffer
                if clean_buffer and self._may_become_function_call(clean_buffer):
                    self.content_buffer = rem
                    break

                # A header can start anywhere, not just at the beginning of
                # the remaining text. Emit ordinary text but retain the longest
                # suffix that a future chunk could complete into a header.
                pending_length = 0
                max_header_length = len(_TRIGGER_WORD) + max(map(len, _TRIGGER_TAILS))
                for length in range(min(len(rem), max_header_length - 1), 0, -1):
                    if self._may_become_function_call(rem[-length:]):
                        pending_length = length
                        break
                split = len(rem) - pending_length
                if split > 0:
                    content_parts.append(rem[:split])
                    self._cursor += split
                self.content_buffer = rem[split:]
                break

            header_start = self._active_header_start if self._active_header_start is not None else self._cursor
            json_start = self._active_json_start
            if json_start is None:
                match = _REGEX_FUNCTION_CALL_HEADER.search(current_text, header_start)
                if not match:
                    self.tool_started = False
                    self._active_header_start = None
                    continue
                header_start = match.start()
                json_start = match.end()
                self._active_header_start = header_start
                self._active_json_start = json_start

            self.content_buffer = current_text[header_start:]

            is_complete = False
            data = None
            safe_data: object = None
            json_end = json_start
            malformed = False
            try:
                data, json_end = dec.raw_decode(current_text, json_start)
                safe_data = data
                is_complete = True
            except (json.JSONDecodeError, ValueError, RecursionError):
                json_tail = current_text[json_start:]
                try:
                    data, _ = partial_json_loads(json_tail, Allow.ALL)
                except (
                    partial_json_parser.core.exceptions.MalformedJSON,
                    ValueError,
                    AssertionError,
                    RecursionError,
                ):
                    malformed = True
                else:
                    try:
                        safe_data, _ = partial_json_loads(json_tail, Allow.ALL & ~Allow.STR)
                    except (
                        partial_json_parser.core.exceptions.MalformedJSON,
                        ValueError,
                        AssertionError,
                        RecursionError,
                    ):
                        safe_data = {}

            if malformed:
                # Non-JSON after "function call{": release prefix as plain content and keep scanning.
                advance_to = min(len(current_text), json_start + 1)
                content_parts.append(current_text[header_start:advance_to])
                self._cursor = advance_to
                self.tool_started = False
                self._active_header_start = None
                self._active_json_start = None
                self.content_buffer = ""
                continue

            if is_complete:
                is_valid_call = (
                    isinstance(data, dict)
                    and isinstance(data.get("name"), str)
                    and bool(data.get("name"))
                    and "arguments" in data
                )
                if not is_valid_call:
                    # Complete non-tool JSON after "function call": emit verbatim as plain content.
                    content_parts.append(current_text[header_start:json_end])
                    self._cursor = json_end
                    self.tool_started = False
                    self._active_header_start = None
                    self._active_json_start = None
                    self.content_buffer = ""
                    continue

            if not isinstance(data, dict):
                break
            if not isinstance(safe_data, dict):
                safe_data = {}

            raw_name = data.get("name") if is_complete else safe_data.get("name")
            func_name = raw_name if isinstance(raw_name, str) and raw_name else None
            if not func_name or "arguments" not in data:
                break

            if not self.prev_tool_call_arr:
                self.prev_tool_call_arr.append({})
            if not self.streamed_args_for_tool:
                self.streamed_args_for_tool.append("")

            raw_args = data.get("arguments")
            safe_args = None
            if raw_args is not None:
                serialized = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
                safe_args = serialized if is_complete else stable_json_stream_prefix(serialized)

            self.tool_name_sent = True
            self.current_tool_id = 0
            self.current_tool_name_sent = True
            self.tool_id = f"chatcmpl-tool-{uuid4()}"
            self.content_buffer = ""
            if is_complete:
                self.tool_started = False
                self._tool_completed = True
                self._cursor = json_end
            self.prev_tool_call_arr[0]["name"] = func_name
            initial_args = safe_args if safe_args else ""
            self.prev_tool_call_arr[0]["arguments"] = initial_args
            self.streamed_args_for_tool[0] = initial_args
            content = "".join(content_parts) or None
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0,
                        id=self.tool_id,
                        type="function",
                        function=DeltaFunctionCall(
                            name=func_name,
                            arguments=initial_args or None,
                        ).model_dump(exclude_none=True),
                    )
                ],
                content=content,
            )

        content = "".join(content_parts)
        return DeltaMessage(content=content) if content else None
