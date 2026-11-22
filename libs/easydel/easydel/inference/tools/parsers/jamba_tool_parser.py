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

"""Jamba tool parser for tool calls with XML-style tokens.

This module provides a tool call parser designed for Jamba models that
generate tool calls wrapped in <tool_calls> and </tool_calls> tokens.
The parser handles JSON arrays of function calls within these delimiters.

Example format:
    <tool_calls>[{"name": "func", "arguments": {...}}]</tool_calls>

Features:
    - Token-based boundary detection
    - JSON array parsing with regex fallback
    - Streaming with partial JSON support
    - Automatic special token configuration
    - Validates tokenizer compatibility (rejects MistralTokenizer)
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from importlib import import_module
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
from ..utils import consume_space, partial_json_loads, stable_json_stream_prefix

logger = get_logger(__name__)


def _strip_trailing_partial_marker(text: str, marker: str) -> str:
    """Withhold any trailing non-empty proper prefix of *marker* from *text*."""
    max_len = min(len(marker) - 1, len(text))
    for k in range(max_len, 0, -1):
        if text.endswith(marker[:k]):
            return text[:-k]
    return text


def _has_open_marker_or_trailing_prefix(text: str, start_token: str, end_token: str) -> bool:
    """Return True if *text* has an unclosed block or ends with a partial start marker."""
    pos = 0
    n = len(text)
    start_len = len(start_token)
    end_len = len(end_token)
    while pos < n:
        start_idx = text.find(start_token, pos)
        if start_idx == -1:
            return ToolParser._has_trailing_marker_prefix(text[pos:], (start_token,))
        end_idx = text.find(end_token, start_idx + start_len)
        if end_idx == -1:
            return True
        pos = end_idx + end_len
    return False


@ToolParserManager.register_module("jamba")  # pyright: ignore[reportUntypedClassDecorator]
class JambaToolParser(ToolParser):
    """Tool parser for Jamba models.

    Handles tool calls wrapped in <tool_calls> and </tool_calls> tokens.
    Validates tokenizer compatibility (not Mistral) and parses JSON arrays
    of function calls.

    Features:
        - Token-based boundary detection
        - JSON array parsing with regex fallback
        - Streaming with partial JSON support
        - Automatic special token configuration

    Format:
        <tool_calls>[{"name": "func", "arguments": {...}}]</tool_calls>

    Attributes:
        tool_calls_start_token: The opening token for tool calls section.
        tool_calls_end_token: The closing token for tool calls section.
        tool_calls_regex: Compiled regex for extracting tool call content.
        tool_calls_start_token_id: Token ID for the start token.
        tool_calls_end_token_id: Token ID for the end token.

    Raises:
        ValueError: If a MistralTokenizer is detected or tokenizer is None.
        RuntimeError: If start/end tokens cannot be found in vocabulary.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the JambaToolParser.

        Sets up the parser with tool call tokens and validates the tokenizer
        is compatible (not a MistralTokenizer).

        Args:
            tokenizer: The tokenizer associated with the Jamba model.
                Must not be a MistralTokenizer.

        Raises:
            ValueError: If tokenizer is a MistralTokenizer or is None.
            RuntimeError: If tool call start/end tokens are not found
                in the tokenizer vocabulary.
        """
        super().__init__(tokenizer)
        try:
            mistral_module = import_module("mistral_common.tokens.tokenizers.mistral")
        except ImportError:
            MistralTokenizer = None
        else:
            MistralTokenizer = getattr(mistral_module, "MistralTokenizer", None)

        if MistralTokenizer is not None and isinstance(self.model_tokenizer, MistralTokenizer):
            raise ValueError("Detected a MistralTokenizer tokenizer when using a Jamba model")

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []
        self._tool_name_sent: list[bool] = []
        self._tool_call_ids: list[str] = []
        self._emitted_content_pos: int | None = None
        self._pending_tail_start: int | None = None
        self._last_current_text: str = ""
        self._streaming_initialized: bool = False
        self._in_open_block: bool = False
        self._scan_pos: int = 0
        self._closed_emitted_idx: int = 0
        self._closed_tracked_calls: list[dict] = []

        self.tool_calls_start_token: str = "<tool_calls>"
        self.tool_calls_end_token: str = "</tool_calls>"

        self.tool_calls_regex = re.compile(rf"{self.tool_calls_start_token}(.*?){self.tool_calls_end_token}", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")
        self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)
        if self.tool_calls_start_token_id is None or self.tool_calls_end_token_id is None:
            raise RuntimeError("Jamba Tool parser could not locate tool calls start/end tokens in the tokenizer!")

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Report whether a ``<tool_calls>`` block or partial start marker is buffered."""
        del delta_text
        if self._streaming_initialized:
            return self._in_open_block or self._pending_tail_start is not None
        return _has_open_marker_or_trailing_prefix(
            current_text,
            self.tool_calls_start_token,
            self.tool_calls_end_token,
        )

    def _reset_streaming_state(self, emitted_pos: int = 0) -> None:
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []
        self._tool_name_sent = []
        self._tool_call_ids = []
        self._emitted_content_pos = emitted_pos
        self._pending_tail_start = None
        self._last_current_text = ""
        self._streaming_initialized = False
        self._in_open_block = False
        self._scan_pos = 0
        self._closed_emitted_idx = 0
        self._closed_tracked_calls = []

    def flush_streaming_content(self) -> str | None:
        """Flush any trailing partial ``<tool_calls>`` prefix withheld at EOS."""
        flushed: str | None = None
        if self._pending_tail_start is not None and self._pending_tail_start < len(self._last_current_text):
            flushed = self._last_current_text[self._pending_tail_start :] or None
            self._emitted_content_pos = len(self._last_current_text)
            self._pending_tail_start = None
        self._last_current_text = ""
        return flushed

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Adjust the request settings for Jamba tool parsing.

        Modifies the request to ensure special tokens are not skipped
        when tools are enabled, as the tool_calls tokens are required
        for proper tool call detection.

        Args:
            request: The chat completion request to adjust.

        Returns:
            The modified request with skip_special_tokens set to False
            when tools are enabled and tool_choice is not 'none'.
        """
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model response.

        Parses the model output to identify and extract tool calls wrapped
        in <tool_calls> and </tool_calls> tokens. Expects a JSON array
        of tool call objects within the delimiters.

        Args:
            model_output: The complete text output from the model.
            request: The chat completion request for context.

        Returns:
            ExtractedToolCallInformation containing:
                - tools_called: True if valid tool calls were found
                - tool_calls: List of ToolCall objects extracted
                - content: Text content before the tool calls, or full
                    content if no tools were called
        """
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            dec = json.JSONDecoder()
            tool_calls: list[ToolCall] = []
            first_tool_pos: int | None = None
            pos = 0
            while pos < len(model_output):
                start = model_output.find(self.tool_calls_start_token, pos)
                if start == -1:
                    break
                payload_start = consume_space(start + len(self.tool_calls_start_token), model_output)
                if payload_start >= len(model_output):
                    break
                try:
                    raw_function_calls, end_pos = dec.raw_decode(model_output, payload_start)
                except (json.JSONDecodeError, ValueError, RecursionError):
                    pos = start + len(self.tool_calls_start_token)
                    continue

                after_json = consume_space(end_pos, model_output)
                if not model_output.startswith(self.tool_calls_end_token, after_json):
                    pos = end_pos
                    continue

                pos = after_json + len(self.tool_calls_end_token)
                if not isinstance(raw_function_calls, list):
                    continue

                block_calls: list[ToolCall] = []
                for function_call in raw_function_calls:
                    if (
                        not isinstance(function_call, dict)
                        or not isinstance(function_call.get("name"), str)
                        or not function_call.get("name")
                        or "arguments" not in function_call
                    ):
                        continue
                    raw_args = function_call["arguments"]
                    block_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=function_call["name"],
                                arguments=(
                                    raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
                                ),
                            ),
                        )
                    )
                if block_calls:
                    if first_tool_pos is None:
                        first_tool_pos = start
                    tool_calls.extend(block_calls)

            if not tool_calls:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            content = model_output[:first_tool_pos] if first_tool_pos is not None else ""
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if (len(content) > 0 and content != " ") else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
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
        """Extract tool calls incrementally during streaming.

        Processes streaming output token by token, parsing partial JSON
        to extract tool call information as it becomes available. Handles
        multi-tool scenarios and tracks argument deltas.

        Args:
            previous_text: Text accumulated before this delta.
            current_text: Complete text including the current delta.
            delta_text: The new text added in this streaming chunk.
            previous_token_ids: Token IDs before this delta.
            current_token_ids: All token IDs including current delta.
            delta_token_ids: Token IDs in the current delta.
            request: The chat completion request for context.

        Returns:
            DeltaMessage containing either:
                - content: Regular text content if no tool call detected
                - tool_calls: Tool call deltas if parsing tool calls
            Returns None if more tokens are needed or if the start token
            is the only content in the delta.
        """
        if not previous_text:
            self._reset_streaming_state(0)
        elif self._emitted_content_pos is None:
            self._reset_streaming_state(len(previous_text))
        elif self._scan_pos > len(current_text):
            self._reset_streaming_state(0)

        self._streaming_initialized = True
        self._last_current_text = current_text
        self._pending_tail_start = None
        self._in_open_block = False

        if (
            self.tool_calls_start_token_id is not None
            and len(delta_token_ids) == 1
            and delta_token_ids[0] == self.tool_calls_start_token_id
            and current_text.find(self.tool_calls_start_token, self._scan_pos) <= (self._emitted_content_pos or 0)
        ):
            self._in_open_block = True
            return None

        try:
            dec = json.JSONDecoder()
            pos = self._scan_pos
            next_scan_pos = pos
            visible_parts: list[tuple[int, int]] = []
            # list of (call_dict, safe_call_dict, call_complete, block_closed)
            all_calls: list[tuple[dict, dict, bool, bool]] = []

            while pos < len(current_text):
                sec_start = current_text.find(self.tool_calls_start_token, pos)
                if sec_start == -1:
                    tail = current_text[pos:]
                    safe_tail = _strip_trailing_partial_marker(tail, self.tool_calls_start_token)
                    safe_end = pos + len(safe_tail)
                    if safe_tail:
                        visible_parts.append((pos, safe_end))
                    if len(safe_tail) < len(tail):
                        self._pending_tail_start = safe_end
                    next_scan_pos = safe_end
                    break

                if sec_start > pos:
                    visible_parts.append((pos, sec_start))

                payload_start = consume_space(sec_start + len(self.tool_calls_start_token), current_text)
                if payload_start >= len(current_text):
                    self._in_open_block = True
                    next_scan_pos = sec_start
                    break

                parsed: object = None
                end_pos = -1
                try:
                    parsed, end_pos = dec.raw_decode(current_text, payload_start)
                except (json.JSONDecodeError, ValueError, RecursionError):
                    parsed = None

                if parsed is not None:
                    after_json = consume_space(end_pos, current_text)
                    if current_text.startswith(self.tool_calls_end_token, after_json):
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict):
                                    all_calls.append((item, item, True, True))
                        pos = after_json + len(self.tool_calls_end_token)
                        next_scan_pos = pos
                        continue
                    if after_json == len(current_text) or self.tool_calls_end_token.startswith(current_text[after_json:]):
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict):
                                    all_calls.append((item, item, True, False))
                        self._in_open_block = True
                        next_scan_pos = sec_start
                        break
                    eot_idx = current_text.find(self.tool_calls_end_token, end_pos)
                    if eot_idx != -1:
                        pos = eot_idx + len(self.tool_calls_end_token)
                        next_scan_pos = pos
                        continue
                    self._in_open_block = True
                    next_scan_pos = sec_start
                    break

                slice_text = current_text[payload_start:]
                try:
                    part_all, _ = partial_json_loads(slice_text, Allow.ALL)
                except (
                    partial_json_parser.core.exceptions.MalformedJSON,
                    ValueError,
                    AssertionError,
                    RecursionError,
                ):
                    logger.debug("not enough tokens to parse into JSON yet")
                    eot_idx = current_text.find(self.tool_calls_end_token, payload_start)
                    if eot_idx != -1:
                        pos = eot_idx + len(self.tool_calls_end_token)
                        next_scan_pos = pos
                        continue
                    self._in_open_block = True
                    next_scan_pos = sec_start
                    break

                if isinstance(part_all, list):
                    try:
                        part_safe, _ = partial_json_loads(slice_text, Allow.ALL & ~Allow.STR)
                        if not isinstance(part_safe, list):
                            part_safe = []
                    except (
                        partial_json_parser.core.exceptions.MalformedJSON,
                        ValueError,
                        AssertionError,
                        RecursionError,
                    ):
                        part_safe = []

                    for i, item in enumerate(part_all):
                        if not isinstance(item, dict):
                            continue
                        safe_item = part_safe[i] if i < len(part_safe) and isinstance(part_safe[i], dict) else {}
                        all_calls.append((item, safe_item, i < len(part_all) - 1, False))
                self._in_open_block = True
                next_scan_pos = sec_start
                break

            content_chunks: list[str] = []
            emitted_pos = self._emitted_content_pos or 0
            for seg_start, seg_end in visible_parts:
                if seg_end > emitted_pos:
                    content_chunks.append(current_text[max(seg_start, emitted_pos) : seg_end])
                    emitted_pos = seg_end
            self._emitted_content_pos = emitted_pos
            content_out = "".join(content_chunks)

            deltas: list[DeltaToolCall] = []
            emitted_idx = self._closed_emitted_idx
            next_closed_emitted_idx = self._closed_emitted_idx
            new_closed_calls: list[dict] = []
            open_tracked_calls: list[dict] = []
            for call, safe_call, call_complete, block_closed in all_calls:
                already_sent = emitted_idx < len(self._tool_name_sent) and self._tool_name_sent[emitted_idx]
                if already_sent or call_complete:
                    func_name = call.get("name")
                else:
                    func_name = safe_call.get("name")

                if not isinstance(func_name, str) or not func_name or "arguments" not in call:
                    if not call_complete and emitted_idx > self.current_tool_id:
                        self.current_tool_id = emitted_idx
                        self.current_tool_name_sent = already_sent
                    continue

                idx = emitted_idx
                emitted_idx += 1
                if block_closed:
                    next_closed_emitted_idx = emitted_idx
                while len(self.streamed_args_for_tool) <= idx:
                    self.streamed_args_for_tool.append("")
                while len(self._tool_name_sent) <= idx:
                    self._tool_name_sent.append(False)
                while len(self._tool_call_ids) <= idx:
                    self._tool_call_ids.append("")

                if idx > self.current_tool_id:
                    self.current_tool_id = idx
                    self.current_tool_name_sent = self._tool_name_sent[idx]

                first = not self._tool_name_sent[idx]
                if first:
                    self._tool_call_ids[idx] = f"chatcmpl-tool-{uuid4()}"
                    self._tool_name_sent[idx] = True
                if idx >= self.current_tool_id:
                    self.current_tool_id = idx
                    self.current_tool_name_sent = True

                cur_arguments = call.get("arguments")
                argument_diff = ""
                if cur_arguments is not None:
                    cur_args_json = (
                        cur_arguments
                        if isinstance(cur_arguments, str)
                        else json.dumps(cur_arguments, ensure_ascii=False)
                    )
                    safe_json = cur_args_json if call_complete else stable_json_stream_prefix(cur_args_json)
                    sent = self.streamed_args_for_tool[idx]
                    if safe_json.startswith(sent):
                        argument_diff = safe_json[len(sent) :]
                        if argument_diff:
                            self.streamed_args_for_tool[idx] = sent + argument_diff

                if block_closed:
                    new_closed_calls.append(call)
                else:
                    open_tracked_calls.append(call)
                if first or argument_diff:
                    deltas.append(
                        DeltaToolCall(
                            index=idx,
                            type="function" if first else None,
                            id=self._tool_call_ids[idx] if first else None,
                            function=DeltaFunctionCall(
                                name=func_name if first else None,
                                arguments=argument_diff if argument_diff else None,
                            ).model_dump(exclude_none=True),
                        )
                    )

            if new_closed_calls:
                self._closed_tracked_calls.extend(new_closed_calls)
            self.prev_tool_call_arr = (
                [*self._closed_tracked_calls, *open_tracked_calls]
                if open_tracked_calls
                else list(self._closed_tracked_calls)
            )
            self._scan_pos = next_scan_pos
            self._closed_emitted_idx = next_closed_emitted_idx

            if content_out and deltas:
                return DeltaMessage(content=content_out, tool_calls=deltas)
            if deltas:
                return DeltaMessage(tool_calls=deltas)
            if content_out:
                return DeltaMessage(content=content_out)
            return None

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug("Skipping chunk as a result of tool streaming extraction error")
            return None
