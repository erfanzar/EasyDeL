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

"""Phi-4 Mini JSON tool parser module.

This module provides the Phi4MiniJsonToolParser class for parsing tool calls
in the ``functools[...]`` format used by Phi-4-mini models in both batch and
streaming modes.

Example format parsed:
    functools[{"name": "func_name", "arguments": {"arg1": "value1"}}]
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import partial_json_parser  # pyright: ignore[reportMissingTypeStubs]
from eformer.loggings import get_logger
from partial_json_parser.core.options import Allow  # pyright: ignore[reportMissingTypeStubs]
from transformers import PreTrainedTokenizerBase

from ...openai_api_modules import (
    ChatCompletionRequest,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from ..abstract_tool import ToolParser, ToolParserManager
from .utils import is_complete_json, partial_json_loads, stream_json_calls

logger = get_logger(__name__)


def _trailing_prefix_len(text: str, marker: str) -> int:
    """Return length of the longest proper prefix of *marker* matching the end of *text*."""
    max_len = min(len(text), len(marker) - 1)
    for size in range(max_len, 0, -1):
        if text.endswith(marker[:size]):
            return size
    return 0


def _is_complete_json_safe(text: str) -> bool:
    """Return ``is_complete_json(text)``, treating ``RecursionError`` as incomplete."""
    try:
        return is_complete_json(text)
    except RecursionError:
        return False


def _normalize_phi4_calls(
    obj: Any,
    *,
    partial_obj: Any = None,
    default_empty_args: bool = False,
) -> list[dict[str, Any]]:
    """Normalize a JSON array of Phi-4 tool call dicts."""
    if not isinstance(obj, list):
        return []
    p_list = partial_obj if isinstance(partial_obj, list) else []
    calls: list[dict[str, Any]] = []
    for idx, item in enumerate(obj):
        if not isinstance(item, dict):
            continue
        p_item = p_list[idx] if idx < len(p_list) and isinstance(p_list[idx], dict) else item
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        if "arguments" in p_item:
            args = p_item["arguments"]
        elif "parameters" in p_item:
            args = p_item["parameters"]
        else:
            args = {} if default_empty_args else None
        calls.append({"name": name, "arguments": args})
    return calls


@ToolParserManager.register_module("phi4_mini_json")  # pyright: ignore[reportUntypedClassDecorator]
class Phi4MiniJsonToolParser(ToolParser):
    """Tool call parser for Phi-4-mini models.

    Handles the ``functools[...]`` format used by Phi-4-mini models in both
    batch and streaming modes. Uses ``bot_token = "functools["`` so ordinary
    prose mentioning Python's ``functools`` module is never buffered or dropped.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Initialize the Phi-4 Mini JSON tool parser."""
        super().__init__(tokenizer)
        self.bot_token: str = "functools["
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state for a new response."""
        self.prev_tool_call_arr: list[dict[str, Any]] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
        self._stream_tool_ids: list[str] = []
        self._streaming_initialized = False
        self._cursor = 0
        self._accumulated_text = ""
        self._in_tool_block = False
        self._block_start = -1
        self._json_start = -1
        self._emitted_tool_call = False
        self._block_emitted_tool_call = False
        self._closed_calls: list[dict[str, Any]] = []

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Return True while inside an unclosed ``functools[...`` payload or partial ``functools[`` prefix."""
        del delta_text
        if self._streaming_initialized:
            if self._in_tool_block:
                return True
            return _trailing_prefix_len(current_text[self._cursor :], self.bot_token) > 0
        idx = current_text.find(self.bot_token)
        if idx != -1:
            json_part = current_text[idx + len("functools") :]
            try:
                json.JSONDecoder().raw_decode(json_part, 0)
                return False
            except RecursionError:
                return False
            except json.JSONDecodeError:
                try:
                    partial_json_loads(json_part, Allow.ALL)
                    return True
                except (
                    ValueError,
                    partial_json_parser.core.exceptions.MalformedJSON,
                    AssertionError,
                    IndexError,
                    RecursionError,
                ):
                    return False
            except ValueError:
                return False
        return _trailing_prefix_len(current_text, self.bot_token) > 0

    def flush_streaming_content(self) -> str:
        """Release withheld non-tool text at end-of-stream."""
        if self._in_tool_block and self._block_emitted_tool_call:
            self._accumulated_text = ""
            self._cursor = 0
            return ""
        self._in_tool_block = False
        self._block_emitted_tool_call = False
        pending = self._accumulated_text[self._cursor :]
        self._accumulated_text = ""
        self._cursor = 0
        return pending

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model response."""
        del request
        logger.debug("Model output: %s", model_output)

        matches = re.search(r"functools\[", model_output)
        if not matches:
            logger.debug("No function calls found")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            json_start = matches.end() - 1
            try:
                decoded, end_pos = json.JSONDecoder().raw_decode(model_output, json_start)
                if not isinstance(decoded, list):
                    raise ValueError("functools payload is not a JSON array")
            except (json.JSONDecodeError, ValueError, RecursionError) as e:
                logger.error("Failed to parse function calls from model output. Error: %s", str(e))
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            normalized = _normalize_phi4_calls(decoded, default_empty_args=True)
            if not normalized:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            tool_calls: list[ToolCall] = []
            for call in normalized:
                raw_args = call["arguments"]
                args_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
                tool_calls.append(
                    ToolCall(
                        id=f"chatcmpl-tool-{uuid4()}",
                        type="function",
                        function=FunctionCall(name=call["name"], arguments=args_str),
                    )
                )

            preamble = model_output[: matches.start()]
            trailing = model_output[end_pos:]
            combined_content = preamble + trailing
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=combined_content if combined_content else None,
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
        """Extract tool calls incrementally from streaming model output."""
        del delta_text, previous_token_ids, current_token_ids, delta_token_ids, request
        if not previous_text:
            self._reset_streaming_state()
            self._streaming_initialized = True
        elif not self._streaming_initialized:
            self._reset_streaming_state()
            self._cursor = len(previous_text)
            self._streaming_initialized = True

        self._accumulated_text = current_text
        content_parts: list[str] = []
        tool_deltas: list[DeltaToolCall] = []

        while True:
            if not self._in_tool_block:
                start_idx = current_text.find(self.bot_token, self._cursor)
                if start_idx != -1:
                    if start_idx > self._cursor:
                        content_parts.append(current_text[self._cursor : start_idx])
                    self._in_tool_block = True
                    self._block_emitted_tool_call = False
                    self._block_start = start_idx
                    self._json_start = start_idx + len("functools")
                    self._cursor = start_idx
                    continue

                rem = current_text[self._cursor :]
                hold = _trailing_prefix_len(rem, self.bot_token)
                safe_end = len(current_text) - hold
                if safe_end > self._cursor:
                    content_parts.append(current_text[self._cursor : safe_end])
                    self._cursor = safe_end
                break

            json_slice = current_text[self._json_start :]
            try:
                decoded, consumed = json.JSONDecoder().raw_decode(json_slice, 0)
            except (json.JSONDecodeError, ValueError, RecursionError):
                decoded = None
                consumed = 0

            if decoded is not None:
                end_abs = self._json_start + consumed
                calls = _normalize_phi4_calls(decoded, default_empty_args=True)
                if calls:
                    combined = [*self._closed_calls, *calls]
                    deltas = stream_json_calls(
                        combined,
                        self.prev_tool_call_arr,
                        self.streamed_args_for_tool,
                        self._stream_tool_ids,
                        True,
                    )
                    if deltas:
                        self._emitted_tool_call = True
                        self._block_emitted_tool_call = True
                        tool_deltas.extend(deltas)
                    self._closed_calls = combined
                    self.current_tool_id = len(combined) - 1
                    self.current_tool_name_sent = bool(self._stream_tool_ids and self._stream_tool_ids[-1])
                else:
                    # Empty functools[] or non-tool array: emit verbatim as content.
                    content_parts.append(current_text[self._block_start : end_abs])
                self._in_tool_block = False
                self._block_emitted_tool_call = False
                self._cursor = end_abs
                continue

            # Still streaming inside functools[...
            try:
                partial_obj, _ = partial_json_loads(json_slice, Allow.ALL)
            except (
                ValueError,
                partial_json_parser.core.exceptions.MalformedJSON,
                AssertionError,
                IndexError,
                RecursionError,
            ):
                if not self._block_emitted_tool_call:
                    content_parts.append(current_text[self._block_start : self._json_start + 1])
                    self._in_tool_block = False
                    self._block_emitted_tool_call = False
                    self._cursor = self._json_start + 1
                    continue
                break

            if not isinstance(partial_obj, list):
                if not self._block_emitted_tool_call:
                    content_parts.append(current_text[self._block_start : self._json_start + 1])
                    self._in_tool_block = False
                    self._block_emitted_tool_call = False
                    self._cursor = self._json_start + 1
                    continue
                break

            try:
                safe_obj, _ = partial_json_loads(json_slice, Allow.ALL & ~Allow.STR)
            except (
                ValueError,
                partial_json_parser.core.exceptions.MalformedJSON,
                AssertionError,
                IndexError,
                RecursionError,
            ):
                break

            calls = _normalize_phi4_calls(
                safe_obj,
                partial_obj=partial_obj,
                default_empty_args=_is_complete_json_safe(json_slice.strip()),
            )
            if calls:
                combined = [*self._closed_calls, *calls]
                deltas = stream_json_calls(
                    combined,
                    self.prev_tool_call_arr,
                    self.streamed_args_for_tool,
                    self._stream_tool_ids,
                    False,
                )
                if deltas:
                    self._emitted_tool_call = True
                    self._block_emitted_tool_call = True
                    tool_deltas.extend(deltas)
                self.current_tool_id = len(combined) - 1
                self.current_tool_name_sent = bool(self._stream_tool_ids and self._stream_tool_ids[-1])
            break

        content = "".join(content_parts)
        if content or tool_deltas:
            return DeltaMessage(content=content or None, tool_calls=tool_deltas or None)
        return None
