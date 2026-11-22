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

"""Llama 3.x and 4 JSON-format tool call parser.

This module provides a tool call parser specifically designed for Meta's Llama
3.x and 4 model families when using JSON-formatted function calling. Supports
single and multiple tool calls separated by semicolons/whitespace and optional
``<|python_tag|>`` prefixes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

import partial_json_parser  # pyright: ignore[reportMissingTypeStubs]
from eformer.loggings import get_logger
from partial_json_parser.core.options import Allow  # pyright: ignore[reportMissingTypeStubs]
from transformers import PreTrainedTokenizerBase

from ...openai_api_modules import (
    ChatCompletionRequest,
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from ..abstract_tool import ToolParser, ToolParserManager
from ..utils import consume_space, is_complete_json, partial_json_loads
from .utils import stream_json_calls

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


def _normalize_llama_obj(obj: object, *, partial_obj: object | None = None) -> dict | None:
    """Normalize a Llama JSON object, requiring ``name`` plus ``arguments`` or ``parameters``."""
    if not isinstance(obj, dict):
        return None
    pdict = partial_obj if isinstance(partial_obj, dict) else obj
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    if "arguments" in pdict:
        return {"name": name, "arguments": pdict["arguments"]}
    if "parameters" in pdict:
        return {"name": name, "arguments": pdict["parameters"]}
    return None


@ToolParserManager.register_module("llama3_json")
@ToolParserManager.register_module("llama4_json")  # pyright: ignore[reportUntypedClassDecorator]
class Llama3JsonToolParser(ToolParser):
    """Tool call parser for Llama 3.x and 4 models with JSON format.

    Supported Formats:
        - Single tool call: ``{"name": "func", "arguments": {...}}``
        - Multiple tool calls: ``{"name": "func1", ...}; {"name": "func2", ...}``
        - With python tag: ``<|python_tag|>{"name": "func", "arguments": {...}}``
        - Parameters alias: Supports both ``"arguments"`` and ``"parameters"``
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        """Initialize the Llama JSON tool parser."""
        super().__init__(tokenizer)

        self.bot_token = "<|python_tag|>"
        self.bot_token_id: int | None = None
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            encoded = tokenizer.encode(self.bot_token, add_special_tokens=False)
            if encoded:
                self.bot_token_id = encoded[0]
        self.tool_call_regex = re.compile(r"\{")
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state for a new response."""
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
        self._stream_tool_ids: list[str] = []
        self._streaming_initialized = False
        self._cursor = 0
        self._accumulated_text = ""
        self._tool_mode = False
        self._explicit_bot = False
        self._tool_payload_start = -1
        self._buffering_tool_call = False
        self._emitted_tool_call = False

    def get_tool_machinery_hints(self) -> Sequence[str]:
        """Include the bare-JSON ``{`` prefix alongside ``<|python_tag|>``."""
        hints = list(super().get_tool_machinery_hints())
        if "{" not in hints:
            hints.append("{")
        return tuple(hints)

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Return True while buffering ``<|python_tag|>`` or an in-flight tool JSON object."""
        if self._streaming_initialized:
            return self._buffering_tool_call
        if self.bot_token in current_text:
            return True
        if _trailing_prefix_len(current_text, self.bot_token) > 0:
            return True
        stripped = current_text.lstrip()
        if stripped.startswith("{"):
            return not _is_complete_json_safe(stripped)
        return False

    def flush_streaming_content(self) -> str:
        """Release withheld non-tool text at end-of-stream."""
        if self._emitted_tool_call or self._explicit_bot:
            self._accumulated_text = ""
            self._cursor = 0
            return ""
        pending = self._accumulated_text[self._cursor :]
        self._accumulated_text = ""
        self._cursor = 0
        self._buffering_tool_call = False
        self._tool_mode = False
        return pending

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete Llama model response."""
        del request
        if not (self.bot_token in model_output or "{" in model_output):
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        match = self.tool_call_regex.search(model_output)
        if not match:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            decoder = json.JSONDecoder()
            json_objects: list[dict] = []
            first_obj_start: int | None = None
            position = match.start()
            while position < len(model_output):
                candidate = self.tool_call_regex.search(model_output, position)
                if candidate is None:
                    break
                try:
                    obj, position = decoder.raw_decode(model_output, candidate.start())
                except (json.JSONDecodeError, ValueError, RecursionError):
                    position = candidate.end()
                    continue
                if (
                    isinstance(obj, dict)
                    and isinstance(obj.get("name"), str)
                    and obj["name"]
                    and ("arguments" in obj or "parameters" in obj)
                ):
                    if first_obj_start is None:
                        first_obj_start = candidate.start()
                    json_objects.append(obj)
            if not json_objects or first_obj_start is None:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            tool_calls: list[ToolCall] = []
            for obj in json_objects:
                raw_args = obj["arguments"] if "arguments" in obj else obj["parameters"]
                args_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(name=obj["name"], arguments=args_str),
                    )
                )

            bot_idx = model_output.find(self.bot_token)
            content_end = bot_idx if (bot_idx != -1 and bot_idx < first_obj_start) else first_obj_start
            preamble = model_output[:content_end]
            content = preamble if preamble.strip() else None
            return ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls, content=content)

        except Exception:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    @staticmethod
    def _parse_streaming_payload(payload: str) -> tuple[list[dict], bool, int | None, bool]:
        """Parse one or more semicolon/whitespace-separated JSON tool calls from *payload*.

        Returns ``(calls, last_complete, first_non_tool_end, malformed_start)``.
        """
        decoder = json.JSONDecoder()
        calls: list[dict] = []
        last_complete = False
        pos = consume_space(0, payload)
        n = len(payload)

        while pos < n:
            if payload[pos] == ";":
                pos = consume_space(pos + 1, payload)
                continue
            if payload[pos] != "{":
                if not calls:
                    return [], False, None, True
                break
            if not calls:
                after_brace = consume_space(pos + 1, payload)
                if after_brace < n and payload[after_brace] not in ('"', "}"):
                    return [], False, None, True
            try:
                obj, next_pos = decoder.raw_decode(payload, pos)
                norm = _normalize_llama_obj(obj)
                if norm is not None and ("arguments" in obj or "parameters" in obj):
                    calls.append(norm)
                    last_complete = True
                    pos = consume_space(next_pos, payload)
                    if pos < n and payload[pos] == ";":
                        pos = consume_space(pos + 1, payload)
                    continue
                if not calls:
                    return [], True, next_pos, False
                break
            except RecursionError:
                if not calls:
                    return [], False, None, True
                break
            except json.JSONDecodeError:
                tail = payload[pos:]
                try:
                    safe_obj, _ = partial_json_loads(tail, Allow.ALL & ~Allow.STR)
                except (
                    ValueError,
                    partial_json_parser.core.exceptions.MalformedJSON,
                    AssertionError,
                    IndexError,
                    RecursionError,
                ):
                    if not calls:
                        if _is_complete_json_safe(tail.strip()):
                            return [], True, len(payload), False
                        try:
                            partial_json_loads(tail, Allow.ALL)
                        except (
                            ValueError,
                            partial_json_parser.core.exceptions.MalformedJSON,
                            AssertionError,
                            IndexError,
                            RecursionError,
                        ):
                            return [], False, None, True
                    break
                try:
                    partial_obj, _ = partial_json_loads(tail, Allow.ALL)
                except (
                    ValueError,
                    partial_json_parser.core.exceptions.MalformedJSON,
                    AssertionError,
                    IndexError,
                    RecursionError,
                ):
                    partial_obj = safe_obj
                norm = _normalize_llama_obj(safe_obj, partial_obj=partial_obj)
                if norm is not None:
                    calls.append(norm)
                    last_complete = _is_complete_json_safe(tail.strip())
                break
            except ValueError:
                if not calls:
                    return [], False, None, True
                break

        return calls, last_complete, None, False

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
        """Extract tool calls incrementally during streaming generation."""
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

        while not self._tool_mode:
            bot_idx = current_text.find(self.bot_token, self._cursor)
            brace_idx = current_text.find("{", self._cursor)

            if bot_idx != -1 and (brace_idx == -1 or bot_idx < brace_idx):
                if bot_idx > self._cursor:
                    content_parts.append(current_text[self._cursor : bot_idx])
                self._tool_mode = True
                self._explicit_bot = True
                self._tool_payload_start = bot_idx + len(self.bot_token)
                self._cursor = self._tool_payload_start
                self._buffering_tool_call = True
                break

            if brace_idx != -1:
                if brace_idx > self._cursor:
                    content_parts.append(current_text[self._cursor : brace_idx])
                    self._cursor = brace_idx
                payload = current_text[brace_idx:]
                calls, last_complete, non_tool_end, malformed_start = self._parse_streaming_payload(payload)
                if non_tool_end is not None:
                    # Complete JSON object without name+arguments/parameters: ordinary content!
                    end_abs = brace_idx + non_tool_end
                    content_parts.append(current_text[brace_idx:end_abs])
                    self._cursor = end_abs
                    self._buffering_tool_call = False
                    continue
                if malformed_start:
                    content_parts.append(current_text[brace_idx : brace_idx + 1])
                    self._cursor = brace_idx + 1
                    self._buffering_tool_call = False
                    continue
                if calls:
                    self._tool_mode = True
                    self._explicit_bot = False
                    self._tool_payload_start = brace_idx
                    self._buffering_tool_call = not last_complete
                    break
                # Could still be an incomplete JSON object (e.g., "{", '{"na') or non-JSON "{x}".
                # Check if a closing '}' already appeared with invalid JSON -> flush as content.
                if "}" in payload:
                    # Try checking whether it can still be valid JSON (e.g., nested or string).
                    try:
                        partial_json_loads(payload, Allow.ALL)
                        self._buffering_tool_call = True
                        break
                    except (
                        ValueError,
                        partial_json_parser.core.exceptions.MalformedJSON,
                        AssertionError,
                        IndexError,
                        RecursionError,
                    ):
                        content_parts.append(current_text[brace_idx : brace_idx + 1])
                        self._cursor = brace_idx + 1
                        self._buffering_tool_call = False
                        continue
                self._buffering_tool_call = True
                break

            # Neither <|python_tag|> nor '{' after self._cursor: withhold only partial <|python_tag|> prefix.
            rem = current_text[self._cursor :]
            hold = _trailing_prefix_len(rem, self.bot_token)
            safe_end = len(current_text) - hold
            if safe_end > self._cursor:
                content_parts.append(current_text[self._cursor : safe_end])
                self._cursor = safe_end
            self._buffering_tool_call = hold > 0
            break

        deltas = []
        if self._tool_mode:
            payload = current_text[self._tool_payload_start :]
            calls, last_complete, non_tool_end, _malformed = self._parse_streaming_payload(payload)
            if non_tool_end is not None and not self._emitted_tool_call and not self._explicit_bot:
                end_abs = self._tool_payload_start + non_tool_end
                content_parts.append(current_text[self._tool_payload_start : end_abs])
                self._cursor = end_abs
                self._tool_mode = False
                self._buffering_tool_call = False
            elif calls:
                deltas = stream_json_calls(
                    calls,
                    self.prev_tool_call_arr,
                    self.streamed_args_for_tool,
                    self._stream_tool_ids,
                    last_complete,
                )
                if deltas:
                    self._emitted_tool_call = True
                self.current_tool_id = len(calls) - 1
                self.current_tool_name_sent = bool(self._stream_tool_ids and self._stream_tool_ids[-1])
                self._buffering_tool_call = not last_complete
            else:
                self._buffering_tool_call = True

        content = "".join(content_parts)
        if content or deltas:
            return DeltaMessage(content=content or None, tool_calls=deltas or None)
        return None
