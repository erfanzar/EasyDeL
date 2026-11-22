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

"""Minimax tool parser with thinking tag support.

This module provides a comprehensive tool call parser for Minimax models
that handles tool calls wrapped in ``<tool_calls>`` tokens while filtering
out tool calls that appear within ``<think>...</think>`` thinking sections.

Example format:
    <tool_calls>
    {"name": "func", "arguments": {...}}
    </tool_calls>
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
from transformers import AutoTokenizer as AnyTokenizer

from ...openai_api_modules import (
    ChatCompletionRequest,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from ..abstract_tool import ToolParser, ToolParserManager
from .utils import partial_json_loads, stream_json_calls

logger = get_logger(__name__)


def _trailing_prefix_len(text: str, marker: str) -> int:
    """Return length of the longest proper prefix of *marker* matching the end of *text*."""
    max_len = min(len(text), len(marker) - 1)
    for size in range(max_len, 0, -1):
        if text.endswith(marker[:size]):
            return size
    return 0


def _find_tag_outside_json_strings(text: str, tag: str, start: int = 0) -> int:
    """Find *tag* in *text* from *start*, ignoring occurrences inside JSON double-quoted strings."""
    in_string = False
    escaped = False
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if text.startswith(tag, i):
            return i
        i += 1
    return -1


def _strip_trailing_partial_tag_outside_json_strings(text: str, tag: str) -> str:
    """Strip an incomplete trailing *tag* prefix only when outside JSON strings."""
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
    if in_string:
        return text
    hold = _trailing_prefix_len(text, tag)
    return text[:-hold] if hold else text


@ToolParserManager.register_module("minimax")  # pyright: ignore[reportUntypedClassDecorator]
class MinimaxToolParser(ToolParser):
    """Tool parser for Minimax models.

    Handles ``<tool_calls>...</tool_calls>`` blocks while ignoring any
    ``<tool_calls>`` markup inside closed or unclosed ``<think>`` sections.
    Supports multiple ``<tool_calls>`` blocks, pretty-printed and multi-object
    JSON payloads, and string-aware streaming without leaking protocol JSON.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the MinimaxToolParser.

        Args:
            tokenizer: The tokenizer associated with the Minimax model.

        Raises:
            ValueError: If tokenizer is None.
        """
        super().__init__(tokenizer)

        self.tool_call_start_token = "<tool_calls>"
        self.tool_call_end_token = "</tool_calls>"
        self.tool_call_regex = re.compile(r"<tool_calls>(.*?)</tool_calls>|<tool_calls>(.*)", re.DOTALL)
        self.thinking_tag_pattern = r"<think>(.*?)</think>"
        self.tool_name_pattern = re.compile(r'"name":\s*"([^"]+)"')
        self.tool_args_pattern = re.compile(r'"arguments":\s*')

        self._reset_streaming_state()

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            logger.warning(
                "Minimax Tool parser could not locate tool call start/end "
                "tokens in the tokenizer. Falling back to string matching."
            )

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state to initial values."""
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool = []
        self.pending_buffer = ""
        self.in_thinking_tag = False
        self.streaming_state: dict[str, Any] = {
            "current_tool_index": -1,
            "tool_ids": [],
            "sent_tools": [],
        }
        self._stream_tool_ids: list[str] = []
        self._streaming_initialized = False
        self._cursor = 0
        self._accumulated_text = ""
        self._in_open_tool_block = False
        self._scan_pos = 0
        self._in_think = False
        self._open_block_start: int | None = None
        self._open_block_scan_pos = 0
        self._open_block_in_string = False
        self._open_block_escaped = False
        self._closed_call_count = 0
        self._active_block_prev: list[dict] = []
        self._active_block_sent: list[str] = []
        self._active_block_ids: list[str] = []

    def flush_streaming_content(self) -> str:
        """Release withheld opening-tag prefix when the stream ends outside a tool block."""
        if self._in_open_tool_block:
            self._accumulated_text = ""
            self._cursor = 0
            return ""
        pending = self._accumulated_text[self._cursor :]
        self._accumulated_text = ""
        self._cursor = 0
        return pending

    def is_buffering_protocol(self, *, current_text: str, delta_text: str) -> bool:
        """Return True only for unclosed ``<tool_calls>`` blocks or prefixes outside ``<think>``."""
        del delta_text
        if self._streaming_initialized:
            if self._in_open_tool_block:
                return True
            if self.in_thinking_tag:
                return False
            return _trailing_prefix_len(current_text[self._cursor :], self.tool_call_start_token) > 0
        blocks, in_open_think = self._scan_tool_blocks(current_text)
        if blocks and blocks[-1][1] is None:
            return True
        if in_open_think:
            return False
        tail_start = blocks[-1][1] + len(self.tool_call_end_token) if blocks and blocks[-1][1] is not None else 0
        return _trailing_prefix_len(current_text[tail_start:], self.tool_call_start_token) > 0

    def _scan_tool_blocks(self, text: str) -> tuple[list[tuple[int, int | None]], bool]:
        """Scan *text* for ``<tool_calls>`` blocks strictly outside ``<think>`` regions.

        Returns ``(blocks, in_open_think)`` where each element of ``blocks`` is
        ``(start_idx, end_idx)`` (with ``end_idx is None`` if still open).
        """
        blocks: list[tuple[int, int | None]] = []
        pos = 0
        n = len(text)
        in_think = False

        while pos < n:
            if in_think:
                close_think = text.find("</think>", pos)
                if close_think == -1:
                    return blocks, True
                pos = close_think + len("</think>")
                in_think = False
                continue

            next_think = text.find("<think>", pos)
            next_tool = text.find(self.tool_call_start_token, pos)

            if next_think != -1 and (next_tool == -1 or next_think < next_tool):
                in_think = True
                pos = next_think + len("<think>")
                continue

            if next_tool == -1:
                break

            body_start = next_tool + len(self.tool_call_start_token)
            end_tool = _find_tag_outside_json_strings(text, self.tool_call_end_token, body_start)
            if end_tool == -1:
                blocks.append((next_tool, None))
                break
            blocks.append((next_tool, end_tool))
            pos = end_tool + len(self.tool_call_end_token)

        return blocks, in_think

    def preprocess_model_output(self, model_output: str) -> str:
        """Remove closed and unclosed ``<tool_calls>`` markup from inside ``<think>`` sections."""

        def _strip_inner(content: str) -> str:
            cleaned = re.sub(r"<tool_calls>.*?(?:</tool_calls>|$)", "", content, flags=re.DOTALL)
            return cleaned.replace("</tool_calls>", "")

        def _replace_closed(match: re.Match[str]) -> str:
            return f"<think>{_strip_inner(match.group(1))}</think>"

        result = re.sub(self.thinking_tag_pattern, _replace_closed, model_output, flags=re.DOTALL)
        # Also sanitize any unclosed trailing <think> block.
        last_open = result.rfind("<think>")
        last_close = result.rfind("</think>")
        if last_open > last_close:
            result = result[: last_open + len("<think>")] + _strip_inner(result[last_open + len("<think>") :])
        return result

    def _clean_duplicate_braces(self, args_text: str) -> str:
        """Clean duplicate closing braces from arguments text."""
        args_text = args_text.strip()
        if not args_text:
            return args_text
        try:
            json.loads(args_text)
            return args_text
        except (json.JSONDecodeError, ValueError, RecursionError):
            pass
        while args_text.endswith("}}"):
            candidate = args_text[:-1]
            try:
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, ValueError, RecursionError):
                args_text = candidate
        return args_text

    def _find_tool_boundaries(self, text: str) -> list[tuple[int, int]]:
        """Find JSON object boundaries in *text* with string/escape-aware brace tracking."""
        boundaries: list[tuple[int, int]] = []
        i = 0
        n = len(text)
        while i < n:
            if text[i] != "{":
                i += 1
                continue
            start = i
            depth = 0
            in_string = False
            escaped = False
            while i < n:
                ch = text[i]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                elif ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        segment = text[start:end]
                        if '"name"' in segment and '"arguments"' in segment:
                            boundaries.append((start, end))
                        i = end
                        break
                i += 1
            else:
                if depth > 0 and '"name"' in text[start:i]:
                    boundaries.append((start, i))
        return boundaries

    def _extract_tool_args(self, tool_content: str, args_match: re.Match[str]) -> str:
        """Extract tool arguments from *tool_content* with string/escape-aware brace scanning."""
        remaining = tool_content[args_match.end() :].lstrip()
        if remaining.startswith("{"):
            depth = 0
            in_string = False
            escaped = False
            for idx, ch in enumerate(remaining):
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                elif ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return remaining[: idx + 1]
            return remaining
        try:
            _, end_idx = json.JSONDecoder().raw_decode(remaining)
            return remaining[:end_idx]
        except (json.JSONDecodeError, ValueError, RecursionError):
            return remaining.rstrip("}").strip()

    @staticmethod
    def _decode_block_calls_batch(body: str) -> list[dict]:
        """Decode all JSON tool call objects/arrays from a ``<tool_calls>`` block."""
        raw_calls: list[dict] = []
        decoder = json.JSONDecoder()
        position = 0
        length = len(body)
        while position < length:
            if body[position].isspace() or body[position] in ",;":
                position += 1
                continue
            try:
                parsed, position = decoder.raw_decode(body, position)
            except (json.JSONDecodeError, ValueError, RecursionError):
                newline = body.find("\n", position)
                if newline < 0:
                    break
                position = newline + 1
                continue
            if isinstance(parsed, dict):
                raw_calls.append(parsed)
            elif isinstance(parsed, list):
                raw_calls.extend(item for item in parsed if isinstance(item, dict))
        return raw_calls

    @staticmethod
    def _decode_block_calls_streaming(body: str, *, closed: bool) -> list[tuple[dict, bool]]:
        """Decode complete and partial JSON tool call objects from a streaming block."""
        payload = body if closed else _strip_trailing_partial_tag_outside_json_strings(body, "</tool_calls>")
        decoder = json.JSONDecoder()
        results: list[tuple[dict, bool]] = []
        pos = 0
        length = len(payload)

        while pos < length:
            if payload[pos].isspace() or payload[pos] in ",;":
                pos += 1
                continue
            try:
                parsed, next_pos = decoder.raw_decode(payload, pos)
            except RecursionError:
                break
            except json.JSONDecodeError:
                # Tail is incomplete JSON: parse safely for names and partially for arguments.
                tail = payload[pos:].strip()
                if not tail:
                    break
                try:
                    safe_obj, _ = partial_json_loads(tail, Allow.ALL & ~Allow.STR)
                except (
                    ValueError,
                    partial_json_parser.core.exceptions.MalformedJSON,
                    AssertionError,
                    IndexError,
                    RecursionError,
                ):
                    break
                safe_items = safe_obj if isinstance(safe_obj, list) else [safe_obj]
                partial_items: list[Any] = []
                try:
                    partial_obj, _ = partial_json_loads(tail, Allow.ALL)
                    partial_items = partial_obj if isinstance(partial_obj, list) else [partial_obj]
                except (
                    ValueError,
                    partial_json_parser.core.exceptions.MalformedJSON,
                    AssertionError,
                    IndexError,
                    RecursionError,
                ):
                    pass

                for idx, item in enumerate(safe_items):
                    if not isinstance(item, dict):
                        continue
                    merged = dict(item)
                    if idx < len(partial_items) and isinstance(partial_items[idx], dict):
                        if "arguments" in partial_items[idx]:
                            merged["arguments"] = partial_items[idx]["arguments"]
                    results.append((merged, closed))
                break
            except ValueError:
                break

            pos = next_pos
            if isinstance(parsed, dict):
                results.append((parsed, True))
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        results.append((item, True))
        return results

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from model output for non-streaming mode."""
        del request
        blocks, _in_think = self._scan_tool_blocks(model_output)
        if not blocks:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            tool_calls: list[ToolCall] = []
            first_valid_block_start: int | None = None

            for block_start, block_end in blocks:
                body_start = block_start + len(self.tool_call_start_token)
                body = model_output[body_start : block_end if block_end is not None else len(model_output)]
                block_calls = []
                for function_call in self._decode_block_calls_batch(body):
                    name = function_call.get("name")
                    if isinstance(name, str) and name and "arguments" in function_call:
                        raw_args = function_call["arguments"]
                        args_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
                        block_calls.append(
                            ToolCall(
                                type="function",
                                function=FunctionCall(name=name, arguments=args_str),
                            )
                        )
                if block_calls:
                    if first_valid_block_start is None:
                        first_valid_block_start = block_start
                    tool_calls.extend(block_calls)

            if not tool_calls or first_valid_block_start is None:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            content = model_output[:first_valid_block_start].strip()
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("An unexpected error occurred during tool call extraction.")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    def _find_tool_start_outside_thinking(self, current_text: str) -> int | None:
        """Find the first ``<tool_calls>`` position outside ``<think>`` blocks."""
        blocks, _ = self._scan_tool_blocks(current_text)
        return blocks[0][0] if blocks else None

    def _find_open_block_end_incremental(self, text: str) -> int:
        """Advance incremental string-aware scan for ``</tool_calls>`` inside the open block."""
        tag = self.tool_call_end_token
        tag_len = len(tag)
        tag_first = tag[0]
        i = self._open_block_scan_pos
        n = len(text)
        in_string = self._open_block_in_string
        escaped = self._open_block_escaped

        while i < n:
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue
            if ch == '"':
                in_string = True
                i += 1
                continue
            if ch == tag_first:
                if text.startswith(tag, i):
                    self._open_block_scan_pos = i + tag_len
                    self._open_block_in_string = False
                    self._open_block_escaped = False
                    return i
                if n - i < tag_len and tag.startswith(text[i:]):
                    break
            i += 1

        self._open_block_scan_pos = i
        self._open_block_in_string = in_string
        self._open_block_escaped = escaped
        return -1

    def _emit_active_block_deltas(
        self,
        block_calls: list[dict],
        *,
        last_complete: bool,
    ) -> list[DeltaToolCall]:
        """Stream calls for the active ``<tool_calls>`` block without touching closed blocks."""
        if not block_calls:
            return []
        base = self._closed_call_count
        deltas = stream_json_calls(
            block_calls,
            self._active_block_prev,
            self._active_block_sent,
            self._active_block_ids,
            last_complete,
        )
        for delta in deltas:
            delta.index += base
        self.prev_tool_call_arr[base:] = self._active_block_prev
        self.streamed_args_for_tool[base:] = self._active_block_sent
        self._stream_tool_ids[base:] = self._active_block_ids
        self.current_tool_id = base + len(block_calls) - 1
        self.current_tool_name_sent = bool(self._stream_tool_ids and self._stream_tool_ids[-1])
        self.streaming_state["current_tool_index"] = self.current_tool_id
        self.streaming_state["tool_ids"] = list(self._stream_tool_ids)
        sent_tools = self.streaming_state["sent_tools"]
        target_len = base + len(self._active_block_ids)
        while len(sent_tools) < target_len:
            sent_tools.append({"sent_name": False, "sent_arguments": "", "id": ""})
        del sent_tools[target_len:]
        for offset, (tid, args) in enumerate(zip(self._active_block_ids, self._active_block_sent, strict=True)):
            sent_tools[base + offset] = {
                "sent_name": bool(tid),
                "sent_arguments": args,
                "id": tid or sent_tools[base + offset]["id"] or f"chatcmpl-tool-{uuid4()}",
            }
        return deltas

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
        """Extract tool calls incrementally during streaming without leaking tool JSON."""
        del delta_text, previous_token_ids, current_token_ids, delta_token_ids, request
        if not previous_text:
            self._reset_streaming_state()
            self._streaming_initialized = True
        elif not self._streaming_initialized:
            self._reset_streaming_state()
            self._cursor = len(previous_text)
            last_open_think = previous_text.rfind("<think>")
            last_close_think = previous_text.rfind("</think>")
            self._in_think = last_open_think > last_close_think
            self.in_thinking_tag = self._in_think
            if self._in_think:
                self._scan_pos = max(0, len(previous_text) - _trailing_prefix_len(previous_text, "</think>"))
            else:
                hold_prev = max(
                    _trailing_prefix_len(previous_text, "<think>"),
                    _trailing_prefix_len(previous_text, self.tool_call_start_token),
                )
                self._scan_pos = max(0, len(previous_text) - hold_prev)
            self._streaming_initialized = True

        self._accumulated_text = current_text
        content_parts: list[str] = []
        tool_deltas: list[DeltaToolCall] = []

        while True:
            if self._open_block_start is None:
                if self._in_think:
                    close_think = current_text.find("</think>", self._scan_pos)
                    if close_think != -1:
                        self._in_think = False
                        self.in_thinking_tag = False
                        self._scan_pos = close_think + len("</think>")
                        if self._scan_pos > self._cursor:
                            content_parts.append(current_text[self._cursor : self._scan_pos])
                            self._cursor = self._scan_pos
                        continue
                    if len(current_text) > self._cursor:
                        content_parts.append(current_text[self._cursor :])
                        self._cursor = len(current_text)
                    self._scan_pos = max(
                        self._scan_pos,
                        len(current_text) - _trailing_prefix_len(current_text[self._scan_pos :], "</think>"),
                    )
                    self.in_thinking_tag = True
                    break

                next_think = current_text.find("<think>", self._scan_pos)
                next_tool = current_text.find(self.tool_call_start_token, self._scan_pos)

                if next_think != -1 and (next_tool == -1 or next_think < next_tool):
                    self._in_think = True
                    self.in_thinking_tag = True
                    self._scan_pos = next_think + len("<think>")
                    if self._scan_pos > self._cursor:
                        content_parts.append(current_text[self._cursor : self._scan_pos])
                        self._cursor = self._scan_pos
                    continue

                if next_tool != -1:
                    if next_tool > self._cursor:
                        content_parts.append(current_text[self._cursor : next_tool])
                    self._cursor = next_tool
                    self._open_block_start = next_tool
                    self._open_block_scan_pos = next_tool + len(self.tool_call_start_token)
                    self._open_block_in_string = False
                    self._open_block_escaped = False
                    self._in_open_tool_block = True
                    self._active_block_prev = []
                    self._active_block_sent = []
                    self._active_block_ids = []
                    continue

                rem = current_text[self._cursor :]
                hold = _trailing_prefix_len(rem, self.tool_call_start_token)
                safe_end = len(current_text) - hold
                if safe_end > self._cursor:
                    content_parts.append(current_text[self._cursor : safe_end])
                    self._cursor = safe_end
                scan_tail = current_text[self._scan_pos :]
                max_marker_hold = max(
                    _trailing_prefix_len(scan_tail, "<think>"),
                    _trailing_prefix_len(scan_tail, self.tool_call_start_token),
                )
                self._scan_pos = max(self._scan_pos, len(current_text) - max_marker_hold)
                self.in_thinking_tag = False
                break

            # Inside an open <tool_calls> block: advance incrementally to find </tool_calls>.
            end_tool = self._find_open_block_end_incremental(current_text)
            body_start = self._open_block_start + len(self.tool_call_start_token)
            closed = end_tool != -1
            body = current_text[body_start : end_tool if closed else len(current_text)]
            decoded = self._decode_block_calls_streaming(body, closed=closed)
            block_calls: list[dict] = []
            last_complete = False
            for call_obj, call_complete in decoded:
                name = call_obj.get("name")
                if isinstance(name, str) and name:
                    block_calls.append(call_obj)
                    last_complete = call_complete
            if block_calls:
                tool_deltas.extend(self._emit_active_block_deltas(block_calls, last_complete=last_complete))

            if closed:
                self._closed_call_count += len(self._active_block_ids)
                self._active_block_prev = []
                self._active_block_sent = []
                self._active_block_ids = []
                end_after_tag = end_tool + len(self.tool_call_end_token)
                self._cursor = end_after_tag
                self._scan_pos = end_after_tag
                self._open_block_start = None
                self._in_open_tool_block = False
                continue
            break

        content = "".join(content_parts)
        if content or tool_deltas:
            return DeltaMessage(content=content or None, tool_calls=tool_deltas or None)
        return None
