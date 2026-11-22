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
formats used by xLAM and similar models, including JSON code blocks,
``[TOOL_CALLS]`` markers, XML-style ``<tool_call>`` tags, ``</think>`` suffixes,
and bare JSON arrays/objects.
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
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from ..abstract_tool import ToolParser, ToolParserManager
from .utils import is_complete_json, partial_json_loads, stream_json_calls

logger = get_logger(__name__)


def _random_uuid() -> str:
    """Generate a random UUID hex string."""
    return uuid4().hex


def _trailing_prefix_len_any(text: str, markers: Sequence[str]) -> int:
    """Return length of the longest proper prefix of any marker matching the end of *text*."""
    best = 0
    for marker in markers:
        max_len = min(len(text), len(marker) - 1)
        for size in range(max_len, best, -1):
            if text.endswith(marker[:size]):
                best = size
                break
    return best


def _find_tag_outside_json_strings(text: str, tag: str, start: int = 0) -> int:
    """Find *tag* in *text* from *start*, ignoring occurrences inside JSON strings."""
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


def _strip_trailing_partial_tag_outside_strings(text: str, tag: str) -> str:
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
    hold = _trailing_prefix_len_any(text, (tag,))
    return text[:-hold] if hold else text


def _decode_json_after_marker(text: str, start: int) -> tuple[Any, int] | None:
    """Decode one JSON object or array starting at/after *start* (skipping whitespace)."""
    pos = start
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text) or text[pos] not in "{[":
        return None
    try:
        return json.JSONDecoder().raw_decode(text, pos)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return None


def _is_complete_json_safe(text: str) -> bool:
    """Return ``is_complete_json(text)``, treating ``RecursionError`` as incomplete."""
    try:
        return is_complete_json(text)
    except RecursionError:
        return False


def _valid_call_dicts(obj: Any, *, partial_obj: Any = None) -> list[dict]:
    """Extract normalized ``{"name": ..., "arguments": ...}`` dicts from a JSON object or list."""
    items = obj if isinstance(obj, list) else ([obj] if isinstance(obj, dict) else [])
    p_items = partial_obj if isinstance(partial_obj, list) else ([partial_obj] if isinstance(partial_obj, dict) else [])
    calls: list[dict] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        p_item = p_items[idx] if idx < len(p_items) and isinstance(p_items[idx], dict) else item
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        if "arguments" in p_item:
            calls.append({"name": name, "arguments": p_item["arguments"]})
        elif "parameters" in p_item:
            calls.append({"name": name, "arguments": p_item["parameters"]})
    return calls


@ToolParserManager.register_module("xlam")  # pyright: ignore[reportUntypedClassDecorator]
class xLAMToolParser(ToolParser):
    """Tool parser for xLAM and similar models with flexible JSON formats.

    Supported formats:
        - `` ```json [...] ``` `` / `` ``` {...} ``` ``
        - ``[TOOL_CALLS] [...]`` (single-line and multiline)
        - ``<tool_call>...</tool_call>`` (single or multiple blocks)
        - ``</think>[...]``
        - Raw JSON arrays ``[...]`` or objects ``{...}``
    """

    _WRAPPER_MARKERS = ("[TOOL_CALLS]", "<tool_call>", "```")
    _TRAILING_SCAN_MARKERS = ("[TOOL_CALLS]", "<tool_call>", "```", "</think>")

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the xLAM tool parser."""
        super().__init__(tokenizer)

        self.json_code_block_patterns = [
            r"```(?:json)?\s*([\s\S]*?)```",
            r"\[TOOL_CALLS\]\s*([\s\S]*)",
            r"<tool_call>([\s\S]*?)</tool_call>",
        ]
        self.thinking_tag_pattern = r"</think>([\s\S]*)"
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        """Reset all streaming and compatibility state for a new response."""
        self.prev_tool_calls: list[dict] = []
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args: list[str] = []
        self.streamed_args_for_tool = self.streamed_args
        self.current_tools_sent: list[bool] = []
        self.streaming_state: dict[str, Any] = {
            "current_tool_index": -1,
            "tool_ids": [],
            "sent_tools": [],
        }
        self._streaming_initialized = False
        self._cursor = 0
        self._accumulated_text = ""
        self._buffering_tool_call = False
        self._emitted_tool_call = False
        self._bare_decided_plain = False
        self._bare_completed = False
        self._active_wrapper: str | None = None
        self._wrapper_start = -1
        self._wrapper_body_start = -1
        self._current_wrapper_emitted = False
        self._awaiting_post_think_json = False
        self._closed_wrapper_calls: list[dict] = []

    def get_tool_machinery_hints(self) -> Sequence[str]:
        """Engage streaming tool machinery on bare JSON delimiters and all xLAM wrappers."""
        hints = list(super().get_tool_machinery_hints())
        for marker in ("[", "{", "[TOOL_CALLS]", "<tool_call>", "```", "</think>"):
            if marker not in hints:
                hints.append(marker)
        return tuple(hints)

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Return True only while buffering a potential xLAM wrapper or leading JSON call."""
        del delta_text
        if self._streaming_initialized:
            return self._buffering_tool_call
        stripped = current_text.lstrip()
        if stripped.startswith(("{", "[")):
            return not _is_complete_json_safe(stripped)
        if "<tool_call>" in current_text and current_text.count("<tool_call>") > current_text.count("</tool_call>"):
            return True
        if current_text.count("```") % 2 == 1:
            return True
        if "[TOOL_CALLS]" in current_text:
            after = current_text.split("[TOOL_CALLS]", 1)[1].strip()
            if not after:
                return True
            if not after.startswith(("[", "{")):
                return False
            return not _is_complete_json_safe(after)
        return _trailing_prefix_len_any(current_text, self._TRAILING_SCAN_MARKERS) > 0

    def flush_streaming_content(self) -> str:
        """Release withheld non-tool text at end-of-stream."""
        if (not self._bare_decided_plain and self._emitted_tool_call) or (
            self._active_wrapper not in (None, "PLAIN_FENCE") and self._current_wrapper_emitted
        ):
            self._accumulated_text = ""
            self._cursor = 0
            return ""
        pending = self._accumulated_text[self._cursor :]
        self._accumulated_text = ""
        self._cursor = 0
        self._buffering_tool_call = False
        self._active_wrapper = None
        self._awaiting_post_think_json = False
        return pending

    def _extract_wrapped_payloads(self, text: str) -> tuple[list[dict], str | None]:
        """Extract all tool-call dicts from wrapped blocks, returning ``(calls, remaining_content)``."""
        calls: list[dict] = []
        content_parts: list[str] = []
        pos = 0
        n = len(text)

        while pos < n:
            candidates = [
                (idx, marker)
                for marker in self._WRAPPER_MARKERS
                if (idx := text.find(marker, pos)) != -1
            ]
            if not candidates:
                content_parts.append(text[pos:])
                break
            start_idx, marker = min(candidates, key=lambda pair: pair[0])

            if marker == "<tool_call>":
                body_start = start_idx + len(marker)
                end_idx = _find_tag_outside_json_strings(text, "</tool_call>", body_start)
                if end_idx == -1:
                    content_parts.append(text[pos:])
                    break
                block_end = end_idx + len("</tool_call>")
                try:
                    parsed = json.loads(text[body_start:end_idx].strip())
                    block_calls = _valid_call_dicts(parsed)
                except (json.JSONDecodeError, ValueError, RecursionError):
                    block_calls = []
                if block_calls:
                    content_parts.append(text[pos:start_idx])
                    calls.extend(block_calls)
                else:
                    content_parts.append(text[pos:block_end])
                pos = block_end
                continue

            if marker == "```":
                after_ticks = start_idx + 3
                nl = text.find("\n", after_ticks)
                brace = min(
                    (i for i in (text.find("{", after_ticks), text.find("[", after_ticks)) if i != -1),
                    default=-1,
                )
                inline_close = text.find("```", after_ticks)
                terminators = [i for i in (nl, brace, inline_close) if i != -1]
                if not terminators:
                    content_parts.append(text[pos:])
                    break
                earliest = min(terminators)
                if earliest == inline_close:
                    block_end = inline_close + 3
                    content_parts.append(text[pos:block_end])
                    pos = block_end
                    continue
                lang = text[after_ticks:earliest].strip().lower()
                body_start = earliest + 1 if text[earliest] == "\n" else earliest
                if lang in ("", "json"):
                    end_idx = _find_tag_outside_json_strings(text, "```", body_start)
                else:
                    end_idx = text.find("```", body_start)
                if end_idx == -1:
                    content_parts.append(text[pos:])
                    break
                block_end = end_idx + 3
                block_calls = []
                if lang in ("", "json"):
                    try:
                        parsed = json.loads(text[body_start:end_idx].strip())
                        block_calls = _valid_call_dicts(parsed)
                    except (json.JSONDecodeError, ValueError, RecursionError):
                        block_calls = []
                if block_calls:
                    content_parts.append(text[pos:start_idx])
                    calls.extend(block_calls)
                else:
                    content_parts.append(text[pos:block_end])
                pos = block_end
                continue

            # marker == "[TOOL_CALLS]"
            decoded = _decode_json_after_marker(text, start_idx + len(marker))
            if decoded is not None:
                parsed, end_pos = decoded
                block_calls = _valid_call_dicts(parsed)
                if block_calls:
                    content_parts.append(text[pos:start_idx])
                    calls.extend(block_calls)
                    pos = end_pos
                    continue
            content_parts.append(text[pos : start_idx + len(marker)])
            pos = start_idx + len(marker)

        content = "".join(content_parts).strip()
        return calls, (content or None)

    def preprocess_model_output(self, model_output: str) -> tuple[str | None, str | None]:
        """Preprocess model output to extract potential tool call JSON and remaining content."""
        thinking_match = re.search(self.thinking_tag_pattern, model_output)
        if thinking_match:
            content = model_output[: thinking_match.start() + len("</think>")].strip()
            thinking_tail = thinking_match.group(1).strip()
            if thinking_tail.startswith(("[", "{")):
                try:
                    json.loads(thinking_tail)
                    return content or None, thinking_tail
                except (json.JSONDecodeError, ValueError, RecursionError):
                    pass
            wrapped_calls, extra_content = self._extract_wrapped_payloads(thinking_tail)
            if wrapped_calls:
                combined_content = f"{content}\n{extra_content}".strip() if extra_content else (content or None)
                return combined_content, json.dumps(wrapped_calls, ensure_ascii=False)

        wrapped_calls, remaining = self._extract_wrapped_payloads(model_output)
        if wrapped_calls:
            return remaining, json.dumps(wrapped_calls, ensure_ascii=False)

        stripped = model_output.strip()
        if stripped.startswith(("[", "{")):
            try:
                json.loads(stripped)
                return None, stripped
            except (json.JSONDecodeError, ValueError, RecursionError):
                if "{" in stripped and '"name"' in stripped and '"arguments"' in stripped:
                    return None, stripped

        return model_output, None

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model output."""
        del request
        try:
            content, potential_tool_calls = self.preprocess_model_output(model_output)
            if not potential_tool_calls:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            tool_calls_data = json.loads(potential_tool_calls)
            normalized = _valid_call_dicts(tool_calls_data)
            if not normalized:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            tool_calls: list[ToolCall] = []
            for idx, call in enumerate(normalized):
                raw_args = call["arguments"]
                args_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
                tool_calls.append(
                    ToolCall(
                        id=f"call_{idx}_{_random_uuid()}",
                        type="function",
                        function=FunctionCall(name=call["name"], arguments=args_str),
                    )
                )

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception as e:
            logger.exception("Error extracting tool calls: %s", str(e))
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    @staticmethod
    def _parse_streaming_json(json_text: str) -> tuple[list[dict], bool, bool]:
        """Parse streaming JSON candidate into ``(calls, is_complete, definitely_non_tool)``."""
        stripped = json_text.strip()
        if not stripped:
            return [], False, False
        if not stripped.startswith(("[", "{")):
            return [], False, True

        complete = _is_complete_json_safe(stripped)
        try:
            safe_obj, _ = partial_json_loads(stripped, Allow.ALL & ~Allow.STR)
        except (
            ValueError,
            partial_json_parser.core.exceptions.MalformedJSON,
            AssertionError,
            IndexError,
            RecursionError,
        ):
            try:
                partial_json_loads(stripped, Allow.ALL)
            except (
                ValueError,
                partial_json_parser.core.exceptions.MalformedJSON,
                AssertionError,
                IndexError,
                RecursionError,
            ):
                return [], complete, True
            return [], complete, complete

        try:
            partial_obj, _ = partial_json_loads(stripped, Allow.ALL)
        except (
            ValueError,
            partial_json_parser.core.exceptions.MalformedJSON,
            AssertionError,
            IndexError,
            RecursionError,
        ):
            partial_obj = safe_obj

        calls = _valid_call_dicts(safe_obj, partial_obj=partial_obj)
        if calls:
            return calls, complete, False

        if complete:
            return [], True, True
        if isinstance(partial_obj, list) and any(not isinstance(elt, dict) for elt in partial_obj):
            return [], False, True
        return [], False, False

    def _emit_calls_delta(self, calls: list[dict], *, complete: bool) -> list[DeltaToolCall]:
        """Stream *calls* via ``stream_json_calls`` and synchronize compatibility attributes."""
        deltas = stream_json_calls(
            calls,
            self.prev_tool_call_arr,
            self.streamed_args,
            self.streaming_state["tool_ids"],
            complete,
        )
        if deltas:
            self._emitted_tool_call = True
        self.prev_tool_calls = self.prev_tool_call_arr
        self.streamed_args_for_tool = self.streamed_args
        self.current_tools_sent = [bool(tool_id) for tool_id in self.streaming_state["tool_ids"]]
        self.current_tool_id = len(calls) - 1
        self.current_tool_name_sent = bool(self.current_tools_sent and self.current_tools_sent[-1])
        self.streaming_state["current_tool_index"] = self.current_tool_id
        self.streaming_state["sent_tools"] = [
            {"sent_name": sent, "sent_arguments_prefix": bool(args), "sent_arguments": args}
            for sent, args in zip(self.current_tools_sent, self.streamed_args, strict=True)
        ]
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
        """Extract tool calls incrementally from streaming model output."""
        del delta_text, previous_token_ids, current_token_ids, delta_token_ids, request
        if not previous_text:
            self._reset_streaming_state()
            self._streaming_initialized = True
        elif not self._streaming_initialized:
            self._reset_streaming_state()
            self._cursor = len(previous_text)
            if previous_text.strip():
                self._bare_decided_plain = True
            if previous_text.rstrip().endswith("</think>"):
                self._awaiting_post_think_json = True
            self._streaming_initialized = True

        self._accumulated_text = current_text

        # Path 1: Bare JSON starting at char 0 (not starting with a wrapper like [TOOL_CALLS]).
        if not self._bare_decided_plain and self._active_wrapper is None:
            stripped = current_text.lstrip()
            if not stripped:
                self._buffering_tool_call = True
                return None
            if stripped.startswith("[TOOL_CALLS]") or (
                stripped.startswith("[") and "[TOOL_CALLS]".startswith(stripped) and stripped != "["
            ):
                self._bare_decided_plain = True
            elif stripped.startswith(("[", "{")):
                if self._bare_completed:
                    self._buffering_tool_call = False
                    return None
                calls, complete, non_tool = self._parse_streaming_json(stripped)
                if non_tool and not self._emitted_tool_call:
                    self._bare_decided_plain = True
                    self._buffering_tool_call = False
                else:
                    if calls:
                        deltas = self._emit_calls_delta(calls, complete=complete)
                        if complete:
                            self._bare_completed = True
                            self._cursor = len(current_text)
                        self._buffering_tool_call = not complete
                        return DeltaMessage(tool_calls=deltas) if deltas else None
                    self._buffering_tool_call = not complete
                    return None
            else:
                self._bare_decided_plain = True

        # Path 2: Wrapper scanning ([TOOL_CALLS], <tool_call>, ```, or </think>[...]).
        content_parts: list[str] = []
        tool_deltas: list[DeltaToolCall] = []

        while True:
            if self._active_wrapper == "PLAIN_FENCE":
                close_idx = current_text.find("```", self._cursor)
                if close_idx != -1:
                    block_end = close_idx + 3
                    content_parts.append(current_text[self._cursor : block_end])
                    self._cursor = block_end
                    self._active_wrapper = None
                    self._buffering_tool_call = False
                    continue

                rem = current_text[self._cursor :]
                hold = _trailing_prefix_len_any(rem, ("```",))
                safe_end = len(current_text) - hold
                if safe_end > self._cursor:
                    content_parts.append(current_text[self._cursor : safe_end])
                    self._cursor = safe_end
                self._buffering_tool_call = hold > 0
                break

            if self._active_wrapper is None:
                if self._awaiting_post_think_json:
                    pos_ws = self._cursor
                    while pos_ws < len(current_text) and current_text[pos_ws].isspace():
                        pos_ws += 1
                    if pos_ws >= len(current_text):
                        self._buffering_tool_call = pos_ws > self._cursor
                        break
                    ch = current_text[pos_ws]
                    if ch == "[":
                        after_ws = current_text[pos_ws:]
                        if after_ws == "[":
                            self._buffering_tool_call = True
                            break
                        if after_ws.startswith("[TOOL_CALLS]") or "[TOOL_CALLS]".startswith(after_ws):
                            self._awaiting_post_think_json = False
                            continue
                        self._active_wrapper = "THINK_JSON"
                        self._wrapper_start = self._cursor
                        self._wrapper_body_start = pos_ws
                        self._current_wrapper_emitted = False
                        self._awaiting_post_think_json = False
                        self._buffering_tool_call = True
                        continue
                    if ch == "{":
                        self._active_wrapper = "THINK_JSON"
                        self._wrapper_start = self._cursor
                        self._wrapper_body_start = pos_ws
                        self._current_wrapper_emitted = False
                        self._awaiting_post_think_json = False
                        self._buffering_tool_call = True
                        continue
                    self._awaiting_post_think_json = False

                candidates: list[tuple[int, str]] = [
                    (idx, marker)
                    for marker in self._WRAPPER_MARKERS
                    if (idx := current_text.find(marker, self._cursor)) != -1
                ]
                think_end = current_text.find("</think>", self._cursor)
                if think_end != -1:
                    candidates.append((think_end, "</think>"))

                if candidates:
                    start_idx, marker = min(candidates, key=lambda pair: pair[0])
                    if marker == "</think>":
                        after_think = start_idx + len("</think>")
                        content_parts.append(current_text[self._cursor : after_think])
                        self._cursor = after_think
                        self._awaiting_post_think_json = True
                        self._buffering_tool_call = False
                        continue
                    if start_idx > self._cursor:
                        content_parts.append(current_text[self._cursor : start_idx])
                    self._cursor = start_idx
                    self._active_wrapper = marker
                    self._wrapper_start = start_idx
                    self._wrapper_body_start = start_idx + len(marker)
                    self._current_wrapper_emitted = False
                    self._buffering_tool_call = True
                    continue

                rem = current_text[self._cursor :]
                hold = _trailing_prefix_len_any(rem, self._TRAILING_SCAN_MARKERS)
                safe_end = len(current_text) - hold
                if safe_end > self._cursor:
                    content_parts.append(current_text[self._cursor : safe_end])
                    self._cursor = safe_end
                self._buffering_tool_call = hold > 0
                break

            if self._active_wrapper == "<tool_call>":
                end_idx = _find_tag_outside_json_strings(current_text, "</tool_call>", self._wrapper_body_start)
                if end_idx != -1:
                    body = current_text[self._wrapper_body_start : end_idx]
                    calls, _c, _nt = self._parse_streaming_json(body)
                    block_end = end_idx + len("</tool_call>")
                    if calls:
                        combined = [*self._closed_wrapper_calls, *calls]
                        tool_deltas.extend(self._emit_calls_delta(combined, complete=True))
                        self._closed_wrapper_calls = combined
                        self._current_wrapper_emitted = True
                    elif not self._current_wrapper_emitted:
                        content_parts.append(current_text[self._wrapper_start : block_end])
                    self._active_wrapper = None
                    self._current_wrapper_emitted = False
                    self._cursor = block_end
                    self._buffering_tool_call = False
                    continue

                raw_body = _strip_trailing_partial_tag_outside_strings(
                    current_text[self._wrapper_body_start :], "</tool_call>"
                )
                calls, _c, _nt = self._parse_streaming_json(raw_body)
                if calls:
                    combined = [*self._closed_wrapper_calls, *calls]
                    deltas = self._emit_calls_delta(combined, complete=False)
                    if deltas:
                        self._current_wrapper_emitted = True
                    tool_deltas.extend(deltas)
                self._buffering_tool_call = True
                break

            if self._active_wrapper == "```":
                after_ticks = current_text[self._wrapper_start + 3 :]
                nl = after_ticks.find("\n")
                brace = min(
                    (i for i in (after_ticks.find("{"), after_ticks.find("[")) if i != -1),
                    default=-1,
                )
                inline_close = after_ticks.find("```")
                terminators = [i for i in (nl, brace, inline_close) if i != -1]
                if not terminators:
                    tag = after_ticks.strip().lower()
                    if not tag or "json".startswith(tag):
                        self._buffering_tool_call = True
                        break
                    content_parts.append(current_text[self._cursor : self._wrapper_start + 3])
                    self._cursor = self._wrapper_start + 3
                    self._active_wrapper = "PLAIN_FENCE"
                    self._buffering_tool_call = False
                    continue

                earliest = min(terminators)
                if earliest == inline_close:
                    block_end = self._wrapper_start + 3 + inline_close + 3
                    content_parts.append(current_text[self._cursor : block_end])
                    self._cursor = block_end
                    self._active_wrapper = None
                    self._current_wrapper_emitted = False
                    self._buffering_tool_call = False
                    continue

                tag = after_ticks[:earliest].strip().lower()
                if tag not in ("", "json"):
                    content_parts.append(current_text[self._cursor : self._wrapper_start + 3])
                    self._cursor = self._wrapper_start + 3
                    self._active_wrapper = "PLAIN_FENCE"
                    self._buffering_tool_call = False
                    continue

                body_start = self._wrapper_start + 3 + (earliest + 1 if after_ticks[earliest] == "\n" else earliest)
                end_idx = _find_tag_outside_json_strings(current_text, "```", body_start)
                if end_idx != -1:
                    body = current_text[body_start:end_idx]
                    calls, _c, _nt = self._parse_streaming_json(body)
                    block_end = end_idx + 3
                    if calls:
                        combined = [*self._closed_wrapper_calls, *calls]
                        tool_deltas.extend(self._emit_calls_delta(combined, complete=True))
                        self._closed_wrapper_calls = combined
                        self._current_wrapper_emitted = True
                    elif not self._current_wrapper_emitted:
                        content_parts.append(current_text[self._wrapper_start : block_end])
                    self._active_wrapper = None
                    self._current_wrapper_emitted = False
                    self._cursor = block_end
                    self._buffering_tool_call = False
                    continue

                raw_body = _strip_trailing_partial_tag_outside_strings(current_text[body_start:], "```")
                calls, _c, non_tool = self._parse_streaming_json(raw_body)
                if calls:
                    combined = [*self._closed_wrapper_calls, *calls]
                    deltas = self._emit_calls_delta(combined, complete=False)
                    if deltas:
                        self._current_wrapper_emitted = True
                    tool_deltas.extend(deltas)
                elif non_tool and not self._current_wrapper_emitted and tag == "":
                    content_parts.append(current_text[self._cursor : self._wrapper_start + 3])
                    self._cursor = self._wrapper_start + 3
                    self._active_wrapper = "PLAIN_FENCE"
                    self._buffering_tool_call = False
                    continue
                self._buffering_tool_call = True
                break

            # self._active_wrapper in ("[TOOL_CALLS]", "THINK_JSON")
            decoded = _decode_json_after_marker(current_text, self._wrapper_body_start)
            if decoded is not None:
                parsed, end_pos = decoded
                calls = _valid_call_dicts(parsed)
                if calls:
                    combined = [*self._closed_wrapper_calls, *calls]
                    tool_deltas.extend(self._emit_calls_delta(combined, complete=True))
                    self._closed_wrapper_calls = combined
                    self._current_wrapper_emitted = True
                elif not self._current_wrapper_emitted:
                    content_parts.append(current_text[self._wrapper_start : end_pos])
                self._active_wrapper = None
                self._current_wrapper_emitted = False
                self._cursor = end_pos
                self._buffering_tool_call = False
                continue

            raw_body = current_text[self._wrapper_body_start :]
            calls, _c, non_tool = self._parse_streaming_json(raw_body)
            if (
                non_tool
                and not self._current_wrapper_emitted
                and self._active_wrapper in ("THINK_JSON", "[TOOL_CALLS]")
            ):
                emit_to = (
                    self._wrapper_body_start + 1
                    if self._active_wrapper == "THINK_JSON"
                    else self._wrapper_body_start
                )
                content_parts.append(current_text[self._cursor : emit_to])
                self._cursor = emit_to
                self._active_wrapper = None
                self._buffering_tool_call = False
                continue
            if calls:
                combined = [*self._closed_wrapper_calls, *calls]
                deltas = self._emit_calls_delta(combined, complete=False)
                if deltas:
                    self._current_wrapper_emitted = True
                tool_deltas.extend(deltas)
            self._buffering_tool_call = True
            break

        content = "".join(content_parts)
        if content or tool_deltas:
            return DeltaMessage(content=content or None, tool_calls=tool_deltas or None)
        return None
