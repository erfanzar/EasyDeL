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
model outputs that follow OpenAI-style JSON formatting conventions, including
raw JSON objects, JSON arrays, nested ``tool_calls`` envelopes, and code-fenced
JSON blocks.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

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
from ..utils import is_complete_json, partial_json_loads
from .utils import stream_json_calls

logger = get_logger(__name__)


def _trailing_backticks_len(text: str) -> int:
    """Return length (0..2) of trailing partial triple-backtick prefix at the end of *text*."""
    if text.endswith("``"):
        return 2
    if text.endswith("`"):
        return 1
    return 0


def _is_complete_json_safe(text: str) -> bool:
    """Return ``is_complete_json(text)``, treating ``RecursionError`` as incomplete."""
    try:
        return is_complete_json(text)
    except RecursionError:
        return False


def _find_fence_close_outside_json_strings(text: str, start: int) -> int:
    """Find closing `` ``` `` outside JSON double-quoted strings starting at *start*."""
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
        if text.startswith("```", i):
            return i
        i += 1
    return -1


def _strip_trailing_backticks_outside_strings(text: str) -> str:
    """Strip 1-2 trailing partial closing backticks when outside JSON strings."""
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
    hold = _trailing_backticks_len(text)
    return text[:-hold] if hold else text


def _normalize_single_call(
    item: object,
    *,
    partial_item: object | None = None,
    allow_missing_args: bool = False,
) -> dict | None:
    """Normalize one dict into ``{"name": ..., "arguments": ...}`` if it looks like a tool call."""
    if not isinstance(item, dict):
        return None
    pdict = partial_item if isinstance(partial_item, dict) else item

    fn = item.get("function")
    if isinstance(fn, dict):
        pfn = pdict.get("function") if isinstance(pdict.get("function"), dict) else fn
        has_args = "arguments" in pfn or "parameters" in pfn
        is_fn_type = item.get("type") == "function" or pdict.get("type") == "function"
        if not (has_args or is_fn_type):
            return None
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            return None
        if "arguments" in pfn:
            args = pfn["arguments"]
        elif "parameters" in pfn:
            args = pfn["parameters"]
        else:
            args = {} if allow_missing_args else None
        return {"name": name, "arguments": args}

    if "name" in item:
        name = item.get("name")
        if not isinstance(name, str) or not name:
            return None
        has_args = "arguments" in pdict or "parameters" in pdict
        is_fn_type = item.get("type") == "function" or pdict.get("type") == "function"
        if not (has_args or is_fn_type):
            return None
        if "arguments" in pdict:
            args = pdict["arguments"]
        elif "parameters" in pdict:
            args = pdict["parameters"]
        else:
            args = {} if allow_missing_args else None
        return {"name": name, "arguments": args}

    return None


@ToolParserManager.register_module("openai")  # pyright: ignore[reportUntypedClassDecorator]
class OpenAIToolParser(ToolParser):
    """Best-effort OpenAI-style tool call parser for local model outputs.

    Supported Formats:
        - JSON list of calls: ``[{"name": ..., "arguments": ...}, ...]``
        - JSON list of OpenAI function items: ``[{"type": "function", "function": {...}}, ...]``
        - Single JSON object: ``{"name": ..., "arguments": ...}``
        - Nested tool_calls: ``{"tool_calls": [{"function": {"name": ..., "arguments": ...}}, ...]}``
        - Code-fenced JSON: `` ```json ... ``` ``
    """

    _json_block_re = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.MULTILINE)

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the OpenAI tool parser.

        Args:
            tokenizer: The tokenizer instance used for encoding/decoding tokens.
        """
        super().__init__(tokenizer)
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state for a fresh response."""
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
        self._stream_tool_ids: list[str] = []
        self._streaming_initialized = False
        self._cursor = 0
        self._accumulated_text = ""
        self._buffering_tool_call = False
        self._emitted_tool_call = False
        self._fence_kind: str | None = None
        self._fence_tag = ""
        self._fence_start = -1
        self._fence_body_start = -1
        self._closed_fence_calls: list[dict] = []
        self._current_fence_emitted = False
        self._bare_json_decided_plain = False
        self._bare_json_completed = False

    def get_tool_machinery_hints(self) -> Sequence[str]:
        """Engage streaming tool machinery on ``[``, ``{``, and fenced `` ``` `` blocks."""
        hints = list(super().get_tool_machinery_hints())
        for marker in ("[", "{", "```"):
            if marker not in hints:
                hints.append(marker)
        return tuple(hints)

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Return True only while buffering a potential JSON/fenced tool call, never on prose ``[``/``{``."""
        del delta_text
        if self._streaming_initialized:
            return self._buffering_tool_call
        stripped = current_text.lstrip()
        if stripped.startswith(("{", "[")):
            return not _is_complete_json_safe(stripped)
        fence_count = current_text.count("```")
        if fence_count % 2 == 1:
            return True
        return _trailing_backticks_len(current_text) > 0

    def flush_streaming_content(self) -> str:
        """Release withheld non-tool text at end-of-stream."""
        if (not self._bare_json_decided_plain and self._emitted_tool_call) or (
            self._fence_kind == "json" and self._current_fence_emitted
        ):
            self._accumulated_text = ""
            self._cursor = 0
            return ""
        pending = self._accumulated_text[self._cursor :]
        self._accumulated_text = ""
        self._cursor = 0
        self._buffering_tool_call = False
        self._fence_kind = None
        return pending

    @staticmethod
    def _extract_json_candidate(text: str) -> tuple[str | None, str | None]:
        """Extract JSON content from model output text."""
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
    def _normalize_tool_call_objects(
        obj: object,
        *,
        partial_obj: object | None = None,
        allow_missing_args: bool = False,
    ) -> list[dict]:
        """Normalize supported OpenAI JSON structures into ``[{"name": ..., "arguments": ...}]``."""
        if isinstance(obj, dict):
            if isinstance(obj.get("tool_calls"), list):
                p_list = (
                    partial_obj.get("tool_calls")
                    if isinstance(partial_obj, dict) and isinstance(partial_obj.get("tool_calls"), list)
                    else None
                )
                tool_calls: list[dict] = []
                for idx, item in enumerate(obj["tool_calls"]):
                    p_item = p_list[idx] if p_list is not None and idx < len(p_list) else None
                    norm = _normalize_single_call(item, partial_item=p_item, allow_missing_args=allow_missing_args)
                    if norm is not None:
                        tool_calls.append(norm)
                return tool_calls
            norm = _normalize_single_call(obj, partial_item=partial_obj, allow_missing_args=allow_missing_args)
            return [norm] if norm is not None else []
        if isinstance(obj, list):
            p_list = partial_obj if isinstance(partial_obj, list) else None
            tool_calls = []
            for idx, item in enumerate(obj):
                p_item = p_list[idx] if p_list is not None and idx < len(p_list) else None
                norm = _normalize_single_call(item, partial_item=p_item, allow_missing_args=allow_missing_args)
                if norm is not None:
                    tool_calls.append(norm)
            return tool_calls
        return []

    @staticmethod
    def _parse_json_candidate_streaming(json_text: str) -> tuple[list[dict], bool, bool]:
        """Parse streaming JSON text.

        Returns ``(tool_call_arr, is_complete, definitely_non_tool)``.
        """
        stripped = json_text.strip()
        if not stripped:
            return [], False, False
        if not stripped.startswith(("{", "[")):
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

        calls = OpenAIToolParser._normalize_tool_call_objects(
            safe_obj,
            partial_obj=partial_obj,
            allow_missing_args=complete,
        )
        if calls:
            return calls, complete, False

        if complete:
            return [], True, True

        # Early rejection for non-dict array elements (e.g. "[1, 2, 3]").
        if isinstance(partial_obj, list) and any(not isinstance(elt, dict) for elt in partial_obj):
            return [], False, True

        return [], False, False

    @classmethod
    def _build_batch_tool_calls(cls, parsed: object) -> list[ToolCall]:
        """Convert normalized OpenAI JSON structures into :class:`ToolCall` items."""
        calls = cls._normalize_tool_call_objects(parsed, allow_missing_args=True)
        tool_calls: list[ToolCall] = []
        for call in calls:
            name = call.get("name")
            if not isinstance(name, str) or not name:
                continue
            args = call.get("arguments")
            if args is None:
                args = {}
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            tool_calls.append(ToolCall(type="function", function=FunctionCall(name=name, arguments=args)))
        return tool_calls

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model response."""
        del request
        stripped = model_output.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, ValueError, RecursionError):
                parsed = None
            if parsed is not None:
                try:
                    tool_calls = self._build_batch_tool_calls(parsed)
                except (TypeError, ValueError, RecursionError):
                    tool_calls = []
                if tool_calls:
                    return ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls, content=None)

        # Scan all fenced code blocks while preserving non-JSON and non-tool fences.
        all_tool_calls: list[ToolCall] = []
        content_parts: list[str] = []
        pos = 0
        n = len(model_output)
        while pos < n:
            start_idx = model_output.find("```", pos)
            if start_idx == -1:
                content_parts.append(model_output[pos:])
                break
            after_ticks = start_idx + 3
            nl = model_output.find("\n", after_ticks)
            brace = min(
                (i for i in (model_output.find("{", after_ticks), model_output.find("[", after_ticks)) if i != -1),
                default=-1,
            )
            inline_close = model_output.find("```", after_ticks)
            terminators = [i for i in (nl, brace, inline_close) if i != -1]
            if not terminators:
                content_parts.append(model_output[pos:])
                break
            earliest = min(terminators)
            if earliest == inline_close:
                block_end = inline_close + 3
                content_parts.append(model_output[pos:block_end])
                pos = block_end
                continue

            lang = model_output[after_ticks:earliest].strip().lower()
            body_start = earliest + 1 if model_output[earliest] == "\n" else earliest
            if lang in ("", "json"):
                close_idx = _find_fence_close_outside_json_strings(model_output, body_start)
            else:
                close_idx = model_output.find("```", body_start)
            if close_idx == -1:
                content_parts.append(model_output[pos:])
                break

            block_end = close_idx + 3
            block_calls: list[ToolCall] = []
            if lang in ("", "json"):
                try:
                    parsed = json.loads(model_output[body_start:close_idx].strip())
                    block_calls = self._build_batch_tool_calls(parsed)
                except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
                    block_calls = []

            if block_calls:
                content_parts.append(model_output[pos:start_idx])
                all_tool_calls.extend(block_calls)
            else:
                content_parts.append(model_output[pos:block_end])
            pos = block_end

        if all_tool_calls:
            content = "".join(content_parts).strip() or None
            return ExtractedToolCallInformation(tools_called=True, tool_calls=all_tool_calls, content=content)

        return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    def _stream_calls(self, calls: list[dict], *, complete: bool) -> list[DeltaToolCall]:
        """Run ``stream_json_calls`` and update public parser tracking attributes."""
        deltas = stream_json_calls(
            calls,
            self.prev_tool_call_arr,
            self.streamed_args_for_tool,
            self._stream_tool_ids,
            complete,
        )
        if deltas:
            self._emitted_tool_call = True
        self.current_tool_id = len(calls) - 1
        self.current_tool_name_sent = bool(self._stream_tool_ids and self._stream_tool_ids[-1])
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
        """Extract tool calls incrementally during streaming generation."""
        del delta_text, previous_token_ids, current_token_ids, delta_token_ids, request
        if not previous_text:
            self._reset_streaming_state()
            self._streaming_initialized = True
        elif not self._streaming_initialized:
            self._reset_streaming_state()
            self._cursor = len(previous_text)
            if previous_text.strip():
                self._bare_json_decided_plain = True
            self._streaming_initialized = True

        self._accumulated_text = current_text

        # Path 1: Bare JSON starting at position 0 (with optional leading whitespace).
        if not self._bare_json_decided_plain and self._fence_kind is None:
            stripped = current_text.lstrip()
            if not stripped:
                self._buffering_tool_call = True
                return None
            if stripped.startswith(("{", "[")):
                if self._bare_json_completed:
                    # Any trailing text after a completed bare JSON call block.
                    self._buffering_tool_call = False
                    return None
                calls, complete, non_tool = self._parse_json_candidate_streaming(stripped)
                if non_tool and not self._emitted_tool_call:
                    self._bare_json_decided_plain = True
                    self._buffering_tool_call = False
                else:
                    if calls:
                        deltas = self._stream_calls(calls, complete=complete)
                        if complete:
                            self._bare_json_completed = True
                            self._cursor = len(current_text)
                        self._buffering_tool_call = not complete
                        return DeltaMessage(tool_calls=deltas) if deltas else None
                    self._buffering_tool_call = not complete
                    return None
            else:
                self._bare_json_decided_plain = True

        # Path 2: Prose and/or code-fenced ```json ... ``` blocks.
        content_parts: list[str] = []
        tool_deltas: list[DeltaToolCall] = []

        while True:
            if self._fence_kind is None:
                fence_idx = current_text.find("```", self._cursor)
                if fence_idx != -1:
                    if fence_idx > self._cursor:
                        content_parts.append(current_text[self._cursor : fence_idx])
                    self._cursor = fence_idx
                    self._fence_kind = "candidate"
                    self._fence_start = fence_idx
                    self._fence_body_start = -1
                    self._current_fence_emitted = False
                    self._buffering_tool_call = True
                    continue

                rem = current_text[self._cursor :]
                hold = _trailing_backticks_len(rem)
                safe_end = len(current_text) - hold
                if safe_end > self._cursor:
                    content_parts.append(current_text[self._cursor : safe_end])
                    self._cursor = safe_end
                self._buffering_tool_call = hold > 0
                break

            if self._fence_kind == "plain":
                close_idx = current_text.find("```", self._cursor)
                if close_idx != -1:
                    fence_end = close_idx + 3
                    content_parts.append(current_text[self._cursor : fence_end])
                    self._cursor = fence_end
                    self._fence_kind = None
                    self._fence_start = -1
                    self._fence_body_start = -1
                    self._buffering_tool_call = False
                    continue

                rem = current_text[self._cursor :]
                hold = _trailing_backticks_len(rem)
                safe_end = len(current_text) - hold
                if safe_end > self._cursor:
                    content_parts.append(current_text[self._cursor : safe_end])
                    self._cursor = safe_end
                self._buffering_tool_call = hold > 0
                break

            if self._fence_kind == "candidate":
                after_ticks = current_text[self._fence_start + 3 :]
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
                    # Definitely not a JSON fence (e.g. ```python): keep tracking until matching closing ```.
                    content_parts.append(current_text[self._cursor : self._fence_start + 3])
                    self._cursor = self._fence_start + 3
                    self._fence_kind = "plain"
                    self._buffering_tool_call = False
                    continue

                earliest = min(terminators)
                if earliest == inline_close:
                    fence_end = self._fence_start + 3 + inline_close + 3
                    content_parts.append(current_text[self._cursor : fence_end])
                    self._cursor = fence_end
                    self._fence_kind = None
                    self._fence_start = -1
                    self._fence_body_start = -1
                    self._buffering_tool_call = False
                    continue

                tag = after_ticks[:earliest].strip().lower()
                if tag not in ("", "json"):
                    content_parts.append(current_text[self._cursor : self._fence_start + 3])
                    self._cursor = self._fence_start + 3
                    self._fence_kind = "plain"
                    self._buffering_tool_call = False
                    continue

                offset = earliest + 1 if after_ticks[earliest] == "\n" else earliest
                self._fence_body_start = self._fence_start + 3 + offset
                self._fence_tag = tag
                self._fence_kind = "json"
                continue

            # self._fence_kind == "json"
            close_idx = _find_fence_close_outside_json_strings(current_text, self._fence_body_start)
            if close_idx != -1:
                json_body = current_text[self._fence_body_start : close_idx]
                calls, _complete, _non_tool = self._parse_json_candidate_streaming(json_body)
                fence_end = close_idx + 3
                if calls:
                    combined = [*self._closed_fence_calls, *calls]
                    deltas = self._stream_calls(combined, complete=True)
                    tool_deltas.extend(deltas)
                    self._closed_fence_calls = combined
                    self._current_fence_emitted = True
                elif not self._current_fence_emitted:
                    # Non-tool JSON inside fence: emit entire fenced block verbatim.
                    content_parts.append(current_text[self._fence_start : fence_end])
                self._fence_kind = None
                self._fence_tag = ""
                self._fence_start = -1
                self._fence_body_start = -1
                self._current_fence_emitted = False
                self._cursor = fence_end
                self._buffering_tool_call = False
                continue

            # Fence still open: stream JSON incrementally without leaking to content.
            raw_body = _strip_trailing_backticks_outside_strings(current_text[self._fence_body_start :])
            calls, _complete, non_tool = self._parse_json_candidate_streaming(raw_body)
            if calls:
                combined = [*self._closed_fence_calls, *calls]
                deltas = self._stream_calls(combined, complete=False)
                if deltas:
                    self._current_fence_emitted = True
                tool_deltas.extend(deltas)
            elif non_tool and not self._current_fence_emitted and self._fence_tag == "":
                content_parts.append(current_text[self._cursor : self._fence_start + 3])
                self._cursor = self._fence_start + 3
                self._fence_kind = "plain"
                self._fence_tag = ""
                self._buffering_tool_call = False
                continue
            self._buffering_tool_call = True
            break

        content = "".join(content_parts)
        if content or tool_deltas:
            return DeltaMessage(content=content or None, tool_calls=tool_deltas or None)
        return None
