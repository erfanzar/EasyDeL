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

"""Gemma4 tool parser.

Gemma4 uses a custom serialization format (not JSON) for tool calls::

    <|tool_call>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>

Strings are delimited by ``<|"|>`` (token 52), keys are unquoted, and
multiple tool calls are concatenated without separators.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from functools import cached_property
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
from ..utils import stable_json_stream_prefix

logger = get_logger(__name__)

# Gemma4 special tokens for tool calls
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
STRING_DELIM = '<|"|>'
_FUNC_NAME_RE = re.compile(r"^[\w\-\.]+$")
_BRACE_OR_DELIM_RE = re.compile(r'[{}]|<\|"\|>')
_BRACKET_OR_DELIM_RE = re.compile(r'[\[\]]|<\|"\|>')
_ARGS_BARE_END_RE = re.compile(r"[,}\]]")
_ARR_BARE_END_RE = re.compile(r"[,\]]")
_MAX_GEMMA4_NESTING_DEPTH = 128


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


def _parse_gemma4_value(value_str: str) -> object:
    """Parse a single Gemma4 value (after ``key:``) into a Python object.

    Recognises ``true``/``false``/``null`` literals, integer / float /
    scientific-notation numbers, and falls back to the raw string when no
    other interpretation applies.

    Args:
        value_str: The bare value substring extracted from a Gemma4 args
            block (no surrounding ``<|"|>`` string delimiters).

    Returns:
        ``True``/``False``/``None`` for boolean/null literals, ``int``/``float``
        for finite numeric literals, otherwise the trimmed source string.
    """
    value_str = value_str.strip()
    if not value_str:
        return value_str
    if value_str == "true":
        return True
    if value_str == "false":
        return False
    if value_str == "null":
        return None
    try:
        if "." in value_str or "e" in value_str or "E" in value_str:
            val = float(value_str)
            return val if math.isfinite(val) else value_str
        return int(value_str)
    except ValueError:
        pass
    return value_str


def _skip_balanced_container(
    text: str,
    pos: int,
    n: int,
    open_ch: str,
    pattern: re.Pattern[str],
) -> int:
    """Linearly skip a container body starting immediately after *open_ch*."""
    depth = 1
    i = pos
    while depth > 0:
        m = pattern.search(text, i)
        if m is None:
            return n
        tok = m.group(0)
        if tok == STRING_DELIM:
            next_delim = text.find(STRING_DELIM, m.end())
            if next_delim == -1:
                return n
            i = next_delim + len(STRING_DELIM)
        elif tok == open_ch:
            depth += 1
            i = m.end()
        else:
            depth -= 1
            i = m.end()
    return i


def _parse_gemma4_object_at(
    args_str: str,
    pos: int,
    n: int,
    *,
    partial: bool = False,
    depth: int = 0,
) -> tuple[dict, int]:
    """Parse a Gemma4 object body in a single linear pass starting at *pos*."""
    result: dict = {}
    i = pos

    while i < n:
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break
        if args_str[i] == "}":
            if depth > 0:
                return result, i + 1
            i += 1
            continue

        # Parse key (unquoted, ends at ':' or closing '}' of nested object)
        colon_idx = args_str.find(":", i)
        if depth > 0:
            close_idx = args_str.find("}", i)
            if close_idx != -1 and (colon_idx == -1 or close_idx < colon_idx):
                return result, close_idx + 1
        if colon_idx == -1:
            return result, n
        key = args_str[i:colon_idx].strip()
        i = colon_idx + 1

        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            if not partial and depth == 0:
                result[key] = ""
            return result, n
        if depth > 0 and args_str[i] == "}":
            result[key] = ""
            return result, i + 1

        # String value: <|"|>...<|"|>
        if args_str[i] == "<" and args_str.startswith(STRING_DELIM, i):
            i += len(STRING_DELIM)
            val_start = i
            end_pos = args_str.find(STRING_DELIM, i)
            if end_pos == -1:
                result[key] = args_str[val_start:]
                return result, n
            result[key] = args_str[val_start:end_pos]
            i = end_pos + len(STRING_DELIM)

        # Nested object: {...}
        elif args_str[i] == "{":
            if depth >= _MAX_GEMMA4_NESTING_DEPTH:
                i = _skip_balanced_container(args_str, i + 1, n, "{", _BRACE_OR_DELIM_RE)
                result[key] = {}
            else:
                result[key], i = _parse_gemma4_object_at(args_str, i + 1, n, partial=partial, depth=depth + 1)

        # Array: [...]
        elif args_str[i] == "[":
            if depth >= _MAX_GEMMA4_NESTING_DEPTH:
                i = _skip_balanced_container(args_str, i + 1, n, "[", _BRACKET_OR_DELIM_RE)
                result[key] = []
            else:
                result[key], i = _parse_gemma4_array_at(args_str, i + 1, n, partial=partial, depth=depth + 1)

        # Bare value (number, boolean, etc.)
        else:
            val_start = i
            m_end = _ARGS_BARE_END_RE.search(args_str, i)
            if m_end is None:
                if partial or depth > 0:
                    return result, n
                result[key] = _parse_gemma4_value(args_str[val_start:n])
                return result, n
            if m_end.start() == i:
                if depth > 0 and args_str[i] == "}":
                    return result, i + 1
                i += 1
                continue
            i = m_end.start()
            result[key] = _parse_gemma4_value(args_str[val_start:i])

    return result, i


def _parse_gemma4_array_at(
    arr_str: str,
    pos: int,
    n: int,
    *,
    partial: bool = False,
    depth: int = 0,
) -> tuple[list, int]:
    """Parse a Gemma4 array body in a single linear pass starting at *pos*."""
    items: list = []
    i = pos

    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break
        if arr_str[i] == "]":
            return items, i + 1

        if arr_str[i] == "<" and arr_str.startswith(STRING_DELIM, i):
            i += len(STRING_DELIM)
            end_pos = arr_str.find(STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                return items, n
            items.append(arr_str[i:end_pos])
            i = end_pos + len(STRING_DELIM)

        elif arr_str[i] == "{":
            if depth >= _MAX_GEMMA4_NESTING_DEPTH:
                i = _skip_balanced_container(arr_str, i + 1, n, "{", _BRACE_OR_DELIM_RE)
                items.append({})
            else:
                obj, i = _parse_gemma4_object_at(arr_str, i + 1, n, partial=partial, depth=depth + 1)
                items.append(obj)

        elif arr_str[i] == "[":
            if depth >= _MAX_GEMMA4_NESTING_DEPTH:
                i = _skip_balanced_container(arr_str, i + 1, n, "[", _BRACKET_OR_DELIM_RE)
                items.append([])
            else:
                sub, i = _parse_gemma4_array_at(arr_str, i + 1, n, partial=partial, depth=depth + 1)
                items.append(sub)

        else:
            val_start = i
            m_end = _ARR_BARE_END_RE.search(arr_str, i)
            if m_end is None:
                if partial or depth > 0:
                    return items, n
                items.append(_parse_gemma4_value(arr_str[val_start:n]))
                return items, n
            if m_end.start() == i:
                i += 1
                continue
            i = m_end.start()
            items.append(_parse_gemma4_value(arr_str[val_start:i]))

    return items, i


def _parse_gemma4_args(args_str: str, *, partial: bool = False) -> dict:
    """Parse Gemma4's custom key:value format into a Python dict.

    Format examples::

        location:<|"|>Tokyo<|"|>
        location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>
        count:42,flag:true
        nested:{inner_key:<|"|>val<|"|>}
        items:[<|"|>a<|"|>,<|"|>b<|"|>]

    Args:
        args_str: The raw Gemma4 argument string.
        partial: When True (streaming), bare values at end of string are
            omitted because they may be incomplete and type-unstable.

    Returns a dict ready for ``json.dumps()``.
    """
    if not args_str or not args_str.strip():
        return {}
    parsed, _ = _parse_gemma4_object_at(args_str, 0, len(args_str), partial=partial, depth=0)
    return parsed


def _parse_gemma4_array(arr_str: str, *, partial: bool = False) -> list:
    """Parse a Gemma4 array content string into a Python list.

    Counterpart to :func:`_parse_gemma4_args` for ``[...]`` bodies. Supports
    string elements delimited by ``<|"|>``, nested objects and arrays, and
    bare scalar values.

    Args:
        arr_str: The array body without the surrounding ``[`` / ``]``.
        partial: When ``True``, the trailing scalar item is dropped because
            it may still be growing in a streaming context.

    Returns:
        The list of decoded Python values.
    """
    if not arr_str:
        return []
    parsed, _ = _parse_gemma4_array_at(arr_str, 0, len(arr_str), partial=partial, depth=0)
    return parsed


@ToolParserManager.register_module("gemma4")  # pyright: ignore[reportUntypedClassDecorator]
class Gemma4ToolParser(ToolParser):
    """Parse Gemma4 ``<|tool_call>call:...<tool_call|>`` tool invocations.

    Supports function names with letters, digits, underscores, hyphens,
    and dots (e.g. ``get-weather``, ``module.func``).

    Streaming uses accumulate-then-parse-then-diff with trailing-character
    withholding to avoid emitting JSON fragments that become invalid as
    more tokens arrive.
    """

    tool_call_start_token = TOOL_CALL_START
    tool_call_end_token = TOOL_CALL_END

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize streaming state and compile the tool-call regex.

        Args:
            tokenizer: HuggingFace tokenizer used to resolve the
                ``<|tool_call>`` / ``<tool_call|>`` token IDs and decode
                streaming tokens.
        """
        super().__init__(tokenizer)

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
        self._in_open_call: bool = False
        self._scan_pos: int = 0
        self._closed_emitted_idx: int = 0

        start_tok = re.escape(TOOL_CALL_START)
        end_tok = re.escape(TOOL_CALL_END)
        self.tool_call_regex = re.compile(
            rf"{start_tok}call:([\w\-\.]+)\{{(.*?)\}}{end_tok}",
            re.DOTALL,
        )

        self.buffered_delta_text = ""

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Report whether a ``<|tool_call>`` block or partial start marker is buffered."""
        del delta_text
        if self._streaming_initialized:
            return self._in_open_call or self._pending_tail_start is not None
        return _has_open_marker_or_trailing_prefix(
            current_text,
            self.tool_call_start_token,
            self.tool_call_end_token,
        )

    def _reset_streaming_state(self, emitted_pos: int = 0) -> None:
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []
        self._tool_name_sent = []
        self._tool_call_ids = []
        self._emitted_content_pos = emitted_pos
        self._pending_tail_start = None
        self._last_current_text = ""
        self.buffered_delta_text = ""
        self._streaming_initialized = False
        self._in_open_call = False
        self._scan_pos = 0
        self._closed_emitted_idx = 0

    def flush_streaming_content(self) -> str | None:
        """Flush any trailing partial ``<|tool_call>`` prefix withheld at EOS."""
        flushed: str | None = None
        if self._pending_tail_start is not None and self._pending_tail_start < len(self._last_current_text):
            flushed = self._last_current_text[self._pending_tail_start :] or None
            self._emitted_content_pos = len(self._last_current_text)
            self._pending_tail_start = None
        self._last_current_text = ""
        return flushed

    @cached_property
    def tool_call_start_token_ids(self) -> list[int]:
        """Encoded opening marker, resolved only when a consumer requests token IDs."""
        return self.model_tokenizer.encode(TOOL_CALL_START, add_special_tokens=False) if self.model_tokenizer else []

    @cached_property
    def tool_call_end_token_ids(self) -> list[int]:
        """Encoded closing marker, resolved only when a consumer requests token IDs."""
        return self.model_tokenizer.encode(TOOL_CALL_END, add_special_tokens=False) if self.model_tokenizer else []

    def _buffer_delta_text(self, delta_text: str) -> str:
        """Withhold partial tool-call markers that may span multiple deltas.

        If the trailing characters of ``delta_text`` form a prefix of either
        the start or end marker, those characters are buffered until enough
        text arrives to disambiguate them.

        Args:
            delta_text: New text delta from the engine.

        Returns:
            The portion of the buffered text that is safe to emit. Buffered
            characters are stored in ``self.buffered_delta_text`` for the
            next call.
        """
        combined = self.buffered_delta_text + delta_text

        if combined.endswith(TOOL_CALL_START) or combined.endswith(TOOL_CALL_END):
            self.buffered_delta_text = ""
            return combined

        for tag in (TOOL_CALL_START, TOOL_CALL_END):
            for i in range(1, len(tag)):
                if combined.endswith(tag[:i]):
                    self.buffered_delta_text = combined[-i:]
                    return combined[:-i]

        self.buffered_delta_text = ""
        return combined

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Disable special-token stripping when tools are enabled.

        Gemma4's tool-call markers are special tokens; the request must
        keep them visible so the parser can find them.

        Args:
            request: Chat completion request to adjust.

        Returns:
            The same request, possibly with ``skip_special_tokens=False``.
        """
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract Gemma4 tool calls from a complete model response.

        Args:
            model_output: Full text produced by the model.
            request: Original chat completion request.

        Returns:
            An :class:`ExtractedToolCallInformation` describing whether tools
            were called, the parsed tool calls, and any leading content text.
        """
        if TOOL_CALL_START not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            tool_calls: list[ToolCall] = []
            start_len = len(TOOL_CALL_START)
            end_len = len(TOOL_CALL_END)
            pos = 0
            n = len(model_output)
            cached_end = -1
            while pos < n:
                call_start = model_output.find(TOOL_CALL_START, pos)
                if call_start == -1:
                    break
                body_start = call_start + start_len
                if cached_end < body_start:
                    cached_end = model_output.find(TOOL_CALL_END, body_start)
                if cached_end == -1:
                    break
                body = model_output[body_start:cached_end]
                func_name, args_str, args_closed = self._split_call_body(body)
                if func_name is not None and args_closed and body.endswith("}"):
                    arguments = _parse_gemma4_args(args_str)
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=func_name,
                                arguments=json.dumps(arguments, ensure_ascii=False),
                            ),
                        )
                    )
                    pos = cached_end + end_len
                else:
                    next_start = model_output.find(TOOL_CALL_START, body_start, cached_end)
                    pos = next_start if next_start != -1 else cached_end + end_len

            if not tool_calls:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            content_end = model_output.find(TOOL_CALL_START)
            content = model_output[:content_end].strip() if content_end > 0 else None
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error extracting tool calls from Gemma4 response")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    @staticmethod
    def _split_call_body(body: str) -> tuple[str | None, str, bool]:
        """Parse ``call:name{args[}]`` into ``(func_name, args_body, args_closed)``.

        Tracks brace depth outside ``<|"|>`` string literals so a trailing ``}``
        is only stripped when it closes the outer call argument object (depth 0),
        never when it closes a nested object or lives inside a string.
        """
        if not body.startswith("call:"):
            return None, "", False
        func_part = body[5:]
        brace_pos = func_part.find("{")
        if brace_pos == -1:
            return None, "", False
        func_name = func_part[:brace_pos].strip()
        if not func_name or _FUNC_NAME_RE.fullmatch(func_name) is None:
            return None, "", False

        args_raw = func_part[brace_pos + 1 :]
        depth = 1
        i = 0
        while depth > 0:
            m = _BRACE_OR_DELIM_RE.search(args_raw, i)
            if m is None:
                return func_name, args_raw, False
            tok = m.group(0)
            if tok == STRING_DELIM:
                end_delim = args_raw.find(STRING_DELIM, m.end())
                if end_delim == -1:
                    return func_name, args_raw, False
                i = end_delim + len(STRING_DELIM)
            elif tok == "{":
                depth += 1
                i = m.end()
            else:
                depth -= 1
                if depth == 0:
                    return func_name, args_raw[: m.start()], True
                i = m.end()
        return func_name, args_raw, False

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
        """Stream Gemma4 tool-call deltas from incremental engine output.

        Args:
            previous_text: Cumulative output up to the previous step.
            current_text: Cumulative output including this step.
            delta_text: Newly produced text since ``previous_text``.
            previous_token_ids: Token IDs corresponding to ``previous_text``.
            current_token_ids: Token IDs corresponding to ``current_text``.
            delta_token_ids: Token IDs corresponding to ``delta_text``.
            request: Original chat completion request.

        Returns:
            A :class:`DeltaMessage` describing the next chunk of visible
            content or tool-call deltas, or ``None`` when nothing should be
            emitted in this step.
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
        self._in_open_call = False

        try:
            return self._extract_streaming(previous_text, current_text, delta_text)
        except Exception:
            logger.exception("Error in Gemma4 streaming tool call extraction")
            return None

    def _extract_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Drive span-based streaming extraction for Gemma4 tool calls.

        Partitions cumulative ``current_text`` into visible content segments and
        ``<|tool_call>...<tool_call|>`` call spans, flushing closing calls before
        opening subsequent calls and never rewinding ``streamed_args_for_tool``.
        """
        del previous_text, delta_text
        pos = self._scan_pos
        next_scan_pos = pos
        spans: list[tuple[str, bool]] = []
        visible_parts: list[tuple[int, int]] = []
        while pos < len(current_text):
            call_start = current_text.find(TOOL_CALL_START, pos)
            if call_start == -1:
                tail = current_text[pos:]
                safe_tail = _strip_trailing_partial_marker(tail, TOOL_CALL_START)
                safe_end = pos + len(safe_tail)
                if safe_tail:
                    visible_parts.append((pos, safe_end))
                if len(safe_tail) < len(tail):
                    self._pending_tail_start = safe_end
                next_scan_pos = safe_end
                break
            if call_start > pos:
                visible_parts.append((pos, call_start))
            body_start = call_start + len(TOOL_CALL_START)
            call_end = current_text.find(TOOL_CALL_END, body_start)
            if call_end == -1:
                spans.append((current_text[body_start:], False))
                self._in_open_call = True
                next_scan_pos = call_start
                break
            spans.append((current_text[body_start:call_end], True))
            pos = call_end + len(TOOL_CALL_END)
            next_scan_pos = pos

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
        for body, is_closed in spans:
            clean_body = body if is_closed else _strip_trailing_partial_marker(body, TOOL_CALL_END)
            func_name, args_part, args_closed = self._split_call_body(clean_body)
            if func_name is None:
                continue

            idx = emitted_idx
            emitted_idx += 1
            if is_closed:
                next_closed_emitted_idx = emitted_idx
            while len(self.prev_tool_call_arr) <= idx:
                self.prev_tool_call_arr.append({})
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
                if idx == self.current_tool_id:
                    self.current_tool_name_sent = True

            prev_streamed = self.streamed_args_for_tool[idx]
            diff = ""

            if is_closed or args_closed:
                final_args = _parse_gemma4_args(args_part, partial=False)
                final_args_json = json.dumps(final_args, ensure_ascii=False)
                self.prev_tool_call_arr[idx] = {"name": func_name, "arguments": final_args}
                if final_args_json.startswith(prev_streamed):
                    diff = final_args_json[len(prev_streamed) :]
                    if diff:
                        self.streamed_args_for_tool[idx] = final_args_json
            else:
                self.prev_tool_call_arr[idx].setdefault("name", func_name)
                if args_part:
                    try:
                        current_args = _parse_gemma4_args(args_part, partial=True)
                    except Exception:
                        current_args = {}
                    if current_args:
                        current_args_json = json.dumps(current_args, ensure_ascii=False)
                        safe_json = stable_json_stream_prefix(current_args_json)
                        while safe_json and safe_json[-1] in ("}", '"', "]", "<", "|", "\\", ">"):
                            safe_json = safe_json[:-1]
                        # Never rewind already-emitted streamed_args_for_tool; only extend on prefix match.
                        if safe_json and safe_json.startswith(prev_streamed) and len(safe_json) > len(prev_streamed):
                            diff = safe_json[len(prev_streamed) :]
                            self.streamed_args_for_tool[idx] = safe_json
                            self.prev_tool_call_arr[idx] = {"name": func_name, "arguments": current_args}

            if first or diff:
                deltas.append(
                    DeltaToolCall(
                        index=idx,
                        type="function" if first else None,
                        id=self._tool_call_ids[idx] if first else None,
                        function=DeltaFunctionCall(
                            name=func_name if first else None,
                            arguments=diff if diff else ("" if first else None),
                        ).model_dump(exclude_none=True),
                    )
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

    def _extract_partial_call(self, current_text: str) -> tuple[str | None, str]:
        """Return the function name and partial argument body of the active call.

        Args:
            current_text: Cumulative output text that may contain the
                in-progress tool call.

        Returns:
            Tuple ``(func_name, args_part)`` where ``func_name`` is ``None``
            when no usable call is detected, and ``args_part`` is the body
            of the partial argument block (without enclosing outer braces).
        """
        last_start = current_text.rfind(TOOL_CALL_START)
        if last_start == -1:
            return None, ""

        partial_call = current_text[last_start + len(TOOL_CALL_START) :]
        if TOOL_CALL_END in partial_call:
            partial_call = partial_call.split(TOOL_CALL_END)[0]

        func_name, args_part, _ = self._split_call_body(partial_call)
        return func_name, args_part

    def _handle_tool_call_middle(self, current_text: str) -> DeltaMessage | None:
        """Emit the streaming delta for an in-progress tool call body.

        Args:
            current_text: Cumulative output containing the open tool call.

        Returns:
            A :class:`DeltaMessage` carrying either the function name (on
            first sight) or an argument-string diff, or ``None`` when no
            update is appropriate yet.
        """
        func_name, args_part = self._extract_partial_call(current_text)
        if func_name is None:
            return None

        if not self.current_tool_name_sent and func_name:
            self.current_tool_name_sent = True
            self.prev_tool_call_arr[self.current_tool_id] = {"name": func_name, "arguments": {}}
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function",
                        id=f"chatcmpl-tool-{uuid4()}",
                        function=DeltaFunctionCall(name=func_name, arguments="").model_dump(exclude_none=True),
                    )
                ]
            )

        if self.current_tool_name_sent and args_part:
            return self._emit_argument_diff(args_part)

        return None

    def _handle_tool_call_end(self, current_text: str) -> DeltaMessage | None:
        """Emit the trailing argument diff when a tool call closes.

        Args:
            current_text: Cumulative output containing the now-closed call.

        Returns:
            A :class:`DeltaMessage` with the final argument diff, or ``None``
            when no diff needs to be sent (e.g. arguments already streamed
            in full).
        """
        if self.current_tool_id < 0 or self.current_tool_id >= len(self.prev_tool_call_arr):
            return None

        all_matches = self.tool_call_regex.findall(current_text)
        if self.current_tool_id < len(all_matches):
            func_name, args_str = all_matches[self.current_tool_id]
            final_args = _parse_gemma4_args(args_str)
            final_args_json = json.dumps(final_args, ensure_ascii=False)
            if not self.current_tool_name_sent:
                self.current_tool_name_sent = True
                self.prev_tool_call_arr[self.current_tool_id] = {"name": func_name, "arguments": final_args}
                self.streamed_args_for_tool[self.current_tool_id] = final_args_json
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            type="function",
                            id=f"chatcmpl-tool-{uuid4()}",
                            function=DeltaFunctionCall(name=func_name, arguments=final_args_json),
                        )
                    ]
                )

            prev_streamed = self.streamed_args_for_tool[self.current_tool_id]
            if final_args_json.startswith(prev_streamed) and len(final_args_json) > len(prev_streamed):
                diff = final_args_json[len(prev_streamed) :]
                self.streamed_args_for_tool[self.current_tool_id] = final_args_json
                self.prev_tool_call_arr[self.current_tool_id]["arguments"] = final_args
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(arguments=diff).model_dump(exclude_none=True),
                        )
                    ]
                )
        return None

    def _emit_argument_diff(self, raw_args_str: str) -> DeltaMessage | None:
        """Parse raw args, convert to JSON, withhold trailing closers, diff and emit.

        Parses the partial Gemma4 argument body, serialises it as JSON,
        strips trailing characters that may still shift as more tokens
        arrive (e.g. ``}``, ``"``, ``]``, fragments of ``<|"|>``), and emits
        only the diff against the previously streamed JSON for the current
        tool.

        Args:
            raw_args_str: The current partial Gemma4 argument body
                (without enclosing braces).

        Returns:
            A :class:`DeltaMessage` carrying the new argument-string diff,
            or ``None`` when there is nothing safe to emit yet (e.g. the
            parser detected that the structure changed and is waiting for
            the final flush).
        """
        try:
            current_args = _parse_gemma4_args(raw_args_str, partial=True)
        except Exception:
            return None

        if not current_args:
            return None

        current_args_json = json.dumps(current_args, ensure_ascii=False)

        # Withhold trailing closing characters that may shift as more
        # tokens arrive (e.g. a string value is still being streamed).
        safe_json = stable_json_stream_prefix(current_args_json)
        while safe_json and safe_json[-1] in ("}", '"', "]", "<", "|", "\\", ">"):
            safe_json = safe_json[:-1]

        prev_streamed = self.streamed_args_for_tool[self.current_tool_id]

        if not safe_json or safe_json == prev_streamed:
            return None

        if prev_streamed:
            if not safe_json.startswith(prev_streamed):
                # Structure changed — never rewind already-emitted text.
                return None

            diff = safe_json[len(prev_streamed) :]
        else:
            diff = safe_json

        if diff:
            self.streamed_args_for_tool[self.current_tool_id] = safe_json
            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = current_args
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        function=DeltaFunctionCall(arguments=diff).model_dump(exclude_none=True),
                    )
                ]
            )
        return None
