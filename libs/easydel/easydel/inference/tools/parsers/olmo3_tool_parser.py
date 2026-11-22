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

"""OLMo-3 Pythonic tool parser module.

This module provides the Olmo3PythonicToolParser class for parsing tool calls
in a Python function call syntax used by OLMo-3 models. The parser uses AST
parsing directly (no backtracking regexes) to extract function names and
keyword arguments across single-line, multiline, bare, and multi-block calls.

Example format parsed:
    <function_calls>
    func_name(arg1="value1", arg2=123)
    another_func(
        param="test"
    )
    </function_calls>

Or without tags:
    func_name(arg1="value1")
"""

from __future__ import annotations

import ast
import re
import textwrap
from collections.abc import Sequence

from eformer.loggings import get_logger
from transformers import AutoTokenizer as AnyTokenizer

from ...openai_api_modules import (
    ChatCompletionRequest,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    ToolCall,
)
from ..abstract_tool import ToolParser, ToolParserManager
from ._pythonic_ast import UnexpectedAstError, compute_tool_delta, handle_single_tool, make_valid_python

logger = get_logger(__name__)

_BARE_IDENT_ONLY_RE = re.compile(r"^[A-Za-z_]\w*$")
_BARE_CALL_START_RE = re.compile(r"^[A-Za-z_]\w*\s*\(")


def _trailing_prefix_len(text: str, marker: str) -> int:
    """Return length of the longest proper prefix of *marker* matching the end of *text*."""
    max_len = min(len(text), len(marker) - 1)
    for size in range(max_len, 0, -1):
        if text.endswith(marker[:size]):
            return size
    return 0


def _find_tag_outside_strings(text: str, tag: str, start: int = 0) -> int:
    """Find *tag* in *text* starting at *start*, ignoring occurrences inside Python strings."""
    quote: str | None = None
    escaped = False
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            i += 1
            continue
        if text.startswith(tag, i):
            return i
        i += 1
    return -1


def _strip_trailing_partial_close_tag(text: str, end_tag: str) -> str:
    """Strip an incomplete trailing *end_tag* prefix only if outside Python strings."""
    quote: str | None = None
    escaped = False
    for ch in text:
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
    if quote is not None:
        return text
    hold = _trailing_prefix_len(text, end_tag)
    return text[:-hold] if hold else text


def _has_balanced_brackets(text: str) -> tuple[bool, bool]:
    """Return ``(is_balanced, is_mismatched)`` ignoring Python string contents."""
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    for ch in text:
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch in {"[", "(", "{"}:
            stack.append(ch)
        elif ch in {"]", ")", "}"}:
            expected = {"]": "[", ")": "(", "}": "{"}[ch]
            if not stack or stack.pop() != expected:
                return False, True
    return (quote is None and not stack), False


def _extract_calls_from_ast(module: ast.Module, *, allow_trailing_name: bool = False) -> list[ToolCall]:
    """Extract validated ``ToolCall`` objects from a parsed Python AST module."""
    if not module.body:
        return []
    calls: list[ast.Call] = []
    for idx, stmt in enumerate(module.body):
        if not isinstance(stmt, ast.Expr):
            raise UnexpectedAstError("Tool output statements must be function call expressions")
        expr = stmt.value
        if isinstance(expr, ast.Call):
            calls.append(expr)
        elif isinstance(expr, ast.Tuple | ast.List) and all(isinstance(elt, ast.Call) for elt in expr.elts):
            calls.extend(expr.elts)
        elif allow_trailing_name and idx == len(module.body) - 1 and isinstance(expr, ast.Name):
            # Incomplete next function name before '(' arrived.
            break
        else:
            raise UnexpectedAstError("Tool output must contain only function calls")

    validated: list[ToolCall] = []
    for call in calls:
        if not isinstance(call.func, ast.Name) or call.args:
            raise UnexpectedAstError("Tool calls require a simple name and keyword-only arguments")
        names = [kw.arg for kw in call.keywords]
        if None in names or len(names) != len(set(names)):
            raise UnexpectedAstError("Tool calls cannot unpack or repeat keyword arguments")
        validated.append(handle_single_tool(call))
    return validated


def _parse_olmo3_block(body: str) -> list[ToolCall] | None:
    """Strictly parse a complete OLMo-3 pythonic call block into ``ToolCall`` objects."""
    stripped = textwrap.dedent(body).strip()
    if not stripped:
        return None
    try:
        module = ast.parse(stripped)
        calls = _extract_calls_from_ast(module, allow_trailing_name=False)
    except (SyntaxError, ValueError, RecursionError, MemoryError, UnexpectedAstError):
        return None
    return calls or None


@ToolParserManager.register_module("olmo3")  # pyright: ignore[reportUntypedClassDecorator]
class Olmo3PythonicToolParser(ToolParser):
    """Parser for OLMo-3 models that emit newline-separated pythonic tool calls.

    Uses Python's AST module directly (no backtracking regexes) to parse
    function call syntax, supporting single-line, multiline, bare, and
    multi-block ``<function_calls>...</function_calls>`` outputs in both batch
    and streaming modes.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the OLMo-3 Pythonic tool parser.

        Args:
            tokenizer: The tokenizer instance used for encoding/decoding tokens.
        """
        super().__init__(tokenizer)
        self.tool_call_start_token = "<function_calls>"
        self.tool_call_end_token = "</function_calls>"
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state for a new response."""
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool = []
        self._streaming_initialized = False
        self._cursor = 0
        self._accumulated_text = ""
        self._in_tool_block = False
        self._block_content_start = -1
        self._block_base_tool_index = 0
        self._block_local_index = 0
        self._next_tool_index = 0
        self._seen_any_block = False
        self._bare_check_done = False
        self._bare_completed = False
        self._bare_last_stripped_len = -1
        self._in_bare_call = False
        self._buffering_bare_call = False
        self._emitted_tool_call = False

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Buffer active ``<function_calls>`` blocks/prefixes or potential bare call prefixes."""
        if self._streaming_initialized:
            return (
                self._in_tool_block
                or self._buffering_bare_call
                or self._has_trailing_marker_prefix(current_text[self._cursor :], (self.tool_call_start_token,))
            )
        return super().is_buffering_protocol(current_text=current_text, delta_text=delta_text)

    def flush_streaming_content(self) -> str:
        """Release withheld plain text at end-of-stream when not inside a tool call."""
        if self._in_tool_block or (self._in_bare_call and self._emitted_tool_call):
            self._accumulated_text = ""
            self._cursor = 0
            return ""
        pending = self._accumulated_text[self._cursor :]
        self._accumulated_text = ""
        self._cursor = 0
        self._in_bare_call = False
        self._buffering_bare_call = False
        return pending

    @property
    def current_tool_index(self) -> int:
        """Get the current tool index being processed."""
        return self.current_tool_id

    @current_tool_index.setter
    def current_tool_index(self, value: int) -> None:
        """Set the current tool index."""
        self.current_tool_id = value

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model output.

        Handles one or more ``<function_calls>...</function_calls>`` blocks
        with optional surrounding prose, as well as bare newline-separated or
        multiline pythonic calls.
        """
        del request
        original_model_output = model_output

        if self.tool_call_start_token in model_output:
            tool_calls: list[ToolCall] = []
            content_parts: list[str] = []
            pos = 0
            n = len(model_output)
            while pos < n:
                start_idx = model_output.find(self.tool_call_start_token, pos)
                if start_idx == -1:
                    content_parts.append(model_output[pos:])
                    break
                body_start = start_idx + len(self.tool_call_start_token)
                end_idx = _find_tag_outside_strings(model_output, self.tool_call_end_token, body_start)
                if end_idx == -1:
                    # Unclosed block: try parsing body, otherwise keep as content.
                    parsed_calls = _parse_olmo3_block(model_output[body_start:])
                    if parsed_calls:
                        content_parts.append(model_output[pos:start_idx])
                        tool_calls.extend(parsed_calls)
                    else:
                        content_parts.append(model_output[pos:])
                    break
                parsed_calls = _parse_olmo3_block(model_output[body_start:end_idx])
                if parsed_calls:
                    content_parts.append(model_output[pos:start_idx])
                    tool_calls.extend(parsed_calls)
                else:
                    content_parts.append(model_output[pos : end_idx + len(self.tool_call_end_token)])
                pos = end_idx + len(self.tool_call_end_token)

            if tool_calls:
                content = "".join(content_parts)
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content.strip() else None,
                )
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=original_model_output)

        # Bare pythonic calls (no <function_calls> wrapper).
        bare_calls = _parse_olmo3_block(model_output)
        if bare_calls:
            return ExtractedToolCallInformation(tools_called=True, tool_calls=bare_calls, content=None)
        return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=original_model_output)

    def _emit_block_calls(
        self,
        raw_body: str,
        *,
        block_closed: bool,
        base_index: int,
    ) -> tuple[list[DeltaToolCall], int, bool]:
        """Parse *raw_body* (inside a block or bare) and emit new tool deltas.

        Returns ``(deltas, next_local_index, had_syntax_error)``.
        """
        body = raw_body if block_closed else _strip_trailing_partial_close_tag(raw_body, self.tool_call_end_token)
        if not body.strip():
            return [], self._block_local_index, False

        try:
            if block_closed:
                valid_text, added_text = textwrap.dedent(body).strip(), ""
            else:
                valid_and_added = make_valid_python(textwrap.dedent(body).lstrip())
                if valid_and_added is None:
                    return [], self._block_local_index, False
                valid_text, added_text = valid_and_added

            module = ast.parse(valid_text)
            tool_calls = _extract_calls_from_ast(module, allow_trailing_name=not block_closed)
        except (SyntaxError, ValueError, RecursionError, MemoryError, UnexpectedAstError):
            return [], self._block_local_index, True
        except Exception:
            logger.debug("Skipping chunk due to OLMo-3 streaming parse error", exc_info=True)
            return [], self._block_local_index, True

        deltas: list[DeltaToolCall] = []
        local_idx = self._block_local_index
        for idx, new_call in enumerate(tool_calls):
            if idx < local_idx:
                continue
            global_index = base_index + idx
            self.current_tool_index = global_index
            while len(self.streamed_args_for_tool) <= global_index:
                self.streamed_args_for_tool.append("")

            new_call_complete = block_closed or idx < len(tool_calls) - 1 or ")" not in added_text
            if new_call_complete:
                local_idx = idx + 1
                self.current_tool_index = global_index + 1

            withheld_suffix = added_text[:-1] if not new_call_complete else ""
            if not new_call_complete and added_text.endswith(")"):
                withheld_suffix = withheld_suffix + "}"
            withheld_suffix = withheld_suffix.replace("'", '"')

            delta = compute_tool_delta(self.streamed_args_for_tool[global_index], new_call, global_index, withheld_suffix)
            if delta is not None:
                deltas.append(delta)
                if delta.function is not None and delta.function.arguments is not None:
                    self.streamed_args_for_tool[global_index] += delta.function.arguments

        if deltas:
            self._emitted_tool_call = True
            self.current_tool_name_sent = True
            if not self.prev_tool_call_arr:
                self.prev_tool_call_arr = [{"arguments": {}}]

        return deltas, local_idx, False

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
            self._bare_check_done = True
            self._streaming_initialized = True

        self._accumulated_text = current_text

        # O(1) short-circuit once bare calls finished and trailing non-call text appeared.
        if self._bare_completed:
            self._buffering_bare_call = False
            self._cursor = len(current_text)
            return None

        # Handle optional bare calls at the start of the response (no <function_calls>).
        if not self._bare_check_done and not self._seen_any_block:
            if self.tool_call_start_token in current_text:
                self._bare_check_done = True
                self._buffering_bare_call = False
                self._in_bare_call = False
            else:
                stripped = current_text.lstrip()
                if not stripped:
                    self._buffering_bare_call = True
                    return None
                if stripped.startswith("<"):
                    self._bare_check_done = True
                    self._buffering_bare_call = False
                elif _BARE_IDENT_ONLY_RE.fullmatch(stripped):
                    self._buffering_bare_call = True
                    return None
                elif _BARE_CALL_START_RE.match(stripped):
                    self._in_bare_call = True
                    stripped_trimmed = stripped.rstrip()
                    if self._emitted_tool_call and len(stripped_trimmed) == self._bare_last_stripped_len:
                        self._buffering_bare_call = False
                        return None
                    balanced, mismatched = _has_balanced_brackets(stripped)
                    deltas, next_local, had_error = self._emit_block_calls(
                        current_text,
                        block_closed=False,
                        base_index=0,
                    )
                    self._block_local_index = next_local
                    if not self._emitted_tool_call and (mismatched or (balanced and had_error)):
                        # Not a valid bare tool call (e.g., "func(x) is ..."): fall through to plain content.
                        self._bare_check_done = True
                        self._in_bare_call = False
                        self._buffering_bare_call = False
                    elif self._emitted_tool_call and (mismatched or (balanced and had_error)):
                        # Bare tool call(s) already completed; trailing text is not valid Python calls.
                        self._bare_check_done = True
                        self._bare_completed = True
                        self._buffering_bare_call = False
                        self._cursor = len(current_text)
                        return DeltaMessage(tool_calls=deltas) if deltas else None
                    else:
                        if balanced and self._emitted_tool_call:
                            self._bare_last_stripped_len = len(stripped_trimmed)
                        self._buffering_bare_call = not balanced or not self._emitted_tool_call
                        if deltas:
                            return DeltaMessage(tool_calls=deltas)
                        return None
                else:
                    self._bare_check_done = True
                    self._buffering_bare_call = False

        content_parts: list[str] = []
        tool_deltas: list[DeltaToolCall] = []

        while True:
            if not self._in_tool_block:
                start_idx = current_text.find(self.tool_call_start_token, self._cursor)
                if start_idx != -1:
                    if start_idx > self._cursor:
                        content_parts.append(current_text[self._cursor : start_idx])
                    self._in_tool_block = True
                    self._seen_any_block = True
                    self._block_content_start = start_idx + len(self.tool_call_start_token)
                    self._block_base_tool_index = self._next_tool_index
                    self._block_local_index = 0
                    self._cursor = self._block_content_start
                    continue

                rem = current_text[self._cursor :]
                hold = _trailing_prefix_len(rem, self.tool_call_start_token)
                safe_end = len(current_text) - hold
                if safe_end > self._cursor:
                    content_parts.append(current_text[self._cursor : safe_end])
                    self._cursor = safe_end
                break

            end_idx = _find_tag_outside_strings(current_text, self.tool_call_end_token, self._block_content_start)
            if end_idx != -1:
                block_body = current_text[self._block_content_start : end_idx]
                deltas, next_local, _had_error = self._emit_block_calls(
                    block_body,
                    block_closed=True,
                    base_index=self._block_base_tool_index,
                )
                tool_deltas.extend(deltas)
                self._block_local_index = next_local
                self._next_tool_index = self._block_base_tool_index + next_local
                self._in_tool_block = False
                self._cursor = end_idx + len(self.tool_call_end_token)
                continue

            raw_body = current_text[self._block_content_start :]
            deltas, next_local, _had_error = self._emit_block_calls(
                raw_body,
                block_closed=False,
                base_index=self._block_base_tool_index,
            )
            tool_deltas.extend(deltas)
            self._block_local_index = next_local
            break

        content = "".join(content_parts)
        if content or tool_deltas:
            return DeltaMessage(content=content or None, tool_calls=tool_deltas or None)
        return None
