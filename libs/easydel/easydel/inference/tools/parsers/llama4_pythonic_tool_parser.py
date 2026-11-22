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

"""Llama 4 Pythonic tool parser with optional delimiter token support.

This module provides a specialized tool call parser for Llama 4 models that
generate tool calls as Python function invocations. It extends pythonic parsing
with support for optional ``<|python_start|>`` and ``<|python_end|>`` delimiter
tokens.

Example formats:
    <|python_start|>[func1(arg1="val1"), func2(arg2="val2")]<|python_end|>
    [func1(arg1="val1"), func2(arg2="val2")]
"""

from __future__ import annotations

import ast
from collections.abc import Sequence

import regex as re
from eformer.loggings import get_logger
from transformers import PreTrainedTokenizerBase

from ...openai_api_modules import (
    ChatCompletionRequest,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
)
from ..abstract_tool import ToolParser, ToolParserManager
from ._pythonic_ast import (
    UnexpectedAstError,
    compute_tool_delta,
    handle_single_tool,
    make_valid_python,
    parse_pythonic_call_list,
)
from .pythonic_tool_parser import _IDENT_PREFIX_RE, _find_matching_list_end

logger = get_logger(__name__)

# Backward-compatible private alias for the shared exception type.
_UnexpectedAstError = UnexpectedAstError


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


@ToolParserManager.register_module("llama4_pythonic")  # pyright: ignore[reportUntypedClassDecorator]
class Llama4PythonicToolParser(ToolParser):
    """Tool call parser for Llama 4 models with Pythonic syntax.

    Specialized version of the pythonic parser for Llama 4 models that
    generate tool calls as Python function invocations. Supports optional
    ``<|python_start|>`` and ``<|python_end|>`` delimiters.

    Attributes:
        TOOL_CALL_REGEX: Compiled structural pre-check. AST parsing validates the syntax.
        regex_timeout: Maximum seconds allowed for the regex pre-check.
    """

    TOOL_CALL_REGEX = re.compile(r"\s*\[.*\]\s*", re.DOTALL)

    def __init__(self, tokenizer: PreTrainedTokenizerBase, regex_timeout: float = 0.05):
        """Initialize the Llama4PythonicToolParser.

        Args:
            tokenizer: The tokenizer associated with the Llama 4 model.
            regex_timeout: Positive time budget in seconds for the structural regex.

        Raises:
            ValueError: If the regex timeout is not positive and finite.
        """
        super().__init__(tokenizer)
        if not 0 < regex_timeout < float("inf"):
            raise ValueError("regex_timeout must be positive and finite")
        self.regex_timeout = regex_timeout
        self.bot_token = "<|python_start|>"
        self.eot_token = "<|python_end|>"
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state for a fresh response."""
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool = []
        self._streaming_initialized = False
        self._stream_content_sent = 0
        self._accumulated_text = ""
        self._plain_content_mode = False
        self._buffering_tool_call = False
        self._emitted_tool_call = False
        self._list_completed = False
        self._in_delimited_block = False
        self._delimited_start = -1
        self._block_base_tool_index = 0
        self._block_local_index = 0
        self._next_tool_index = 0

    def get_tool_machinery_hints(self) -> Sequence[str]:
        """Engage streaming tool machinery on ``[`` in addition to ``<|python_start|>``."""
        hints = list(super().get_tool_machinery_hints())
        if "[" not in hints:
            hints.append("[")
        return tuple(hints)

    @staticmethod
    def _is_potential_bare_call_prefix(text: str) -> bool:
        """Return whether *text* could grow into a bare ``[fn(...)]`` call list."""
        stripped = text.lstrip()
        if not stripped.startswith("["):
            return False
        match = _IDENT_PREFIX_RE.match(stripped)
        if match is None:
            return False
        ident, rest = match.group(1), match.group(2)
        if ident is None or not rest:
            return True
        if not rest.startswith("("):
            return False
        close_idx, mismatched = _find_matching_list_end(stripped)
        if mismatched:
            return False
        if close_idx != -1:
            return False
        return True

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Buffer active ``<|python_start|>`` blocks/prefixes or potential bare ``[fn(`` calls."""
        if self._streaming_initialized:
            if self._in_delimited_block:
                return True
            if self._has_trailing_marker_prefix(current_text[self._stream_content_sent :], (self.bot_token,)):
                return True
            if self._plain_content_mode:
                return False
            if self._emitted_tool_call:
                return not self._list_completed
            return self._buffering_tool_call
        if super().is_buffering_protocol(current_text=current_text, delta_text=delta_text):
            return True
        return self._is_potential_bare_call_prefix(current_text)

    def flush_streaming_content(self) -> str:
        """Release withheld text at end-of-stream when no tool calls were emitted."""
        if self._in_delimited_block or (self._emitted_tool_call and not self._plain_content_mode):
            self._accumulated_text = ""
            self._stream_content_sent = 0
            return ""
        pending = self._accumulated_text[self._stream_content_sent :]
        self._accumulated_text = ""
        self._stream_content_sent = 0
        self._buffering_tool_call = False
        return pending

    @property
    def current_tool_index(self) -> int:
        """Get the current tool index being processed."""
        return self.current_tool_id

    @current_tool_index.setter
    def current_tool_index(self, value: int) -> None:
        """Set the current tool index being processed."""
        self.current_tool_id = value

    @staticmethod
    def _validate_tool_call_shape(module: ast.Module) -> list[ast.Call]:
        """Validate structure before converting the supported keyword-only calls."""
        if len(module.body) != 1 or not isinstance(module.body[0], ast.Expr):
            raise UnexpectedAstError("Tool output must be one list of function calls")
        parsed = module.body[0].value
        if not isinstance(parsed, ast.List) or not all(isinstance(call, ast.Call) for call in parsed.elts):
            raise UnexpectedAstError("Tool output must be a list of function calls")
        for call in parsed.elts:
            if not isinstance(call.func, ast.Name) or call.args:
                raise UnexpectedAstError("Tool calls require a simple name and keyword-only arguments")
            names = [keyword.arg for keyword in call.keywords]
            if None in names or len(names) != len(set(names)):
                raise UnexpectedAstError("Tool calls cannot unpack or repeat keyword arguments")
        return parsed.elts

    def _extract_delimited_body(self, text: str) -> tuple[str | None, str]:
        """Strip optional preamble/trailing prose and ``<|python_start|>``/``<|python_end|>`` wrappers."""
        bot_idx = text.find(self.bot_token)
        if bot_idx != -1:
            preamble = text[:bot_idx]
            rest = text[bot_idx + len(self.bot_token) :].lstrip()
            eot_idx = _find_tag_outside_strings(rest, self.eot_token, 0)
            if eot_idx != -1:
                body = rest[:eot_idx].strip()
                trailing = rest[eot_idx + len(self.eot_token) :]
            else:
                body = rest.strip()
                trailing = ""
            combined_content = preamble + trailing
            return (combined_content if combined_content.strip() else None), body

        return None, text.strip()

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model response."""
        del request
        original_output = model_output
        preamble, stripped = self._extract_delimited_body(model_output)

        if not (stripped.startswith("[") and stripped.endswith("]")):
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=original_output)

        try:
            if self.TOOL_CALL_REGEX.fullmatch(stripped, timeout=self.regex_timeout) is None:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=original_output)
            calls = self._validate_tool_call_shape(ast.parse(stripped))
            if not calls:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=original_output)
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=[handle_single_tool(call) for call in calls],
                content=preamble,
            )
        except TimeoutError:
            logger.warning("Regex timed out while detecting Llama 4 tool calls")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=original_output)
        except (SyntaxError, ValueError, RecursionError, MemoryError, UnexpectedAstError):
            logger.debug("Model output is not valid Python; treating it as plain content.")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=original_output)
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=original_output)

    def _emit_safe_plain_content(self, current_text: str) -> DeltaMessage | None:
        """Emit plain text while withholding any trailing partial ``<|python_start|>`` prefix."""
        pending = current_text[self._stream_content_sent :]
        if not pending:
            return None
        hold = _trailing_prefix_len(pending, self.bot_token)
        safe_end = len(current_text) - hold
        if safe_end > self._stream_content_sent:
            out = current_text[self._stream_content_sent : safe_end]
            self._stream_content_sent = safe_end
            self._buffering_tool_call = hold > 0
            return DeltaMessage(content=out)
        self._buffering_tool_call = hold > 0
        return None

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
        """Extract tool calls incrementally during streaming."""
        del delta_text, previous_token_ids, current_token_ids, delta_token_ids, request
        if not previous_text:
            self._reset_streaming_state()
            self._streaming_initialized = True
        elif not self._streaming_initialized:
            self._reset_streaming_state()
            # Keep start-marker-only test calls (where previous_text == current_text == "<|python_start|>") working.
            if previous_text.startswith(self.bot_token) or previous_text.lstrip().startswith("["):
                self._stream_content_sent = 0
            else:
                self._stream_content_sent = len(previous_text)
            self._streaming_initialized = True

        self._accumulated_text = current_text

        # O(1) short-circuit after a bare pythonic call list has completed.
        if not self._in_delimited_block and not self._plain_content_mode and self._list_completed:
            self._buffering_tool_call = False
            self._stream_content_sent = len(current_text)
            return None

        content_parts: list[str] = []
        tool_deltas: list[DeltaToolCall] = []
        closed_empty_delimited = False

        while True:
            if self._in_delimited_block:
                eot_idx = _find_tag_outside_strings(current_text, self.eot_token, self._delimited_start)
                if eot_idx != -1:
                    block_body = current_text[self._delimited_start : eot_idx]
                    deltas, _plain, _completed = self._parse_streaming_pythonic(
                        block_body,
                        delimited=True,
                        block_closed=True,
                    )
                    tool_deltas.extend(deltas)
                    if not deltas and not self._emitted_tool_call:
                        closed_empty_delimited = True
                    self._in_delimited_block = False
                    self._delimited_start = -1
                    self._stream_content_sent = eot_idx + len(self.eot_token)
                    self._plain_content_mode = True
                    self._buffering_tool_call = False
                    self._list_completed = self._emitted_tool_call
                    continue

                deltas, _plain, completed = self._parse_streaming_pythonic(
                    current_text[self._delimited_start :],
                    delimited=True,
                    block_closed=False,
                )
                tool_deltas.extend(deltas)
                self._list_completed = completed
                self._buffering_tool_call = True
                break

            bot_idx = current_text.find(self.bot_token, self._stream_content_sent)
            if bot_idx != -1 and (self._plain_content_mode or not self._emitted_tool_call):
                if bot_idx > self._stream_content_sent:
                    content_parts.append(current_text[self._stream_content_sent : bot_idx])
                self._stream_content_sent = bot_idx
                self._in_delimited_block = True
                self._delimited_start = bot_idx + len(self.bot_token)
                self._block_base_tool_index = self._next_tool_index
                self._block_local_index = 0
                self._plain_content_mode = False
                self._buffering_tool_call = True
                continue

            if self._plain_content_mode:
                pending = current_text[self._stream_content_sent :]
                if pending:
                    hold = _trailing_prefix_len(pending, self.bot_token)
                    safe_end = len(current_text) - hold
                    if safe_end > self._stream_content_sent:
                        content_parts.append(current_text[self._stream_content_sent : safe_end])
                        self._stream_content_sent = safe_end
                    self._buffering_tool_call = hold > 0
                else:
                    self._buffering_tool_call = False
                break

            stripped = current_text.lstrip()
            if not stripped:
                self._buffering_tool_call = True
                break

            if not stripped.startswith("["):
                self._plain_content_mode = True
                continue

            deltas, decided_plain, completed = self._parse_streaming_pythonic(
                stripped,
                delimited=False,
                block_closed=False,
            )
            if decided_plain:
                self._plain_content_mode = True
                continue
            tool_deltas.extend(deltas)
            self._list_completed = completed
            self._buffering_tool_call = not completed
            if completed:
                self._stream_content_sent = len(current_text)
            break

        content = "".join(content_parts)
        if content or tool_deltas:
            return DeltaMessage(content=content or None, tool_calls=tool_deltas or None)
        if closed_empty_delimited:
            return DeltaMessage(content="")
        if not self._in_delimited_block and self._list_completed and self.current_tool_id > 0:
            return DeltaMessage(content="")
        return None

    def _parse_streaming_pythonic(
        self,
        raw_body: str,
        *,
        delimited: bool,
        block_closed: bool = False,
    ) -> tuple[list[DeltaToolCall], bool, bool]:
        """Parse streaming pythonic list payload.

        Returns ``(tool_deltas, decided_plain, completed_list)``.
        """
        body = raw_body.lstrip()
        if not body:
            return [], False, False

        if not body.startswith("["):
            if delimited:
                return [], False, False
            return [], True, False

        close_idx, mismatched = _find_matching_list_end(body)
        parse_target = body
        list_closed = False
        if not mismatched and close_idx != -1:
            tail = body[close_idx + 1 :].strip()
            if delimited:
                if block_closed:
                    if not tail:
                        parse_target = body[: close_idx + 1]
                    list_closed = True
                elif (
                    not tail
                    or tail == self.eot_token
                    or self.eot_token.startswith(tail)
                    or tail.startswith(self.eot_token)
                ):
                    parse_target = body[: close_idx + 1]
                    list_closed = True
            else:
                if not tail:
                    parse_target = body[: close_idx + 1]
                    list_closed = True
                elif not self._emitted_tool_call:
                    return [], True, False

        if not self._emitted_tool_call and not delimited:
            match = _IDENT_PREFIX_RE.match(body)
            if match is None:
                return [], True, False
            ident, rest = match.group(1), match.group(2)
            if ident is not None and rest and not rest.startswith("("):
                return [], True, False
            if mismatched:
                return [], True, False

        try:
            if not self._emitted_tool_call and not delimited and list_closed:
                if parse_pythonic_call_list(parse_target) is None:
                    return [], True, False

            if block_closed:
                valid_text, added_text = parse_target, ""
            else:
                valid_and_added_text = make_valid_python(parse_target)
                if valid_and_added_text is None:
                    return [], False, False
                valid_text, added_text = valid_and_added_text

            module = ast.parse(valid_text)
            if not module.body:
                return [], False, False
            calls = self._validate_tool_call_shape(module)
            if not calls:
                if not added_text and not delimited:
                    return [], True, False
                return [], False, False
            tool_calls = [handle_single_tool(call) for call in calls]
        except (SyntaxError, ValueError, RecursionError, MemoryError, UnexpectedAstError):
            if not self._emitted_tool_call and not delimited and list_closed:
                return [], True, False
            return [], False, False
        except Exception:
            logger.debug("Skipping chunk as a result of tool streaming extraction error", exc_info=True)
            return [], False, False

        base_idx = self._block_base_tool_index if delimited else 0
        local_idx = self._block_local_index if delimited else self.current_tool_index
        tool_deltas: list[DeltaToolCall] = []
        for idx, new_call in enumerate(tool_calls):
            if idx < local_idx:
                continue

            global_index = base_idx + idx
            self.current_tool_index = global_index
            while len(self.streamed_args_for_tool) <= global_index:
                self.streamed_args_for_tool.append("")

            new_call_complete = block_closed or idx < len(tool_calls) - 1 or ")]" not in added_text
            if new_call_complete:
                local_idx = idx + 1
                self.current_tool_index = global_index + 1

            withheld_suffix = added_text[:-2] if not new_call_complete else ""
            if not new_call_complete and len(added_text) >= 2 and added_text[-2] == ")":
                withheld_suffix = withheld_suffix + "}"
            withheld_suffix = withheld_suffix.replace("'", '"')
            delta = compute_tool_delta(self.streamed_args_for_tool[global_index], new_call, global_index, withheld_suffix)

            if delta is not None:
                tool_deltas.append(delta)
                if delta.function is not None and delta.function.arguments is not None:
                    self.streamed_args_for_tool[global_index] += delta.function.arguments

        if delimited:
            self._block_local_index = local_idx
        self._next_tool_index = max(self._next_tool_index, base_idx + max(local_idx, 0))

        if tool_deltas:
            self._emitted_tool_call = True
            self.current_tool_name_sent = True
            if not self.prev_tool_call_arr:
                self.prev_tool_call_arr = [{"arguments": {}}]

        completed = (not added_text and list_closed) or block_closed
        return tool_deltas, False, completed
