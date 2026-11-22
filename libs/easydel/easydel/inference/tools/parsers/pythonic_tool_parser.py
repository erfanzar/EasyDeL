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

"""Pythonic tool parser for models using Python-style function call syntax.

This module provides a tool call parser designed for models like Llama 3.2 and
Llama 4 that generate tool calls as Python function invocations within list
literals. The parser uses Python's AST module to safely extract and validate
function calls from model outputs.

Example format:
    [function1(param1="value1", param2="value2"), function2(arg="val")]

Features:
    - Python AST-based parsing for safe evaluation
    - Linear-time bracket and AST shape validation (no backtracking ReDoS)
    - Support for nested data structures (lists, dicts) in arguments
    - Streaming support with bracket/quote tracking and plain-prose passthrough
    - Multiple tool calls in single list
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from typing import Any

from eformer.loggings import get_logger
from transformers import PreTrainedTokenizerBase

from ...openai_api_modules import (
    ChatCompletionRequest,
    DeltaMessage,
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

try:  # ``regex`` (a transformers dependency) supports match timeouts; stdlib ``re`` does not.
    import regex as _regex_engine  # pyright: ignore[reportMissingImports]

    _HAS_REGEX_TIMEOUT = True
except ImportError:  # pragma: no cover - transformers always installs ``regex``
    _regex_engine = re
    _HAS_REGEX_TIMEOUT = False

logger = get_logger(__name__)

_IDENT_PREFIX_RE = re.compile(r"^\[\s*(?:([A-Za-z_]\w*)\s*(.*))?$", re.DOTALL)


def _find_matching_list_end(text: str) -> tuple[int, bool]:
    """Locate the matching closing ``]`` for a list starting at index 0.

    Returns ``(close_idx, mismatched)`` where ``close_idx`` is ``-1`` if the
    outer list is still open, and ``mismatched`` is ``True`` if brackets are
    imbalanced outside string literals.
    """
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    for idx, ch in enumerate(text):
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
                return -1, True
            if not stack:
                return idx, False
    return -1, False


def _validate_call_list_ast(module: ast.Module) -> list[ast.Call]:
    """Validate that *module* is a single expression containing a list of calls."""
    if len(module.body) != 1 or not isinstance(module.body[0], ast.Expr):
        raise UnexpectedAstError("Tool output must be one list of function calls")
    parsed = module.body[0].value
    if not isinstance(parsed, ast.List) or not all(isinstance(elt, ast.Call) for elt in parsed.elts):
        raise UnexpectedAstError("Tool output must be a list of function calls")
    for call in parsed.elts:
        if not isinstance(call.func, ast.Name) or call.args:
            raise UnexpectedAstError("Tool calls require a simple name and keyword-only arguments")
        names = [kw.arg for kw in call.keywords]
        if None in names or len(names) != len(set(names)):
            raise UnexpectedAstError("Tool calls cannot unpack or repeat keyword arguments")
    return parsed.elts


@ToolParserManager.register_module("pythonic")  # pyright: ignore[reportUntypedClassDecorator]
class PythonicToolParser(ToolParser):
    """Tool call parser for models using Python-style function call syntax.

    Designed for models like Llama 3.2 and Llama 4 that generate tool calls
    as Python function invocations within list literals, e.g.:
    [function1(param1="value1", param2="value2"), function2(...)]

    Attributes:
        TOOL_CALL_REGEX: Optional regex pattern for subclasses; ``None`` by
            default because detection uses a linear-time bracket and AST check.
        REGEX_TIMEOUT_S: Positive time budget in seconds; non-positive values
            disable pattern matching and preserve original content.
    """

    REGEX_TIMEOUT_S: float = 1.0
    TOOL_CALL_REGEX: Any = None

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        """Initialize the PythonicToolParser.

        Args:
            tokenizer: The tokenizer associated with the model. Used for
                token-level operations during streaming.
        """
        super().__init__(tokenizer)
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

    def get_tool_machinery_hints(self) -> Sequence[str]:
        """Engage streaming tool machinery on ``[`` without marking it a buffering protocol token."""
        hints = list(super().get_tool_machinery_hints())
        if "[" not in hints:
            hints.append("[")
        return tuple(hints)

    @staticmethod
    def _is_potential_call_list_prefix(text: str) -> bool:
        """Return whether *text* could still grow into a valid pythonic call list."""
        stripped = text.lstrip()
        if not stripped.startswith("["):
            return False
        match = _IDENT_PREFIX_RE.match(stripped)
        if match is None:
            return False
        ident, rest = match.group(1), match.group(2)
        if ident is None:
            return True
        if not rest:
            return True
        if not rest.startswith("("):
            return False
        close_idx, mismatched = _find_matching_list_end(stripped)
        if mismatched:
            return False
        if close_idx != -1:
            if stripped[close_idx + 1 :].strip():
                return False
            return False
        return True

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Return True only while buffering a potential leading pythonic tool call list."""
        del delta_text
        if self._streaming_initialized:
            if self._plain_content_mode:
                return False
            if self._emitted_tool_call:
                return not self._list_completed
            return self._buffering_tool_call
        return self._is_potential_call_list_prefix(current_text)

    def flush_streaming_content(self) -> str:
        """Release withheld text at end-of-stream when no tool calls were emitted."""
        if self._emitted_tool_call:
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

    def _matches_tool_call_pattern(self, model_output: str) -> bool:
        """Check whether *model_output* looks like a pythonic tool-call list.

        Uses a linear-time bracket check by default, honoring ``REGEX_TIMEOUT_S``
        and any custom ``TOOL_CALL_REGEX`` configured on a subclass.
        """
        if not (0 < self.REGEX_TIMEOUT_S < float("inf")):
            return False
        stripped = model_output.strip()
        if not (stripped.startswith("[") and stripped.endswith("]")):
            return False
        if self.TOOL_CALL_REGEX is None:
            return True
        try:
            if _HAS_REGEX_TIMEOUT:
                return self.TOOL_CALL_REGEX.fullmatch(stripped, timeout=self.REGEX_TIMEOUT_S) is not None
            return self.TOOL_CALL_REGEX.fullmatch(stripped) is not None
        except TimeoutError:
            logger.warning("Regex timeout occurred when matching tool call pattern.")
            logger.debug("Regex timeout occurred when matching user input: %s", model_output)
            return False

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model response.

        Parses the model output to identify and extract Python-style function
        calls using linear bracket checking and strict AST validation.
        """
        del request
        if not self._matches_tool_call_pattern(model_output):
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            tool_calls = parse_pythonic_call_list(model_output)
        except (SyntaxError, ValueError, RecursionError, MemoryError, UnexpectedAstError):
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        if not tool_calls:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=None,
        )

    def _emit_plain_content(self, current_text: str) -> DeltaMessage | None:
        self._plain_content_mode = True
        self._buffering_tool_call = False
        pending = current_text[self._stream_content_sent :]
        self._stream_content_sent = len(current_text)
        return DeltaMessage(content=pending) if pending else None

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
            self._stream_content_sent = len(previous_text)
            self._streaming_initialized = True

        self._accumulated_text = current_text

        if self._plain_content_mode:
            return self._emit_plain_content(current_text)

        if self._list_completed:
            self._buffering_tool_call = False
            return None

        stripped = current_text.lstrip()
        if not stripped:
            self._buffering_tool_call = True
            return None

        if not stripped.startswith("["):
            return self._emit_plain_content(current_text)

        close_idx, mismatched = _find_matching_list_end(stripped)
        if not self._emitted_tool_call:
            match = _IDENT_PREFIX_RE.match(stripped)
            if match is None:
                return self._emit_plain_content(current_text)
            ident, rest = match.group(1), match.group(2)
            if ident is not None and rest and not rest.startswith("("):
                return self._emit_plain_content(current_text)
            if mismatched:
                return self._emit_plain_content(current_text)
            if close_idx != -1 and stripped[close_idx + 1 :].strip():
                return self._emit_plain_content(current_text)

        try:
            if not self._emitted_tool_call and close_idx != -1:
                if parse_pythonic_call_list(stripped) is None:
                    return self._emit_plain_content(current_text)
            valid_and_added_text = make_valid_python(stripped)
            if valid_and_added_text is None:
                self._buffering_tool_call = True
                return None
            valid_text, added_text = valid_and_added_text

            module = ast.parse(valid_text)
            calls = _validate_call_list_ast(module)
            if not calls:
                if not added_text:
                    return self._emit_plain_content(current_text)
                self._buffering_tool_call = True
                return None

            tool_calls = [handle_single_tool(call) for call in calls]
        except (SyntaxError, ValueError, RecursionError, MemoryError, UnexpectedAstError):
            if not self._emitted_tool_call and close_idx != -1:
                return self._emit_plain_content(current_text)
            self._buffering_tool_call = not self._emitted_tool_call or not self._list_completed
            return None
        except Exception:
            logger.debug("Skipping chunk as a result of tool streaming extraction error", exc_info=True)
            return None

        tool_deltas = []
        for index, new_call in enumerate(tool_calls):
            if index < self.current_tool_index:
                continue

            self.current_tool_index = index
            while len(self.streamed_args_for_tool) <= index:
                self.streamed_args_for_tool.append("")

            new_call_complete = index < len(tool_calls) - 1 or ")]" not in added_text
            if new_call_complete:
                self.current_tool_index += 1

            withheld_suffix = added_text[:-2] if not new_call_complete else ""
            if not new_call_complete and len(added_text) >= 2 and added_text[-2] == ")":
                withheld_suffix = withheld_suffix + "}"
            withheld_suffix = withheld_suffix.replace("'", '"')
            delta = compute_tool_delta(self.streamed_args_for_tool[index], new_call, index, withheld_suffix)

            if delta is not None:
                tool_deltas.append(delta)
                if delta.function is not None and delta.function.arguments is not None:
                    self.streamed_args_for_tool[index] += delta.function.arguments

        if tool_deltas:
            self._emitted_tool_call = True
            self.current_tool_name_sent = True
            if not self.prev_tool_call_arr:
                self.prev_tool_call_arr = [{"arguments": {}}]

        self._list_completed = not added_text and close_idx != -1
        self._buffering_tool_call = not self._list_completed

        if tool_deltas:
            return DeltaMessage(tool_calls=tool_deltas)
        if not added_text and self.current_tool_id > 0:
            return DeltaMessage(content="")
        return None
