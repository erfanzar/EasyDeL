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
import re
from collections.abc import Sequence
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

logger = get_logger(__name__)

# Gemma4 special tokens for tool calls
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
STRING_DELIM = '<|"|>'


def _parse_gemma4_value(value_str: str) -> object:
    """Parse a single Gemma4 value (after key:) into a Python object."""
    value_str = value_str.strip()
    if not value_str:
        return value_str
    if value_str == "true":
        return True
    if value_str == "false":
        return False
    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass
    return value_str


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

    result: dict = {}
    i = 0
    n = len(args_str)

    while i < n:
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # Parse key (unquoted, ends at ':')
        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        i += 1  # skip ':'

        if i >= n:
            if not partial:
                result[key] = ""
            break

        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            if not partial:
                result[key] = ""
            break

        # String value: <|"|>...<|"|>
        if args_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            val_start = i
            end_pos = args_str.find(STRING_DELIM, i)
            if end_pos == -1:
                result[key] = args_str[val_start:]
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + len(STRING_DELIM)

        # Nested object: {...}
        elif args_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = n if next_delim == -1 else next_delim + len(STRING_DELIM)
                    continue
                if args_str[i] == "{":
                    depth += 1
                elif args_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_args(args_str[obj_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_args(args_str[obj_start : i - 1])

        # Array: [...]
        elif args_str[i] == "[":
            depth = 1
            arr_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = n if next_delim == -1 else next_delim + len(STRING_DELIM)
                    continue
                if args_str[i] == "[":
                    depth += 1
                elif args_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_array(args_str[arr_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_array(args_str[arr_start : i - 1])

        # Bare value (number, boolean, etc.)
        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            if partial and i >= n:
                break
            result[key] = _parse_gemma4_value(args_str[val_start:i])

    return result


def _parse_gemma4_array(arr_str: str, *, partial: bool = False) -> list:
    """Parse a Gemma4 array content string into a Python list."""
    items: list = []
    i = 0
    n = len(arr_str)

    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        if arr_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            end_pos = arr_str.find(STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + len(STRING_DELIM)

        elif arr_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    nd = arr_str.find(STRING_DELIM, i)
                    i = nd + len(STRING_DELIM) if nd != -1 else n
                    continue
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_args(arr_str[obj_start:i], partial=True))
            else:
                items.append(_parse_gemma4_args(arr_str[obj_start : i - 1]))

        elif arr_str[i] == "[":
            depth = 1
            sub_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_array(arr_str[sub_start:i], partial=True))
            else:
                items.append(_parse_gemma4_array(arr_str[sub_start : i - 1]))

        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            if partial and i >= n:
                break
            items.append(_parse_gemma4_value(arr_str[val_start:i]))

    return items


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

        start_tok = re.escape(TOOL_CALL_START)
        end_tok = re.escape(TOOL_CALL_END)
        self.tool_call_regex = re.compile(
            rf"{start_tok}call:([\w\-\.]+)\{{(.*?)\}}{end_tok}",
            re.DOTALL,
        )

        if self.model_tokenizer:
            self.tool_call_start_token_ids = self.model_tokenizer.encode(TOOL_CALL_START, add_special_tokens=False)
            self.tool_call_end_token_ids = self.model_tokenizer.encode(TOOL_CALL_END, add_special_tokens=False)
        else:
            self.tool_call_start_token_ids = []
            self.tool_call_end_token_ids = []

        self.buffered_delta_text = ""

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
            matches = self.tool_call_regex.findall(model_output)
            if not matches:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            tool_calls: list[ToolCall] = []
            for func_name, args_str in matches:
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
        delta_text = self._buffer_delta_text(delta_text)

        if TOOL_CALL_START not in current_text:
            return DeltaMessage(content=delta_text) if delta_text else None

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
        """Drive the streaming state machine for Gemma4 tool calls.

        Counts open/close markers between ``previous_text`` and
        ``current_text`` to decide whether the current step opens a new
        call, continues a call's body, or closes a call.

        Args:
            previous_text: Cumulative output up to the previous step.
            current_text: Cumulative output including the current step.
            delta_text: Newly produced text in this step.

        Returns:
            The :class:`DeltaMessage` to forward to the client, or ``None``
            when no client-visible event should be emitted.
        """
        start_count = current_text.count(TOOL_CALL_START)
        end_count = current_text.count(TOOL_CALL_END)
        prev_start_count = previous_text.count(TOOL_CALL_START)
        prev_end_count = previous_text.count(TOOL_CALL_END)

        # Not inside any tool call
        if start_count == end_count and prev_end_count == end_count and TOOL_CALL_END not in delta_text:
            if delta_text:
                return DeltaMessage(content=delta_text)
            return None

        # Starting a new tool call
        if start_count > prev_start_count and start_count > end_count:
            self.current_tool_id += 1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool.append("")
            self.prev_tool_call_arr.append({})
            if len(delta_text) <= len(TOOL_CALL_START):
                return None

        # Tool call just ended
        if end_count > prev_end_count:
            return self._handle_tool_call_end(current_text)

        # In the middle of a tool call
        if start_count > end_count:
            return self._handle_tool_call_middle(current_text)

        if delta_text:
            text = delta_text.replace(TOOL_CALL_START, "").replace(TOOL_CALL_END, "")
            if text:
                return DeltaMessage(content=text)
        return None

    def _extract_partial_call(self, current_text: str) -> tuple[str | None, str]:
        """Return the function name and partial argument body of the active call.

        Args:
            current_text: Cumulative output text that may contain the
                in-progress tool call.

        Returns:
            Tuple ``(func_name, args_part)`` where ``func_name`` is ``None``
            when no usable call is detected, and ``args_part`` is the body
            of the partial argument block (without enclosing braces).
        """
        last_start = current_text.rfind(TOOL_CALL_START)
        if last_start == -1:
            return None, ""

        partial_call = current_text[last_start + len(TOOL_CALL_START) :]
        if TOOL_CALL_END in partial_call:
            partial_call = partial_call.split(TOOL_CALL_END)[0]

        if not partial_call.startswith("call:"):
            return None, ""

        func_part = partial_call[5:]
        if "{" not in func_part:
            return None, ""

        func_name, _, args_part = func_part.partition("{")
        func_name = func_name.strip()

        if args_part.endswith("}"):
            args_part = args_part[:-1]

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
            _, args_str = all_matches[self.current_tool_id]
            final_args = _parse_gemma4_args(args_str)
            final_args_json = json.dumps(final_args, ensure_ascii=False)

            prev_streamed = self.streamed_args_for_tool[self.current_tool_id]
            if len(final_args_json) > len(prev_streamed):
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
        """Parse raw args, convert to JSON, withhold trailing closers, diff and emit."""
        try:
            current_args = _parse_gemma4_args(raw_args_str, partial=True)
        except Exception:
            return None

        if not current_args:
            return None

        current_args_json = json.dumps(current_args, ensure_ascii=False)

        # Withhold trailing closing characters that may shift as more
        # tokens arrive (e.g. a string value is still being streamed).
        safe_json = current_args_json
        while safe_json and safe_json[-1] in ("}", '"', "]", "<", "|", "\\", ">"):
            safe_json = safe_json[:-1]

        prev_streamed = self.streamed_args_for_tool[self.current_tool_id]

        if not safe_json or safe_json == prev_streamed:
            return None

        if prev_streamed:
            prefix_len = 0
            min_len = min(len(prev_streamed), len(safe_json))
            while prefix_len < min_len and prev_streamed[prefix_len] == safe_json[prefix_len]:
                prefix_len += 1

            if prefix_len < len(prev_streamed):
                # Structure changed — wait for final flush.
                self.streamed_args_for_tool[self.current_tool_id] = safe_json[:prefix_len]
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
