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

"""Qwen3 Coder tool parser aligned with reference serving behavior.

Parses tool calls emitted by Qwen3 Coder models in the ``<tool_call>``
XML envelope (as well as bare ``<function=...>`` blocks)::

    <tool_call><function=name><parameter=x>value</parameter></function></tool_call>

Supports both complete (non-streaming) extraction via
:meth:`Qwen3CoderToolParser.extract_tool_calls` and incremental
streaming extraction via :meth:`Qwen3CoderToolParser.extract_tool_calls_streaming`,
including schema-aware coercion of parameter string values to their
JSON-Schema declared types using the tool definitions on the request.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

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
    ToolDefinition,
)
from ..abstract_tool import ToolParser, ToolParserManager
from ._schema_params import convert_param_value, get_arguments_config

logger = get_logger(__name__)


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


def _find_function_close(
    text: str,
    start_after_open: int,
    limit: int,
    *,
    param_prefix: str = "<parameter=",
    param_end: str = "</parameter>",
    func_end: str = "</function>",
    env_end: str | None = None,
    allow_unclosed_param: bool = True,
) -> int:
    """Find ``</function>`` in ``text[start_after_open:limit]``, ending parameters at the earliest boundary."""
    if env_end:
        close_env = text.find(env_end, start_after_open, limit)
        if close_env != -1:
            limit = close_env
    pos = start_after_open
    while pos < limit:
        close_fn = text.find(func_end, pos, limit)
        next_param = text.find(param_prefix, pos, limit)
        if close_fn != -1 and (next_param == -1 or close_fn < next_param):
            return close_fn
        if next_param == -1:
            return -1
        param_name_start = next_param + len(param_prefix)
        param_gt = text.find(">", param_name_start, limit)
        if param_gt == -1:
            return -1
        lt_before_gt = text.find("<", param_name_start, param_gt)
        if lt_before_gt != -1:
            pos = lt_before_gt
            continue
        val_start = param_gt + 1
        param_close = text.find(param_end, val_start, limit)
        next_param2 = text.find(param_prefix, val_start, limit)
        close_fn2 = text.find(func_end, val_start, limit)
        candidates: list[tuple[int, str]] = []
        if param_close != -1:
            candidates.append((param_close, "param_end"))
        if next_param2 != -1:
            candidates.append((next_param2, "next_param"))
        if close_fn2 != -1:
            candidates.append((close_fn2, "func_end"))
        if not candidates:
            return -1
        earliest_pos, kind = min(candidates, key=lambda pair: pair[0])
        if kind == "param_end":
            pos = earliest_pos + len(param_end)
            continue
        if kind == "next_param":
            pos = earliest_pos
            continue
        return earliest_pos if allow_unclosed_param else -1
    return -1


def _iter_parameters(
    params_text: str,
    *,
    param_prefix: str = "<parameter=",
    param_end: str = "</parameter>",
) -> list[tuple[str, str]]:
    """Extract ``(name, value)`` pairs from *params_text*, safely skipping truncated tags."""
    results: list[tuple[str, str]] = []
    pos = 0
    n = len(params_text)
    while pos < n:
        p_start = params_text.find(param_prefix, pos)
        if p_start == -1:
            break
        name_start = p_start + len(param_prefix)
        gt_idx = params_text.find(">", name_start)
        if gt_idx == -1:
            break
        lt_before_gt = params_text.find("<", name_start, gt_idx)
        if lt_before_gt != -1:
            pos = lt_before_gt
            continue
        param_name = params_text[name_start:gt_idx]
        val_start = gt_idx + 1
        p_close = params_text.find(param_end, val_start)
        next_p = params_text.find(param_prefix, val_start)
        if p_close != -1 and (next_p == -1 or p_close < next_p):
            param_value = params_text[val_start:p_close]
            pos = p_close + len(param_end)
        elif next_p != -1:
            param_value = params_text[val_start:next_p]
            pos = next_p
        else:
            param_value = params_text[val_start:]
            pos = n
        if param_value.startswith("\n"):
            param_value = param_value[1:]
        if param_value.endswith("\n"):
            param_value = param_value[:-1]
        results.append((param_name, param_value))
    return results


@ToolParserManager.register_module(["qwen3_coder"])  # pyright: ignore[reportUntypedClassDecorator]
class Qwen3CoderToolParser(ToolParser):
    """Tool parser for Qwen3 Coder XML tool calls.

    Parses tool calls emitted in ``<tool_call><function=name>...</function></tool_call>``
    and bare ``<function=name>...</function>`` XML format with typed
    ``<parameter=name>value</parameter>`` tags.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the Qwen3 Coder tool parser.

        Args:
            tokenizer: The tokenizer associated with the Qwen3 Coder model.

        Raises:
            ValueError: If the tokenizer is not provided.
            RuntimeError: If tool call start/end tokens are not found in
                the tokenizer vocabulary.
        """
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        # Override the base class type: string call IDs are used here.
        self.current_tool_id: str | None = None  # type: ignore[assignment]
        self.streamed_args_for_tool: list[str] = []

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"
        self.is_tool_call_started: bool = False
        self.failed_count: int = 0

        self._reset_streaming_state()

        self.tool_call_complete_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL)
        self.tool_call_function_regex = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError("Qwen3 XML Tool parser could not locate tool call start/end tokens in the tokenizer!")

        logger.debug("Successfully imported tool parser %s !", self.__class__.__name__)

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state for a new message."""
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = None
        self.current_tool_name_sent = False
        self.current_function_name = None
        self.current_param_name = None
        self.current_param_value = ""
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False
        self.accumulated_params: dict[str, Any] = {}
        self.streaming_request: ChatCompletionRequest | None = None
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()
        self._streaming_initialized = False
        self._cursor = 0
        self._in_tool_wrapper = False
        self._func_body_start = -1
        self._func_param_cursor = -1

    def get_streaming_buffer_marker_pairs(self) -> Sequence[tuple[str, str | None]]:
        """Scope protocol buffering to top-level ``<tool_call>`` and ``<function=`` markers."""
        return (
            (self.tool_call_start_token, self.tool_call_end_token),
            (self.tool_call_prefix, self.function_end_token),
        )

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Return True while inside an active tool/function block or trailing marker prefix."""
        if self._streaming_initialized:
            if self.is_tool_call_started or self._in_tool_wrapper or self.in_function:
                return True
            rem = current_text[self._cursor :]
            return _trailing_prefix_len_any(rem, (self.tool_call_start_token, self.tool_call_prefix)) > 0
        return super().is_buffering_protocol(current_text=current_text, delta_text=delta_text)

    def flush_streaming_content(self) -> str:
        """Release withheld opening-tag prefix when the stream ends outside a confirmed tool call."""
        self.streaming_request = None
        if self._in_tool_wrapper or self.in_function:
            self.accumulated_text = ""
            self._cursor = 0
            return ""
        pending = self.accumulated_text[self._cursor :]
        self.accumulated_text = ""
        self._cursor = 0
        self.is_tool_call_started = False
        return pending

    def _get_arguments_config(self, func_name: str, tools: list[ToolDefinition] | None) -> dict:
        """Extract the parameter properties schema for a function."""
        return get_arguments_config(func_name, tools)

    def _convert_param_value(self, param_value: str, param_name: str, param_config: dict, func_name: str) -> Any:
        """Convert a raw string parameter value to its schema-defined type."""
        try:
            return convert_param_value(param_value, param_name, param_config, func_name)
        except (RecursionError, MemoryError):
            return param_value

    @staticmethod
    def _is_valid_function_name(name: str) -> bool:
        """Return True if *name* is a non-empty XML function name without tag/newline characters."""
        return bool(name and name.strip()) and not any(ch in name for ch in "<>\r\n")

    def _parse_xml_function_call(self, function_call_str: str, tools: list[ToolDefinition] | None) -> ToolCall | None:
        """Parse a single XML function call string into a ToolCall object."""
        end_index = function_call_str.find(">")
        if end_index == -1:
            return None
        function_name = function_call_str[:end_index]
        if not self._is_valid_function_name(function_name):
            return None
        param_config = self._get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1 :]
        param_dict: dict[str, Any] = {}
        for param_name, param_value in _iter_parameters(
            parameters,
            param_prefix=self.parameter_prefix,
            param_end=self.parameter_end_token,
        ):
            param_dict[param_name] = self._convert_param_value(param_value, param_name, param_config, function_name)
        return ToolCall(
            type="function",
            function=FunctionCall(name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False)),
        )

    def _get_function_calls(self, model_output: str) -> list[str]:
        """Extract all ``<function=...>`` bodies (both wrapped and bare) in document order."""
        function_calls: list[str] = []
        pos = 0
        n = len(model_output)
        while pos < n:
            fn_start = model_output.find(self.tool_call_prefix, pos)
            if fn_start == -1:
                break
            body_start = fn_start + len(self.tool_call_prefix)
            gt_idx = model_output.find(">", body_start)
            lt_before_gt = model_output.find("<", body_start, gt_idx if gt_idx != -1 else n)
            if gt_idx == -1 or lt_before_gt != -1:
                # Malformed/truncated <function=name without '>' before the next tag or EOF.
                if lt_before_gt != -1:
                    pos = lt_before_gt
                    continue
                break
            if not self._is_valid_function_name(model_output[body_start:gt_idx]):
                pos = gt_idx + 1
                continue
            fn_close = _find_function_close(
                model_output,
                gt_idx + 1,
                n,
                param_prefix=self.parameter_prefix,
                param_end=self.parameter_end_token,
                func_end=self.function_end_token,
                env_end=self.tool_call_end_token,
            )
            if fn_close != -1:
                function_calls.append(model_output[body_start:fn_close])
                pos = fn_close + len(self.function_end_token)
            else:
                tc_close = model_output.find(self.tool_call_end_token, gt_idx + 1)
                end_pos = tc_close if tc_close != -1 else n
                function_calls.append(model_output[body_start:end_pos])
                pos = end_pos + (len(self.tool_call_end_token) if tc_close != -1 else 0)
        return function_calls

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete (non-streaming) model output (stateless)."""
        if self.tool_call_prefix not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            function_calls = self._get_function_calls(model_output)
            tools = request.tools if request else None
            valid_tool_calls = [
                tool_call
                for function_call_str in function_calls
                if (tool_call := self._parse_xml_function_call(function_call_str, tools)) is not None
            ]
            if not valid_tool_calls:
                tc_idx = model_output.find(self.tool_call_start_token)
                if tc_idx != -1:
                    preamble = model_output[:tc_idx]
                    return ExtractedToolCallInformation(
                        tools_called=False,
                        tool_calls=[],
                        content=preamble if preamble else None,
                    )
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            marker_indices = [
                i
                for i in (model_output.find(self.tool_call_start_token), model_output.find(self.tool_call_prefix))
                if i >= 0
            ]
            content_index = min(marker_indices) if marker_indices else 0
            content = model_output[:content_index]

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=valid_tool_calls,
                content=content if content else None,
            )
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    def _append_arg_delta(self, tool_deltas: list[DeltaToolCall], tool_index: int, fragment: str) -> None:
        """Append *fragment* to *tool_index*'s streamed args and merge into *tool_deltas*."""
        while len(self.streamed_args_for_tool) <= tool_index:
            self.streamed_args_for_tool.append("")
        self.streamed_args_for_tool[tool_index] += fragment
        if tool_deltas and tool_deltas[-1].index == tool_index and tool_deltas[-1].function is not None:
            prev_arg = tool_deltas[-1].function.arguments or ""
            tool_deltas[-1].function.arguments = prev_arg + fragment
        else:
            tool_deltas.append(
                DeltaToolCall(index=tool_index, function=DeltaFunctionCall(arguments=fragment))
            )

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
        del previous_token_ids, current_token_ids
        if not previous_text:
            self._reset_streaming_state()
            self._streaming_initialized = True
        elif not self._streaming_initialized:
            self._reset_streaming_state()
            self._cursor = len(previous_text)
            self._streaming_initialized = True

        if request is not None:
            self.streaming_request = request

        self.accumulated_text = current_text

        if not delta_text and self._cursor >= len(current_text) and not self.is_tool_call_started:
            if delta_token_ids and self.tool_call_end_token_id not in delta_token_ids:
                complete_calls = len(self.tool_call_complete_regex.findall(current_text))
                if complete_calls > 0 and len(self.prev_tool_call_arr) > 0:
                    open_calls = current_text.count(self.tool_call_start_token) - current_text.count(
                        self.tool_call_end_token
                    )
                    if open_calls == 0:
                        return DeltaMessage(content="")
                elif not self.is_tool_call_started and current_text:
                    return DeltaMessage(content="")
            return None

        content_parts: list[str] = []
        tool_deltas: list[DeltaToolCall] = []
        open_markers = (self.tool_call_start_token, self.tool_call_prefix)

        while True:
            if not self._in_tool_wrapper and not self.in_function:
                tc_idx = current_text.find(self.tool_call_start_token, self._cursor)
                fn_idx = current_text.find(self.tool_call_prefix, self._cursor)
                candidates = [(i, tag) for i, tag in ((tc_idx, "tc"), (fn_idx, "fn")) if i != -1]
                if candidates:
                    start_idx, kind = min(candidates, key=lambda pair: pair[0])
                    if start_idx > self._cursor:
                        content_parts.append(current_text[self._cursor : start_idx])
                    self.is_tool_call_started = True
                    if kind == "tc":
                        self._in_tool_wrapper = True
                        self._cursor = start_idx + len(self.tool_call_start_token)
                    else:
                        self._in_tool_wrapper = False
                        self._cursor = start_idx
                        # Enter bare function immediately below.
                        open_status = self._try_open_function(current_text, len(current_text), tool_deltas)
                        if open_status == "incomplete":
                            break
                        if open_status == "invalid":
                            content_parts.append(self.tool_call_prefix)
                            self._cursor = start_idx + len(self.tool_call_prefix)
                            self.is_tool_call_started = False
                            continue
                        if not self._has_body_activity(current_text, len(current_text)):
                            break
                    continue

                rem = current_text[self._cursor :]
                hold = _trailing_prefix_len_any(rem, open_markers)
                safe_end = len(current_text) - hold
                if safe_end > self._cursor:
                    content_parts.append(current_text[self._cursor : safe_end])
                    self._cursor = safe_end
                break

            if self._in_tool_wrapper and not self.in_function:
                fn_idx = current_text.find(self.tool_call_prefix, self._cursor)
                tc_end = current_text.find(self.tool_call_end_token, self._cursor)
                if tc_end != -1 and (fn_idx == -1 or tc_end < fn_idx):
                    self._in_tool_wrapper = False
                    self.is_tool_call_started = False
                    self._cursor = tc_end + len(self.tool_call_end_token)
                    continue
                if fn_idx != -1:
                    self._cursor = fn_idx
                    limit = tc_end if tc_end != -1 else len(current_text)
                    open_status = self._try_open_function(current_text, limit, tool_deltas)
                    if open_status == "incomplete":
                        if tc_end != -1:
                            self._in_tool_wrapper = False
                            self.is_tool_call_started = False
                            self._cursor = tc_end + len(self.tool_call_end_token)
                            continue
                        break
                    if open_status == "invalid":
                        name_start = fn_idx + len(self.tool_call_prefix)
                        next_lt = current_text.find("<", name_start, limit)
                        self._cursor = next_lt if next_lt != -1 else name_start
                        continue
                    if not self._has_body_activity(current_text, limit):
                        break
                    continue
                break

            # Inside an active function (<function=name>...).
            tc_end_candidate = (
                current_text.find(self.tool_call_end_token, self._func_body_start)
                if self._in_tool_wrapper
                else -1
            )
            fn_search_limit = tc_end_candidate if tc_end_candidate != -1 else len(current_text)
            fn_close = _find_function_close(
                current_text,
                self._func_body_start,
                fn_search_limit,
                param_prefix=self.parameter_prefix,
                param_end=self.parameter_end_token,
                func_end=self.function_end_token,
                env_end=self.tool_call_end_token if self._in_tool_wrapper else None,
                allow_unclosed_param=True,
            )
            tc_end = tc_end_candidate if (self._in_tool_wrapper and fn_close == -1) else -1
            closed_by_tc = fn_close == -1 and tc_end != -1
            fn_limit = fn_close if fn_close != -1 else (tc_end if closed_by_tc else len(current_text))
            is_closed = fn_close != -1 or closed_by_tc

            arg_parts: list[str] = []
            tools_def = self.streaming_request.tools if self.streaming_request else None
            param_config = self._get_arguments_config(self.current_function_name or "", tools_def)

            while True:
                p_start = current_text.find(self.parameter_prefix, self._func_param_cursor, fn_limit)
                if p_start == -1:
                    break
                name_start = p_start + len(self.parameter_prefix)
                gt_idx = current_text.find(">", name_start, fn_limit)
                if gt_idx == -1:
                    break
                lt_before_gt = current_text.find("<", name_start, gt_idx)
                if lt_before_gt != -1:
                    self._func_param_cursor = lt_before_gt
                    continue
                val_start = gt_idx + 1
                p_end = current_text.find(self.parameter_end_token, val_start, fn_limit)
                next_p = current_text.find(self.parameter_prefix, val_start, fn_limit)
                if p_end != -1 and (next_p == -1 or p_end < next_p):
                    next_cursor = p_end + len(self.parameter_end_token)
                elif next_p != -1:
                    p_end = next_p
                    next_cursor = next_p
                elif is_closed:
                    p_end = fn_limit
                    next_cursor = fn_limit
                else:
                    break

                param_name = current_text[name_start:gt_idx]
                param_value = current_text[val_start:p_end]
                if param_value.startswith("\n"):
                    param_value = param_value[1:]
                if param_value.endswith("\n"):
                    param_value = param_value[:-1]

                self.current_param_name = param_name
                self.current_param_value = param_value
                converted = self._convert_param_value(
                    param_value,
                    param_name,
                    param_config,
                    self.current_function_name or "",
                )
                self.accumulated_params[param_name] = converted

                if not self.json_started:
                    self.json_started = True
                    arg_parts.append("{")
                encoded_key = json.dumps(param_name, ensure_ascii=False)
                encoded_val = json.dumps(converted, ensure_ascii=False)
                if self.param_count == 0:
                    arg_parts.append(f"{encoded_key}: {encoded_val}")
                else:
                    arg_parts.append(f", {encoded_key}: {encoded_val}")
                self.param_count += 1
                self._func_param_cursor = next_cursor

            if is_closed:
                if not self.json_started:
                    self.json_started = True
                    arg_parts.append("{")
                arg_parts.append("}")
                combined = "".join(arg_parts)
                tool_idx = self.current_tool_index
                self._append_arg_delta(tool_deltas, tool_idx, combined)
                if tool_idx < len(self.prev_tool_call_arr):
                    self.prev_tool_call_arr[tool_idx]["arguments"] = json.dumps(
                        self.accumulated_params, ensure_ascii=False
                    )
                self.in_function = False
                self.json_closed = True
                if fn_close != -1:
                    self._cursor = fn_close + len(self.function_end_token)
                else:
                    self._cursor = tc_end
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.accumulated_params = {}
                if not self._in_tool_wrapper:
                    self.is_tool_call_started = False
                continue

            if arg_parts:
                self._append_arg_delta(tool_deltas, self.current_tool_index, "".join(arg_parts))
            break

        content = "".join(content_parts)
        if content or tool_deltas:
            return DeltaMessage(content=content or None, tool_calls=tool_deltas or None)
        return None

    def _try_open_function(
        self,
        current_text: str,
        limit: int,
        tool_deltas: list[DeltaToolCall],
    ) -> str:
        """Open a ``<function=name>`` at ``self._cursor`` if valid ``>`` has arrived.

        Returns ``"opened"``, ``"incomplete"``, or ``"invalid"``.
        """
        name_start = self._cursor + len(self.tool_call_prefix)
        gt_idx = current_text.find(">", name_start, limit)
        scan_end = gt_idx if gt_idx != -1 else limit
        candidate_name = current_text[name_start:scan_end]
        if any(ch in candidate_name for ch in "<\r\n"):
            return "invalid"
        if gt_idx == -1:
            return "incomplete"
        function_name = candidate_name
        if not self._is_valid_function_name(function_name):
            return "invalid"
        self.current_function_name = function_name
        self.current_tool_id = self._generate_tool_call_id()
        self.header_sent = True
        self.current_tool_name_sent = True
        self.in_function = True
        self.json_started = False
        self.json_closed = False
        self.param_count = 0
        self.accumulated_params = {}
        self._func_body_start = gt_idx + 1
        self._func_param_cursor = gt_idx + 1

        while len(self.prev_tool_call_arr) <= self.current_tool_index:
            self.prev_tool_call_arr.append({"name": function_name, "arguments": "{}"})
        self.prev_tool_call_arr[self.current_tool_index] = {"name": function_name, "arguments": "{}"}
        while len(self.streamed_args_for_tool) <= self.current_tool_index:
            self.streamed_args_for_tool.append("")

        tool_deltas.append(
            DeltaToolCall(
                index=self.current_tool_index,
                id=self.current_tool_id,
                function=DeltaFunctionCall(name=function_name, arguments=""),
                type="function",
            )
        )
        return "opened"

    def _has_body_activity(self, current_text: str, limit: int) -> bool:
        """Return whether the newly opened function already has parameters or a closing tag."""
        return (
            current_text.find(self.parameter_prefix, self._func_body_start, limit) != -1
            or current_text.find(self.function_end_token, self._func_body_start, limit) != -1
            or (self._in_tool_wrapper and current_text.find(self.tool_call_end_token, self._func_body_start, limit) != -1)
        )
