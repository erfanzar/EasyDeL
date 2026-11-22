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

"""Qwen3 XML tool parser module.

Supports both ``<tool_call>``-wrapped and bare XML tool calls:
    - ``<tool_call><function=lookup>...</function></tool_call>``
    - ``<function=lookup>...</function>``
    - ``<function name="lookup">...</function>``
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

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
from .qwen3coder_tool_parser import Qwen3CoderToolParser, _trailing_prefix_len_any

_PARAM_OPEN_RE = re.compile(
    r"<parameter(?:=([^<>\s]+)|\s+name=\"([^\"<>]+)\"|\s+name='([^'<>]+)')\s*>",
    re.DOTALL,
)


def _find_xml_function_close(
    text: str,
    start_after_open: int,
    limit: int,
    *,
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
        close_fn = text.find("</function>", pos, limit)
        param_match = _PARAM_OPEN_RE.search(text, pos, limit)
        if close_fn != -1 and (param_match is None or close_fn < param_match.start()):
            return close_fn
        if param_match is None:
            return -1
        val_start = param_match.end()
        param_end = text.find("</parameter>", val_start, limit)
        next_param = _PARAM_OPEN_RE.search(text, val_start, limit)
        next_param_pos = next_param.start() if next_param is not None else -1
        close_fn2 = text.find("</function>", val_start, limit)
        candidates: list[tuple[int, str]] = []
        if param_end != -1:
            candidates.append((param_end, "param_end"))
        if next_param_pos != -1:
            candidates.append((next_param_pos, "next_param"))
        if close_fn2 != -1:
            candidates.append((close_fn2, "func_end"))
        if not candidates:
            return -1
        earliest_pos, kind = min(candidates, key=lambda pair: pair[0])
        if kind == "param_end":
            pos = earliest_pos + len("</parameter>")
            continue
        if kind == "next_param":
            pos = earliest_pos
            continue
        return earliest_pos if allow_unclosed_param else -1
    return -1


@ToolParserManager.register_module(["qwen3_xml"])  # pyright: ignore[reportUntypedClassDecorator]
class Qwen3XMLToolParser(Qwen3CoderToolParser):
    """Dedicated parser for Qwen3 XML outputs.

    Unlike ``qwen3_coder``, this parser does not require ``<tool_call>``/``</tool_call>``
    to exist as single tokenizer vocabulary entries, accepts both wrapped and bare
    ``<function=...>`` / ``<function name="...">`` calls (even mixed together), and
    preserves ``</function>`` appearing inside ``<parameter>`` values.
    """

    _OPEN_PREFIXES = ("<tool_call>", "<function=", "<function ")

    def __init__(self, tokenizer):
        """Initialize the Qwen3 XML tool parser without requiring ``<tool_call>`` in vocab."""
        ToolParser.__init__(self, tokenizer)

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
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

        self.tool_call_complete_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL)
        self.tool_call_function_regex = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )
        self.function_start_regex = re.compile(
            r"<function(?:=([^<>\s]+)|\s+name=\"([^\"<>]+)\"|\s+name='([^'<>]+)')\s*>",
            re.DOTALL,
        )
        self.parameter_regex = re.compile(
            r"<parameter(?:=([^<>\s]+)|\s+name=\"([^\"<>]+)\"|\s+name='([^'<>]+)')\s*>(.*?)(?:</parameter>|(?=<parameter(?:=|\s+name=))|(?=</function>)|$)",
            re.DOTALL,
        )

        # Optional token IDs (never raise RuntimeError when absent).
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        self._reset_streaming_state()

    @staticmethod
    def _first_nonempty(*values: str | None) -> str | None:
        """Return the first non-empty string from the arguments, or None."""
        for value in values:
            if value:
                return value
        return None

    def _find_first_tool_start(self, text: str, start: int = 0) -> int:
        """Find the earliest ``<tool_call>`` or ``<function=`` / ``<function `` tag at/after *start*."""
        indices = [
            idx
            for idx in (
                text.find(self.tool_call_start_token, start),
                text.find("<function=", start),
                text.find("<function ", start),
            )
            if idx >= 0
        ]
        return min(indices) if indices else -1

    def get_streaming_buffer_marker_pairs(self) -> Sequence[tuple[str, str | None]]:
        """Scope protocol buffering to top-level ``<tool_call>``, ``<function=``, and ``<function name=`` markers."""
        return (
            (self.tool_call_start_token, self.tool_call_end_token),
            (self.tool_call_prefix, self.function_end_token),
            ("<function name=", self.function_end_token),
        )

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Return True while inside an active tool/function block or trailing open-marker prefix."""
        if self._streaming_initialized:
            if self.is_tool_call_started or self._in_tool_wrapper or self.in_function:
                return True
            rem = current_text[self._cursor :]
            return _trailing_prefix_len_any(rem, self._OPEN_PREFIXES) > 0
        return super().is_buffering_protocol(current_text=current_text, delta_text=delta_text)

    def flush_streaming_content(self) -> str:
        """Release withheld prose or an unfinished non-confirmed bare ``<function`` prefix at EOF."""
        self.streaming_request = None
        if self._in_tool_wrapper:
            self.accumulated_text = ""
            self._cursor = 0
            return ""
        if self.in_function:
            if self.function_start_regex.match(self.accumulated_text, self._cursor) is not None:
                self.accumulated_text = ""
                self._cursor = 0
                return ""
        elif self.is_tool_call_started:
            self.accumulated_text = ""
            self._cursor = 0
            return ""
        pending = self.accumulated_text[self._cursor :]
        self.accumulated_text = ""
        self._cursor = 0
        self.in_function = False
        self.is_tool_call_started = False
        return pending

    @staticmethod
    def _is_possible_function_open_prefix(tail: str) -> bool:
        """Return True if *tail* (starting with ``<function``) could still complete a valid opening tag."""
        if tail.startswith("<function="):
            rest = tail[len("<function=") :]
            if "<" in rest or ">" in rest or "\n" in rest or "\r" in rest:
                return False
            if not rest:
                return True
            if rest[0].isspace():
                return False
            return len(rest.split()) <= 1
        if tail.startswith("<function") and len(tail) > len("<function") and tail[len("<function")].isspace():
            after_ws = tail[len("<function") :].lstrip()
            if not after_ws:
                return True
            if "name=".startswith(after_ws):
                return True
            if after_ws.startswith("name="):
                val = after_ws[len("name=") :]
                if not val:
                    return True
                quote = val[0]
                if quote not in {'"', "'"}:
                    return False
                inner = val[1:]
                close_q = inner.find(quote)
                if close_q == -1:
                    return "<" not in inner and ">" not in inner
                after_q = inner[close_q + 1 :]
                return ">" not in after_q and after_q.strip() == ""
            return False
        return False

    def _extract_function_name_from_match(self, match: re.Match[str]) -> str | None:
        """Extract the function name from a ``function_start_regex`` match."""
        return self._first_nonempty(match.group(1), match.group(2), match.group(3))

    def _extract_parameter_name_from_match(self, match: re.Match[str]) -> str | None:
        """Extract the parameter name from a ``_PARAM_OPEN_RE`` / ``parameter_regex`` match."""
        return self._first_nonempty(match.group(1), match.group(2), match.group(3))

    def _iter_function_blocks_from_chunk(
        self,
        chunk: str,
        *,
        complete_only: bool,
    ) -> list[str]:
        """Extract function block substrings from *chunk*, respecting ``<parameter>`` boundaries."""
        function_blocks: list[str] = []
        pos = 0
        n = len(chunk)
        while pos < n:
            start_match = self.function_start_regex.search(chunk, pos)
            if start_match is None:
                break
            close_idx = _find_xml_function_close(
                chunk,
                start_match.end(),
                n,
                env_end=self.tool_call_end_token,
            )
            if close_idx != -1:
                end_idx = close_idx + len(self.function_end_token)
                function_blocks.append(chunk[start_match.start() : end_idx])
                pos = end_idx
                continue
            tc_close = chunk.find(self.tool_call_end_token, start_match.end())
            next_match = self.function_start_regex.search(chunk, start_match.end())
            if tc_close != -1 and (next_match is None or tc_close < next_match.start()):
                function_blocks.append(chunk[start_match.start() : tc_close])
                pos = tc_close + len(self.tool_call_end_token)
                continue
            if complete_only:
                break
            # Incomplete trailing block: extend to next function open or end of chunk.
            end_idx = next_match.start() if next_match is not None else n
            function_blocks.append(chunk[start_match.start() : end_idx])
            pos = end_idx
        return function_blocks

    def _iter_function_blocks(self, text: str, *, complete_only: bool) -> list[str]:
        """Extract all wrapped and bare function blocks from *text* in document order."""
        return self._iter_function_blocks_from_chunk(text, complete_only=complete_only)

    def _parse_xml_function_call(self, function_block: str, tools) -> ToolCall | None:
        """Parse an XML function block into a ToolCall object."""
        start_match = self.function_start_regex.search(function_block)
        if start_match is None:
            return None

        function_name = self._extract_function_name_from_match(start_match)
        if not function_name:
            return None

        parameters_text = function_block[start_match.end() :]
        if parameters_text.endswith(self.function_end_token):
            parameters_text = parameters_text[: -len(self.function_end_token)]
        elif parameters_text.endswith(self.tool_call_end_token):
            parameters_text = parameters_text[: -len(self.tool_call_end_token)]

        param_config = self._get_arguments_config(function_name, tools)
        param_dict = {}
        pos = 0
        n = len(parameters_text)
        while pos < n:
            match = _PARAM_OPEN_RE.search(parameters_text, pos)
            if match is None:
                break
            param_name = self._extract_parameter_name_from_match(match)
            val_start = match.end()
            p_close = parameters_text.find(self.parameter_end_token, val_start)
            next_match = _PARAM_OPEN_RE.search(parameters_text, val_start)
            next_pos = next_match.start() if next_match is not None else -1
            fn_close = parameters_text.find(self.function_end_token, val_start)
            tc_close = parameters_text.find(self.tool_call_end_token, val_start)
            candidates: list[tuple[int, str]] = []
            if p_close != -1:
                candidates.append((p_close, "param_end"))
            if next_pos != -1:
                candidates.append((next_pos, "next_param"))
            if fn_close != -1:
                candidates.append((fn_close, "func_end"))
            if tc_close != -1:
                candidates.append((tc_close, "env_end"))
            if candidates:
                earliest_pos, kind = min(candidates, key=lambda pair: pair[0])
                param_value = parameters_text[val_start:earliest_pos]
                if kind == "param_end":
                    pos = earliest_pos + len(self.parameter_end_token)
                elif kind == "next_param":
                    pos = earliest_pos
                else:
                    pos = n
            else:
                param_value = parameters_text[val_start:]
                pos = n

            if not param_name:
                continue
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = self._convert_param_value(param_value, param_name, param_config, function_name)

        return ToolCall(
            type="function",
            function=FunctionCall(name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False)),
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete (non-streaming) model output (stateless)."""
        if "<function" not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            function_blocks = self._iter_function_blocks(model_output, complete_only=False)
            if not function_blocks:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            tools = request.tools if request else None
            tool_calls = [self._parse_xml_function_call(function_block, tools) for function_block in function_blocks]
            valid_tool_calls = [tool_call for tool_call in tool_calls if tool_call is not None]
            if not valid_tool_calls:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            first_fn_match = self.function_start_regex.search(model_output)
            content_index = first_fn_match.start() if first_fn_match is not None else self._find_first_tool_start(model_output)
            prefix_before_fn = model_output[:content_index].rstrip()
            if prefix_before_fn.endswith(self.tool_call_start_token):
                content_index = len(prefix_before_fn) - len(self.tool_call_start_token)
            content = model_output[:content_index] if content_index > 0 else None
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=valid_tool_calls,
                content=content,
            )
        except Exception:
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
        """Extract tool calls incrementally from streaming model output."""
        del previous_token_ids, current_token_ids

        if not previous_text:
            self._reset_streaming_state()
            self._streaming_initialized = True
        elif not self._streaming_initialized:
            self._reset_streaming_state()
            self._cursor = len(previous_text)
            self._streaming_initialized = True

        self.accumulated_text = current_text

        content_parts: list[str] = []
        delta_tool_calls: list[DeltaToolCall] = []
        tools = request.tools if request else None

        while True:
            if not self._in_tool_wrapper and not self.in_function:
                start_idx = self._find_first_tool_start(current_text, self._cursor)
                if start_idx != -1:
                    if start_idx > self._cursor:
                        content_parts.append(current_text[self._cursor : start_idx])
                    self.is_tool_call_started = True
                    if current_text.startswith(self.tool_call_start_token, start_idx):
                        self._in_tool_wrapper = True
                        self._cursor = start_idx + len(self.tool_call_start_token)
                    else:
                        self.in_function = True
                        self._cursor = start_idx
                    continue

                rem = current_text[self._cursor :]
                hold = _trailing_prefix_len_any(rem, self._OPEN_PREFIXES)
                safe_end = len(current_text) - hold
                if safe_end > self._cursor:
                    content_parts.append(current_text[self._cursor : safe_end])
                    self._cursor = safe_end
                break

            if self._in_tool_wrapper and not self.in_function:
                fn_match = self.function_start_regex.search(current_text, self._cursor)
                tc_end = current_text.find(self.tool_call_end_token, self._cursor)
                if tc_end != -1 and (fn_match is None or tc_end < fn_match.start()):
                    self._in_tool_wrapper = False
                    self.is_tool_call_started = False
                    self._cursor = tc_end + len(self.tool_call_end_token)
                    continue
                if fn_match is not None:
                    self.in_function = True
                    self._cursor = fn_match.start()
                    continue
                break

            # Inside a function starting at self._cursor.
            start_match = self.function_start_regex.match(current_text, self._cursor)
            if start_match is None:
                tail = current_text[self._cursor :]
                if self._is_possible_function_open_prefix(tail):
                    break
                # Not a valid function open tag (e.g. "<function invalid>"): release "<function" and rescan.
                advance = len("<function") if tail.startswith("<function") else 1
                if not self._in_tool_wrapper:
                    self.in_function = False
                    self.is_tool_call_started = False
                    content_parts.append(current_text[self._cursor : self._cursor + advance])
                    self._cursor += advance
                    continue
                self.in_function = False
                self._cursor += advance
                continue

            tc_end = (
                current_text.find(self.tool_call_end_token, start_match.end())
                if self._in_tool_wrapper
                else -1
            )
            search_limit = tc_end if tc_end != -1 else len(current_text)
            close_idx = _find_xml_function_close(
                current_text,
                start_match.end(),
                search_limit,
                env_end=self.tool_call_end_token if self._in_tool_wrapper else None,
                allow_unclosed_param=True,
            )
            if close_idx != -1:
                block_end = close_idx + len(self.function_end_token)
                function_block = current_text[start_match.start() : block_end]
            elif tc_end != -1:
                block_end = tc_end
                function_block = current_text[start_match.start() : tc_end]
            else:
                break
            parsed_tool = self._parse_xml_function_call(function_block, tools)
            if parsed_tool is not None:
                tool_idx = self.current_tool_index
                self.current_tool_index += 1
                tool_id = self._generate_tool_call_id()
                delta_tool_calls.append(
                    DeltaToolCall(
                        index=tool_idx,
                        id=tool_id,
                        function=DeltaFunctionCall(
                            name=parsed_tool.function.name,
                            arguments=parsed_tool.function.arguments,
                        ),
                        type="function",
                    )
                )
                self.prev_tool_call_arr.append(
                    {
                        "name": parsed_tool.function.name,
                        "arguments": parsed_tool.function.arguments,
                    }
                )
                self.streamed_args_for_tool.append(parsed_tool.function.arguments)

            self.in_function = False
            self._cursor = block_end
            if not self._in_tool_wrapper:
                self.is_tool_call_started = False
            continue

        content = "".join(content_parts)
        if content or delta_tool_calls:
            return DeltaMessage(content=content or None, tool_calls=delta_tool_calls or None)
        if not delta_text and delta_token_ids and self.prev_tool_call_arr:
            return DeltaMessage(content="")
        return None
