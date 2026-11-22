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

"""Tool parser implementation for Seed OSS models.

Example format:
    <seed:think>reasoning content</seed:think>
    <seed:tool_call>
    <function=function_name>
    <parameter=param_name>value</parameter>
    </function>
    </seed:tool_call>
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence

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
from .qwen3coder_tool_parser import _find_function_close, _iter_parameters, _trailing_prefix_len_any

logger = get_logger(__name__)


def _find_tag_skipping_parameters(
    text: str,
    target_tag: str,
    start: int,
    limit: int,
    *,
    param_prefix: str = "<parameter=",
    param_end: str = "</parameter>",
    func_end: str = "</function>",
    env_end: str = "</seed:tool_call>",
    allow_unclosed_param: bool = True,
) -> int:
    """Find *target_tag* in ``text[start:limit]`` outside ``<parameter=...>`` values.

    Each parameter value ends at the earliest of ``param_end``, ``param_prefix``,
    ``func_end``, or ``env_end``.
    """
    pos = start
    while pos < limit:
        target_idx = text.find(target_tag, pos, limit)
        if target_idx == -1:
            return -1
        param_idx = text.find(param_prefix, pos, target_idx)
        if param_idx == -1:
            return target_idx
        name_start = param_idx + len(param_prefix)
        gt_idx = text.find(">", name_start, limit)
        if gt_idx == -1:
            return -1
        lt_idx = text.find("<", name_start, gt_idx)
        if lt_idx != -1:
            pos = lt_idx
            continue
        val_start = gt_idx + 1
        p_close = text.find(param_end, val_start, limit)
        next_p = text.find(param_prefix, val_start, limit)
        fn_close = text.find(func_end, val_start, limit)
        tc_close = text.find(env_end, val_start, limit)
        candidates: list[tuple[int, str]] = []
        if p_close != -1:
            candidates.append((p_close, "param_end"))
        if next_p != -1:
            candidates.append((next_p, "next_param"))
        if fn_close != -1:
            candidates.append((fn_close, "func_end"))
        if tc_close != -1:
            candidates.append((tc_close, "env_end"))
        if not candidates:
            return -1
        earliest_pos, kind = min(candidates, key=lambda pair: pair[0])
        if kind == "param_end":
            pos = earliest_pos + len(param_end)
            continue
        if kind == "next_param":
            pos = earliest_pos
            continue
        if not allow_unclosed_param:
            return -1
        if earliest_pos == target_idx:
            return target_idx
        pos = earliest_pos + (len(func_end) if kind == "func_end" else len(env_end))
    return -1


@ToolParserManager.register_module("seed_oss")  # pyright: ignore[reportUntypedClassDecorator]
class SeedOssToolParser(ToolParser):
    """Tool parser for Seed OSS model outputs.

    Handles ``<seed:think>...</seed:think>`` (including unclosed, repeated, and
    prompt-injected thinking), ``<seed:tool_call>...</seed:tool_call>``, and
    bare ``<function=name>...</function>`` with schema-aware parameter coercion.
    """

    TOOL_CALL_START = "<seed:tool_call>"
    TOOL_CALL_END = "</seed:tool_call>"

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the Seed OSS tool parser."""
        super().__init__(tokenizer)

        self.prev_tool_call_arr: list[dict] = []

        self.tool_call_start_token: str = self.TOOL_CALL_START
        self.tool_call_end_token: str = self.TOOL_CALL_END
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"
        self.think_start_token: str = "<seed:think>"
        self.think_end_token: str = "</seed:think>"
        self.is_tool_call_started: bool = False
        self.is_thinking_end: bool = False
        self.failed_count: int = 0
        self._reset_streaming_state()

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        self.think_end_token_id = self.vocab.get(self.think_end_token)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError("Seed_Oss XML parser: tokenizer did not include <seed:tool_call> or its closing tag.")

        tool_start_re = re.escape(self.tool_call_start_token)
        tool_end_re = re.escape(self.tool_call_end_token)

        self.tool_call_complete_regex = re.compile(rf"{tool_start_re}(.*?){tool_end_re}", re.DOTALL)
        self.tool_call_regex = re.compile(rf"{tool_start_re}(.*?){tool_end_re}|{tool_start_re}(.*?)$", re.DOTALL)
        self.tool_call_function_regex = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

    def get_streaming_buffer_marker_pairs(self) -> Sequence[tuple[str, str | None]]:
        """Scope protocol buffering to top-level ``<seed:tool_call>`` and ``<function=`` markers."""
        return (
            (self.tool_call_start_token, self.tool_call_end_token),
            (self.tool_call_prefix, self.function_end_token),
        )

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state variables."""
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = -1
        self.current_function_name = None
        self.current_param_name = None
        self.param_count = 0
        self.in_function = False
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool = []
        self._stream_calls: list[dict] = []
        self._stream_content_sent = 0
        self._streaming_initialized = False
        self.is_thinking_end = False
        self._scan_pos = 0
        self._in_think_block = False
        self._think_search_pos = 0
        self._saw_any_think_open = False
        self._saw_any_tool_region = False
        self._closed_emit_index = 0

    def _scan_regions(
        self,
        text: str,
        *,
        streaming: bool = False,
    ) -> tuple[list[tuple[int, int | None, str]], bool, bool]:
        """Scan *text* for tool regions outside ``<seed:think>`` blocks.

        Handles:
        - Prompt-injected thinking (leading ``</seed:think>`` outside parameters before any ``<seed:think>``)
        - Closed and repeated ``<seed:think>...</seed:think>`` blocks
        - Unclosed ``<seed:think>...`` blocks (suppresses tool calls inside)
        - ``<seed:think>`` / ``</seed:think>`` appearing inside ``<parameter=...>`` values
        - Both ``<seed:tool_call>...</seed:tool_call>`` and bare ``<function=...>...</function>``

        Returns ``(regions, active_thinking, saw_think_end)`` where each region is
        ``(start_idx, end_idx_or_None, kind)`` with ``kind`` in ``("envelope", "bare")``.
        """
        regions: list[tuple[int, int | None, str]] = []
        n = len(text)
        pos = 0
        saw_think_end = False

        # Detect prompt-injected thinking: an unmatched </seed:think> outside parameter values
        # occurring before any <seed:think> opening tag.
        del streaming
        first_think_open = _find_tag_skipping_parameters(
            text,
            self.think_start_token,
            0,
            n,
            param_prefix=self.parameter_prefix,
            param_end=self.parameter_end_token,
            func_end=self.function_end_token,
            env_end=self.tool_call_end_token,
            allow_unclosed_param=True,
        )
        first_think_close = _find_tag_skipping_parameters(
            text,
            self.think_end_token,
            0,
            n,
            param_prefix=self.parameter_prefix,
            param_end=self.parameter_end_token,
            func_end=self.function_end_token,
            env_end=self.tool_call_end_token,
            allow_unclosed_param=True,
        )
        if first_think_close != -1 and (first_think_open == -1 or first_think_close < first_think_open):
            pos = first_think_close + len(self.think_end_token)
            saw_think_end = True

        while pos < n:
            next_think = text.find(self.think_start_token, pos)
            next_env = text.find(self.tool_call_start_token, pos)
            next_bare = text.find(self.tool_call_prefix, pos)

            candidates = [
                (idx, kind)
                for idx, kind in ((next_think, "think"), (next_env, "envelope"), (next_bare, "bare"))
                if idx != -1
            ]
            if not candidates:
                break
            start_idx, kind = min(candidates, key=lambda pair: pair[0])

            if kind == "think":
                close_think = text.find(self.think_end_token, start_idx + len(self.think_start_token))
                if close_think == -1:
                    return regions, True, saw_think_end
                saw_think_end = True
                pos = close_think + len(self.think_end_token)
                continue

            if kind == "envelope":
                body_start = start_idx + len(self.tool_call_start_token)
                end_env = _find_tag_skipping_parameters(
                    text,
                    self.tool_call_end_token,
                    body_start,
                    n,
                    param_prefix=self.parameter_prefix,
                    param_end=self.parameter_end_token,
                    func_end=self.function_end_token,
                    env_end=self.tool_call_end_token,
                    allow_unclosed_param=True,
                )
                if end_env == -1:
                    regions.append((start_idx, None, "envelope"))
                    break
                end_after = end_env + len(self.tool_call_end_token)
                regions.append((start_idx, end_after, "envelope"))
                pos = end_after
                continue

            # kind == "bare": bare <function=...>...</function>
            name_start = start_idx + len(self.tool_call_prefix)
            gt_idx = text.find(">", name_start)
            if gt_idx == -1:
                candidate = text[name_start:]
                bad_lt = candidate.find("<")
                if bad_lt != -1:
                    pos = name_start + bad_lt
                    continue
                if "\n" in candidate or "\r" in candidate:
                    pos = name_start
                    continue
                regions.append((start_idx, None, "bare"))
                break
            lt_idx = text.find("<", name_start, gt_idx)
            if lt_idx != -1:
                pos = lt_idx
                continue
            fn_name = text[name_start:gt_idx]
            if not fn_name or not fn_name.strip() or any(ch in fn_name for ch in "<>\r\n"):
                pos = gt_idx + 1
                continue
            close_fn = _find_function_close(
                text,
                gt_idx + 1,
                n,
                param_prefix=self.parameter_prefix,
                param_end=self.parameter_end_token,
                func_end=self.function_end_token,
                env_end=self.tool_call_end_token,
                allow_unclosed_param=True,
            )
            if close_fn == -1:
                regions.append((start_idx, None, "bare"))
                break
            end_after = close_fn + len(self.function_end_token)
            regions.append((start_idx, end_after, "bare"))
            pos = end_after

        return regions, False, saw_think_end

    def _extract_functions_from_region(
        self,
        region_text: str,
        kind: str,
        *,
        allow_unclosed_param: bool = True,
        region_closed: bool = False,
    ) -> list[tuple[str, str, bool]]:
        """Extract ``(func_name, params_text, is_closed)`` tuples from one scanned region."""
        if kind == "envelope":
            body = region_text[len(self.tool_call_start_token) :]
            if body.endswith(self.tool_call_end_token):
                body = body[: -len(self.tool_call_end_token)]
        else:
            body = region_text

        functions: list[tuple[str, str, bool]] = []
        pos = 0
        n = len(body)
        while pos < n:
            fn_start = body.find(self.tool_call_prefix, pos)
            if fn_start == -1:
                break
            name_start = fn_start + len(self.tool_call_prefix)
            gt_idx = body.find(">", name_start)
            if gt_idx == -1:
                break
            lt_idx = body.find("<", name_start, gt_idx)
            if lt_idx != -1:
                pos = lt_idx
                continue
            fn_name = body[name_start:gt_idx]
            if not fn_name or not fn_name.strip() or any(ch in fn_name for ch in "<>\r\n"):
                pos = gt_idx + 1
                continue
            close_fn = _find_function_close(
                body,
                gt_idx + 1,
                n,
                param_prefix=self.parameter_prefix,
                param_end=self.parameter_end_token,
                func_end=self.function_end_token,
                env_end=self.tool_call_end_token,
                allow_unclosed_param=allow_unclosed_param,
            )
            if close_fn != -1:
                functions.append((fn_name, body[gt_idx + 1 : close_fn], True))
                pos = close_fn + len(self.function_end_token)
            else:
                functions.append((fn_name, body[gt_idx + 1 :], region_closed))
                break
        return functions

    def _parse_xml_function_call(
        self,
        function_call_str: str,
        tools: list[ToolDefinition] | None,
    ) -> ToolCall | None:
        """Parse an XML-formatted function call string into a ToolCall object."""
        end_index = function_call_str.find(">")
        if end_index == -1:
            return None
        function_name = function_call_str[:end_index]
        if not function_name or not function_name.strip() or any(ch in function_name for ch in "<>\r\n"):
            return None
        param_config = get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        for param_name, param_value in _iter_parameters(
            parameters,
            param_prefix=self.parameter_prefix,
            param_end=self.parameter_end_token,
        ):
            try:
                param_dict[param_name] = convert_param_value(param_value, param_name, param_config, function_name)
            except (RecursionError, MemoryError):
                param_dict[param_name] = param_value
        return ToolCall(
            type="function",
            function=FunctionCall(name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False)),
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model output (stateless)."""
        if self.tool_call_prefix not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            regions, _active_thinking, _saw_think_end = self._scan_regions(model_output)
            if not regions:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            tools = request.tools if request else None
            tool_calls: list[ToolCall] = []
            first_valid_region_start: int | None = None

            for start_idx, end_idx, kind in regions:
                region_text = model_output[start_idx : end_idx if end_idx is not None else len(model_output)]
                region_calls: list[ToolCall] = []
                for fn_name, params_text, _closed in self._extract_functions_from_region(
                    region_text,
                    kind,
                    region_closed=end_idx is not None,
                ):
                    parsed = self._parse_xml_function_call(f"{fn_name}>{params_text}", tools)
                    if parsed is not None:
                        region_calls.append(parsed)
                if region_calls:
                    if first_valid_region_start is None:
                        first_valid_region_start = start_idx
                    tool_calls.extend(region_calls)

            if not tool_calls or first_valid_region_start is None:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            content = model_output[:first_valid_region_start]
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    def is_buffering_protocol(self, *, current_text: str, delta_text: str) -> bool:
        """Return True for unclosed tool regions or partial opening markers outside ``<seed:think>``."""
        del delta_text
        if self._streaming_initialized:
            if self.is_tool_call_started:
                return True
            if self._in_think_block:
                return False
            return (
                _trailing_prefix_len_any(
                    current_text[self._stream_content_sent :],
                    (self.tool_call_start_token, self.tool_call_prefix),
                )
                > 0
            )
        regions, active_thinking, _saw = self._scan_regions(current_text)
        if regions and regions[-1][1] is None:
            return True
        if active_thinking:
            return False
        tail_start = regions[-1][1] if regions and regions[-1][1] is not None else 0
        return _trailing_prefix_len_any(current_text[tail_start:], (self.tool_call_start_token, self.tool_call_prefix)) > 0

    def flush_streaming_content(self) -> str:
        """Release a withheld, unfinished opening marker when no tool call followed."""
        if self.is_tool_call_started:
            self.accumulated_text = ""
            self._stream_content_sent = 0
            return ""
        pending = self.accumulated_text[self._stream_content_sent :]
        self.accumulated_text = ""
        self._stream_content_sent = 0
        return pending

    def _scan_regions_incremental(self, text: str) -> tuple[list[tuple[int, int | None, str]], bool]:
        """Incrementally scan *text* from ``self._scan_pos`` for new/open tool regions."""
        new_regions: list[tuple[int, int | None, str]] = []
        n = len(text)
        pos = self._scan_pos
        max_marker_len = max(
            len(self.think_start_token),
            len(self.think_end_token),
            len(self.tool_call_start_token),
            len(self.tool_call_prefix),
        )

        # Resume an active unclosed <seed:think> block without rescanning from 0.
        if self._in_think_block:
            close_think = text.find(self.think_end_token, self._think_search_pos)
            if close_think == -1:
                self._think_search_pos = max(self._think_search_pos, n - len(self.think_end_token))
                return new_regions, True
            self.is_thinking_end = True
            self._in_think_block = False
            pos = close_think + len(self.think_end_token)
            self._scan_pos = pos

        # Detect prompt-injected leading </seed:think> arriving before any <seed:think> or tool call.
        if (
            not self._saw_any_think_open
            and not self._saw_any_tool_region
            and pos == 0
            and self.think_end_token in text
        ):
            first_think_open = _find_tag_skipping_parameters(
                text,
                self.think_start_token,
                0,
                n,
                param_prefix=self.parameter_prefix,
                param_end=self.parameter_end_token,
                func_end=self.function_end_token,
                env_end=self.tool_call_end_token,
                allow_unclosed_param=True,
            )
            first_think_close = _find_tag_skipping_parameters(
                text,
                self.think_end_token,
                0,
                n,
                param_prefix=self.parameter_prefix,
                param_end=self.parameter_end_token,
                func_end=self.function_end_token,
                env_end=self.tool_call_end_token,
                allow_unclosed_param=True,
            )
            if first_think_close != -1 and (first_think_open == -1 or first_think_close < first_think_open):
                pos = first_think_close + len(self.think_end_token)
                self._scan_pos = pos
                self.is_thinking_end = True

        while pos < n:
            next_think = text.find(self.think_start_token, pos)
            next_env = text.find(self.tool_call_start_token, pos)
            next_bare = text.find(self.tool_call_prefix, pos)
            next_unmatched_close = (
                text.find(self.think_end_token, pos)
                if not self._saw_any_think_open and not self._saw_any_tool_region
                else -1
            )

            candidates = [
                (idx, kind)
                for idx, kind in (
                    (next_unmatched_close, "unmatched_think_close"),
                    (next_think, "think"),
                    (next_env, "envelope"),
                    (next_bare, "bare"),
                )
                if idx != -1
            ]
            if not candidates:
                self._scan_pos = max(pos, n - max_marker_len)
                break
            start_idx, kind = min(candidates, key=lambda pair: pair[0])

            if kind == "unmatched_think_close":
                self.is_thinking_end = True
                pos = start_idx + len(self.think_end_token)
                self._scan_pos = pos
                continue

            if kind == "think":
                self._saw_any_think_open = True
                body_start = start_idx + len(self.think_start_token)
                close_think = text.find(self.think_end_token, body_start)
                if close_think == -1:
                    self._in_think_block = True
                    self._think_search_pos = max(body_start, n - len(self.think_end_token))
                    self._scan_pos = start_idx
                    return new_regions, True
                self.is_thinking_end = True
                pos = close_think + len(self.think_end_token)
                self._scan_pos = pos
                continue

            if kind == "envelope":
                self._saw_any_tool_region = True
                body_start = start_idx + len(self.tool_call_start_token)
                end_env = _find_tag_skipping_parameters(
                    text,
                    self.tool_call_end_token,
                    body_start,
                    n,
                    param_prefix=self.parameter_prefix,
                    param_end=self.parameter_end_token,
                    func_end=self.function_end_token,
                    env_end=self.tool_call_end_token,
                    allow_unclosed_param=True,
                )
                if end_env == -1:
                    new_regions.append((start_idx, None, "envelope"))
                    self._scan_pos = start_idx
                    break
                end_after = end_env + len(self.tool_call_end_token)
                new_regions.append((start_idx, end_after, "envelope"))
                pos = end_after
                self._scan_pos = end_after
                continue

            # kind == "bare": bare <function=...>...</function>
            name_start = start_idx + len(self.tool_call_prefix)
            gt_idx = text.find(">", name_start)
            if gt_idx == -1:
                candidate = text[name_start:]
                bad_lt = candidate.find("<")
                if bad_lt != -1:
                    pos = name_start + bad_lt
                    self._scan_pos = pos
                    continue
                if "\n" in candidate or "\r" in candidate:
                    pos = name_start
                    self._scan_pos = pos
                    continue
                new_regions.append((start_idx, None, "bare"))
                self._scan_pos = start_idx
                break
            lt_idx = text.find("<", name_start, gt_idx)
            if lt_idx != -1:
                pos = lt_idx
                self._scan_pos = pos
                continue
            fn_name = text[name_start:gt_idx]
            if not fn_name or not fn_name.strip() or any(ch in fn_name for ch in "<>\r\n"):
                pos = gt_idx + 1
                self._scan_pos = pos
                continue
            self._saw_any_tool_region = True
            close_fn = _find_function_close(
                text,
                gt_idx + 1,
                n,
                param_prefix=self.parameter_prefix,
                param_end=self.parameter_end_token,
                func_end=self.function_end_token,
                env_end=self.tool_call_end_token,
                allow_unclosed_param=True,
            )
            if close_fn == -1:
                new_regions.append((start_idx, None, "bare"))
                self._scan_pos = start_idx
                break
            end_after = close_fn + len(self.function_end_token)
            new_regions.append((start_idx, end_after, "bare"))
            pos = end_after
            self._scan_pos = end_after

        return new_regions, False

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
        del delta_text, previous_token_ids, current_token_ids, delta_token_ids
        if not previous_text:
            self._reset_streaming_state()
            self._streaming_initialized = True
        elif not self._streaming_initialized:
            self._reset_streaming_state()
            self._stream_content_sent = len(previous_text)
            self._streaming_initialized = True

        self.accumulated_text = current_text
        regions, active_thinking = self._scan_regions_incremental(current_text)

        in_open_region = bool(regions and regions[-1][1] is None)
        self.is_tool_call_started = in_open_region

        # Emit only content strictly outside tool regions.
        content_parts: list[str] = []
        for start_idx, end_idx, _kind in regions:
            if self._stream_content_sent < start_idx:
                content_parts.append(current_text[self._stream_content_sent : start_idx])
            if end_idx is not None:
                if self._stream_content_sent < end_idx:
                    self._stream_content_sent = end_idx
            else:
                self._stream_content_sent = start_idx
                break

        if not in_open_region and self._stream_content_sent < len(current_text):
            rem = current_text[self._stream_content_sent :]
            hold = (
                0
                if active_thinking
                else _trailing_prefix_len_any(rem, (self.tool_call_start_token, self.tool_call_prefix))
            )
            safe_end = len(current_text) - hold
            if safe_end > self._stream_content_sent:
                content_parts.append(current_text[self._stream_content_sent : safe_end])
                self._stream_content_sent = safe_end

        # Extract and stream function calls from newly closed and open regions only.
        deltas: list[DeltaToolCall] = []
        tools = request.tools if request else None
        emit_index = self._closed_emit_index
        for start_idx, end_idx, kind in regions:
            region_text = current_text[start_idx : end_idx if end_idx is not None else len(current_text)]
            region_functions = self._extract_functions_from_region(
                region_text,
                kind,
                allow_unclosed_param=True,
                region_closed=end_idx is not None,
            )
            for name, params_body, closed in region_functions:
                if not name or not name.strip() or any(ch in name for ch in "<>\r\n"):
                    continue
                index = emit_index
                emit_index += 1
                first = index >= len(self._stream_calls)
                if first:
                    self._stream_calls.append({"id": self._generate_tool_call_id(), "count": 0, "closed": False})
                    self.streamed_args_for_tool.append("")
                    self.prev_tool_call_arr.append({"name": name, "arguments": "{}"})
                state = self._stream_calls[index]
                fragment = "{" if first else ""

                # Extract completed parameters (explicit </parameter>, next <parameter=, or closed function).
                completed_params: list[tuple[str, str]] = []
                pos = 0
                n_body = len(params_body)
                while pos < n_body:
                    p_start = params_body.find(self.parameter_prefix, pos)
                    if p_start == -1:
                        break
                    p_name_start = p_start + len(self.parameter_prefix)
                    gt_idx = params_body.find(">", p_name_start)
                    if gt_idx == -1:
                        break
                    lt_idx = params_body.find("<", p_name_start, gt_idx)
                    if lt_idx != -1:
                        pos = lt_idx
                        continue
                    val_start = gt_idx + 1
                    p_close = params_body.find(self.parameter_end_token, val_start)
                    next_p = params_body.find(self.parameter_prefix, val_start)
                    if p_close != -1 and (next_p == -1 or p_close < next_p):
                        val = params_body[val_start:p_close]
                        pos = p_close + len(self.parameter_end_token)
                    elif next_p != -1:
                        val = params_body[val_start:next_p]
                        pos = next_p
                    elif closed:
                        val = params_body[val_start:]
                        pos = n_body
                    else:
                        break
                    if val.startswith("\n"):
                        val = val[1:]
                    if val.endswith("\n"):
                        val = val[:-1]
                    completed_params.append((params_body[p_name_start:gt_idx], val))

                config = get_arguments_config(name, tools)
                for key, value in completed_params[state["count"] :]:
                    try:
                        converted = convert_param_value(value, key, config, name)
                    except (RecursionError, MemoryError):
                        converted = value
                    fragment += (", " if state["count"] else "") + json.dumps(key, ensure_ascii=False)
                    fragment += ": " + json.dumps(converted, ensure_ascii=False)
                    state["count"] += 1

                if closed and not state["closed"]:
                    fragment += "}"
                    state["closed"] = True

                if first or fragment:
                    self.streamed_args_for_tool[index] += fragment
                    deltas.append(
                        DeltaToolCall(
                            index=index,
                            id=state["id"] if first else None,
                            type="function" if first else None,
                            function=DeltaFunctionCall(name=name if first else None, arguments=fragment or None),
                        )
                    )
                if closed:
                    self.prev_tool_call_arr[index]["arguments"] = self.streamed_args_for_tool[index]
                self.current_tool_index = index
                self.current_function_name = name
                self.current_tool_id = state["id"]
                self.header_sent = True
                self.json_started = True
                self.json_closed = state["closed"]
                self.in_function = not state["closed"]
                self.param_count = state["count"]
            if end_idx is not None:
                self._closed_emit_index = emit_index

        content = "".join(content_parts)
        return DeltaMessage(content=content or None, tool_calls=deltas or None) if content or deltas else None
