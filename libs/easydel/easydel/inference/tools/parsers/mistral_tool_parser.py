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
"""Mistral model tool call parser.

This module provides a specialized tool call parser for Mistral AI models
(7B Instruct v0.3 and later). The parser handles Mistral's specific tool call
format which uses the ``[TOOL_CALLS]`` token followed by a JSON array of
function calls or v11+ ``fn_name{...}`` invocations.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from random import choices
from string import ascii_letters, digits

import partial_json_parser  # pyright: ignore[reportMissingTypeStubs]
from eformer.loggings import get_logger
from partial_json_parser.core.options import Allow  # pyright: ignore[reportMissingTypeStubs]
from pydantic import Field
from transformers import AutoTokenizer as AnyTokenizer

try:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer  # pyright: ignore[reportMissingImports]
except ImportError:
    MistralTokenizer = None

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
from ..utils import consume_space, is_complete_json, partial_json_loads, stable_json_stream_prefix

logger = get_logger(__name__)

ALPHANUMERIC = ascii_letters + digits
_V11_FN_HEAD_RE = re.compile(r"([a-zA-Z0-9_.-]+)\s*(?=\{)")
_V11_FN_PARTIAL_RE = re.compile(r"^[a-zA-Z0-9_.-]+\s*$")


def _is_complete_json_safe(text: str) -> bool:
    """Return ``is_complete_json(text)``, treating ``RecursionError`` as incomplete."""
    try:
        return is_complete_json(text)
    except RecursionError:
        return False


def _trailing_prefix_len(text: str, marker: str) -> int:
    """Return length of the longest proper prefix of *marker* matching the end of *text*."""
    max_len = min(len(text), len(marker) - 1)
    for size in range(max_len, 0, -1):
        if text.endswith(marker[:size]):
            return size
    return 0


class MistralToolCall(ToolCall):
    """A specialized ToolCall class for Mistral models with auto-generated 9-char alphanumeric IDs."""

    id: str = Field(default_factory=lambda: MistralToolCall.generate_random_id())

    @staticmethod
    def generate_random_id() -> str:
        """Generate a random 9-character alphanumeric ID."""
        return "".join(choices(ALPHANUMERIC, k=9))

    @staticmethod
    def is_valid_id(id: str) -> bool:  # noqa
        """Validate whether a string is a valid 9-character alphanumeric Mistral tool call ID."""
        return id.isalnum() and len(id) == 9


def _is_fn_name_regex_support(model_tokenizer: AnyTokenizer) -> bool:
    """Check if the tokenizer is a MistralTokenizer with version >= 11."""
    return bool(MistralTokenizer and isinstance(model_tokenizer, MistralTokenizer) and model_tokenizer.version >= 11)


@ToolParserManager.register_module("mistral")  # pyright: ignore[reportUntypedClassDecorator]
class MistralToolParser(ToolParser):
    """Tool call parser for Mistral models (7B Instruct v0.3+).

    Supported Formats:
        - Standard JSON array: ``[TOOL_CALLS][{"name": "func", "arguments": {...}}]``
        - Multiple calls: ``[TOOL_CALLS][{"name": "func1", ...}, {"name": "func2", ...}]``
        - v11+ format: ``[TOOL_CALLS]function_name{...}`` (including arbitrarily nested objects)
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the Mistral tool parser.

        Raises:
            RuntimeError: If the ``[TOOL_CALLS]`` token cannot be found in vocab.
        """
        super().__init__(tokenizer)

        self.bot_token = "[TOOL_CALLS]"
        self.bot_token_id = self.vocab.get(self.bot_token)
        self.tool_call_regex = re.compile(r"\[[\s\S]*\]", re.DOTALL)
        if _is_fn_name_regex_support(self.model_tokenizer):
            self.fn_name_regex = _V11_FN_HEAD_RE
        else:
            self.fn_name_regex = None

        if self.bot_token_id is None:
            raise RuntimeError("Mistral Tool Parser could not locate the tool call token in the tokenizer!")

        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state for a new response."""
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
        self._streaming_initialized = False
        self._stream_content_sent = 0
        self._accumulated_text = ""
        self._buffering_tool_call = False
        self._in_tool_block = False
        self._tool_block_start = -1
        self._tool_payload_completed = False
        self._emitted_tool_call = False

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Return True only while inside an unfinished ``[TOOL_CALLS]`` payload or prefix."""
        del delta_text
        if self._streaming_initialized:
            if self._buffering_tool_call:
                return True
            return self._has_trailing_marker_prefix(current_text[self._stream_content_sent :], (self.bot_token,))

        bot_idx = current_text.find(self.bot_token)
        if bot_idx != -1:
            after = current_text[bot_idx + len(self.bot_token) :].lstrip()
            if not after:
                return True
            if after.startswith("["):
                try:
                    json.JSONDecoder().raw_decode(after, 0)
                    return False
                except RecursionError:
                    return False
                except json.JSONDecodeError:
                    try:
                        partial_json_loads(after, Allow.ALL)
                        return True
                    except (
                        ValueError,
                        partial_json_parser.core.exceptions.MalformedJSON,
                        AssertionError,
                        IndexError,
                        RecursionError,
                    ):
                        return False
                except ValueError:
                    return False
            if _V11_FN_HEAD_RE.match(after) is not None:
                v11 = self._parse_v11_calls_streaming(after, Allow.ALL)
                if v11 is None:
                    return True
                calls, complete_flags = v11
                return not (bool(calls) and all(complete_flags))
            if _V11_FN_PARTIAL_RE.fullmatch(after) is not None:
                return True
            return False

        return self._has_trailing_marker_prefix(current_text, (self.bot_token,))

    def flush_streaming_content(self) -> str:
        """Release withheld opening-tag prefix when the stream ends without a confirmed tool call."""
        if self._emitted_tool_call or self._in_tool_block:
            self._accumulated_text = ""
            self._stream_content_sent = 0
            self._buffering_tool_call = False
            return ""
        pending = self._accumulated_text[self._stream_content_sent :]
        self._accumulated_text = ""
        self._stream_content_sent = 0
        self._buffering_tool_call = False
        return pending

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Ensure special tokens are preserved for non-Mistral tokenizers (including HF tokenizers)."""
        if (
            (MistralTokenizer is None or not isinstance(self.model_tokenizer, MistralTokenizer))
            and request.tools
            and request.tool_choice != "none"
        ):
            request.skip_special_tokens = False
        return request

    def _parse_v11_calls_batch(self, text: str) -> list[dict]:
        """Parse one or more v11+ ``fn_name{...}`` calls with brace/string-aware ``raw_decode``."""
        decoder = json.JSONDecoder()
        calls: list[dict] = []
        pos = 0
        n = len(text)
        while pos < n:
            pos = consume_space(pos, text)
            if pos >= n:
                break
            if text[pos] == ",":
                pos += 1
                continue
            if text.startswith(self.bot_token, pos):
                pos += len(self.bot_token)
                continue
            match = _V11_FN_HEAD_RE.match(text, pos)
            if match is None:
                break
            fn_name = match.group(1)
            json_start = match.end()
            try:
                args_obj, next_pos = decoder.raw_decode(text, json_start)
            except (json.JSONDecodeError, ValueError, RecursionError):
                break
            calls.append({"name": fn_name, "arguments": args_obj})
            pos = next_pos
        return calls

    def _parse_v11_calls_streaming(self, text: str, flags: Allow) -> tuple[list[dict], list[bool]] | None:
        """Parse streaming v11+ ``fn_name{...}`` calls incrementally."""
        decoder = json.JSONDecoder()
        calls: list[dict] = []
        complete_flags: list[bool] = []
        pos = 0
        n = len(text)
        while pos < n:
            pos = consume_space(pos, text)
            if pos >= n:
                break
            if text[pos] == ",":
                pos += 1
                continue
            if text.startswith(self.bot_token, pos):
                pos += len(self.bot_token)
                continue
            match = _V11_FN_HEAD_RE.match(text, pos)
            if match is None:
                # Could still be typing function name before '{'.
                if not calls:
                    return None
                break
            fn_name = match.group(1)
            json_start = match.end()
            try:
                args_obj, next_pos = decoder.raw_decode(text, json_start)
                calls.append({"name": fn_name, "arguments": args_obj})
                complete_flags.append(True)
                pos = next_pos
            except RecursionError:
                break
            except json.JSONDecodeError:
                tail = text[json_start:]
                try:
                    args_obj, _ = partial_json_loads(tail, flags)
                except (
                    ValueError,
                    partial_json_parser.core.exceptions.MalformedJSON,
                    AssertionError,
                    IndexError,
                    RecursionError,
                ):
                    calls.append({"name": fn_name})
                    complete_flags.append(False)
                    break
                calls.append({"name": fn_name, "arguments": args_obj})
                complete_flags.append(_is_complete_json_safe(tail.strip()))
                break
            except ValueError:
                break
        return calls, complete_flags

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete Mistral model response."""
        del request
        if self.bot_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        content, after_bot = model_output.split(self.bot_token, 1)
        tool_content = after_bot.strip()

        try:
            function_call_arr: list[dict] = []
            if tool_content.startswith("["):
                decoded, _ = json.JSONDecoder().raw_decode(tool_content, 0)
                if isinstance(decoded, list):
                    function_call_arr = decoded
            elif self.fn_name_regex is not None or _V11_FN_HEAD_RE.match(tool_content) is not None:
                function_call_arr = self._parse_v11_calls_batch(tool_content)
            else:
                bracket_idx = tool_content.find("[")
                if bracket_idx != -1:
                    decoded, _ = json.JSONDecoder().raw_decode(tool_content, bracket_idx)
                    if isinstance(decoded, list):
                        function_call_arr = decoded

            tool_calls: list[MistralToolCall] = []
            for raw_function_call in function_call_arr:
                if not isinstance(raw_function_call, dict):
                    continue
                name = raw_function_call.get("name")
                if not isinstance(name, str) or not name:
                    continue
                raw_args = raw_function_call.get("arguments", {})
                args_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
                tool_calls.append(
                    MistralToolCall(
                        type="function",
                        function=FunctionCall(name=name, arguments=args_str),
                    )
                )

            if not tool_calls:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if len(content) > 0 else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    @staticmethod
    def _serialize_args(arguments: object) -> str:
        return arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)

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
        del delta_text, previous_token_ids, current_token_ids, request
        if not previous_text:
            self._reset_streaming_state()
            self._streaming_initialized = True
        elif not self._streaming_initialized:
            self._reset_streaming_state()
            self._stream_content_sent = len(previous_text)
            self._streaming_initialized = True

        self._accumulated_text = current_text

        # O(1) short-circuit once a JSON-array payload has completed.
        if self._tool_payload_completed:
            self._buffering_tool_call = False
            self._stream_content_sent = len(current_text)
            return None

        content_parts: list[str] = []
        deltas: list[DeltaToolCall] = []

        while True:
            if not self._in_tool_block:
                bot_idx = current_text.find(self.bot_token, self._stream_content_sent)
                if bot_idx == -1:
                    rem = current_text[self._stream_content_sent :]
                    hold = _trailing_prefix_len(rem, self.bot_token)
                    safe_end = len(current_text) - hold
                    if safe_end > self._stream_content_sent:
                        content_parts.append(current_text[self._stream_content_sent : safe_end])
                        self._stream_content_sent = safe_end
                    self._buffering_tool_call = hold > 0
                    break

                if bot_idx > self._stream_content_sent:
                    content_parts.append(current_text[self._stream_content_sent : bot_idx])
                    self._stream_content_sent = bot_idx

                self._in_tool_block = True
                self._tool_block_start = bot_idx
                self._buffering_tool_call = True

                if self.bot_token_id is not None and self.bot_token_id in delta_token_ids and len(delta_token_ids) == 1:
                    break

            bot_idx = self._tool_block_start
            try:
                parsable_arr = current_text[bot_idx + len(self.bot_token) :]
                stripped_arr = parsable_arr.lstrip()
                leading_ws = len(parsable_arr) - len(stripped_arr)
                if not stripped_arr:
                    self._buffering_tool_call = True
                    break

                tool_call_arr: list[dict]
                safe_call_arr: list[dict]
                per_call_complete: list[bool]
                arr_complete = False
                is_array_payload = stripped_arr.startswith("[")

                if is_array_payload:
                    try:
                        raw_parsed, consumed = partial_json_loads(stripped_arr, Allow.ALL)
                    except (
                        partial_json_parser.core.exceptions.MalformedJSON,
                        ValueError,
                        AssertionError,
                        IndexError,
                        RecursionError,
                    ):
                        logger.debug("Malformed or unsupported Mistral JSON array payload")
                        if not self._emitted_tool_call:
                            emit_end = bot_idx + len(self.bot_token)
                            content_parts.append(current_text[self._stream_content_sent : emit_end])
                            self._stream_content_sent = emit_end
                            self._in_tool_block = False
                            self._tool_block_start = -1
                            self._buffering_tool_call = False
                            continue
                        self._buffering_tool_call = False
                        break
                    if not isinstance(raw_parsed, list):
                        if not self._emitted_tool_call:
                            emit_end = bot_idx + len(self.bot_token)
                            content_parts.append(current_text[self._stream_content_sent : emit_end])
                            self._stream_content_sent = emit_end
                            self._in_tool_block = False
                            self._tool_block_start = -1
                            self._buffering_tool_call = False
                            continue
                        self._buffering_tool_call = False
                        break
                    json_slice = stripped_arr[:consumed]
                    try:
                        raw_safe, _ = partial_json_loads(json_slice, Allow.ALL & ~Allow.STR)
                        if not isinstance(raw_safe, list):
                            raw_safe = []
                    except (
                        partial_json_parser.core.exceptions.MalformedJSON,
                        ValueError,
                        AssertionError,
                        IndexError,
                        RecursionError,
                    ):
                        raw_safe = []

                    arr_complete = _is_complete_json_safe(json_slice)
                    last_raw_idx = len(raw_parsed) - 1
                    tool_call_arr = []
                    safe_call_arr = []
                    per_call_complete = []
                    for idx, item in enumerate(raw_parsed):
                        if not isinstance(item, dict):
                            continue
                        safe_item = raw_safe[idx] if idx < len(raw_safe) and isinstance(raw_safe[idx], dict) else {}
                        item_complete = True if idx < last_raw_idx else arr_complete
                        if item_complete:
                            item_name = item.get("name")
                            if not isinstance(item_name, str) or not item_name:
                                continue
                        tool_call_arr.append(item)
                        safe_call_arr.append(safe_item)
                        per_call_complete.append(item_complete)

                    if arr_complete and len(tool_call_arr) == 0:
                        if not self._emitted_tool_call:
                            emit_end = bot_idx + len(self.bot_token) + leading_ws + consumed
                            content_parts.append(current_text[self._stream_content_sent : emit_end])
                            self._stream_content_sent = emit_end
                            self._in_tool_block = False
                            self._tool_block_start = -1
                            self._buffering_tool_call = False
                            continue
                        self._tool_payload_completed = True
                        self._in_tool_block = False
                        self._buffering_tool_call = False
                        break
                elif _V11_FN_HEAD_RE.match(stripped_arr) is not None:
                    v11 = self._parse_v11_calls_streaming(stripped_arr, Allow.ALL)
                    if v11 is None:
                        self._buffering_tool_call = True
                        break
                    tool_call_arr, per_call_complete = v11
                    safe_call_arr = tool_call_arr
                elif _V11_FN_PARTIAL_RE.fullmatch(stripped_arr) is not None:
                    self._buffering_tool_call = True
                    break
                else:
                    if not self._emitted_tool_call:
                        emit_end = bot_idx + len(self.bot_token)
                        content_parts.append(current_text[self._stream_content_sent : emit_end])
                        self._stream_content_sent = emit_end
                        self._in_tool_block = False
                        self._tool_block_start = -1
                        self._buffering_tool_call = False
                        continue
                    self._buffering_tool_call = False
                    break

                if len(tool_call_arr) == 0:
                    self._buffering_tool_call = True
                    break

                if len(tool_call_arr) > self.current_tool_id + 1:
                    if self.current_tool_id >= 0:
                        prev_call = tool_call_arr[self.current_tool_id]
                        closing_name = prev_call.get("name")
                        announce = not self.current_tool_name_sent and bool(closing_name)
                        closing_arguments = prev_call.get("arguments")
                        remaining = ""
                        if closing_arguments is not None and (self.current_tool_name_sent or announce):
                            closing_args_json = self._serialize_args(closing_arguments)
                            already_streamed = self.streamed_args_for_tool[self.current_tool_id]
                            remaining = (
                                closing_args_json[len(already_streamed) :]
                                if closing_args_json.startswith(already_streamed)
                                else ""
                            )
                        if remaining or announce:
                            deltas.append(
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    type="function" if announce else None,
                                    id=MistralToolCall.generate_random_id() if announce else None,
                                    function=DeltaFunctionCall(
                                        name=closing_name if announce else None,
                                        arguments=remaining or None,
                                    ).model_dump(exclude_none=True),
                                )
                            )
                            self.streamed_args_for_tool[self.current_tool_id] += remaining

                    while self.current_tool_id < len(tool_call_arr) - 1:
                        self.current_tool_id += 1
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")
                        if self.current_tool_id >= len(tool_call_arr) - 1:
                            break
                        closed_tool_call = tool_call_arr[self.current_tool_id]
                        closed_name = closed_tool_call.get("name")
                        closed_arguments = closed_tool_call.get("arguments")
                        closed_args_json = None
                        if closed_arguments is not None:
                            closed_args_json = self._serialize_args(closed_arguments)
                            self.streamed_args_for_tool[self.current_tool_id] = closed_args_json
                        deltas.append(
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=MistralToolCall.generate_random_id(),
                                function=DeltaFunctionCall(
                                    name=closed_name,
                                    arguments=closed_args_json,
                                ).model_dump(exclude_none=True),
                            )
                        )

                    self.current_tool_name_sent = False

                current_tool_call = tool_call_arr[self.current_tool_id]
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                announced_now = False
                if not self.current_tool_name_sent:
                    name_source = (
                        current_tool_call
                        if per_call_complete[self.current_tool_id]
                        else safe_call_arr[self.current_tool_id]
                    )
                    raw_name = name_source.get("name")
                    function_name = raw_name if isinstance(raw_name, str) and raw_name else None
                    if function_name:
                        deltas.append(
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=MistralToolCall.generate_random_id(),
                                function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True),
                            )
                        )
                        self.current_tool_name_sent = True
                        announced_now = True

                if self.current_tool_name_sent:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments is not None:
                        cur_args_json = self._serialize_args(cur_arguments)
                        is_cur_complete = per_call_complete[self.current_tool_id]
                        safe_json = cur_args_json if is_cur_complete else stable_json_stream_prefix(cur_args_json)
                        sent = self.streamed_args_for_tool[self.current_tool_id]
                        arguments_delta = safe_json[len(sent) :] if safe_json.startswith(sent) else ""
                        if arguments_delta:
                            self.streamed_args_for_tool[self.current_tool_id] += arguments_delta
                            if announced_now and deltas and deltas[-1].index == self.current_tool_id:
                                fn_dict = deltas[-1].function
                                if isinstance(fn_dict, dict):
                                    fn_dict["arguments"] = arguments_delta
                                elif fn_dict is not None:
                                    fn_dict.arguments = arguments_delta
                            else:
                                deltas.append(
                                    DeltaToolCall(
                                        index=self.current_tool_id,
                                        function=DeltaFunctionCall(arguments=arguments_delta).model_dump(
                                            exclude_none=True
                                        ),
                                    )
                                )

                if deltas:
                    self._emitted_tool_call = True
                self.prev_tool_call_arr = tool_call_arr
                if is_array_payload and arr_complete:
                    self._tool_payload_completed = True
                    self._in_tool_block = False
                    self._buffering_tool_call = False
                    self._stream_content_sent = len(current_text)
                elif not is_array_payload:
                    self._buffering_tool_call = not all(per_call_complete)
                else:
                    self._buffering_tool_call = True
                break

            except Exception:
                logger.exception("Error trying to handle streaming tool call.")
                logger.debug("Skipping chunk as a result of tool streaming extraction error")
                self._buffering_tool_call = False
                break

        content = "".join(content_parts)
        if content or deltas:
            return DeltaMessage(content=content or None, tool_calls=deltas or None)
        return None
