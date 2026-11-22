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

"""DeepSeek V3 tool parser module for parsing tool calls from DeepSeek V3 model outputs.

This module provides the DeepSeekV3ToolParser class which handles the specific
tool call format used by DeepSeek V3 models. The format uses custom Unicode
delimiters for tool calls and arguments, with JSON-formatted argument blocks.

Example tool call format:
    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
    ```json
    {"location": "Beijing"}
    ```<｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>
"""

from __future__ import annotations

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
from ..utils import stable_json_stream_prefix

logger = get_logger(__name__)


def _strip_trailing_partial_marker(text: str, marker: str) -> str:
    """Withhold any trailing non-empty proper prefix of *marker* from *text*."""
    max_len = min(len(marker) - 1, len(text))
    for k in range(max_len, 0, -1):
        if text.endswith(marker[:k]):
            return text[:-k]
    return text


def _has_open_marker_or_trailing_prefix(text: str, start_token: str, end_token: str) -> bool:
    """Return True if *text* has an unclosed outer section or ends with a partial start marker."""
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


_TRAILING_FENCE_RE = re.compile(r"\n```[ \t\r\n]*$")


def _strip_open_v3_args_suffix(raw_args: str, end_token: str) -> str:
    """Strip any partial closing code fence / end-token suffix from open V3 args."""
    cleaned = _strip_trailing_partial_marker(raw_args, end_token)
    cleaned = _TRAILING_FENCE_RE.sub("", cleaned)
    cleaned = _strip_trailing_partial_marker(cleaned, "\n```")
    return cleaned


@ToolParserManager.register_module("deepseek_v3")  # pyright: ignore[reportUntypedClassDecorator]
class DeepSeekV3ToolParser(ToolParser):
    """Tool parser for DeepSeek V3 models.

    This parser handles the specific tool call format used by DeepSeek V3 models,
    which uses custom delimiters for tool calls and arguments. The format includes:
    - Tool calls wrapped in <｜tool▁calls▁begin｜> and <｜tool▁calls▁end｜>
    - Individual calls wrapped in <｜tool▁call▁begin｜> and <｜tool▁call▁end｜>
    - Tool type and name separated by <｜tool▁sep｜>
    - Arguments in JSON format wrapped in ```json blocks

    Attributes:
        current_tool_name_sent (bool): Tracks if tool name has been sent in streaming.
        prev_tool_call_arr (list): Previous tool calls for comparison in streaming.
        current_tool_id (int): Index of current tool being processed.
        streamed_args_for_tool (list): Arguments streamed so far for each tool.
        tool_calls_start_token (str): Token marking start of tool calls section.
        tool_calls_end_token (str): Token marking end of tool calls section.
        tool_call_start_token (str): Token marking start of individual tool call.
        tool_call_end_token (str): Token marking end of individual tool call.
        tool_call_regex (re.Pattern): Regex pattern for parsing complete tool calls.
        stream_tool_call_portion_regex (re.Pattern): Regex for parsing streaming portions.
        stream_tool_call_name_regex (re.Pattern): Regex for extracting tool names in streaming.
        tool_calls_start_token_id (int): Token ID for tool calls start marker.
        tool_calls_end_token_id (int): Token ID for tool calls end marker.
        tool_call_start_token_id (int): Token ID for individual tool call start marker.
        tool_call_end_token_id (int): Token ID for individual tool call end marker.

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-v3")
        >>> parser = DeepSeekV3ToolParser(tokenizer)
        >>> result = parser.extract_tool_calls(model_output, request)
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the DeepSeek V3 tool parser.

        Sets up the token markers, regex patterns, and token IDs required for
        parsing DeepSeek V3 tool call format.

        Args:
            tokenizer: The tokenizer associated with the DeepSeek V3 model.
                Must contain the special tool call tokens in its vocabulary.

        Raises:
            ValueError: If the tokenizer is not provided.
            RuntimeError: If the tool call start/end tokens cannot be found
                in the tokenizer vocabulary.
        """
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] | None = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []
        self._tool_name_sent: list[bool] = []
        self._tool_call_ids: list[str] = []
        self._emitted_content_pos: int | None = None
        self._pending_tail_start: int | None = None
        self._last_current_text: str = ""
        self._streaming_initialized: bool = False
        self._in_open_section: bool = False
        self._scan_pos: int = 0
        self._closed_span_count: int = 0

        self.tool_calls_start_token: str = "<｜tool▁calls▁begin｜>"
        self.tool_calls_end_token: str = "<｜tool▁calls▁end｜>"

        self.tool_call_start_token: str = "<｜tool▁call▁begin｜>"
        self.tool_call_end_token: str = "<｜tool▁call▁end｜>"

        self.tool_call_regex = re.compile(
            r"<｜tool▁call▁begin｜>(?P<type>[^<\n]*)<｜tool▁sep｜>(?P<function_name>[^\n]*)\n```json\n(?P<function_arguments>.*?)\n```<｜tool▁call▁end｜>",
            re.DOTALL,
        )

        self._closed_call_body_regex = re.compile(
            r"^(?P<type>[^<\n]*)<｜tool▁sep｜>(?P<function_name>[^\n]*)\n```json\n(?P<function_arguments>.*?)\n```\s*$",
            re.DOTALL,
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"(?P<type>[^<\n]*)<｜tool▁sep｜>(?P<function_name>[^\n]*)\n```json\n(?P<function_arguments>.*)",
            re.DOTALL,
        )

        self.stream_tool_call_name_regex = re.compile(r"(?P<type>[^<\n]*)<｜tool▁sep｜>(?P<function_name>[^\n]*)\n")

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")
        self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_calls_start_token_id is None or self.tool_calls_end_token_id is None:
            raise RuntimeError("DeepSeek-V3 Tool parser could not locate tool call start/end tokens in the tokenizer!")
        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            logger.debug(
                "DeepSeek-V3 Tool parser could not locate per-call start/end token IDs; using string markers."
            )

    def get_streaming_buffer_marker_pairs(self) -> Sequence[tuple[str, str | None]]:
        """Scope protocol buffering strictly to outer ``<｜tool▁calls▁begin｜>`` sections."""
        return ((self.tool_calls_start_token, self.tool_calls_end_token),)

    def is_buffering_protocol(
        self,
        *,
        current_text: str,
        delta_text: str,
    ) -> bool:
        """Report whether an outer tool-calls section or partial start marker is buffered."""
        del delta_text
        if self._streaming_initialized:
            return self._in_open_section or self._pending_tail_start is not None
        return _has_open_marker_or_trailing_prefix(
            current_text,
            self.tool_calls_start_token,
            self.tool_calls_end_token,
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
        self._streaming_initialized = False
        self._in_open_section = False
        self._scan_pos = 0
        self._closed_span_count = 0

    def flush_streaming_content(self) -> str | None:
        """Flush any trailing partial marker prefix withheld at end-of-stream."""
        flushed: str | None = None
        if self._pending_tail_start is not None and self._pending_tail_start < len(self._last_current_text):
            flushed = self._last_current_text[self._pending_tail_start :] or None
            self._emitted_content_pos = len(self._last_current_text)
            self._pending_tail_start = None
        self._last_current_text = ""
        return flushed

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output.

        Parses the DeepSeek V3 format to extract function names and arguments
        from the model's response. Handles the specific token structure with
        tool type, separator, and JSON argument blocks.

        Args:
            model_output: Complete text output from the model containing
                potential tool calls in DeepSeek V3 format.
            request: Original chat completion request with tool definitions.
                Used for context but not directly accessed in parsing.

        Returns:
            ExtractedToolCallInformation: Contains the following fields:
                - tools_called (bool): True if any tools were invoked.
                - tool_calls (list[ToolCall]): List of parsed tool calls with
                  function names and JSON argument strings.
                - content (str | None): Text content before tool calls, or
                  None if no content precedes the tool calls.

        Example:
            >>> output = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\\n```json\\n{\"city\": \"NYC\"}\\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
            >>> result = parser.extract_tool_calls(output, request)
            >>> result.tools_called
            True
            >>> result.tool_calls[0].function.name
            'get_weather'
        """
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        else:
            try:
                tool_calls: list[ToolCall] = []
                sep_token = "<｜tool▁sep｜>"
                fence_open = "\n```json\n"
                fence_close = "\n```"
                start_len = len(self.tool_call_start_token)
                end_len = len(self.tool_call_end_token)
                pos = 0
                n = len(model_output)
                cached_end = -1
                while pos < n:
                    call_start = model_output.find(self.tool_call_start_token, pos)
                    if call_start == -1:
                        break
                    body_start = call_start + start_len
                    if cached_end < body_start:
                        cached_end = model_output.find(self.tool_call_end_token, body_start)
                    if cached_end == -1:
                        break
                    body = model_output[body_start:cached_end]
                    sep_idx = body.find(sep_token)
                    matched = False
                    if sep_idx != -1:
                        tool_type = body[:sep_idx]
                        if "<" not in tool_type and "\n" not in tool_type:
                            rest_start = sep_idx + len(sep_token)
                            fence_idx = body.find(fence_open, rest_start)
                            if fence_idx != -1:
                                function_name = body[rest_start:fence_idx]
                                if "\n" not in function_name and body.endswith(fence_close):
                                    args_start = fence_idx + len(fence_open)
                                    args_end = len(body) - len(fence_close)
                                    if args_start <= args_end:
                                        function_args = body[args_start:args_end]
                                        tool_calls.append(
                                            ToolCall(
                                                type=tool_type,
                                                function=FunctionCall(name=function_name, arguments=function_args),
                                            )
                                        )
                                        matched = True
                    if matched:
                        pos = cached_end + end_len
                    else:
                        next_start = model_output.find(self.tool_call_start_token, body_start, cached_end)
                        pos = next_start if next_start != -1 else cached_end + end_len

                if not tool_calls:
                    return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
                content = model_output[: model_output.find(self.tool_calls_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            except Exception:
                logger.exception("Error in extracting tool call from response.")
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
        """Extract tool calls from streaming model output.

        Handles incremental parsing of DeepSeek V3 tool call format during
        streaming generation. Maintains state across chunks to properly
        identify tool boundaries and stream arguments progressively.

        This method tracks the state of tool call parsing including:
        - When new tool calls begin (start token detection)
        - Tool names as they become available
        - Arguments as they are incrementally generated
        - When tool calls complete (end token detection)

        Args:
            previous_text: Text generated up to the previous chunk.
            current_text: All text generated so far including the current chunk.
            delta_text: New text added in this chunk only.
            previous_token_ids: Sequence of token IDs up to the previous chunk.
            current_token_ids: Sequence of all token IDs generated so far.
            delta_token_ids: Sequence of new token IDs in this chunk.
            request: Original chat completion request with tool definitions.

        Returns:
            DeltaMessage | None: A delta message containing incremental tool
                call information (tool name, argument fragments), content text,
                or None if more data is needed before emitting a message.

        Note:
            This method modifies internal state including current_tool_id,
            current_tool_name_sent, prev_tool_call_arr, and streamed_args_for_tool
            to track progress across streaming chunks.
        """
        logger.debug("delta_text: %s", delta_text)
        logger.debug("delta_token_ids: %s", delta_token_ids)
        if not previous_text:
            self._reset_streaming_state(0)
        elif self._emitted_content_pos is None:
            self._reset_streaming_state(len(previous_text))
        elif self._scan_pos > len(current_text):
            self._reset_streaming_state(0)

        self._streaming_initialized = True
        self._last_current_text = current_text
        self._pending_tail_start = None
        self._in_open_section = False

        try:
            pos = self._scan_pos
            next_scan_pos = pos
            next_closed_span_count = self._closed_span_count
            spans: list[tuple[str, bool]] = []
            visible_parts: list[tuple[int, int]] = []
            while pos < len(current_text):
                sec_start = current_text.find(self.tool_calls_start_token, pos)
                if sec_start == -1:
                    tail = current_text[pos:]
                    safe_tail = _strip_trailing_partial_marker(tail, self.tool_calls_start_token)
                    safe_end = pos + len(safe_tail)
                    if safe_tail:
                        visible_parts.append((pos, safe_end))
                    if len(safe_tail) < len(tail):
                        self._pending_tail_start = safe_end
                    next_scan_pos = safe_end
                    break
                if sec_start > pos:
                    visible_parts.append((pos, sec_start))
                sec_body_start = sec_start + len(self.tool_calls_start_token)
                sec_end = current_text.find(self.tool_calls_end_token, sec_body_start)
                scan_limit = sec_end if sec_end != -1 else len(current_text)

                call_pos = sec_body_start
                sec_span_count = 0
                while call_pos < scan_limit:
                    call_start = current_text.find(self.tool_call_start_token, call_pos, scan_limit)
                    if call_start == -1:
                        break
                    body_start = call_start + len(self.tool_call_start_token)
                    call_end = current_text.find(self.tool_call_end_token, body_start, scan_limit)
                    if call_end == -1:
                        spans.append((current_text[body_start:scan_limit], False))
                        sec_span_count += 1
                        break
                    spans.append((current_text[body_start:call_end], True))
                    sec_span_count += 1
                    call_pos = call_end + len(self.tool_call_end_token)

                if sec_end == -1:
                    self._in_open_section = True
                    next_scan_pos = sec_start
                    break
                pos = sec_end + len(self.tool_calls_end_token)
                next_scan_pos = pos
                next_closed_span_count += sec_span_count

            content_chunks: list[str] = []
            emitted_pos = self._emitted_content_pos or 0
            for seg_start, seg_end in visible_parts:
                if seg_end > emitted_pos:
                    content_chunks.append(current_text[max(seg_start, emitted_pos) : seg_end])
                    emitted_pos = seg_end
            self._emitted_content_pos = emitted_pos
            content_out = "".join(content_chunks)

            if self.prev_tool_call_arr is None:
                self.prev_tool_call_arr = []

            tool_deltas: list[DeltaToolCall] = []
            base_idx = self._closed_span_count
            for offset, (body, is_closed) in enumerate(spans):
                idx = base_idx + offset
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

                func_name: str | None = None
                target_args: str | None = None
                if is_closed:
                    closed_match = self._closed_call_body_regex.match(body)
                    if not closed_match:
                        continue
                    func_name = closed_match.group("function_name")
                    target_args = closed_match.group("function_arguments")
                else:
                    portion_match = self.stream_tool_call_portion_regex.match(body)
                    if portion_match:
                        func_name = portion_match.group("function_name")
                        raw_args = _strip_open_v3_args_suffix(
                            portion_match.group("function_arguments"),
                            self.tool_call_end_token,
                        )
                        target_args = stable_json_stream_prefix(raw_args)
                    else:
                        name_match = self.stream_tool_call_name_regex.match(body)
                        if not name_match:
                            continue
                        func_name = name_match.group("function_name")
                        target_args = ""

                if not func_name:
                    continue

                self.prev_tool_call_arr[idx] = {"name": func_name, "arguments": target_args}

                first = not self._tool_name_sent[idx]
                if first:
                    self._tool_call_ids[idx] = f"chatcmpl-tool-{uuid4()}"
                    self._tool_name_sent[idx] = True
                    if idx == self.current_tool_id:
                        self.current_tool_name_sent = True

                streamed = self.streamed_args_for_tool[idx]
                diff = ""
                if target_args is not None and target_args.startswith(streamed):
                    diff = target_args[len(streamed) :]
                if diff:
                    self.streamed_args_for_tool[idx] = streamed + diff

                if first or diff:
                    tool_deltas.append(
                        DeltaToolCall(
                            index=idx,
                            id=self._tool_call_ids[idx] if first else None,
                            type="function" if first else None,
                            function=DeltaFunctionCall(
                                name=func_name if first else None,
                                arguments=diff if diff else None,
                            ).model_dump(exclude_none=True),
                        )
                    )

            self._scan_pos = next_scan_pos
            self._closed_span_count = next_closed_span_count

            if content_out and tool_deltas:
                return DeltaMessage(content=content_out, tool_calls=tool_deltas)
            if tool_deltas:
                return DeltaMessage(tool_calls=tool_deltas)
            if content_out:
                return DeltaMessage(content=content_out)
            return None

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None
